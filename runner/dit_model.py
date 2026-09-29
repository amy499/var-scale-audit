"""DiT wrapper: builds, loads and samples via unmodified third_party/DiT code."""

import os

import torch

from runner import upstream
from runner.config import resolve_path
from runner.runtime import autocast_ctx, sdpa_ctx, sdpa_record


class DiTModel:
    name = "dit"

    def __init__(self, cfg: dict):
        upstream.activate("dit")
        os.environ["HF_HUB_OFFLINE"] = "1"  # compute nodes: load the VAE from checkpoints/ only
        import diffusion
        import download
        import models
        from diffusers.models import AutoencoderKL
        from diffusion import create_diffusion
        from download import find_model
        from models import DiT_models
        for mod in (models, diffusion, download):
            upstream.check_origin(mod, "dit")

        self.cfg = cfg
        self.device = torch.device(cfg["device"])
        b = cfg["build"]
        self.latent_size = b["image_size"] // 8
        self.model = DiT_models[b["model"]](input_size=self.latent_size, num_classes=b["num_classes"]).to(self.device)
        # Absolute path -> find_model takes its local-file branch (torch.load, unwraps "ema"), never downloads.
        self.model.load_state_dict(find_model(str(resolve_path(cfg["checkpoints"]["dit"]["path"]))))
        self.model.eval()
        self.num_classes = b["num_classes"]
        self.diffusion = create_diffusion(str(cfg["sampler"]["num_sampling_steps"]))
        self.vae = AutoencoderKL.from_pretrained(str(resolve_path(cfg["checkpoints"]["vae"]["path"]))).to(self.device)

        self.attention = self._attention_record()

    def _attention_record(self) -> dict:
        from timm.models.vision_transformer import Attention
        attns = [m for m in self.model.modules() if isinstance(m, Attention)]
        fused = sum(a.fused_attn for a in attns)
        return {
            "impl": "timm Attention -> torch SDPA" if fused == len(attns) else "timm Attention (manual matmul/softmax)",
            "blocks": len(attns),
            "blocks_fused_attn": fused,
        }

    def sample(self, class_ids: list[int], seed: int):
        """One batch (one seed for the whole batch).

        Returns (BxHxWx3 uint8 images, raw decoded tensor on CPU, attention record).
        Mirrors the per-batch body of DiT sample_ddp.py (the FID script).
        """
        n = len(class_ids)
        cfg_scale = self.cfg["sampler"]["cfg_scale"]
        assert cfg_scale >= 1.0, "In almost all cases, cfg_scale be >= 1.0"  # as sample_ddp.py
        using_cfg = cfg_scale > 1.0
        backend = self.cfg["attention"]["sdpa_backend"]
        torch.manual_seed(seed)

        with torch.no_grad(), autocast_ctx(self.cfg["precision"], self.device), sdpa_ctx(backend, self.device):
            sdpa = sdpa_record(backend, self.device)
            z = torch.randn(n, self.model.in_channels, self.latent_size, self.latent_size, device=self.device)
            y = torch.tensor(class_ids, device=self.device)
            if using_cfg:
                z = torch.cat([z, z], 0)
                y_null = torch.tensor([self.num_classes] * n, device=self.device)
                y = torch.cat([y, y_null], 0)
                model_kwargs = dict(y=y, cfg_scale=cfg_scale)
                sample_fn = self.model.forward_with_cfg
            else:
                model_kwargs = dict(y=y)
                sample_fn = self.model.forward
            samples = self.diffusion.p_sample_loop(
                sample_fn, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=False, device=self.device
            )
            if using_cfg:
                samples, _ = samples.chunk(2, dim=0)
            samples = self.vae.decode(samples / 0.18215).sample
        imgs = torch.clamp(127.5 * samples + 128.0, 0, 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()
        return imgs, samples.cpu(), {**self.attention, "sdpa_kernel": sdpa}
