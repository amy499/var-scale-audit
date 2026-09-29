"""DiT wrapper: builds, loads and samples via unmodified third_party/DiT code."""

import os

import torch

from runner import upstream
from runner.config import resolve_path
from runner.runtime import autocast_ctx, row_generators, sdpa_ctx, sdpa_record


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

    def stage_table(self) -> list[dict]:
        """Per-stage fields (docs/hook_interface.md §3), one per step j in sampling order.

        Read from the schedule's numpy arrays only; independent of hooks and of the run's outputs.
        """
        d = self.diffusion
        n = d.num_timesteps
        table = []
        for j, i in enumerate(range(n - 1, -1, -1)):
            ab_in, ab_out = float(d.alphas_cumprod[i]), float(d.alphas_cumprod_prev[i])
            table.append({"stage": int(d.timestep_map[i]), "p": j / (n - 1) if n > 1 else 1.0, "step": j,
                          "timestep_in": int(d.timestep_map[i]),
                          "timestep_out": int(d.timestep_map[i - 1]) if i > 0 else None,
                          "alpha_bar_in": ab_in, "alpha_bar_out": ab_out,
                          "sigma_in": (1.0 - ab_in) ** 0.5, "sigma_out": (1.0 - ab_out) ** 0.5})
        return table

    def _p_sample_loop(self, sample_fn, z, model_kwargs, gens, using_cfg):
        """Upstream p_sample_loop (gaussian_diffusion.py, p_sample_loop_progressive + p_sample),
        except that each step's noise is drawn per row from that row's generator.

        Upstream draws one th.randn_like(x) per step from the global RNG for the whole batch.
        Here row i draws a tensor of the shape upstream would draw at B=1 ((2, C, H, W) with CFG,
        else (1, C, H, W)); with CFG, its [0] goes to row i and its [1] to row i's unconditional
        copy (n + i). At B=1 this consumes the generator exactly as upstream consumes the global
        RNG after torch.manual_seed(seed).
        """
        k = 2 if using_cfg else 1
        img = z
        for i in list(range(self.diffusion.num_timesteps))[::-1]:
            t = torch.tensor([i] * img.shape[0], device=self.device)
            out = self.diffusion.p_mean_variance(sample_fn, img, t, clip_denoised=False, model_kwargs=model_kwargs)
            draws = [torch.randn((k, *img.shape[1:]), generator=g, device=self.device) for g in gens]
            noise = torch.cat([d[j:j + 1] for j in range(k) for d in draws])
            nonzero_mask = (t != 0).float().view(-1, *([1] * (len(img.shape) - 1)))  # no noise when t == 0
            img = out["mean"] + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise
        return img

    def sample(self, class_ids: list[int], seeds: list[int], gen_states: list | None = None):
        """One batch; row i is generated from seeds[i] only.

        Returns (BxHxWx3 uint8 images, raw decoded tensor on CPU, attention record).
        Mirrors the per-batch body of DiT sample_ddp.py (the FID script), with per-row noise.
        gen_states: if a list, each row's generator state (Generator.get_state()) after the batch is
          appended. Read after sampling; does not touch it.
        """
        if len(class_ids) != len(seeds):
            raise ValueError("class_ids and seeds must have the same length")
        n = len(class_ids)
        cfg_scale = self.cfg["sampler"]["cfg_scale"]
        assert cfg_scale >= 1.0, "In almost all cases, cfg_scale be >= 1.0"  # as sample_ddp.py
        using_cfg = cfg_scale > 1.0
        backend = self.cfg["attention"]["sdpa_backend"]
        gens = row_generators(seeds, self.device)

        with torch.no_grad(), autocast_ctx(self.cfg["precision"], self.device), sdpa_ctx(backend, self.device):
            sdpa = sdpa_record(backend, self.device)
            shape = (self.model.in_channels, self.latent_size, self.latent_size)
            z = torch.cat([torch.randn((1, *shape), generator=g, device=self.device) for g in gens])
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
            samples = self._p_sample_loop(sample_fn, z, model_kwargs, gens, using_cfg)
            if using_cfg:
                samples, _ = samples.chunk(2, dim=0)
            samples = self.vae.decode(samples / 0.18215).sample
        if gen_states is not None:
            gen_states.extend(g.get_state() for g in gens)
        imgs = torch.clamp(127.5 * samples + 128.0, 0, 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()
        return imgs, samples.cpu(), {**self.attention, "sdpa_kernel": sdpa}
