"""VAR wrapper: builds, loads and samples via unmodified third_party/VAR code."""

import contextlib

import numpy as np
import torch
import torch.nn.functional as F

from runner import upstream
from runner.config import ConfigError, resolve_path
from runner.runtime import autocast_ctx, row_generators, sdpa_ctx, sdpa_record

_FORCED_BUILD_KWARGS = ("flash_if_available", "fused_if_available")


class VARModel:
    name = "var"
    # Every function autoregressive_infer_cfg passes its rng to (models/var.py).
    _SAMPLING_HELPERS = ("sample_with_top_k_top_p_", "gumbel_softmax_with_rng")

    def __init__(self, cfg: dict):
        upstream.activate("var")
        import dist as var_dist
        import models
        from models import basic_var, build_vae_var
        from models import helpers as var_helpers
        from models import var as var_module
        upstream.check_origin(models, "var")
        # The helpers _per_row_rng swaps must be the ones var.py imported from helpers.py.
        for name in self._SAMPLING_HELPERS:
            if getattr(var_module, name) is not getattr(var_helpers, name):
                raise RuntimeError(f"models.var.{name} is not models.helpers.{name}; upstream changed")
        self._var_module = var_module

        self.cfg = cfg
        self.device = torch.device(cfg["device"])
        # VAR.__init__ creates its sampling Generator on dist.get_device() (cuda if available).
        if torch.device(var_dist.get_device()).type != self.device.type:
            raise ConfigError(f"VAR's rng device is {var_dist.get_device()}, config device is {self.device}")

        build = dict(cfg["build"])
        if any(k in build for k in _FORCED_BUILD_KWARGS):
            raise ConfigError(f"{_FORCED_BUILD_KWARGS} are forced off by the runner; remove them from [build]")
        build["patch_nums"] = tuple(build["patch_nums"])

        # flash/xformers/fused kernels off -> attention goes through torch SDPA.
        self.vae, self.var = build_vae_var(
            device=self.device, flash_if_available=False, fused_if_available=False, **build
        )
        ckpts = cfg["checkpoints"]
        self.vae.load_state_dict(torch.load(resolve_path(ckpts["vae"]["path"]), map_location="cpu"), strict=True)
        self.var.load_state_dict(torch.load(resolve_path(ckpts["var"]["path"]), map_location="cpu"), strict=True)
        for m in (self.vae, self.var):
            m.eval()
            m.requires_grad_(False)

        self.attention = self._verify_attention(basic_var)

    def _verify_attention(self, basic_var) -> dict:
        blocks = self.var.blocks
        rec = {
            "impl": "torch.nn.functional.scaled_dot_product_attention",
            "upstream_slow_attn_is_torch_sdpa": basic_var.slow_attn is F.scaled_dot_product_attention,
            "flash_attn_installed": basic_var.flash_attn_func is not None,
            "xformers_installed": basic_var.memory_efficient_attention is not None,
            "blocks": len(blocks),
            "blocks_using_flash": sum(b.attn.using_flash for b in blocks),
            "blocks_using_xformers": sum(b.attn.using_xform for b in blocks),
            "blocks_fused_mlp": sum(b.ffn.fused_mlp_func is not None for b in blocks),
            "blocks_fused_add_norm": sum(b.fused_add_norm_fn is not None for b in blocks),
        }
        non_sdpa = [k for k in ("blocks_using_flash", "blocks_using_xformers", "blocks_fused_mlp",
                                "blocks_fused_add_norm") if rec[k]]
        if not rec["upstream_slow_attn_is_torch_sdpa"] or non_sdpa:
            raise RuntimeError(f"VAR is not using torch SDPA only: {rec}")
        return rec

    @contextlib.contextmanager
    def _per_row_rng(self, gens: list[torch.Generator]):
        """Make each image's token sampling use only its own generator.

        Upstream autoregressive_infer_cfg seeds one generator (self.rng, via g_seed) and samples
        all B*l tokens of a scale in one torch.multinomial call, so a row's draws depend on B and
        on the rows before it. While this is active, the two sampling helpers that models/var.py
        imported are replaced (in that module's namespace only; no file changes) by wrappers that
        call the unmodified helper once per row with that row's generator. At B=1 this is the
        same call upstream makes with g_seed=seed. The transformer forward stays batched.
        """
        mod = self._var_module
        originals = {name: getattr(mod, name) for name in self._SAMPLING_HELPERS}

        def per_row(fn):
            def wrapped(logits_BlV, *args, rng=None, **kwargs):
                if rng is not None:
                    raise RuntimeError("upstream passed its shared rng; call with g_seed=None")
                if logits_BlV.shape[0] != len(gens):
                    raise RuntimeError(f"batch {logits_BlV.shape[0]} != {len(gens)} generators")
                # Row slices are views, so in-place top-k/top-p masking still reaches logits_BlV.
                return torch.cat([fn(logits_BlV[i:i + 1], *args, rng=g, **kwargs) for i, g in enumerate(gens)])
            return wrapped

        for name, fn in originals.items():
            setattr(mod, name, per_row(fn))
        try:
            yield
        finally:
            for name, fn in originals.items():
                setattr(mod, name, fn)

    def sample(self, class_ids: list[int], seeds: list[int]):
        """One batch; row i is generated from seeds[i] only.

        Returns (BxHxWx3 uint8 images, raw output tensor on CPU, attention record).
        """
        if len(class_ids) != len(seeds):
            raise ValueError("class_ids and seeds must have the same length")
        s = self.cfg["sampler"]
        label_B = torch.tensor(class_ids, device=self.device)
        backend = self.cfg["attention"]["sdpa_backend"]
        gens = row_generators(seeds, self.device)
        with torch.inference_mode(), autocast_ctx(self.cfg["precision"], self.device), sdpa_ctx(backend, self.device), \
                self._per_row_rng(gens):
            sdpa = sdpa_record(backend, self.device)
            img_B3HW = self.var.autoregressive_infer_cfg(
                B=len(class_ids), label_B=label_B, g_seed=None,   # randomness comes from gens (see _per_row_rng)
                cfg=s["cfg"], top_k=s["top_k"], top_p=s["top_p"], more_smooth=s["more_smooth"],
            )  # [0, 1], in the autocast dtype
        # Exactly as demo_sample.ipynb: x*255 on device in the output dtype, then numpy astype(uint8) (truncation).
        imgs = img_B3HW.permute(0, 2, 3, 1).mul(255).cpu().numpy().astype(np.uint8)
        return imgs, img_B3HW.cpu(), {**self.attention, "sdpa_kernel": sdpa}
