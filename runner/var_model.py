"""VAR wrapper: builds, loads and samples via unmodified third_party/VAR code."""

import contextlib

import numpy as np
import torch
import torch.nn.functional as F

from runner import upstream
from runner.config import ConfigError, require_checkpoints, resolve_path
from runner.hooks import VARScaleState, check_hooks, has_point, run_point, var_stage_table
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
        from models.quant import VectorQuantizer2
        upstream.check_origin(models, "var")
        # The helpers _per_row_rng swaps must be the ones var.py imported from helpers.py.
        for name in self._SAMPLING_HELPERS:
            if getattr(var_module, name) is not getattr(var_helpers, name):
                raise RuntimeError(f"models.var.{name} is not models.helpers.{name}; upstream changed")
        self._var_module = var_module

        self.cfg = cfg
        require_checkpoints(cfg)
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
        var_sd = torch.load(resolve_path(ckpts["var"]["path"]), map_location="cpu")
        self.depth_record = self._check_depth(var_sd, cfg["build"]["depth"], ckpts["var"]["path"])
        self.var.load_state_dict(var_sd, strict=True)
        del var_sd
        for m in (self.vae, self.var):
            m.eval()
            m.requires_grad_(False)

        self.attention = self._verify_attention(basic_var)

        # The quantizer _hooked_quantizer shadows must be the one var.py calls, with upstream's method.
        quant = self.vae.quantize
        if self.var.vae_quant_proxy[0] is not quant or type(quant) is not VectorQuantizer2 \
                or "get_next_autoregressive_input" in vars(quant):
            raise RuntimeError("var.vae_quant_proxy[0] is not the upstream VectorQuantizer2 of self.vae; upstream changed")
        self.patch_nums = tuple(build["patch_nums"])

    def _check_depth(self, sd: dict, depth: int, path) -> dict:
        """Fail clearly if the VAR checkpoint does not have the config's depth.

        Checkpoint depth = its number of transformer blocks; width = word_embed's output size
        (build_vae_var uses width = 64 * depth). Both must equal the model built from the config.
        load_state_dict(strict=True) afterwards still checks every other parameter.
        """
        blocks = {int(k.split(".")[1]) for k in sd if k.startswith("blocks.")}
        ck_blocks = max(blocks) + 1 if blocks else 0
        w = sd.get("word_embed.weight")
        ck_width = int(w.shape[0]) if w is not None else None
        n_blocks, width = len(self.var.blocks), int(self.var.word_embed.weight.shape[0])
        if blocks != set(range(ck_blocks)) or ck_blocks != n_blocks or ck_width != width:
            raise ConfigError(
                f"{self.cfg['_path']} says VAR depth {depth} ({n_blocks} blocks, width {width}), but the checkpoint "
                f"{path} has {ck_blocks} blocks and width {ck_width}"
                + (f", i.e. it is a VAR-d{ck_blocks} checkpoint" if ck_width == 64 * ck_blocks else "")
                + ". Use the config that matches the checkpoint (configs/var_d<depth>.yaml).")
        return {"config_depth": depth, "checkpoint_blocks": ck_blocks, "checkpoint_width": ck_width}

    def stage_table(self) -> list[dict]:
        """Per-stage fields (docs/hook_interface.md §3); independent of hooks and of the run's outputs."""
        return var_stage_table(self.patch_nums)

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
    def _per_row_rng(self, gens: list[torch.Generator], capture: list | None = None, idx_slot: dict | None = None):
        """Make each image's token sampling use only its own generator.

        Upstream autoregressive_infer_cfg seeds one generator (self.rng, via g_seed) and samples
        all B*l tokens of a scale in one torch.multinomial call, so a row's draws depend on B and
        on the rows before it. While this is active, the two sampling helpers that models/var.py
        imported are replaced (in that module's namespace only; no file changes) by wrappers that
        call the unmodified helper once per row with that row's generator. At B=1 this is the
        same call upstream makes with g_seed=seed. The transformer forward stays batched.
        idx_slot: if a dict (hooks on), its "idx" is set to each scale's sampled ids (B, l) for the hook state.
        """
        mod = self._var_module
        originals = {name: getattr(mod, name) for name in self._SAMPLING_HELPERS}

        def per_row(fn, record, slot):
            def wrapped(logits_BlV, *args, rng=None, **kwargs):
                if rng is not None:
                    raise RuntimeError("upstream passed its shared rng; call with g_seed=None")
                if logits_BlV.shape[0] != len(gens):
                    raise RuntimeError(f"batch {logits_BlV.shape[0]} != {len(gens)} generators")
                if record is not None:
                    # Read-only: a new tensor, computed before the helper masks logits_BlV in place.
                    probs = logits_BlV.softmax(dim=-1)
                # Row slices are views, so in-place top-k/top-p masking still reaches logits_BlV.
                out = torch.cat([fn(logits_BlV[i:i + 1], *args, rng=g, **kwargs) for i, g in enumerate(gens)])
                if record is not None:
                    top2_p, top2_i = probs.topk(2, dim=-1)
                    record.append({"idx": out[..., 0].cpu(), "top2_p": top2_p.float().cpu(), "top2_i": top2_i.cpu(),
                                   "p_chosen": probs.gather(-1, out[..., :1])[..., 0].float().cpu()})
                if slot is not None:
                    slot["idx"] = out[..., 0]
                return out
            return wrapped

        for name, fn in originals.items():
            is_sampler = name == "sample_with_top_k_top_p_"
            setattr(mod, name, per_row(fn, capture if is_sampler else None, idx_slot if is_sampler else None))
        try:
            yield
        finally:
            for name, fn in originals.items():
                setattr(mod, name, fn)

    @contextlib.contextmanager
    def _hooked_quantizer(self, hooks: tuple, rows: tuple, idx_slot: dict):
        """Run hooks around each scale's accumulation (docs/hook_interface.md §1, §8). Only used with hooks.

        Shadows get_next_autoregressive_input on the quantizer instance var.py calls (instance
        attribute, removed on exit; the class and file are untouched). All random draws of a scale
        happen before this call, in _per_row_rng's wrappers, so nothing here can reach a generator.
        """
        quant = self.vae.quantize
        original = quant.get_next_autoregressive_input   # bound upstream method
        table = self.stage_table()

        def wrapped(si, SN, f_hat, h_BChw):
            if SN != len(table):
                raise RuntimeError(f"upstream SN={SN}, runner patch_nums has {len(table)} scales")
            fields = table[si]
            p = fields["p"]
            state = VARScaleState(rows, fields, h_BChw=h_BChw, f_hat=f_hat, idx_Bl=idx_slot.pop("idx"))
            if has_point(hooks, "before", si):
                state, _ = run_point(hooks, "var", si, p, "before", state)
                f_hat, h_BChw = state.f_hat, state.h_BChw
            if not has_point(hooks, "after", si):
                return original(si, SN, f_hat, h_BChw)
            f_hat_before = f_hat.clone()   # upstream adds this scale into f_hat in place
            f_hat, next_input = original(si, SN, f_hat, h_BChw)
            state = VARScaleState(rows, fields, h_BChw=h_BChw, f_hat=f_hat, idx_Bl=state.idx_Bl,
                                  f_hat_before=f_hat_before)
            state, modified = run_point(hooks, "var", si, p, "after", state)
            if not modified:
                return f_hat, next_input
            f_hat = state.f_hat   # recompute the next scale's input as models/quant.py:192 / :196 do
            if si == SN - 1:
                return f_hat, f_hat
            pn_next = quant.v_patch_nums[si + 1]
            return f_hat, F.interpolate(f_hat, size=(pn_next, pn_next), mode="area")

        quant.get_next_autoregressive_input = wrapped
        try:
            yield
        finally:
            del quant.get_next_autoregressive_input

    def sample(self, class_ids: list[int], seeds: list[int], capture: list | None = None, hooks=(),
               gen_states: list | None = None):
        """One batch; row i is generated from seeds[i] only.

        Returns (BxHxWx3 uint8 images, raw output tensor on CPU, attention record).
        capture: if a list, one dict per scale is appended (all CPU tensors, batch-first):
          idx (B, l) sampled token ids; top2_p / top2_i (B, l, 2) the two most likely tokens;
          p_chosen (B, l) probability of the sampled token. Probabilities are the model's post-CFG
          softmax before top-k/top-p filtering. Read-only: outputs are the same with or without it.
        hooks: Observe/Modify hooks (runner/hooks.py), stage = scale index si. With none, the sampling
          path is exactly the hook-free one (no quantizer override, no clones).
        gen_states: if a list, each row's generator state (Generator.get_state()) after the batch is
          appended. Read after sampling; does not touch it.
        """
        if len(class_ids) != len(seeds):
            raise ValueError("class_ids and seeds must have the same length")
        hooks = check_hooks(hooks, range(len(self.patch_nums)), "var")
        s = self.cfg["sampler"]
        label_B = torch.tensor(class_ids, device=self.device)
        backend = self.cfg["attention"]["sdpa_backend"]
        gens = row_generators(seeds, self.device)
        idx_slot = {} if hooks else None
        hooked = self._hooked_quantizer(hooks, tuple(zip(class_ids, seeds)), idx_slot) if hooks \
            else contextlib.nullcontext()
        with torch.inference_mode(), autocast_ctx(self.cfg["precision"], self.device), sdpa_ctx(backend, self.device), \
                self._per_row_rng(gens, capture, idx_slot), hooked:
            sdpa = sdpa_record(backend, self.device)
            img_B3HW = self.var.autoregressive_infer_cfg(
                B=len(class_ids), label_B=label_B, g_seed=None,   # randomness comes from gens (see _per_row_rng)
                cfg=s["cfg"], top_k=s["top_k"], top_p=s["top_p"], more_smooth=s["more_smooth"],
            )  # [0, 1], in the autocast dtype
        if gen_states is not None:
            gen_states.extend(g.get_state() for g in gens)
        # Exactly as demo_sample.ipynb: x*255 on device in the output dtype, then numpy astype(uint8) (truncation).
        imgs = img_B3HW.permute(0, 2, 3, 1).mul(255).cpu().numpy().astype(np.uint8)
        return imgs, img_B3HW.cpu(), {**self.attention, "sdpa_kernel": sdpa}
