"""DiT wrapper: builds, loads and samples via unmodified third_party/DiT code."""

import os

import numpy as np
import torch

from runner import upstream
from runner.config import resolve_path
from runner.hooks import DiTStepState, Hook, HookError, check_hooks, has_point, run_point
from runner.runtime import autocast_ctx, row_generators, sdpa_ctx, sdpa_record

# Schedule arrays GaussianDiffusion.p_mean_variance / q_posterior_mean_variance read for a step
# (LEARNED_RANGE variance, EPSILON mean): the coefficients a step actually uses.
COEFFICIENT_ARRAYS = ("betas", "alphas_cumprod", "alphas_cumprod_prev", "posterior_variance",
                      "posterior_log_variance_clipped", "posterior_mean_coef1", "posterior_mean_coef2",
                      "sqrt_recip_alphas_cumprod", "sqrt_recipm1_alphas_cumprod")


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
        from diffusion import gaussian_diffusion
        from diffusion.respace import SpacedDiffusion
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

        # skip_timesteps builds more SpacedDiffusion objects the way create_diffusion does; check that
        # doing so for the baseline timesteps reproduces self.diffusion exactly.
        self._gd, self._SpacedDiffusion = gaussian_diffusion, SpacedDiffusion
        rebuilt = self._spaced(self.diffusion.use_timesteps)
        if rebuilt.timestep_map != self.diffusion.timestep_map or any(
                not np.array_equal(getattr(rebuilt, a), getattr(self.diffusion, a)) for a in COEFFICIENT_ARRAYS):
            raise RuntimeError("rebuilding the DiT schedule does not reproduce create_diffusion; upstream changed")

    def _attention_record(self) -> dict:
        from timm.models.vision_transformer import Attention
        attns = [m for m in self.model.modules() if isinstance(m, Attention)]
        fused = sum(a.fused_attn for a in attns)
        return {
            "impl": "timm Attention -> torch SDPA" if fused == len(attns) else "timm Attention (manual matmul/softmax)",
            "blocks": len(attns),
            "blocks_fused_attn": fused,
        }

    def _spaced(self, use_timesteps):
        """A SpacedDiffusion over `use_timesteps`, with create_diffusion's defaults (diffusion/__init__.py)."""
        d = self.diffusion
        return self._SpacedDiffusion(
            use_timesteps=use_timesteps,
            betas=self._gd.get_named_beta_schedule("linear", d.original_num_steps),
            model_mean_type=d.model_mean_type, model_var_type=d.model_var_type, loss_type=d.loss_type)

    def plan(self, skip_timesteps=()) -> list[dict]:
        """The steps of one batch in sampling order (j = 0 .. S-1 of the baseline schedule).

        Each entry: step (j), skipped, and for executed steps diffusion + index (the schedule object
        and respaced index whose coefficients the step uses) and fields (per-stage fields, §3).
        Every executed step uses the baseline schedule, except the step just before a skipped window
        (docs/hook_interface.md §11b): it maps its latent straight to the level after the window,
        with the coefficients of a schedule that lacks only that window.
        """
        base = self.diffusion
        S = base.num_timesteps
        tmap = base.timestep_map                      # respaced index -> original timestep, increasing
        skip = set(skip_timesteps)
        bad = sorted(skip - set(tmap))
        if bad:
            raise ValueError(f"skip_timesteps {bad} are not timesteps of this {S}-step schedule")
        if tmap[-1] in skip:
            raise ValueError(f"cannot skip the first step (timestep {tmap[-1]}): there is no earlier step to merge into")
        plan = []
        for j, i in enumerate(range(S - 1, -1, -1)):
            tau = tmap[i]
            if tau in skip:
                plan.append({"step": j, "skipped": True, "timestep": tau})
                continue
            d, idx, merged = base, i, False
            if i > 0 and tmap[i - 1] in skip:         # next step is skipped: merge up to the next kept timestep
                window, m = set(), i - 1
                while m >= 0 and tmap[m] in skip:
                    window.add(tmap[m])
                    m -= 1
                d = self._spaced(set(tmap) - window)
                idx, merged = d.timestep_map.index(tau), True
            ab_in, ab_out = float(d.alphas_cumprod[idx]), float(d.alphas_cumprod_prev[idx])
            fields = {"stage": tau, "p": j / (S - 1) if S > 1 else 1.0, "step": j, "merged": merged,
                      "timestep_in": tau, "timestep_out": int(d.timestep_map[idx - 1]) if idx > 0 else None,
                      "alpha_bar_in": ab_in, "alpha_bar_out": ab_out,
                      "sigma_in": (1.0 - ab_in) ** 0.5, "sigma_out": (1.0 - ab_out) ** 0.5}
            plan.append({"step": j, "skipped": False, "diffusion": d, "index": idx, "fields": fields})
        return plan

    def stage_table(self, skip_timesteps=()) -> list[dict]:
        """Per-stage fields (docs/hook_interface.md §3) of every executed step, in sampling order.

        Read from the schedule's numpy arrays only; independent of hooks and of the run's outputs.
        """
        return [s["fields"] for s in self.plan(skip_timesteps) if not s["skipped"]]

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

    def _p_sample_loop_hooked(self, sample_fn, z, model_kwargs, gens, using_cfg, plan, hooks, rows, burn):
        """_p_sample_loop with hooks around each step and skipped steps (docs/hook_interface.md §1, §8, §11b).

        Only used when hooks or skip_timesteps are given; without them sample() runs _p_sample_loop.
        Per executed step, the same operations as _p_sample_loop, using the step's schedule object
        from plan(). Hooks see and replace the conditional half img[:n] only; img[n:] (the CFG copies,
        never read by forward_with_cfg) stays as computed. A skipped step makes no model call but
        still draws (and discards) its noise, so every row makes the same draws as in the baseline.
        """
        k = 2 if using_cfg else 1
        n = len(gens)
        img = z
        for step in plan:
            if step["skipped"]:
                if burn:
                    for g in gens:
                        torch.randn((k, *img.shape[1:]), generator=g, device=self.device)
                continue
            d, fields = step["diffusion"], step["fields"]
            stage, p = fields["stage"], fields["p"]
            if has_point(hooks, "before", stage):
                state, modified = run_point(hooks, "dit", stage, p, "before", DiTStepState(rows, fields, x=img[:n]))
                if modified:
                    img = torch.cat([state.x, img[n:]])
            x_before = img[:n]
            t = torch.tensor([step["index"]] * img.shape[0], device=self.device)
            out = d.p_mean_variance(sample_fn, img, t, clip_denoised=False, model_kwargs=model_kwargs)
            draws = [torch.randn((k, *img.shape[1:]), generator=g, device=self.device) for g in gens]
            noise = torch.cat([dr[j:j + 1] for j in range(k) for dr in draws])
            nonzero_mask = (t != 0).float().view(-1, *([1] * (len(img.shape) - 1)))  # no noise when t == 0
            img = out["mean"] + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise
            if has_point(hooks, "after", stage):
                state = DiTStepState(rows, fields, x=img[:n], x_before=x_before, pred_xstart=out["pred_xstart"][:n])
                state, modified = run_point(hooks, "dit", stage, p, "after", state)
                if modified:
                    img = torch.cat([state.x, img[n:]])
        return img

    def sample(self, class_ids: list[int], seeds: list[int], gen_states: list | None = None, hooks=(),
               skip_timesteps=(), _burn_skipped: bool = True):
        """One batch; row i is generated from seeds[i] only.

        Returns (BxHxWx3 uint8 images, raw decoded tensor on CPU, attention record).
        Mirrors the per-batch body of DiT sample_ddp.py (the FID script), with per-row noise.
        gen_states: if a list, each row's generator state (Generator.get_state()) after the batch is
          appended. Read after sampling; does not touch it.
        hooks: Observe/Modify hooks (runner/hooks.py), stage = timestep_in (original timestep).
        skip_timesteps: original timesteps whose model evaluation is skipped (§11b); not the first step.
        _burn_skipped: test only. False drops the skipped steps' draws, which unpairs the run (§6 test 6).
        With no hooks and no skips, the loop is exactly _p_sample_loop.
        """
        if len(class_ids) != len(seeds):
            raise ValueError("class_ids and seeds must have the same length")
        hooks, skip_timesteps = tuple(hooks), tuple(skip_timesteps)
        plan = self.plan(skip_timesteps) if (hooks or skip_timesteps) else None
        if plan is not None:
            skipped = {s["timestep"] for s in plan if s["skipped"]}
            for h in hooks:
                on_skipped = sorted(set(h.stages or ()) & skipped) if isinstance(h, Hook) else []
                if on_skipped:
                    raise HookError(f"hook {h.name!r} ({h.when}) registered for skipped timesteps {on_skipped}; "
                                    "it would never fire")
            hooks = check_hooks(hooks, [s["fields"]["stage"] for s in plan if not s["skipped"]], "dit")
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
            if plan is None:
                samples = self._p_sample_loop(sample_fn, z, model_kwargs, gens, using_cfg)
            else:
                samples = self._p_sample_loop_hooked(sample_fn, z, model_kwargs, gens, using_cfg, plan, hooks,
                                                     tuple(zip(class_ids, seeds)), _burn_skipped)
            if using_cfg:
                samples, _ = samples.chunk(2, dim=0)
            samples = self.vae.decode(samples / 0.18215).sample
        if gen_states is not None:
            gen_states.extend(g.get_state() for g in gens)
        imgs = torch.clamp(127.5 * samples + 128.0, 0, 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()
        return imgs, samples.cpu(), {**self.attention, "sdpa_kernel": sdpa}
