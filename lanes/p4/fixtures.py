"""P4: reference `stages` tables for CPU checks, built without torch, a GPU or a checkpoint.

Run from the repo root. Standard library plus PyYAML (what `runner/config.py` itself needs).

The authoritative stages table is the one `runner/generate.py` writes into `run.json`. This module
reproduces the *baseline* table of a given schedule so the frozen band lists (lanes/p4/bands.py) and
lanes/p4/check_p4_shared.py can be checked on the laptop, where no 256px checkpoint is loaded.
`check_p4_shared.py` asserts this reproduction against a real `run.json` whenever one is given, so a
drift from upstream is caught rather than assumed away.

    python -m lanes.p4.fixtures var     # the configs/var_d20.yaml table
    python -m lanes.p4.fixtures dit     # the configs/dit_xl2_256.yaml table
"""

import sys
from pathlib import Path

sys.dont_write_bytecode = True

REPO = Path(__file__).resolve().parents[2]
CONFIGS = {"var": REPO / "configs" / "var_d20.yaml", "dit": REPO / "configs" / "dit_xl2_256.yaml"}

# Upstream DiT: diffusion/gaussian_diffusion.py get_named_beta_schedule("linear", N).
BETA_START, BETA_END, BETA_REFERENCE_STEPS = 1e-4, 0.02, 1000
ORIGINAL_STEPS = 1000   # the trained schedule DiT respaces down to num_sampling_steps


def frozen_config(model: str) -> dict:
    """The frozen config for `model`, so fixtures cannot drift from configs/ (docs/HANDOVER.md section 3)."""
    import yaml
    if model not in CONFIGS:
        raise ValueError(f"model must be one of {tuple(CONFIGS)}, got {model!r}")
    return yaml.safe_load(CONFIGS[model].read_text())


# ---------------------------------------------------------------- VAR

def var_stages(patch_nums) -> list[dict]:
    """The VAR stages table for `patch_nums`: the fields runner.hooks.var_stage_table writes."""
    patch_nums = list(patch_nums)
    if not patch_nums:
        raise ValueError("patch_nums is empty")
    sn = len(patch_nums)
    total = sum(pn * pn for pn in patch_nums)
    table, cum = [], 0
    for si, pn in enumerate(patch_nums):
        cum += pn * pn
        table.append({"stage": si, "p": si / (sn - 1) if sn > 1 else 1.0, "pn": pn, "n_tokens": pn * pn,
                      "cum_tokens": cum, "total_tokens": total})
    return table


# ---------------------------------------------------------------- DiT

def linear_betas(original_steps: int = ORIGINAL_STEPS) -> list[float]:
    """get_named_beta_schedule("linear", N), as np.linspace computes it (i*step + start, last = stop)."""
    scale = BETA_REFERENCE_STEPS / original_steps
    start, stop = scale * BETA_START, scale * BETA_END
    step = (stop - start) / (original_steps - 1)
    betas = [i * step + start for i in range(original_steps)]
    betas[-1] = stop
    return betas


def alphas_cumprod(original_steps: int = ORIGINAL_STEPS) -> list[float]:
    out, running = [], 1.0
    for beta in linear_betas(original_steps):
        running *= 1.0 - beta
        out.append(running)
    return out


def timestep_map(num_sampling_steps: int, original_steps: int = ORIGINAL_STEPS) -> list[int]:
    """Upstream diffusion/respace.py space_timesteps(original, [count]), increasing (one section)."""
    if num_sampling_steps < 1:
        raise ValueError(f"num_sampling_steps must be >= 1, got {num_sampling_steps}")
    if num_sampling_steps > original_steps:
        raise ValueError(f"cannot respace {original_steps} steps up to {num_sampling_steps}")
    if num_sampling_steps == 1:
        return [0]
    stride = (original_steps - 1) / (num_sampling_steps - 1)
    return sorted({round(k * stride) for k in range(num_sampling_steps)})


def spaced_alphas_cumprod(tmap, ab) -> list[float]:
    """The alphas_cumprod of upstream's SpacedDiffusion over `tmap`, indexed by respaced index.

    diffusion/respace.py does not index the base schedule: it derives one new beta per kept timestep
    from consecutive kept alpha_cumprod values, and GaussianDiffusion then recomputes
    alphas_cumprod = cumprod(1 - new_betas). That is mathematically alphas_cumprod[tmap[i]] but rounds
    differently (docs/hook_interface.md section 11b), so reproducing the arithmetic -- not the
    shortcut -- is what makes this table bit-identical to the runner's.
    """
    new_betas, last = [], 1.0
    for t in tmap:
        new_betas.append(1.0 - ab[t] / last)
        last = ab[t]
    out, running = [], 1.0
    for beta in new_betas:
        running *= 1.0 - beta
        out.append(running)
    return out


def dit_stages(num_sampling_steps: int, original_steps: int = ORIGINAL_STEPS) -> list[dict]:
    """The DiT baseline stages table: the fields runner.dit_model.DiTModel.stage_table writes.

    Baseline only -- no skipped timesteps. A skip run's table is whatever the runner wrote; this is
    the reference the frozen bands are cut from, and bands are cut on the baseline enumeration.
    """
    tmap = timestep_map(num_sampling_steps, original_steps)
    spaced = spaced_alphas_cumprod(tmap, alphas_cumprod(original_steps))
    steps = len(tmap)
    table = []
    for j, i in enumerate(range(steps - 1, -1, -1)):
        ab_in = spaced[i]
        ab_out = spaced[i - 1] if i > 0 else 1.0
        table.append({"stage": tmap[i], "p": j / (steps - 1) if steps > 1 else 1.0, "step": j, "merged": False,
                      "timestep_in": tmap[i], "timestep_out": int(tmap[i - 1]) if i > 0 else None,
                      "alpha_bar_in": ab_in, "alpha_bar_out": ab_out,
                      "sigma_in": (1.0 - ab_in) ** 0.5, "sigma_out": (1.0 - ab_out) ** 0.5})
    return table


# ---------------------------------------------------------------- the frozen schedules

def frozen_stages(model: str) -> list[dict]:
    """The stages table of the frozen config: VAR-d20's 10 scales, DiT's 250 steps."""
    cfg = frozen_config(model)
    if model == "var":
        return var_stages(cfg["build"]["patch_nums"])
    return dit_stages(cfg["sampler"]["num_sampling_steps"])


def stages_for_config(config_path) -> tuple[str, list[dict]]:
    """(model, baseline stages table) for any run config, so a driver can plan arms without a GPU."""
    import yaml
    cfg = yaml.safe_load(Path(config_path).read_text())
    model = cfg.get("model")
    if model == "var":
        return model, var_stages(cfg["build"]["patch_nums"])
    if model == "dit":
        return model, dit_stages(cfg["sampler"]["num_sampling_steps"])
    raise ValueError(f"{config_path}: model must be 'var' or 'dit', got {model!r}")


def run_record(model: str, stages: list[dict], rows=((207, 0),), **extra) -> dict:
    """A minimal run record shaped like run.json, for checks that read one through load_run()."""
    return {"model": model, "stages": stages,
            "rows": [{"class_id": c, "seed": s, "file": f"class{c:04d}_seed{s}",
                      "tensor_sha256": "0" * 64, "generator_state_sha256": "0" * 64} for c, s in rows],
            **extra}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 1 or argv[0] not in CONFIGS:
        print(__doc__.strip(), file=sys.stderr)
        return 2
    model = argv[0]
    stages = frozen_stages(model)
    print(f"{model}: {len(stages)} stages from {CONFIGS[model].relative_to(REPO)}")
    for row in stages:
        print(row)
    return 0


if __name__ == "__main__":
    sys.exit(main())
