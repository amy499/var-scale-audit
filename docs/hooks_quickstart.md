# Hooks quickstart

A hook is a Python function the runner calls at every sampling stage, so you can read (observe) or
change (modify) the model's state mid-generation. The full design is in `docs/hook_interface.md`; this
page is what you need to write one.

## The function

```python
def my_hook(model, stage, p, when, state):
    ...
```

| Argument | VAR | DiT |
|---|---|---|
| `model` | `"var"` | `"dit"` |
| `stage` | scale index `si`, 0..9 | the step's original timestep `timestep_in`, 999..0 |
| `p` | progress in [0, 1]: 0 at the first stage, 1 at the last | same |
| `when` | `"before"` or `"after"` the stage | same |
| `state` | `VARScaleState` | `DiTStepState` |

Every state has `state.rows` (`(class_id, seed)` per batch row) and `state.fields` (per-stage
numbers: VAR `pn`, `n_tokens`, `cum_tokens`, `total_tokens`; DiT `timestep_in/out`,
`alpha_bar_in/out`, `sigma_in/out`, `step`, `merged`).

| Tensor | VAR `before` | VAR `after` | DiT `before` | DiT `after` |
|---|---|---|---|---|
| modifiable | `h_BChw` (this scale's embedding), `f_hat` (accumulator without this scale) | `f_hat` (with this scale) | `x` (latent x_t) | `x` (latent x_{t-1}) |
| read-only | `idx_Bl` (tokens sampled at this scale) | `h_BChw`, `idx_Bl`, `f_hat_before` | — | `x_before` (x_t), `pred_xstart` |

Tensors are batch-first. DiT's `x` is the conditional half only (one row per image); the runner
handles the CFG copies.

## An observe hook

Returns `None`. It gets copies, so it cannot change anything. This one logs the accumulator's norm at
every VAR scale:

```python
# my_hooks.py
import json
from runner.hooks import Observe

def log_fhat_norm(model, stage, p, when, state):
    norms = state.f_hat.float().flatten(1).norm(dim=1).tolist()   # one value per row
    with open("fhat_norms.jsonl", "a") as f:
        f.write(json.dumps({"stage": stage, "p": p, "rows": state.rows, "norm": norms}) + "\n")

fhat_norm = Observe(log_fhat_norm, when="after", name="fhat_norm")
```

## A modify hook

Returns a state of the same class, usually `dataclasses.replace(state, <field>=new_tensor)`.
Replaced tensors must keep their shape, dtype and device, and read-only fields must stay untouched,
or the run stops with an error. This one removes scale 2's contribution for every image (VAR
"skip-scale", version b):

```python
import dataclasses
from runner.hooks import Modify

def drop_scale(model, stage, p, when, state):
    return dataclasses.replace(state, f_hat=state.f_hat_before)

skip_scale_2 = Modify(drop_scale, when="after", stages=[2], name="skip_scale_2")
```

`stages=[...]` limits a hook to those stages (default: every stage). A stage the run does not have is
an error, never silently ignored. To change only some images, check `state.rows` and edit those rows
(see `scripts/phase3/hooks_lib.py` for a row-0-only example).

## Running

```bash
python -m runner.generate --config configs/var_d20.yaml --manifest manifest/provisional_4x4.csv \
    --batch-size 16 --out-dir outputs/my_exp --hook my_hooks.py:skip_scale_2 --hook my_hooks.py:fhat_norm
```

`--hook FILE.py:NAME` is repeatable, and `NAME` may be a list of hooks. At one stage, modify hooks
run in the order given and observe hooks then see the result. `run.json` records every hook (name,
kind, when, stages) and each image's generator state. DiT's "skip a timestep" is a flag, not a hook:
`--skip-timesteps 497` (it must be one of the run's timesteps, listed as `stage` in `run.json`'s
`stages` table; the first step, 999, cannot be skipped).

Always also make a **baseline run with no hooks**, using the same config, manifest and batch size,
and compare against it.

## Frozen settings (do not change)

- **Configs:** `configs/var_d20.yaml` and `configs/dit_xl2_256.yaml` exactly as committed:
  - VAR: cfg 1.5, top_k 900, top_p 0.96, `more_smooth: false`, **strict fp32** (autocast off, TF32
    off). Don't use `configs/var_d20_fp16.yaml`: it exists only for the historical fp16 jobs and a timing
    comparison.
  - VAR depth: **d20 is the default and primary**. The other 256px depths have their own configs,
    identical except the depth and checkpoint:

    | Config | Depth | Params | Checkpoint | Status |
    |---|---|---|---|---|
    | `configs/var_d20.yaml` | 20 | 600M | 2.4 GB | default, primary |
    | `configs/var_d24.yaml` | 24 | 1.0B | 4.1 GB | in use |
    | `configs/var_d16.yaml` | 16 | 310M | 1.2 GB | not in use for now |
    | `configs/var_d30.yaml` | 30 | 2.0B | 8.0 GB | not in use for now |

    Every depth has the same 10 scales and 680 tokens, so hooks work unchanged. **Never mix depths
    within a comparison**: the baseline and the hooked run use the same config. `run.json` records
    the `depth`.
  - DiT: cfg_scale 1.5, 250 steps, fp32 with TF32.
  - Both: SDPA `math`, `cudnn_deterministic: true`.
- **Batch size 16.** Compare only runs made at the same batch size: a different batch size changes
  images slightly by float rounding, which can fake or hide an effect. The manifest size must be a
  multiple of 16. Don't use `--allow-partial-batch` for experiments.
- **Same manifest, same seeds** for the baseline and the hooked run. Each image's randomness comes
  only from its own seed, so the two runs stay paired.
- **Never touch the random numbers.** Hooks are not given the image generators; don't try to reach
  them. If a hook needs randomness, create its own `torch.Generator` with its own seed. Don't change
  tensor sizes (e.g. dropping tokens or rows): mask or replace values instead.
- **Don't keep references** to tensors you return; the runner may update them in place later.
- **Inside the sampler:** hooks run under `inference_mode` (VAR) / `no_grad` (DiT), so no gradients,
  and under whatever autocast the config sets (none for either model now). Keep hooks cheap: DiT calls
  them 250 times per side per batch.
- **Don't edit** `third_party/`.

## Where your code goes

Put your own code in lanes/pN/; don't edit runner/. If you need a change there, ask P1.
