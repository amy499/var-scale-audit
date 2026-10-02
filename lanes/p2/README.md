# P2: corruption and recovery

Question: where can VAR / DiT recover from a corrupted intermediate state, and where does damage
become irreversible? (Figure 2: stage × severity maps and recovery curves.)

| File | What |
|---|---|
| `hooks.py` | Corruption hooks (noise at chosen stages) and state-trace observers |
| `schema.py`, `SCHEMA.md` | Shared logging / output schema (P2's shared-work item): `experiment.json`, pairing check, merged tables |
| `analyze.py` | Image damage vs the baseline, and the summary tables behind Figure 2 |
| `jobs/pilot.pbs` | Pilot: baseline + stage × severity sweep on each model, recorded, merged and summarised |
| `check_p2.py` | CPU tests of all of the above with tiny random-weight models |

Nothing here edits `runner/`, `configs/`, `scripts/` or `jobs/`.

## Corruption hooks

Gaussian noise added to one tensor at the stage(s) in `P2_STAGE`, with strength `P2_SEVERITY`:

| Hook | Model | Point | Tensor | Meaning |
|---|---|---|---|---|
| `noise_h` | VAR | before scale k is accumulated | `h_BChw` (pn × pn) | Corrupt scale k's own contribution, at its own resolution |
| `noise_fhat` | VAR | after scale k is accumulated | `f_hat` (16 × 16) | Corrupt everything built so far |
| `noise_x` | DiT | after the step at timestep t | `x` | Corrupt the latent the next step starts from |

```bash
P2_STAGE=3 P2_SEVERITY=0.5 python -m runner.generate --config configs/var_d20.yaml \
    --manifest manifest/provisional_4x4.csv --batch-size 16 --out-dir outputs/p2/demo/noise_h_s3 \
    --hook lanes/p2/hooks.py:noise_h
```

- **`P2_STAGE`**: native stage id, or several separated by commas: VAR scale `0..9`, DiT `timestep_in`
  (the `stage` column of a DiT `run.json` `stages` table; an id the run does not have is an error).
- **`P2_SEVERITY`** with `P2_SCALE=rel` (default): noise std = severity × that image's own std of the
  tensor. The same rule for both models ("noise-to-signal ratio of the state"), which is the matched
  severity logic of the plan. `P2_SCALE=abs`: noise std = severity.
- **`P2_NOISE_SEED`** (default 0): another value gives an independent noise draw.
- The hook name in `run.json` carries the settings, e.g. `p2_noise_h_s3_sev0.5_rel`.

Properties (each one is a check in `check_p2.py`):

- Pairing is kept: every image's generator state equals the baseline's.
- An image's noise depends only on (class, seed, stage, tensor, noise seed): not on the batch, the
  device or the severity. A severity sweep therefore scales one fixed noise pattern.
- Severity 0 is bit-identical to the baseline; repeat runs are bit-identical.
- A hook on the wrong model, or without `P2_STAGE` / `P2_SEVERITY`, fails instead of doing nothing.

### Why two VAR hooks

`f_hat` is a running sum: later scales only add to it. Noise put into `f_hat` stays in what the decoder
finally sees unless later scales cancel it, and the transformer only sees `f_hat` averaged down to the
next scale's grid, so most i.i.d. 16 × 16 noise never reaches it. `noise_h` corrupts scale k's
contribution at scale k's own resolution, which then passes through the same upsampling as real
content. `noise_h` is the default for the pilot; `noise_fhat` is kept as the "corrupt the carried
state" variant that mirrors DiT's `noise_x`. Which one is the headline result is still open.

At the last stage (VAR scale 9, DiT timestep 0) nothing runs after the corruption, so that run is the
"no recovery possible" reference for the same severity.

## State traces (for recovery curves)

| Hook | Use on | Does |
|---|---|---|
| `save_states` | the baseline | Saves the state after each traced stage to `P2_TRACE_DIR` (VAR `f_hat`; DiT `x` and `pred_xstart`). Read-only: images are identical to a run with no hooks. |
| `compare_states` | a corrupted run | Compares its state with the baseline's in `P2_TRACE_REF` and appends, per image and stage, `state_l2:<tensor>` = ‖run − base‖, `state_rel_l2:<tensor>` = ‖run − base‖ / ‖base‖ and `state_cos:<tensor>` to the `metrics.csv` named by `P2_METRICS` |

`P2_TRACE_EVERY` (default 1) traces every n-th stage; use e.g. 25 for DiT's 250 steps. The last stage
is always traced. Because the runs are paired, the gap is exactly 0 before the injection stage; its
size at the injection stage and at the end is the recovery curve.

Size: VAR about 0.3 MB per image for a full baseline trace; DiT about 0.03 MB per image per traced
stage. Corrupted runs store only numbers.

Use `state_l2` (absolute) to judge recovery in VAR: `f_hat` grows as scales are added, so the relative
gap shrinks even when nothing is repaired.

## Analysis

```bash
python -m lanes.p2.analyze images <run dir> --baseline <baseline dir>   # per image, into the run's metrics.csv
python -m lanes.p2.schema collect <folder of runs>                      # merged tables in <folder>/tables
python -m lanes.p2.analyze summary <folder>/tables                      # p2_summary.csv, p2_curves.csv
```

| Output | Content |
|---|---|
| `img_mse`, `img_mad255` (in `metrics.csv`) | Pixel distance of each image from its baseline image, computed on the PNGs (same 0..255 scale for both models; the raw `.pt` tensors are not on a common scale) |
| `p2_summary.csv` | One row per corrupted run: `stage`, `p`, `severity`, mean `img_mse` / `img_psnr_db` / `img_mad255`, and the state gap right after injection (`gap_l2_injection`), at the end (`gap_l2_final`) and their ratio. This is the stage × severity damage map. |
| `p2_curves.csv` | Per run and traced stage, the mean of each `state_*` metric: the recovery curves |

`gap_l2_ratio` < 1 means the gap to the baseline shrank after the corruption, > 1 that it grew. These
are plain pixel and state distances for pilots; they say how much an image changed, not whether it
still looks right. That needs P3's calibrated metrics.

## Running

On the laptop (CPU, tiny models; tests mechanics only, about 5 minutes):

```bash
python lanes/p2/check_p2.py            # --skip-dit for the VAR and schema checks only
```

On NSCC (from the repo root, with your own project):

```bash
mkdir -p outputs/p2/pbs
qsub -P <your-project-id> lanes/p2/jobs/pilot.pbs
tail -f outputs/p2/pilot/<JOBID>/job.log      # ends with "PILOT DONE: all runs recorded and paired"
```

Defaults: 16 images (`manifest/provisional_4x4.csv`); VAR `noise_h` at scales 0, 3, 6, 9 and DiT
`noise_x` at timesteps 798, 497, 197, each at severities 0.25 and 1.0. The settings at the top of the
job file are environment variables, so they can be overridden at submit time, e.g.
`qsub -P <id> -v RUN_DIT=0 lanes/p2/jobs/pilot.pbs` (not yet tried on NSCC; lists such as
`VAR_STAGES` contain spaces and need PBS's quoting rules).
Outputs: `outputs/p2/pilot/<JOBID>/{var,dit}/<arm>/` (images, `run.json`, `experiment.json`,
`metrics.csv`) and `outputs/p2/pilot/<JOBID>/tables/*.csv`, including `p2_summary.csv`.

## Not done yet

- Perceptual / semantic metrics on the final images (P3's calibrated metrics; they go into `metrics.csv`).
- Severity schedule and stage grid for the full sweep (decide after the pilot).
- Plots of the maps and recovery curves (P4's plotting template; `var-dit` has no matplotlib).
- The final manifest (P3); everything so far uses `manifest/provisional_4x4.csv`.
