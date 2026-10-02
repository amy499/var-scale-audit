# Logging and output schema (proposal, v1)

Owner: P2. Status: proposal for the team; implemented in `lanes/p2/schema.py`, tested on CPU by
`lanes/p2/check_p2.py`. Nothing in `runner/` changes.

**The idea in one line:** `run.json` stays the record of *what was generated*; a small
`experiment.json` next to it records *what the run was for*; and one command merges any number of
runs into four CSV tables that every lane and Figure 5 read.

```
<run dir>/
    class0207_seed0.png, .pt, ...   written by runner.generate (unchanged)
    run.json                        written by runner.generate (unchanged, never edited)
    experiment.json                 written by `schema record`   <- new
    metrics.csv                     written by lane / metric code <- new, optional
<any folder of runs>/tables/
    runs.csv  images.csv  stages.csv  metrics.csv    written by `schema collect`  <- new
```

Everything is standard-library Python and plain JSON/CSV: no new packages in `var-dit`, and it runs
on a login node.

## 1. What `run.json` already records, and what is missing

Already there (per run): model, config path, VAR depth and checkpoint check, sampler, build,
precision, torch/CUDA/GPU, submodule commits, manifest, batch size, hooks (name, kind, when, stages),
skipped timesteps, attention kernel, load and sampling seconds, the `stages` table (VAR: `pn`,
`n_tokens`, `cum_tokens`; DiT: timesteps, ᾱ, σ; both: `p`), and per image `tensor_sha256` and
`generator_state_sha256`.

Missing, and added by this schema:

| Missing | Where it goes |
|---|---|
| Which baseline a run is compared with, and proof that the two are paired | `experiment.json`: `baseline`, `pairing` |
| Intervention parameters (severity, target, ...): today only inside the hook's name string | `experiment.json`: `params` |
| Commit and branch of *this* repo (`run.json` has only the submodules), and whether it was clean | `experiment.json`: `code` |
| PBS job id, host | `experiment.json`: `job` |
| Seconds per image, peak GPU memory | `experiment.json`: `resources` |
| Per-image and per-stage numbers (metrics, hook measurements) | `metrics.csv` |
| One table over many runs | `schema collect` |

## 2. `experiment.json`

Written after a run by:

```bash
# the baseline
python -m lanes.p2.schema record <run dir> --lane p2 --experiment var_pilot_<JOBID> --arm baseline
# a run compared with it
python -m lanes.p2.schema record <run dir> --lane p2 --experiment var_pilot_<JOBID> --arm noise_h_s3_sev0.5 \
    --baseline <baseline run dir> --param hook=noise_h --param stage=3 --param severity=0.5 \
    [--vram-log <file>] [--note "..."]
```

| Field | Meaning |
|---|---|
| `schema_version` | `1` |
| `run_id` | `<lane>/<experiment>/<arm>`: the key used in every table. Must be unique; include the job id in `experiment`. |
| `lane`, `experiment`, `arm` | `experiment` groups one baseline with its interventions; `arm` names this run within it. |
| `role` | `baseline` (no `--baseline` given) or `intervention`. |
| `baseline` | `null` for a baseline. Otherwise `run_id`, relative `dir` and `run_json_sha256` of the baseline it is compared with. |
| `params` | Free-form intervention parameters from `--param KEY=VALUE` (numbers are stored as numbers). Each lane chooses its own keys. |
| `pairing` | Result of the pairing check (§3) against the baseline: `ok`, `errors`, `warnings`, `n_rows`, `rows_changed`, `rows_identical`. |
| `code` | `commit`, `branch`, `dirty` (true if the working tree had edited or untracked files, i.e. the commit alone does not reproduce the run), `dirty_files`. |
| `job` | `pbs_jobid`, `host`. |
| `resources` | `n_images`, `seconds_load`, `seconds_sample`, `seconds_per_image`, `peak_vram_mb` (max of an `nvidia-smi` log, if one was given). |
| `run_json_sha256` | Hash of the `run.json` it describes; `collect` refuses a record whose `run.json` changed afterwards. |
| `created_utc`, `note` | |

`record` exits 1 if the pairing check fails, so a job script notices.

## 3. Pairing check

`python -m lanes.p2.schema check <baseline dir> <run dir>` (also run by `record --baseline`). It turns
the rules of `docs/HANDOVER.md` §3 into a test that reads only the two `run.json` files:

| Error (comparison invalid) | Why |
|---|---|
| `model`, `depth`, `sampler`, `build`, `precision`, `device`, attention kernel differ | Frozen settings; different models or numerics |
| `batch_size` differs, or either run used `--allow-partial-batch` | Batch size changes float rounding |
| torch / CUDA version or submodule commits differ | Different code |
| The image list differs (count or order) | Not the same manifest |
| The baseline has a modify hook or skipped timesteps | It is not a clean baseline (observe hooks are fine) |
| Any image's `generator_state_sha256` differs | The runs did not consume the same random numbers |

Warnings only: GPU model, cuDNN or Python version differ; config file names differ.
It also reports how many images changed (`tensor_sha256` differs from the baseline's).

## 4. `metrics.csv` (per run, long format)

One number per row, appended by whoever computes it (`schema.append_metrics(run_dir, rows)`, or any
code writing the same header):

```
class_id,seed,metric,value,stage
207,0,lpips,0.3127,
207,0,state_rel_l2:f_hat,0.2113,3
```

| Column | Meaning |
|---|---|
| `class_id`, `seed` | The image (the manifest key). |
| `metric` | Free-form name. Suggested: `<name>` for image metrics (`lpips`, `clip_sim`, ...), `<name>:<tensor>` for state measurements. |
| `value` | A float. |
| `stage` | The **native** stage the number belongs to (VAR scale, DiT timestep); empty for a number about the final image. Join with `stages.csv` for `p`, token counts or noise level. |

Whether a metric is "vs the baseline" or absolute is part of its definition; the baseline of a run is in
`runs.csv`. Long format means a new metric needs no schema change.

## 5. Merged tables

`python -m lanes.p2.schema collect <folder> [--out <dir>]` finds every `run.json` below the folder and
writes (default `<folder>/tables/`):

| Table | One row per | Columns |
|---|---|---|
| `runs.csv` | run | `run_id`, `lane`, `experiment`, `arm`, `role`, `baseline_run_id`, `pairing_ok`, `model`, `config`, `depth`, `manifest`, `n_images`, `batch_size`, `hooks`, `skip_timesteps`, `params` (JSON), `seconds_load`, `seconds_sample`, `seconds_per_image`, `peak_vram_mb`, `gpu`, `torch`, `code_commit`, `code_dirty`, `pbs_jobid`, `created_utc`, `run_dir` |
| `images.csv` | run × image | `run_id`, `class_id`, `seed`, `file`, `tensor_sha256`, `generator_state_sha256`, `identical_to_baseline` |
| `stages.csv` | run × stage | `run_id` + the run's `stages` table: `stage`, `p`, and VAR `pn`, `n_tokens`, `cum_tokens`, `total_tokens` / DiT `step`, `merged`, `timestep_in/out`, `alpha_bar_in/out`, `sigma_in/out` |
| `metrics.csv` | run × image × metric (× stage) | `run_id` + the columns of §4 |

A run without `experiment.json` is still included (its `run_id` is its path), so P1's canonical
baselines can be merged as they are. The tables are derived: delete and rebuild at any time.
`stages.csv` carries both the native stage and `p`, which is what P4's progress mapping needs.

## 6. Reproducing a run

A run is reproduced by: checking out `code.commit` (if `dirty` is false), then running
`runner.generate` with the `config`, `manifest`, `batch_size` and hooks in `run.json`, with the hook
environment variables in `params`, on the same GPU type. The result must have the same
`tensor_sha256` for every image.

## 7. Requests to P1 (runner), none blocking

1. **Peak GPU memory in `run.json`.** `torch.cuda.max_memory_allocated()` after sampling, next to
   `seconds`. Until then `peak_vram_mb` comes from sampling `nvidia-smi` once a second during the run
   (whole-GPU "memory used", so it includes the CUDA context), as `lanes/p2/jobs/pilot.pbs` does.
2. **This repo's commit in `run.json`** (`git rev-parse HEAD` and a dirty flag), next to the submodule
   commits. Until then it is in `experiment.json`.
3. If the schema is adopted, `schema.py` could move from `lanes/p2/` to a shared place (e.g.
   `scripts/schema.py`) so other lanes don't import from `lanes.p2`.

## 8. Open questions for the team

- **P3:** metric names and whether a metric row needs more keys (e.g. a reference class).
- **P4:** is `stage` + join on `stages.csv` enough for the plotting template, or should `p` be copied
  into `metrics.csv`?
- **All:** `run_id` = `<lane>/<experiment>/<arm>`; is `<experiment>` = `<name>_<JOBID>` acceptable as
  the uniqueness rule?
