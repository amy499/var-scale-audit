# P1: stage removal

Remove one group of generation stages and measure what the image loses, on VAR (scales) and DiT
(denoising steps). Every removal run is paired with a clean baseline from the same job and recorded with
P2's schema (`lanes/p2/schema.py`, `--lane p1`).

| File | What |
|---|---|
| `hooks.py` | `remove_scales`: VAR removal, variant (b), scales from `P1_VAR_SCALES` |
| `jobs/gateb.pbs` | Baseline + removal arms per model, scoring, tables, `summary.txt` |
| `jobs/gateb_var.env`, `jobs/gateb_dit.env` | Gate B settings: the mid arm only |
| `jobs/sweep_var.env`, `jobs/sweep_dit.env` | Necessity sweep: early, mid and late arms per model |
| `jobs/sanity_var.env` | Positive control for the VAR hook: scale 9 alone, scale 0 alone |

## What "removal" means

**VAR, variant (b)** (`hooks.py`, as `scripts/examples/skip_scale_2.py`): a `Modify(when="after")` hook
sets `f_hat = f_hat_before` at each removed scale. That scale's contribution is dropped from the
accumulator, so later scales (whose input is read from `f_hat`) and the decoder never see it. Its tokens
are still sampled, so every row draws the same random numbers as the baseline and the run stays paired.

**DiT, skip + burn** (`runner.generate --skip-timesteps`, `docs/hook_interface.md` §11b): the model is
not evaluated at the skipped timesteps; the step before the window maps its latent straight to the level
after it, and each skipped step's noise is still drawn ("burned"), so the run stays paired.

## Stage groups and the equal-cuts rule

Each arm removes the same number of stages: **3 VAR scales**, **83 DiT steps** (a third of 250).

| Group | VAR scales (tokens of 680) | DiT step indices (0 = first of 250) |
|---|---|---|
| early | {0, 1, 2} (14) | 1-83 |
| mid | {3, 4, 5} (77) | 84-166 |
| late | {7, 8, 9} (525) | 167-249, the last 83 steps |

- VAR scale 6 (64 tokens) is never cut, so that each group is exactly 3 scales.
- DiT step 0 is never cut (the runner cannot skip it: there is no earlier step to merge into),
  mirroring VAR scale 6.
- The cut is equal in stages, not in work: the VAR groups remove very different token budgets
  (14 / 77 / 525 of 680 tokens; scale 9 alone is 256), while every DiT arm removes 83 model calls.
  Compare the arms with this in mind.
- The late DiT arm skips the final step, so the last kept step (166) goes straight to x_0.

DiT step indices are converted to timesteps from this job's own DiT baseline `run.json` (`stages`
table, `step` -> `timestep_in`). The job stops that arm if an index or timestep is not in the table,
or if the range does not give exactly 83 timesteps. VAR arms must list exactly 3 scales (`VAR_CUT`;
`sanity_var.env` sets it to 1 for its single-scale arms).

## Controls

- **Clean baseline.** Each job's baselines must be byte-identical (`tensor_sha256`, every row) to job
  25665469; `summary.txt` reports the comparison and the job fails on a mismatch.
- **Matched groups.** The three equal-cut groups are matched to each other: same number of stages
  (3 scales, 83 steps), only their position differs.
- **No random-cut arm.** Scales and steps are not exchangeable (each has its own resolution or noise
  level), so a random set of stages is not a meaningful null.
- **Known confound.** The VAR groups remove 14 / 77 / 525 of 680 tokens: position and token budget move
  together, and VAR results cannot separate them.
- **Hook positive control** (`jobs/sanity_var.env`). Removing scale 9 alone should give a near-normal
  dog. If it does not, the hook is wrong and no VAR result is valid. Scale 0 alone is reported alongside
  it.
- **Gate B observation** (jobs 25891860 VAR / 25891861 DiT). VAR `remove_mid` gave fur texture with no
  dog shape; DiT `remove_mid` gave disfigured but recognisable dogs. Interpretation is pending the
  positive control.

## How to run

On NSCC, from the repo root. Weights must already be on disk (the job sets `HF_HUB_OFFLINE=1` and never
downloads):

```bash
mkdir -p outputs/p1/pbs
python -m lanes.p2.quality --download          # once, on the login node: classifier weights for QUALITY=1
qsub -v SETTINGS=lanes/p1/jobs/gateb_var.env lanes/p1/jobs/gateb.pbs
qsub -v SETTINGS=lanes/p1/jobs/gateb_dit.env lanes/p1/jobs/gateb.pbs

qsub -v SETTINGS=lanes/p1/jobs/sanity_var.env lanes/p1/jobs/gateb.pbs   # hook positive control, ~4 min
qsub -v SETTINGS=lanes/p1/jobs/sweep_var.env lanes/p1/jobs/gateb.pbs    # VAR early/mid/late, ~5 min
qsub -v SETTINGS=lanes/p1/jobs/sweep_dit.env lanes/p1/jobs/gateb.pbs    # DiT early/mid/late, ~26 min
tail -f outputs/p1/<JOBID>/job.log             # ends with "P1 DONE: all runs recorded and paired"
```

The project (`personal-gura0001`) is in the job header; override it with `qsub -P <id>`. Per model the
job runs the baseline, then each arm in `VAR_ARMS` / `DIT_ARMS`; after each run: `schema record`
(pairing check), `analyze images` (not for the baseline) and `quality`; at the end `schema collect`
into `outputs/p1/<JOBID>/tables/`. Each step's exit code and duration go to `steps.tsv`.

`summary.txt` lists every run with its step exit codes and pairing result, and compares each baseline
row by row (`tensor_sha256`) with the reference baseline `BASELINE_REF` (default
`outputs/baseline/25665469`): PASS, FAIL (the job exits 1), or SKIPPED if the reference is not there.
The per-row tables are at the end of the summary and in `outputs/p1/<JOBID>/<model>/hash_vs_ref.tsv`.

Options (environment, or in the `SETTINGS` file): `VAR_ARMS="NAME:S,S,S ..."`,
`DIT_ARMS="NAME:A-B ..."`, `RUN_VAR`, `RUN_DIT`, `QUALITY`, `MANIFEST`, `BASELINE_REF`, `EXP_TAG`.

**P3 ride-along.** `P3_RIDEALONG=1` adds P3's observe hook (`lanes/p3/p3_observe.py:var_stages`) to the
VAR baseline only, writing to `outputs/p1/<JOBID>/p3_var_stages`. It is an Observe hook, so the
baseline images must still match the reference; the hash comparison checks this. DiT has no ride-along.
