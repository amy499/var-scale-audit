# P2 Phase C: full corruption maps, robustness checks and the recovery ranking

Completes P2's sweep on the frozen manifest v1.1 (7 dog classes × 32 = 224 images; sha256
`caf0535d9cb0…`). Builds on Gate B (`../gateb/README.md`), which has the VAR map at noise seed 0.

| Job | What | Result |
|---|---|---|
| `25891324`-`27` | DiT `noise_x` at timesteps 798, 598, 497, 397, 297, 197, 96, 36 × severity 0.25, 0.5, 1, 2, 4 (4 jobs of 2 timesteps) | 30 of 40 runs done; the home quota (50 GB) ran out |
| `25900881` | DiT: the 10 missing runs, plus 2 repeats | Done, all paired |
| `25891328` | VAR `noise_h` at every scale × 5 severities, **second noise draw** (`P2_NOISE_SEED=1`) | Done, 50 / 50 paired |

Since then `jobs/pilot.pbs` deletes each run's raw `.pt` tensors once it is scored (`KEEP_RAW=0`;
their sha256 stays in `run.json`), so a sweep no longer fills the quota.

Metrics (per run, means over images): classifier probability of the image's own class and KID vs
the baseline (validity), `incep_cos` (fidelity, P3's definition), and P3's outcome label
(unchanged / re-routed / degraded-in-place / damaged; `lanes/p3/p3_metrics.py`, thresholds 0.30 /
0.50 / 0.005), plus the fraction of images with each label. See `lanes/p2/README.md`.

## Reproducibility checks

| Check | Result |
|---|---|
| The 5 DiT baselines (one per job) | Bit-identical (all 224 tensor hashes) |
| VAR baseline, Phase C vs Gate B (two jobs, days apart) | Bit-identical |
| Runs repeated in two jobs (598 / sev 1, 397 / sev 1) | Bit-identical |
| Every corrupted run vs its baseline | Paired (generator states equal) |

So runs from different jobs combine safely, and every number here can be regenerated exactly.

## DiT: the full map (224 images; baseline class probability 0.76)

**Outcome** (P3's rule, run level):

| Timestep (p) | 0.25 | 0.5 | 1 | 2 | 4 |
|---|---|---|---|---|---|
| 798 (0.20) | unchanged | unchanged | unchanged | unchanged | damaged |
| 598 (0.40) | unchanged | unchanged | degraded | damaged | damaged |
| 497 (0.50) | unchanged | unchanged | damaged | damaged | damaged |
| 397 (0.60) | unchanged | unchanged | damaged | damaged | damaged |
| 297 (0.70) | unchanged | degraded | damaged | damaged | damaged |
| 197 (0.80) | unchanged | degraded | damaged | damaged | damaged |
| 96 (0.90) | unchanged | degraded | damaged | damaged | damaged |
| 36 (0.96) | unchanged | damaged | damaged | damaged | damaged |

**KID vs the baseline** (0 = indistinguishable from clean):

| Timestep (p) | 0.25 | 0.5 | 1 | 2 | 4 |
|---|---|---|---|---|---|
| 798 (0.20) | 0.000 | 0.000 | 0.000 | 0.001 | 0.034 |
| 598 (0.40) | 0.000 | 0.000 | 0.007 | 0.117 | 0.253 |
| 497 (0.50) | 0.000 | 0.001 | 0.056 | 0.184 | 0.297 |
| 397 (0.60) | 0.000 | 0.004 | 0.131 | 0.287 | 0.580 |
| 297 (0.70) | 0.000 | 0.011 | 0.181 | 0.338 | 0.609 |
| 197 (0.80) | 0.000 | 0.014 | 0.187 | 0.325 | 0.633 |
| 96 (0.90) | 0.001 | 0.033 | 0.198 | 0.358 | 0.632 |
| 36 (0.96) | 0.004 | 0.070 | 0.219 | 0.398 | 0.630 |

**Class probability:** 0.70-0.74 at every timestep for severity 0.25 except 36 (0.64); at severity 1 it falls
from 0.74 (798) to 0.59, 0.31, 0.06 and then 0.00 from 297 on. Full tables: `p2_summary.csv`.

Grids (first image of each class): `dit_timesteps_sev0.5.jpg`, `dit_timesteps_sev1.jpg`.

- **DiT's tolerance falls steadily with progress:** severity 2 is still absorbed at p 0.2, then 0.5
  up to p 0.6, then 0.25 from p 0.7. KID rises monotonically along both axes.
- **What failure looks like moves with the stage** (severity 1 grid): early, the scene is
  re-composed with harsher contrast (598); middle, oversaturated and cartoon-like (497); late, the
  injected noise is not removed and the image dissolves (397-36).
- **It repairs towards the original:** the state gap to the baseline shrinks after the corruption
  (ratio 0.25-0.86 for most cells), and re-routing is rare (≤ 21 % of images, only early).

## VAR: robust across noise draws

The Gate B map (seed 0) and Phase C (seed 1) agree:

| | Across all 50 stage × severity cells |
|---|---|
| Outcome label | **Identical in 50 / 50** |
| Class probability | Mean difference 0.014, max 0.049 |
| KID | Mean difference 0.002, max 0.022 |

So VAR's map is a property of the stage and severity, not of the particular noise pattern.

## Broken baselines

`p2_baselines.csv` flags every baseline image: `ok` (own class top-1), `other-top1` (own class in
the top 5, usually a neighbouring breed), `broken` (not in the top 5).

- **VAR:** no broken baselines (13 of 224 are a neighbouring breed).
- **DiT:** 9 broken: Chihuahua images 1, 10, 12, 20 (classified as non-dogs: 568, 836, 283, 674),
  Chihuahua 7 and 28, beagle 15, golden retriever 13, Samoyed 17.
- **Excluding them changes nothing that matters:** class probability by ≤ 0.03, KID by ≤ 0.005,
  and no outcome label (42 / 42 DiT runs; `p2_summary_valid.csv`).

## Recovery ranking (for P4 / Figure 5)

`recovery_ranking.csv`, one row per stage: the largest severity that still leaves the run
unchanged or re-routed, the first that fails and how, and KID / damaged fraction at severities 1
and 2. VAR from Gate B (seed 0), DiT from this sweep.

| Model | Stage (p) | Tolerates up to | Fails at | Damaged at sev 2 |
|---|---|---|---|---|
| VAR | every scale 0-9 | 1 | 2 | 0.42 (s0) → **0.09 (s5)** → 0.18 (s8) → 1.00 (s9) |
| DiT | 798 (0.20) | 2 | 4 | 0.09 |
| DiT | 598 (0.40) | 0.5 | 1 (degraded) | 0.89 |
| DiT | 497, 397 (0.50-0.60) | 0.5 | 1 (damaged) | 1.00 |
| DiT | 297, 197, 96 (0.70-0.90) | 0.25 | 0.5 (degraded) | 1.00 |
| DiT | 36 (0.96) | 0.25 | 0.5 (damaged) | 1.00 |

**Which stages matter most for robustness:**

- **DiT:** importance rises monotonically with progress. The last ~40 % of the denoising is where
  corruption cannot be undone; the first ~20 % repairs almost anything.
- **VAR:** a flat threshold (every scale tolerates severity 1), but past it the damage is
  **U-shaped**. The first scales (0-1) are fragile through *semantic drift* (the dog disappears,
  KID stays low), the middle scales (4-6) are the most robust, and the last scale is the most
  fragile (*structural* damage, nothing left to repair it).

## For Figure 2

- **Panel A, maps:** outcome / KID per stage × severity, VAR and DiT side by side on `p`.
  Data: `p2_summary.csv` (+ `../gateb/var_p2_summary.csv` for VAR seed 0).
- **Panel B, recovery curves:** state gap to the baseline at every traced stage after the
  corruption. Data: `p2_curves.csv` (`state_l2:f_hat` for VAR, `state_l2:x` for DiT).
- **Panel C, examples:** the grids.

Compare the trends across models, not absolute severities: VAR's noise goes into one scale's
embedding, DiT's into the whole latent (the plan's warning against equating scales and timesteps).

## Caveats

- One VAR depth (d20), Gaussian noise only, one corruption point per run.
- Metrics are interim: P3's thresholds (0.30 / 0.50 / 0.005) are provisional until re-derived on
  the real baselines. VAR scale 0 at severity 1 has mean `incep_cos` 0.25, below 0.30, so it is
  labelled "unchanged" although 12-18 % of its images are clearly different dogs (`frac_re_routed`).
- DiT timestep 36 has no state-gap ratio: it is not one of the traced steps (every 25th).

## Files

| File | Content |
|---|---|
| `p2_summary.csv`, `p2_summary_valid.csv` | One row per corrupted run (all 92 Phase C runs), with and without broken baselines |
| `p2_curves.csv` | State gap to the baseline per run and traced stage |
| `p2_baselines.csv` | Every baseline image's classifier result and flag |
| `recovery_ranking.csv` | Per model and stage: tolerated severity, failure mode, damage at severity 1 and 2 |
| `runs.csv` | Every run: settings, pairing, timing, VRAM, code commit |
| `dit_timesteps_sev0.5.jpg`, `dit_timesteps_sev1.jpg` | DiT grids |

Full outputs are on NSCC in `outputs/p2/phasec/<JOBID>/`.
