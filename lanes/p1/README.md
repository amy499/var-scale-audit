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
| `jobs/single_var.env` | Single-scale sweep: each of VAR scales 1-8 alone |
| `jobs/window2_var.env`, `jobs/window3_var.env` | Sliding windows of 2 and 3 neighbouring VAR scales |
| `summarize.py` | One row per arm: means, paired drops vs baseline with bootstrap CIs, KID; `--compare` for arm vs arm; `--ranking-out` for the ranking |
| `plot_fig1.py` | Figure 1: VAR top-1 by removed scale / window, DiT top-1 per arm, with CIs |
| `failure_grid.py` | Figure 1b: four fixed example images under each VAR and DiT arm |
| `plot_heatmap.py` | Figure 1c: per-class drop in top-1 for every VAR single-scale, VAR 2-scale and DiT arm |
| `jobs/p3_both.env` | Baselines only, both models, with P3's observe hooks (P3 ride-along) |
| `compare_p3_var.py` | Checks two P3 VAR snapshot folders are identical, file by file |
| `make_seed2_manifest.py`, `manifests/seed2.csv` | Phase E: the second seed set (same classes and layout as v1.1, master seed 61320018) |
| `jobs/seed2_var_single.env`, `jobs/seed2_var_groups.env`, `jobs/seed2_dit.env` | Phase E: single scales, VAR groups and DiT bands on seed set 2 |
| `compare_seeds.py` | Phase E: seed set 1 vs seed set 2, per arm, plus rank agreement |
| `select_crystals.py` | Phase F: picks the 64 OpenCrystalData images (32 per magnification, evenly spaced over the sorted file list); the images are not in the repo |
| `make_dog64.py` | Phase F: copies 64 VAR baseline PNGs (all 7 classes) to a folder outside the repo, with `dog64_list.csv` |
| `jobs/transfer_var.pbs` | Phase F: VAR scale removal on real images, one dataset per job, with codec and hook checks |
| `transfer_summary.py` | Phase F: `pair` (per-arm change vs the reference, run in the job) and `summary` (CSV + Figure 1d) |
| `results/` | `results_sweep.csv`, `ranking.csv`, `compare_dit.csv`, `heatmap_by_class.csv`, `seed2_results.csv`, `seed_compare.csv`, `fig1.png`, `fig1b_failures.png`, `fig1c_heatmap.png` (all quoted below); Phase F adds `transfer_var.csv`, `fig1d_transfer.png` |

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
- **P1's DiT arms are offset by one step from P4's frozen bands** (`lanes/p4/bands.py`: early 0-82,
  middle 83-165, late 166-249), because the runner cannot skip step 0. P1 early (1-83) and mid (84-166)
  each overlap their P4 band in 82 of 83 steps. P1 late (167-249, 83 steps) lies entirely inside P4's
  late band, which has 84 steps (166-249), so the overlap is 83 of 83. P4's Figure 5 maps P1's DiT arms
  onto P4's bands.

DiT step indices are converted to timesteps from this job's own DiT baseline `run.json` (`stages`
table, `step` -> `timestep_in`). The job stops that arm if an index or timestep is not in the table,
or if the range does not give exactly 83 timesteps. VAR arms must list exactly 3 scales (`VAR_CUT`;
`sanity_var.env` and `single_var.env` set it to 1, `window2_var.env` to 2).

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

## Sliding-window design (VAR)

VAR scales are removed in windows of 1, 2 and 3 neighbouring scales, sliding along the ladder:

| Cut size | Windows | Jobs |
|---|---|---|
| 1 | {0} ... {9}: 10 | `sanity_var.env` (0, 9; job 25893211), `single_var.env` (1-8) |
| 2 | {0,1} ... {8,9}: 9 | `window2_var.env` |
| 3 | {0,1,2} ... {7,8,9}: 8 | `sweep_var.env` (012, 345, 789; job 25893212), `window3_var.env` (123, 234, 456, 567, 678) |

- Only neighbouring scales, not every combination (45 pairs, 120 triples): a window asks what a
  contiguous stretch of the coarse-to-fine ladder contributes, and stays affordable.
- Within each job every arm removes the same number of scales (`VAR_CUT` is checked per arm), so arms
  in one job differ only in position. Across cut sizes the token budget still differs (see Controls).
- Arms are compared with each other, within one job and over the same images, by
  `summarize.py --compare JOB:ARM_A,ARM_B` (paired difference A - B, 95% paired-bootstrap CI).

Arm vs arm, DiT sweep (`results/compare_dit.csv`, from
`python -m lanes.p1.summarize --compare 25893213:remove_mid,remove_late --compare 25893213:remove_early,remove_mid --compare-out lanes/p1/results/compare_dit.csv`).
Positive = arm A scores higher:

| job | A | B | n | top-1 A | top-1 B | A - B top-1 [95% CI] | cls_prob A | cls_prob B | A - B cls_prob [95% CI] |
|---|---|---|---|---|---|---|---|---|---|
| 25893213 | remove_mid | remove_late | 224 | 0.737 | 0.656 | 0.080 [0.000, 0.165] | 0.572 | 0.493 | 0.079 [0.014, 0.144] |
| 25893213 | remove_early | remove_mid | 224 | 0.826 | 0.737 | 0.089 [0.013, 0.165] | 0.681 | 0.572 | 0.109 [0.048, 0.170] |

Observations (not conclusions): early is above mid on both metrics, with both CIs above 0. Mid is above
late on cls_prob (CI above 0), but on top-1 the CI's lower end is 0.000, so the top-1 difference
between mid and late is not clearly non-zero.

## Results so far

| Job | What | Arms |
|---|---|---|
| 25893211 | VAR sanity (`sanity_var.env`) | scale 0 alone, scale 9 alone |
| 25893212 | VAR sweep (`sweep_var.env`) | early {0,1,2}, mid {3,4,5}, late {7,8,9} |
| 25893213 | DiT sweep (`sweep_dit.env`) | early 1-83, mid 84-166, late 167-249 |
| 25916920 | VAR single scales (`single_var.env`) | each of scales 1-8 alone |
| 25916921 | VAR 2-scale windows (`window2_var.env`) | {0,1}, {1,2}, ... {8,9} |
| 25983856 | P3 ride-along (`p3_both.env`, commit 1fa31ca) | clean VAR + DiT baselines with P3's observe hooks; no removal arms |
| 25985813 | P3 ride-along (`p3_both.env`, commit 58ec5e5, after P3's PR #8 merged) | clean VAR + DiT baselines with P3's observe hooks; no removal arms |

The two P3 ride-along jobs passed every check, and the snapshots were delivered to P3:
- baselines byte-identical to 25665469 (224/224 rows, both models);
- 140 VAR and 336 DiT `.npz` per job, with no `_failures.log`;
- `lanes/p3/verify_extraction.py` checks 1-2 PASS;
- snapshots bit-identical across runs: VAR against Gate B's 25891860 140/140 (`compare_p3_var.py`),
  and DiT 25985813 against 25983856 336/336.

`window3_var.env` (the five 3-scale windows not yet run) is held back: the three 3-scale groups already
run span top-1 0.000-0.107, so further 3-scale windows are unlikely to separate positions.

![Figure 1](results/fig1.png)

Figure 1 (`results/fig1.png`, from `python -m lanes.p1.plot_fig1`; needs matplotlib). Left: VAR top-1
vs the removed scale (single scales), with 2-scale windows and the three labelled 3-scale groups at the
mean of their removed scales; each scale index shows its token-grid size. Right: DiT top-1 per arm.
Error bars: 95% bootstrap CI of each arm's mean over its 224 images (not the paired CI of the drop in
the table). Dashed line: clean baseline.

![Figure 1b](results/fig1b_failures.png)

Figure 1b (`results/fig1b_failures.png`, from `python -m lanes.p1.failure_grid`; needs matplotlib +
Pillow): the same four images (first manifest row of each of the first four classes) under each VAR
and DiT arm, headed by the arm's mean top-1 over all 224 images.

![Figure 1c](results/fig1c_heatmap.png)

Figure 1c (`results/fig1c_heatmap.png` and `results/heatmap_by_class.csv`, from
`python -m lanes.p1.plot_heatmap`; needs matplotlib): drop in top-1 per class (baseline - arm, paired,
against the same job's baseline) for every VAR single scale (A), VAR 2-scale window (B) and DiT arm (C),
on one shared colour scale. Each cell is **n = 32 images**, so per-class CIs are wide: the 95% CI
half-width has a median of 0.16 and goes up to 0.22 (`heatmap_by_class.csv` has every CI).

Observations from `heatmap_by_class.csv` (not conclusions):
- Single scales: scale 3 has the largest drop for 6 of 7 classes. The exception is beagle, where scale 2
  is larger (0.375 vs 0.344), with heavily overlapping CIs. For every class, the drops at scales 2 and
  3 are at least as large as at any other scale; another scale ties scale 2 for Chihuahua (scale 6,
  0.344) and German shepherd (scale 4, 0.188).
- 2-scale windows: {2,3} has the largest drop for every class (tied with {1,2} for German shepherd,
  both 0.875; pug reaches 1.000).
- DiT: late has the largest drop for 6 of 7 classes. Chihuahua is the exception (mid 0.344
  [0.12, 0.56] vs late 0.156 [0.00, 0.31]; CIs overlap).
- The 7 negative cells (arm better than baseline for that class) are all at scale 0, scale 1 or {0,1},
  and none is below -0.094.

Table from `lanes/p1/results/results_sweep.csv`; `lanes/p1/results/ranking.csv` from the same command:
`python -m lanes.p1.summarize 25893211 25893212 25893213 25916920 25916921 --out lanes/p1/results/results_sweep.csv --ranking-out lanes/p1/results/ranking.csv`.

`results/ranking.csv`: every removal arm per model, ordered by top-1 drop (largest first), with what it
removed, how many scales or steps, the VAR tokens removed, and the drop CIs and KID from this table.
Ties are broken by cls_prob drop, then job and arm: VAR {0,1,2} and {3,4,5} tie on both (0.942 / 0.811).
Means are over the 224 images. Drops are baseline - arm (positive = worse), with 95% paired-bootstrap
CIs over images (10000 resamples, seed 0). `img_mad255` is the distance from the baseline image
(0..255). KID is paired, against the job's own baseline.

| job | model | arm | n | top-1 | top-5 | cls_prob | img_mad255 | drop top-1 [95% CI] | drop cls_prob [95% CI] | KID |
|---|---|---|---|---|---|---|---|---|---|---|
| 25893211 | var | baseline | 224 | 0.942 | 1.000 | 0.812 | | | | |
| 25893211 | var | remove_s0 | 224 | 0.911 | 0.978 | 0.789 | 66.19 | 0.031 [-0.013, 0.076] | 0.023 [-0.016, 0.064] | 0.00315 |
| 25893211 | var | remove_s9 | 224 | 0.835 | 0.969 | 0.663 | 13.33 | 0.107 [0.058, 0.156] | 0.149 [0.115, 0.184] | 0.01216 |
| 25893212 | var | baseline | 224 | 0.942 | 1.000 | 0.812 | | | | |
| 25893212 | var | remove_early | 224 | 0.000 | 0.004 | 0.001 | 69.64 | 0.942 [0.911, 0.969] | 0.811 [0.781, 0.839] | 0.12059 |
| 25893212 | var | remove_mid | 224 | 0.000 | 0.004 | 0.001 | 41.39 | 0.942 [0.911, 0.969] | 0.811 [0.781, 0.839] | 0.11306 |
| 25893212 | var | remove_late | 224 | 0.107 | 0.254 | 0.064 | 22.03 | 0.835 [0.786, 0.880] | 0.748 [0.714, 0.781] | 0.15318 |
| 25893213 | dit | baseline | 224 | 0.906 | 0.960 | 0.759 | | | | |
| 25893213 | dit | remove_early | 224 | 0.826 | 0.938 | 0.681 | 50.59 | 0.080 [0.027, 0.134] | 0.078 [0.027, 0.129] | 0.00029 |
| 25893213 | dit | remove_mid | 224 | 0.737 | 0.884 | 0.572 | 46.45 | 0.170 [0.107, 0.237] | 0.187 [0.134, 0.240] | 0.01775 |
| 25893213 | dit | remove_late | 224 | 0.656 | 0.808 | 0.493 | 20.85 | 0.250 [0.188, 0.317] | 0.266 [0.217, 0.315] | 0.03127 |
| 25916920 | var | baseline | 224 | 0.942 | 1.000 | 0.812 | | | | |
| 25916920 | var | remove_s1 | 224 | 0.920 | 0.982 | 0.786 | 60.52 | 0.022 [-0.022, 0.071] | 0.026 [-0.016, 0.070] | 0.00209 |
| 25916920 | var | remove_s2 | 224 | 0.589 | 0.754 | 0.467 | 51.72 | 0.353 [0.281, 0.424] | 0.346 [0.290, 0.402] | 0.02491 |
| 25916920 | var | remove_s3 | 224 | 0.469 | 0.603 | 0.346 | 42.77 | 0.473 [0.402, 0.545] | 0.466 [0.413, 0.520] | 0.02846 |
| 25916920 | var | remove_s4 | 224 | 0.710 | 0.897 | 0.566 | 32.99 | 0.232 [0.174, 0.295] | 0.246 [0.202, 0.293] | 0.00963 |
| 25916920 | var | remove_s5 | 224 | 0.844 | 0.951 | 0.694 | 27.38 | 0.098 [0.049, 0.147] | 0.118 [0.082, 0.155] | 0.00451 |
| 25916920 | var | remove_s6 | 224 | 0.732 | 0.902 | 0.583 | 24.61 | 0.210 [0.156, 0.268] | 0.230 [0.187, 0.273] | 0.01244 |
| 25916920 | var | remove_s7 | 224 | 0.812 | 0.964 | 0.673 | 19.60 | 0.130 [0.080, 0.183] | 0.139 [0.105, 0.175] | 0.00501 |
| 25916920 | var | remove_s8 | 224 | 0.893 | 0.969 | 0.756 | 15.90 | 0.049 [0.009, 0.094] | 0.056 [0.030, 0.083] | 0.00231 |
| 25916921 | var | baseline | 224 | 0.942 | 1.000 | 0.812 | | | | |
| 25916921 | var | remove_s01 | 224 | 0.906 | 0.996 | 0.761 | 70.57 | 0.036 [-0.013, 0.085] | 0.051 [0.009, 0.093] | 0.00414 |
| 25916921 | var | remove_s12 | 224 | 0.058 | 0.103 | 0.051 | 61.04 | 0.884 [0.835, 0.929] | 0.761 [0.722, 0.799] | 0.06539 |
| 25916921 | var | remove_s23 | 224 | 0.009 | 0.022 | 0.004 | 50.93 | 0.933 [0.897, 0.964] | 0.808 [0.778, 0.837] | 0.08372 |
| 25916921 | var | remove_s34 | 224 | 0.076 | 0.125 | 0.046 | 43.43 | 0.866 [0.821, 0.911] | 0.766 [0.731, 0.801] | 0.08095 |
| 25916921 | var | remove_s45 | 224 | 0.277 | 0.438 | 0.190 | 34.80 | 0.665 [0.603, 0.723] | 0.622 [0.577, 0.666] | 0.06008 |
| 25916921 | var | remove_s56 | 224 | 0.255 | 0.402 | 0.179 | 30.00 | 0.688 [0.625, 0.746] | 0.634 [0.589, 0.678] | 0.07265 |
| 25916921 | var | remove_s67 | 224 | 0.473 | 0.732 | 0.346 | 25.60 | 0.469 [0.402, 0.536] | 0.466 [0.420, 0.514] | 0.04702 |
| 25916921 | var | remove_s78 | 224 | 0.696 | 0.875 | 0.529 | 21.02 | 0.245 [0.188, 0.304] | 0.283 [0.241, 0.326] | 0.02102 |
| 25916921 | var | remove_s89 | 224 | 0.482 | 0.732 | 0.309 | 17.90 | 0.460 [0.393, 0.527] | 0.503 [0.461, 0.547] | 0.07275 |

Observations (not conclusions):
- The hook positive control passes: removing scale 9 alone keeps top-1 at 0.835, against 0.942 for the
  baseline.
- VAR collapses under any 3-scale removal: top-1 is 0.000 for early and mid, and 0.107 for late.
- Early removes 14 tokens yet collapses; late removes 525 and is the least harmful by the classifier
  (top-1 0.107, cls_prob 0.064, against 0.000 / 0.001 for early). Token count does not explain the VAR
  damage. By KID, though, late is the furthest from the baseline (0.153, against 0.121 early and
  0.113 mid), so "least harmful" holds for the classifier metrics only.
- DiT degrades but survives: top-1 is 0.826 (early), 0.737 (mid) and 0.656 (late), against 0.906 for the
  baseline.
- Single scales (s0-s9): the largest top-1 drops are at s3 (0.473, top-1 0.469) and s2 (0.353, top-1
  0.589). Only s0 and s1 have a top-1 drop whose CI includes 0. The drop is not monotone in scale: s6
  (0.732) is below both s5 (0.844) and s7 (0.812).
- 2-scale windows: {1,2}, {2,3} and {3,4} give top-1 0.058, 0.009 and 0.076. {0,1} keeps 0.906, its
  top-1 drop CI including 0. The highest of the other windows is {7,8} at 0.696, and {8,9} is lower at
  0.482.
- Every 2-scale window has a lower top-1 than either of its two single-scale arms (point estimates;
  for {0,1}, 0.906 against 0.911 and 0.920, within the CIs).

## Phase E: second seed set

**Why.** Every result above uses one set of 224 images (manifest v1.1). Their CIs cover resampling
those images, not drawing new ones. A second, independent seed set checks whether the scale ranking, the
top scales and the DiT band order are properties of the models or of these 224 images.

**Manifest.** `make_seed2_manifest.py` writes `manifests/seed2.csv`:
- same 7 classes and per-image layout as v1.1: 32 images per class, class 151 first, header `class_id,seed`;
- master seed 61320017 + 1 = 61320018, seeds from v1.1's own `derive_seeds`;
- 224 unique seeds, none shared with `manifest/frozen_v1_1.csv` (the script checks this and stops
  otherwise);
- LF line endings, byte-identical on re-run (sha256 `85fe1715fb71…`).

**Jobs.** All three use `KEEP_RAW=0`, `QUALITY=1` and `P3_RIDEALONG=0`. There is no reference baseline
for these seeds, so `BASELINE_REF` points at a path that does not exist (`outputs/baseline/none_for_seed2`)
and the hash check reports SKIPPED, which `gateb.pbs` does not count as a failure. Pairing within each
job is still checked against that job's own baseline.

```bash
python -m lanes.p1.make_seed2_manifest                                        # once; already in the repo
qsub -v SETTINGS=lanes/p1/jobs/seed2_var_single.env lanes/p1/jobs/gateb.pbs   # scales 0-9 alone, ~8 min
qsub -v SETTINGS=lanes/p1/jobs/seed2_var_groups.env lanes/p1/jobs/gateb.pbs   # {0,1,2} {3,4,5} {7,8,9}, ~3 min
qsub -v SETTINGS=lanes/p1/jobs/seed2_dit.env lanes/p1/jobs/gateb.pbs          # DiT 1-83 84-166 167-249, ~25 min
```

These estimates come from the seed-1 jobs' `steps.tsv` and exclude job start-up.

**What will be compared.** Seed 1 uses jobs 25893211 + 25916920 (single scales), 25893212 (groups) and
25893213 (DiT):

```bash
python -m lanes.p1.compare_seeds --single <JOB> --groups <JOB> --dit <JOB>
```

The jobs can be given as they finish. It writes two new files and never touches the existing results:
- `results/seed2_results.csv`: same columns as `results_sweep.csv`.
- `results/seed_compare.csv`: one row per arm, matched on the scales or steps removed. For top-1 and
  `cls_prob` it gives the drop on each seed set with its paired 95% CI, and the difference
  (seed 2 − seed 1) with a 95% CI from resampling the two image sets independently.

It also prints:
- the Spearman rho between the two seed sets' rankings of the 10 single scales by `cls_prob` drop;
- the top-3 scales on each seed set (seed 1: 3, 2, 4);
- the DiT band order on each seed set (seed 1: late > mid > early).

The script stops if the two seed sets share an image.

**Jobs run.** All three report `RESULT: OK` with manifest sha256 `85fe1715fb71…` and the hash check
SKIPPED, as intended.

| Job | Settings | Arms |
|---|---|---|
| 26203723 | `seed2_var_single.env` | scales 0-9 alone |
| 26203724 | `seed2_var_groups.env` | {0,1,2}, {3,4,5}, {7,8,9} |
| 26203725 | `seed2_dit.env` | DiT 1-83, 84-166, 167-249 |

The seed-2 VAR baselines of 26203723 and 26203724 are byte-identical: 224/224 `tensor_sha256`, checked on
NSCC and again from the two `run.json` files. That is the same check the seed-1 jobs pass against
`outputs/baseline/25665469`.

**Results** (`results/seed2_results.csv`: 19 runs; `results/seed_compare.csv`: 16 arms).

```bash
python -m lanes.p1.compare_seeds --single 26203723 --groups 26203724 --dit 26203725
```

- Baselines: VAR top-1 0.929 on seed 2 vs 0.942 on seed 1; DiT 0.862 vs 0.906.
- Single VAR scales:
  - Ranked by `cls_prob` drop, the order is the same on both seed sets: 3, 2, 4, 6, 9, 7, 5, 8, 1, 0
    (Spearman rho = 1.000).
  - Two neighbouring pairs are close on seed 2: scale 4 0.244 vs scale 6 0.243, and scale 5 0.111 vs
    scale 8 0.110.
  - Top-3: scales 3, 2, 4 on both (seed 1: 0.466, 0.346, 0.247; seed 2: 0.488, 0.315, 0.244).
  - Scales 0 and 1 have a top-1 drop whose CI includes 0 on both seed sets.
- VAR groups (top-1, seed 1 → seed 2):
  - {0,1,2} and {3,4,5} give top-1 0.000 on both seed sets, so their drop equals the baseline: 0.942 on
    seed 1, 0.929 on seed 2.
  - {7,8,9} gives top-1 0.107 on seed 1 and 0.094 on seed 2, a drop of 0.835 on both.
  - All six differences (3 groups × top-1 and `cls_prob`) have CIs that include 0.
- DiT order by `cls_prob` drop:
  - seed 1: late (0.266) > mid (0.187) > early (0.078);
  - seed 2: late (0.238) > mid (0.219) > early (−0.026).
  - On seed 2, skipping the early steps leaves top-1 at or above baseline (drop −0.022, CI −0.085 to
    0.040).
- 5 of the 32 seed 2 − seed 1 differences (16 arms × top-1 and `cls_prob`) have a 95% CI that excludes
  0, without multiple-comparison correction; about 1-2 are expected by chance at 95%. The single-scale
  ranking is unchanged (rho = 1.0). The five are:

  | Arm | Measure | Difference [95% CI] |
  |---|---|---|
  | scale 8 | top-1 | +0.080 [0.018, 0.143] |
  | scale 8 | `cls_prob` | +0.054 [0.012, 0.096] |
  | scale 9 | top-1 | +0.103 [0.027, 0.179] |
  | DiT early | top-1 | −0.103 [−0.183, −0.022] |
  | DiT early | `cls_prob` | −0.104 [−0.176, −0.029] |

  Scale 9's `cls_prob` difference has a CI that includes 0 (+0.040 [−0.011, 0.091]). All the other
  differences, including scales 2 and 3 and DiT mid and late, have CIs that include 0.

## Phase F: transfer to crystals

**Question.** Does removing one VAR scale do the same thing to a real image as to VAR's own generations?
The design matches P2's VAR Phase F (`lanes/p2/results/phasef/README.md`), so the two lanes' results can
be compared.

**Design** (`jobs/transfer_var.pbs`, VAR only):
- **Real tokens:** the real image's VQ tokens are teacher-forced at scales 0-5 (`lanes/p2/transfer.py:real_h`,
  `P2_REAL_STAGE=5`), and VAR-d20 generates scales 6-9.
- **Model settings:** cfg 0 (`lanes/p2/jobs/var_cfg0.yaml`), every row conditioned on class 0, as P2. The
  manifest comes from `python -m lanes.p2.transfer manifest`.
- **Arms:** a reference with no removal, and scale j removed alone for each j = 0..9
  (`lanes/p1/hooks.py:remove_scales`, `P1_VAR_SCALES=j`). Both hooks are on the same run. Removing j ≤ 5
  drops a real scale; removing j ≥ 6 drops a generated one.
- **Checks per dataset, before the arms:**
  - **(a) Codec:** `transfer recon --model var`, the VQ tokenizer's reconstruction.
  - **(b) Real scales 0-9, no removal:** must reproduce (a) to within 0.1 dB of mean PSNR (`CHECK_TOL`).
    P2's pilot gave 26.16 vs 26.19 dB. If (b) fails, the job stops and exits 1.
  - **Hooks:** the j = 3 arm runs first. Its `run.json` must list both hooks, and every image must differ
    from the reference (`tensor_sha256`) with the same random draws (`generator_state_sha256`).
    Otherwise the job stops.
- **Scores per image:**
  - PSNR to the real image and `hf_ratio` (`transfer score`), as P2;
  - PSNR and mean absolute difference (0-255) between each arm's output and the reference's output for
    the same image. This is how much removing scale j changes the result, whatever VAR's fidelity to
    the real image (`transfer_summary.py pair`).
- **Raw tensors:** each run's `*.pt` are deleted once it is scored.

**Datasets.** Both use identical settings (class 0, cfg 0, real scales 0-5), so only the image domain
differs.
- **Crystals** (`~/data_crystal_64_p1` on NSCC): OpenCrystalData, "In-line Image Characterization using
  EasyViewer" (Barhate et al., Digital Chemical Engineering, 2024; CC-BY-SA-4.0;
  https://www.kaggle.com/datasets/opencrystaldata/easyviewer-based-image-characterization).
  - `select_crystals.py` takes 32 images per magnification (x100, x400), evenly spaced over each
    magnification's sorted file list: 5 Glutamic Acid + 27 Wollastonite per magnification.
  - Its `sources.csv` sits in the same folder. It is not an image, and `transfer.py` lists only image
    files, so it is ignored.
- **Dogs** (`~/data_dog64_p1`): 64 of VAR-d20's clean baseline PNGs (`outputs/baseline/25665469/var_d20`).
  - `make_dog64.py` sorts the rows of `manifest/frozen_v1_1.csv` by (class_id, seed) and takes positions
    round(i × 223 / 63), i = 0..63. That gives 9 or 10 images from each of the 7 classes.
  - It names them `dog01_…` to `dog64_…` so name order is pick order, and writes `dog64_list.csv`. The
    script uses the standard library only and runs on the login node.

**Submit** (from the repo root on NSCC; use `$HOME`, not `~`, inside `-v`):

```bash
python lanes/p1/make_dog64.py ~/data_dog64_p1                     # login node, once
qsub -v REAL_DIR=$HOME/data_crystal_64_p1,DATASET=crystal lanes/p1/jobs/transfer_var.pbs
qsub -v REAL_DIR=$HOME/data_dog64_p1,DATASET=dog lanes/p1/jobs/transfer_var.pbs
python -m lanes.p1.transfer_summary summary --crystal <JOB> --dog <JOB>   # laptop, after copying both jobs
```

Estimated runtime is about 6-8 min per job, plus start-up. This is scaled from P1's 224-image VAR runs
(~29 s per generate step):
- 12 generate runs of 64 images at ~15-20 s each, including model load and tokenizing;
- the codec check, ~0.5 min;
- scoring and pairing, a few seconds per run.

Each job writes `outputs/p1/<JOBID>/`: `dataset.txt`, `manifest.csv`, `recon/`, `check_k9/`, `reference/`
and `remove_s0` … `remove_s9`, each with PNGs, `run.json`, `transfer.csv` and, for arms,
`vs_reference.csv`. It also writes `steps.tsv` and `summary.txt`, which reports check (b), the hook check
and per-run means.

`transfer_summary.py summary` writes:
- **`results/transfer_var.csv`:** one row per dataset and run. Columns: mean PSNR to the real image with
  its CI; the PSNR drop vs the reference with a 95% paired bootstrap CI; mean `hf_ratio`; and PSNR and MAD
  vs the reference, each a mean with its CI. All CIs use 10000 resamples, seed 0.
- **`results/fig1d_transfer.png`:** change vs removed scale for crystals and dogs, with CI bars and
  scales 0-5 shaded as real. Panel A is MAD vs the reference; panel B is the PSNR drop to the real image.

**Limits.**
- VAR only. P2's DiT transfer starts at t197, which comes after P1's early and mid DiT bands, so those
  bands cannot be removed from a real-image start.
- PSNR, not a classifier: the crystals have no ImageNet class, so the classifier top-1 drop used
  elsewhere in P1 does not apply.
- Class 0 conditioning for both datasets; the model is pulled towards class 0 in both.
- The two image sets are not alike. The dogs are VAR's own outputs, which are in distribution and
  already lie on the model's manifold. The crystals are real photographs. A difference between the two
  therefore mixes "real vs generated" with "dogs vs crystals".
- The reconstruction ceiling is the VQ codec (P2: about 26.2-26.5 dB on crystals), so compare drops and
  changes, not absolute PSNR.
- `psnr_vs_ref` is capped at 99 dB for an image identical to the reference; `n_identical` in the CSV
  counts these.

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
qsub -v SETTINGS=lanes/p1/jobs/single_var.env lanes/p1/jobs/gateb.pbs    # VAR scales 1-8 alone, ~10 min
qsub -v SETTINGS=lanes/p1/jobs/window2_var.env lanes/p1/jobs/gateb.pbs   # VAR windows of 2, ~11 min
qsub -v SETTINGS=lanes/p1/jobs/window3_var.env lanes/p1/jobs/gateb.pbs   # VAR windows of 3, ~7 min
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
`DIT_ARMS="NAME:A-B ..."`, `VAR_CUT`, `DIT_CUT`, `RUN_VAR`, `RUN_DIT`, `QUALITY`, `MANIFEST`,
`BASELINE_REF`, `EXP_TAG`, `KEEP_RAW`, `P3_RIDEALONG`.

**Raw tensors (`KEEP_RAW`).** Each run saves one raw output tensor per image (`*.pt`, about 170 MB per
224-image run). Nothing downstream reads them: scoring uses the PNGs, and each tensor's `tensor_sha256`
stays in `run.json` for the hash checks. `KEEP_RAW=1` (the default) keeps them, so the gateb, sweep and
sanity env files behave as before. With `KEEP_RAW=0` (`single_var.env`, `window2_var.env`,
`window3_var.env`):
- an arm's `*.pt` are deleted as a `drop_raw` step only after its record, image-distance and quality
  steps have all succeeded; a failed arm keeps them;
- the baseline's `*.pt` are deleted only after every arm of the job has finished, and only if no step
  of the job failed;
- only `*.pt` directly inside a run directory (one with a `run.json`) under this job's
  `outputs/p1/<JOBID>/` are deleted. PNGs, `run.json`, metrics and features are kept, and the reference
  baseline `outputs/baseline/25665469` (`BASELINE_REF`) is never touched.

**P3 ride-along (both models).** `P3_RIDEALONG=1` adds P3's observe hooks to the baselines only, never
to a removal arm, each with its own `P3_OUT` set on that one `runner.generate` command:

| Model | Hook | Output | Expected (224 images, 14 batches) |
|---|---|---|---|
| VAR | `lanes/p3/p3_observe.py:var_stages` | `outputs/p1/<JOBID>/p3_var_stages/var/` | 140 .npz (10 scales x 14), ~70 MB |
| DiT | `lanes/p3/p3_observe.py:dit_stages` | `outputs/p1/<JOBID>/p3_dit_stages/dit/` | 336 .npz (24 kept steps x 14), ~90 MB |

They are Observe hooks, so the baseline images must still match the reference; the hash comparison
checks this. `summary.txt` reports each model's .npz count against the expected count (stages kept x
batches: VAR from the baseline's stages table, DiT from the hook's own step selection). A wrong count
or a `_failures.log` fails the job: the hook never fails the run itself, so this is where a silent loss
shows up. A DiT count of 16 is the old call-counter bug.

`jobs/p3_both.env` runs only the two baselines with both hooks (`VAR_ARMS=""`, `DIT_ARMS=""`: an empty
list means baseline only; an unset one still takes the default), `QUALITY=0`, `KEEP_RAW=0`:

```bash
qsub -v SETTINGS=lanes/p1/jobs/p3_both.env lanes/p1/jobs/gateb.pbs
python -m lanes.p1.compare_p3_var outputs/p1/25891860/p3_var_stages outputs/p1/<JOBID>/p3_var_stages
```

`compare_p3_var.py` (numpy only) checks the new VAR snapshots against Gate B's (job 25891860). It needs
the same .npz names, and in each pair equal stage, class_id and seed and a bit-identical latent. It
prints the counts and PASS or FAIL (exit 0 or 1); other fields are reported but don't decide.
