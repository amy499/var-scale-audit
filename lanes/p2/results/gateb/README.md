# P2 Gate B: noise corruption on manifest v1.1

Gate B (11 Oct) asks for one valid experiment per lane on VAR and on DiT. This is P2's: noise injected
at each stage and severity, on the 224-image dog manifest, scored with pixel distance, an ImageNet
classifier and KID. **Provisional:** manifest v1.1 is P1's draft pending P3's sign-off; one noise draw
per image; interim metrics (`lanes/p2/quality.py`) until P3's calibrated ones exist.

| Part | Job | Status |
|---|---|---|
| VAR | `25689627` | Done, below |
| DiT | `25689628` | Done, below |

**Gate B for P2: met (provisionally).** Both models: corruption runs on 224 images, every run paired
with its baseline, scored, and showing a clear stage × severity effect.

## VAR (job 25689627)

| | |
|---|---|
| Command | `qsub -P personal-riddhima -l walltime=02:00:00 -v SETTINGS=lanes/p2/jobs/gateb_var.env lanes/p2/jobs/pilot.pbs` |
| Code | `d414064` (on `main`); `dirty` only because of an untracked copy of `lanes/p2/jobs/smoke.pbs`, not used by this job |
| Manifest | `frozen_v1_1.csv` draft, sha256 `caf0535d9cb0…`: classes 151 Chihuahua, 153 Maltese, 162 beagle, 207 golden retriever, 235 German shepherd, 254 pug, 258 Samoyed × 32 images |
| Model | `configs/var_d20.yaml` (strict fp32), batch 16, NSCC A100-SXM4-40GB |
| Corruption | `noise_h` (noise on scale k's own contribution) at every scale 0-9 × severity 0.25, 0.5, 1, 2, 4 (noise std / that image's std of the tensor) |
| Runs | 1 baseline + 50 corrupted, all pass the pairing check |
| Cost | 0.10-0.12 s/image; 39 min for the job; peak GPU memory 17.1 GB (`nvidia-smi`) |
| Baseline | Class probability 0.81, top-1 94 %, top-5 100 % |

### Stage × severity maps (means over 224 images)

**Classifier probability of the image's own class** (baseline 0.81; lower = the image lost its class):

| Scale \ severity | 0.25 | 0.5 | 1 | 2 | 4 |
|---|---|---|---|---|---|
| 0 | 0.79 | 0.80 | 0.72 | 0.45 | 0.06 |
| 1 | 0.81 | 0.79 | 0.73 | 0.56 | 0.13 |
| 2 | 0.81 | 0.82 | 0.77 | 0.57 | 0.06 |
| 3 | 0.81 | 0.83 | 0.80 | 0.54 | 0.06 |
| 4 | 0.83 | 0.83 | 0.80 | 0.60 | 0.06 |
| 5 | 0.82 | 0.82 | 0.82 | 0.67 | 0.06 |
| 6 | 0.83 | 0.81 | 0.80 | 0.61 | 0.01 |
| 7 | 0.83 | 0.82 | 0.80 | 0.52 | 0.00 |
| 8 | 0.81 | 0.81 | 0.80 | 0.52 | 0.00 |
| 9 | 0.82 | 0.82 | 0.77 | **0.00** | 0.00 |

**KID against the baseline** (0 = the set looks like the clean set; higher = visibly damaged):

| Scale \ severity | 0.25 | 0.5 | 1 | 2 | 4 |
|---|---|---|---|---|---|
| 0 | 0.000 | 0.000 | 0.002 | 0.022 | 0.071 |
| 1 | 0.000 | 0.000 | 0.003 | 0.025 | 0.165 |
| 2 | 0.000 | 0.000 | 0.003 | 0.040 | 0.205 |
| 3 | 0.000 | 0.000 | 0.002 | 0.042 | 0.237 |
| 4 | 0.000 | 0.000 | 0.001 | 0.028 | 0.224 |
| 5 | 0.000 | 0.000 | 0.001 | 0.016 | 0.226 |
| 6 | 0.000 | 0.000 | 0.001 | 0.033 | 0.288 |
| 7 | 0.000 | 0.000 | 0.001 | 0.055 | 0.340 |
| 8 | 0.000 | 0.000 | 0.001 | 0.049 | 0.423 |
| 9 | 0.000 | 0.000 | 0.004 | **0.342** | 0.588 |

**Pixel PSNR against the baseline image** (dB; lower = more changed): from 12.2 (scale 0) to 33.6
(scale 9) at severity 0.25, and between 8.9 and 12.4 at severity 4. Full table: `var_p2_summary.csv`.

### What it shows

Grids (first image of each class): `var_scales_sev2.jpg` (severity 2 at every scale) and
`var_severity_s0_s9.jpg` (every severity at the first and the last scale).

1. **Up to severity 1, VAR absorbs the corruption at every scale.** Class probability stays 0.72-0.83
   and KID ≤ 0.004. Early corruption changes *which* image is drawn (PSNR 10.7-12.2 dB at scale 0) but
   it is still a clean image of the right breed.
2. **Severity 2 is where damage starts to stick, and it sticks more the later it happens.**
   - At scale 9 nothing runs afterwards: class probability falls to 0.00 and KID to 0.34.
   - At scales 0-8 the later scales repair part of it: class probability 0.45-0.67, KID 0.02-0.06.
   - In the grid, early scales give re-composed scenes, middle and late scales keep the layout with
     growing mosaic artifacts, and scale 9 is fully broken.
3. **Two different failure modes:**
   - **Early (scale 0-1), high severity: semantic drift.** The image stays clean but the dog
     disappears: at severity 4, scale 0 gives textures and patterns (class probability 0.06, but KID
     only 0.07, because they are clean images, just not of a dog).
   - **Late (scale 6-9), high severity: structural damage.** The layout survives, but the image is
     broken into mosaic artifacts (KID 0.29-0.59).
4. **The state gap grows after early corruption** (`gap_l2_ratio` 28 at scale 0, severity 0.25; 1.0
   at scale 9 by definition). VAR does not pull the trajectory back to the baseline; later scales
   build a different image on top. "Recovery" for VAR means "the result is still a good image",
   not "the result is the original image".

The classifier and KID disagree in exactly the useful place: semantic drift (classifier low, KID
low) vs structural damage (both bad). Both are needed.

### Caveats

- Manifest v1.1 is a draft; if P3 changes it, the run is repeated (~40 min).
- One noise draw per image and stage; no confidence intervals yet.
- The classifier is an ImageNet Inception-v3, a proxy for "still the right breed"; KID on 224 images
  is a set-level estimate. Both are interim until P3's calibrated metrics.
- Severity is relative to each image's own `h_BChw` at that scale; scale 0's contribution is a
  single 1×1 token embedding, so "severity 2" means different absolute changes at different scales.

## DiT (job 25689628)

| | |
|---|---|
| Command | `qsub -P personal-riddhima -l walltime=02:00:00 -v SETTINGS=lanes/p2/jobs/gateb_dit.env lanes/p2/jobs/pilot.pbs` |
| Code, manifest | As for VAR (`d414064`, v1.1 draft `caf0535d9cb0…`) |
| Model | `configs/dit_xl2_256.yaml` (fp32 + TF32, 250 steps, cfg 1.5), batch 16, NSCC A100-SXM4-40GB |
| Corruption | `noise_x` (noise on the latent after the step) at timesteps 798 (p 0.20), 497 (p 0.50), 197 (p 0.80) × severity 0.25, 0.5, 1 |
| Runs | 1 baseline + 9 corrupted, all pass the pairing check |
| Cost | 1.95 s/image; 1 h 19 min for the job; peak GPU memory 10.1 GB (`nvidia-smi`) |
| Baseline | Class probability 0.76, top-1 91 % |

| Timestep (p) | Severity | PSNR (dB) | Class prob (base 0.76) | Same top-1 | KID | Gap ratio |
|---|---|---|---|---|---|---|
| 798 (0.20) | 0.25 | 21.9 | 0.74 | 0.86 | 0.000 | 0.85 |
| 798 (0.20) | 0.5 | 18.7 | 0.74 | 0.86 | 0.000 | 0.64 |
| 798 (0.20) | 1 | 15.6 | 0.74 | 0.83 | 0.000 | 0.46 |
| 497 (0.50) | 0.25 | 20.5 | 0.73 | 0.83 | 0.000 | 1.13 |
| 497 (0.50) | 0.5 | 16.6 | 0.72 | 0.83 | 0.001 | 0.86 |
| 497 (0.50) | 1 | 11.1 | **0.32** | 0.38 | **0.056** | 0.72 |
| 197 (0.80) | 0.25 | 25.8 | 0.74 | 0.90 | 0.000 | 0.67 |
| 197 (0.80) | 0.5 | 19.1 | **0.50** | 0.64 | **0.014** | 0.77 |
| 197 (0.80) | 1 | 11.9 | **0.00** | 0.00 | **0.187** | 0.75 |

Grid: `dit_timesteps_severity.jpg` (first image of each class).

1. **Early (798): repaired at every severity tested.** Class probability and KID stay at the
   baseline up to severity 1. Most images keep their scene; a few are re-composed.
2. **The tolerated severity falls as generation progresses:** about ≥ 1 at 798, between 0.5 and 1 at
   497, between 0.25 and 0.5 at 197.
3. **The damage looks different by stage.** At 497 / severity 1, images become oversaturated and
   cartoon-like, often with the dog still there (class probability 0.32, KID 0.056). At 197, leftover
   noise: speckle at 0.5, destroyed at 1 (0.00, KID 0.187).
4. **DiT pulls the trajectory back towards the original.** The state gap shrinks after the
   corruption (ratio 0.46-0.86, except 1.13 at 497 / 0.25), the opposite of VAR's ratios above 1.

## VAR vs DiT at matched progress

Severity is matched by rule (noise std = severity × that image's std of the corrupted tensor), not
by mechanism: VAR's `h_BChw` and DiT's latent are different objects, and the plan warns against
treating scales and timesteps as equivalent. Compare the **trends**, not the absolute thresholds.
VAR scales are the nearest by `p` (2, 5, 7 for p 0.22, 0.56, 0.78).

| Progress | Severity | VAR class prob (base 0.81) | VAR KID | DiT class prob (base 0.76) | DiT KID |
|---|---|---|---|---|---|
| early (VAR s2 / DiT 798) | 0.5 | 0.82 | 0.000 | 0.74 | 0.000 |
| early | 1 | 0.77 | 0.003 | 0.74 | 0.000 |
| middle (s5 / 497) | 0.5 | 0.82 | 0.000 | 0.72 | 0.001 |
| middle | 1 | 0.82 | 0.001 | **0.32** | **0.056** |
| late (s7 / 197) | 0.5 | 0.82 | 0.000 | **0.50** | **0.014** |
| late | 1 | 0.80 | 0.001 | **0.00** | **0.187** |

- **Common to both:** the later the corruption, the less of it is repaired (VAR shows this from
  severity 2, DiT from 0.5).
- **Different:** within severity 1, VAR is unaffected at every stage, while DiT's late stages break.
  DiT repairs towards the original image; VAR builds a different, still valid image.
- DiT has not been tested above severity 1, and only at 3 timesteps, so its full map is open.

## Next (Phase C, 12-18 Oct)

1. DiT: severities 2 and 4 at least early, and about 8 timesteps between 999 and 0 to place the
   threshold (~5 GPU-hours at 224 images: split into several jobs).
2. A second noise draw (`P2_NOISE_SEED=1`) for variability; confidence intervals over images.
3. P3's calibrated metrics next to the interim ones; rerun if manifest v1.1 changes at sign-off.
4. Plots on P4's template once it exists (stage tables below).

## For P4: stage tables and progress

`stages_var.csv` and `stages_dit.csv` are the per-stage tables the runner writes into every
`run.json` (`stages`), one copy per model. They are identical for every run of a model with the same
config (checked: 51 VAR runs, 7 DiT runs, one distinct table each) and do not depend on the manifest.
A run with `--skip-timesteps` would have fewer DiT rows, with `merged = True` on the step before a gap.

| Column | Model | Meaning |
|---|---|---|
| `stage` | both | Native stage id: VAR scale 0-9; DiT timestep 999 … 0. The key used by every P2 table (`metrics.csv`, `p2_curves.csv`, `p2_summary.csv`) |
| `p` | both | Progress in [0, 1]: VAR `scale / 9`; DiT `step / 249`. The runner's default axis |
| `pn`, `n_tokens`, `cum_tokens`, `total_tokens` | VAR | Token grid side, tokens at this scale, tokens so far, 680 in total. `cum_tokens / total_tokens` is a token-weighted progress (scale 9 alone is 256 of 680 tokens) |
| `step`, `merged`, `timestep_in`, `timestep_out` | DiT | Step index 0-249; timesteps into and out of the step |
| `alpha_bar_in/out`, `sigma_in/out` | DiT | Noise level of the latent into and out of the step; `1 - sigma_in` is a noise-based progress |

So the three candidate progress axes for the early/mid/late mapping are all here: step-uniform `p`,
VAR token-weighted progress, and DiT noise-based progress. `var_p2_curves.csv` is an example of
per-stage data to plot against them: the mean state gap to the baseline (`state_l2:f_hat`,
`state_rel_l2:f_hat`, `state_cos:f_hat`) at every scale, for every corrupted VAR run, with `stage`
and `p` columns.

## Files

| File | Content |
|---|---|
| `var_p2_summary.csv` | One row per corrupted VAR run: stage, p, severity, pixel damage, state gaps, classifier and KID columns |
| `var_p2_curves.csv` | Per corrupted VAR run and scale: mean state gap to the baseline (recovery curves) |
| `var_runs.csv` | Every VAR run: settings, pairing, timing, VRAM, code commit |
| `dit_p2_summary.csv`, `dit_p2_curves.csv`, `dit_runs.csv` | The same for DiT |
| `dit_timesteps_severity.jpg` | DiT comparison grid |
| `stages_var.csv`, `stages_dit.csv` | Stage tables for P4's progress mapping (above) |
| `var_scales_sev2.jpg`, `var_severity_s0_s9.jpg` | Comparison grids |

Full outputs are on NSCC in `outputs/p2/gateb/25689627/` and `outputs/p2/gateb/25689628/`. The Mac copy in `outputs/p2/gateb/`
keeps the tables, every run's records, and 2 sample images per class per run.
