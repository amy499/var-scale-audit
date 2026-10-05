# P2 pilot: noise corruption on VAR and DiT (job 25656645)

First real run of the P2 corruption hooks. It shows the hooks work on GPU and how the two models
respond. **Provisional:** 16 images on `manifest/provisional_4x4.csv`, one noise draw, two severities.
It is a feasibility check (plan Phase B), not a result for Figure 2.

## What was run

| | |
|---|---|
| Job | `lanes/p2/jobs/pilot.pbs`, NSCC A100-SXM4-40GB, 2 Oct 2026 |
| Code | `riddhimajain/b2-logging-reproducibility` @ `dd8997f` (merged into `main` as PR #2) |
| Configs | `configs/var_d20.yaml` (strict fp32), `configs/dit_xl2_256.yaml`, batch 16 |
| Images | 16: classes 207 golden retriever, 360 otter, 387 lesser panda, 974 geyser × seeds 0-3 |
| VAR | `noise_h` (noise on scale k's own contribution) at scales 0, 3, 6, 9 |
| DiT | `noise_x` (noise on the latent after the step) at timesteps 798, 497, 197 |
| Severity | 0.25 and 1.0: noise std = severity × that image's std of the corrupted tensor (`P2_SCALE=rel`) |
| Pairing | All 14 corrupted runs pass the pairing check against their baseline (`runs.csv`, `pairing_ok`) |

## Files

| File | Content |
|---|---|
| `grid_var.jpg`, `grid_dit.jpg` | Rows: 207 seed 0, 360 seed 1, 387 seed 2, 974 seed 3. Columns: baseline, then each stage × severity |
| `p2_summary.csv` | One row per corrupted run (`lanes/p2/README.md`, "Analysis"), including the quality columns |
| `p2_curves.csv` | State gap to the baseline at every traced stage (recovery curves) |
| `runs.csv` | Every run: settings, pairing, timing, peak VRAM, code commit |

Full outputs (images, `run.json`, `experiment.json`, `metrics.csv`, logs) are on NSCC in
`outputs/p2/pilot/25656645/`. They are not in git.

## Numbers

Means over the 16 images. PSNR and MAD compare each image with its own baseline image (higher PSNR /
lower MAD = closer to the baseline). Gap ratio = state distance to the baseline at the end ÷ right
after the corruption (< 1: the gap shrank; > 1: it grew).

| Model | Hook | Stage | p | Severity | PSNR (dB) | MAD (0-255) | Gap ratio |
|---|---|---|---|---|---|---|---|
| VAR | noise_h | 0 | 0.00 | 0.25 | 12.6 | 39.1 | 28.13 |
| VAR | noise_h | 0 | 0.00 | 1.0 | 10.8 | 53.0 | 7.64 |
| VAR | noise_h | 3 | 0.33 | 0.25 | 15.5 | 26.5 | 19.73 |
| VAR | noise_h | 3 | 0.33 | 1.0 | 13.9 | 34.4 | 5.58 |
| VAR | noise_h | 6 | 0.67 | 0.25 | 19.1 | 16.3 | 9.89 |
| VAR | noise_h | 6 | 0.67 | 1.0 | 16.5 | 23.1 | 3.15 |
| VAR | noise_h | 9 | 1.00 | 0.25 | 31.9 | 3.4 | 1.00 |
| VAR | noise_h | 9 | 1.00 | 1.0 | 21.1 | 13.4 | 1.00 |
| DiT | noise_x | 798 | 0.20 | 0.25 | 22.1 | 11.0 | 0.78 |
| DiT | noise_x | 798 | 0.20 | 1.0 | 17.1 | 22.3 | 0.37 |
| DiT | noise_x | 497 | 0.50 | 0.25 | 20.5 | 13.8 | 1.03 |
| DiT | noise_x | 497 | 0.50 | 1.0 | 12.0 | 47.1 | 0.69 |
| DiT | noise_x | 197 | 0.80 | 0.25 | 26.0 | 7.7 | 0.64 |
| DiT | noise_x | 197 | 0.80 | 1.0 | 12.6 | 44.6 | 0.71 |

Cost: VAR 0.11 s/image, DiT 1.96 s/image (batch 16, after model load). Peak GPU memory (`nvidia-smi`,
includes cache): VAR 17.1 GB, DiT 10.1 GB.

## What the pilot shows (from the grids and the table)

- **VAR re-routes, it does not break.** Noise at scale 0 gives a different but clean image of the
  same class (another dog, another otter scene). At scale 3 the scene stays and its layout shifts; at
  scales 6 and 9 the image is nearly unchanged, even at severity 1.0. The state gap grows after
  early corruption (ratio up to 28): the generation takes another path, not a damaged one.
- **DiT repairs early and breaks late.** At timestep 798 the image stays clean (gap shrinks to 0.37×
  at severity 1.0). At 497 and 197, severity 1.0 gives oversaturated colours and artifacts (497) or
  visible leftover noise (197), while severity 0.25 stays clean.
- **Pixel distance cannot separate "changed" from "damaged".** VAR at scale 0 has the lowest PSNR of
  all runs, yet its images are clean; DiT at 197 / severity 1.0 has similar PSNR and is destroyed.
  Recovery has to be measured with a quality / semantic metric, with pixel distance only as "how much
  changed".

## Quality scores (added 5 Oct, computed afterwards on these images)

Scored on CPU with `lanes/p2/quality.py` (Inception-v3 ImageNet classifier and features; see
`lanes/p2/README.md`, "Quality"). Class prob = mean probability of the image's own class (baseline in
brackets); same top-1 = fraction of images whose top prediction matches the baseline image's; KID =
paired KID against the baseline (0 = indistinguishable as a set).

| Model | Stage | Severity | PSNR (dB) | Class prob (baseline) | Same top-1 | KID |
|---|---|---|---|---|---|---|
| VAR | 0 | 0.25 | 12.6 | 0.82 (0.75) | 0.88 | +0.0002 |
| VAR | 0 | 1.0 | 10.8 | 0.74 (0.75) | 0.69 | −0.0030 |
| VAR | 3 | 0.25 | 15.5 | 0.76 (0.75) | 0.81 | −0.0005 |
| VAR | 3 | 1.0 | 13.9 | 0.81 (0.75) | 0.81 | +0.0006 |
| VAR | 6 | 0.25 | 19.1 | 0.74 (0.75) | 0.81 | +0.0000 |
| VAR | 6 | 1.0 | 16.5 | 0.74 (0.75) | 0.88 | −0.0003 |
| VAR | 9 | 0.25 | 31.9 | 0.75 (0.75) | 1.00 | −0.0000 |
| VAR | 9 | 1.0 | 21.1 | 0.71 (0.75) | 0.75 | +0.0025 |
| DiT | 798 | 0.25 | 22.1 | 0.85 (0.82) | 0.94 | −0.0001 |
| DiT | 798 | 1.0 | 17.1 | 0.81 (0.82) | 0.94 | +0.0000 |
| DiT | 497 | 0.25 | 20.5 | 0.77 (0.82) | 0.88 | +0.0009 |
| DiT | 497 | 1.0 | 12.0 | 0.73 (0.82) | 0.81 | **+0.0185** |
| DiT | 197 | 0.25 | 26.0 | 0.84 (0.82) | 0.94 | −0.0002 |
| DiT | 197 | 1.0 | 12.6 | **0.00** (0.82) | **0.00** | **+0.1872** |

- The classifier keeps every VAR run near its baseline, including scale 0 with the lowest PSNR: the
  re-routed images are still good images of their class.
- It catches DiT 197 / severity 1.0 (0.00) but **not** DiT 497 / severity 1.0 (0.73), whose images
  are visibly oversaturated: it measures meaning, not visual quality.
- KID catches both damaged DiT runs and keeps every clean-looking run within ±0.003. With 16 images
  it is noisy, so these values show the direction, not a reliable size.

## Decisions for the next run

1. **Severities 0.25, 0.5, 1, 2, 4.** DiT changes from clean to broken between 0.25 and 1.0; VAR's
   late scales barely react at 1.0.
2. **Stages:** all 10 VAR scales; about 8 DiT timesteps, denser between 798 and 197.
3. **Metric:** classifier scores + KID (`lanes/p2/quality.py`) from now on, as P2's interim measure,
   to be checked against P3's calibrated metrics.
4. **Manifest:** rerun on `manifest/frozen_v1_1.csv` (7 dog classes × 32 images) once P3 signs it.
5. The corrupted images here are graded, known degradations. They could help P3 calibrate metrics
   ("calibrate on known mild/heavy degradations", plan §8).
