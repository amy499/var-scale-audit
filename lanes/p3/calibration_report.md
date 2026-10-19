# P3 Metric Calibration Report

**AI6132 Group 17 — VAR vs DiT stage audit**
Owner: P3 (information/semantics lane) · 1 Oct 2026, addendum 20 Oct 2026

Reproduce with `python -m lanes.p3.calibrate_metrics` (no model checkpoints needed).

---

## Method

Six candidate metrics applied to five varied natural images (256×256, grayscale) under
three degradations at six severity levels each, with the direction **predicted in advance**
for every metric/degradation pair.

| Degradation | Severities | Simulates |
|---|---|---|
| Gaussian blur | σ = 0, 0.5, 1, 2, 4, 8 | early VAR scale (missing high frequency) |
| Gaussian noise | σ = 0, 2, 5, 10, 20, 40 (/255) | early DiT timestep (excess high frequency) |
| Downsample→upsample | factor 1, 2, 4, 8, 16, 32 | VAR's coarse token grids |

Pass requires all three: direction matches the prediction on 100% of images; mean
|Spearman ρ| ≥ 0.90 with the correct sign; median relative change at the *mildest* non-zero
severity ≥ 2%. Everything is measured against each image's **own** clean baseline, because
between-image variance in natural images otherwise swamps the effect.

## Results

| Degradation | Metric | Expected | ρ | Direction | Sensitivity | Verdict |
|---|---|---|---|---|---|---|
| blur | hf_ratio | down | −1.000 | 100% | 42.6% | PASS |
| blur | lf_ratio | up | +1.000 | 100% | 3.5% | PASS |
| blur | spectral_centroid | down | −1.000 | 100% | 17.9% | PASS |
| blur | lap_var | down | −1.000 | 100% | 58.4% | PASS |
| blur | grad_energy | down | −1.000 | 100% | 15.6% | PASS |
| blur | pixel_entropy | — | −0.497 | — | 0.3% | NO PREDICTION |
| noise | hf_ratio | up | +1.000 | 100% | 2.0% | PASS |
| noise | lf_ratio | down | −1.000 | 100% | 0.1% | WEAK |
| noise | spectral_centroid | up | +1.000 | 100% | 0.7% | WEAK |
| noise | lap_var | up | +1.000 | 100% | 6.1% | PASS |
| noise | grad_energy | up | +1.000 | 100% | 5.0% | PASS |
| noise | pixel_entropy | — | +0.657 | — | 0.6% | NO PREDICTION |
| downsample | hf_ratio | down | −1.000 | 100% | 72.4% | PASS |
| downsample | lf_ratio | up | +1.000 | 100% | 5.2% | PASS |
| downsample | spectral_centroid | down | −1.000 | 100% | 28.5% | PASS |
| downsample | lap_var | down | −1.000 | 100% | 84.7% | PASS |
| downsample | grad_energy | down | −1.000 | 100% | 22.9% | PASS |
| downsample | pixel_entropy | — | −0.211 | — | 0.3% | NO PREDICTION |

## Findings

**1. `pixel_entropy` is dropped.** No consistent direction under any degradation (ρ between
−0.50 and +0.66), movement under 1% throughout, curves crossing between test images.

**2. `hf_ratio` and `lap_var` saturate under coarsening.** Both collapse to ≈0 by blur σ=2 /
downsample factor 4 and flatline. Their large sensitivity numbers are misleading: almost all
of the movement is spent in the first step.

**3. `spectral_centroid` and `grad_energy` stay graded** across the full severity range in
both directions.

**4. The same metric means opposite things in the two models.** `hf_ratio` falls under blur
and rises under noise. VAR's early scales are blur-like; DiT's early timesteps are
noise-like. A single "high-frequency energy" axis carrying both curves is not a comparison.

---

## Addendum, 20 Oct 2026

### 5. On VAR's actual scale ladder, the pixel metrics are worse than the severity sweep suggests

Downsampling real photos to VAR-d20's own grids (1, 2, 3, 4, 5, 6, 8, 10, 13, 16) and
measuring gives the *best case* at each scale — an upper bound on what the model could have
committed, with no GPU time:

| si | grid | tokens | cum % | spec_centroid | grad_energy | hf_ratio |
|---|---|---|---|---|---|---|
| 0 | 1×1 | 1 | 0.1% | NaN | 0.00000 | NaN |
| 1 | 2×2 | 4 | 0.7% | 0.0157 | 0.00106 | 0.000000 |
| 2 | 3×3 | 9 | 2.1% | 0.0157 | 0.00171 | 0.000000 |
| 3 | 4×4 | 16 | 4.4% | 0.0157 | 0.00227 | 0.000000 |
| 4 | 5×5 | 25 | 8.1% | 0.0159 | 0.00275 | 0.000000 |
| 5 | 6×6 | 36 | 13.4% | 0.0162 | 0.00330 | 0.000000 |
| 6 | 8×8 | 64 | 22.8% | 0.0179 | 0.00447 | 0.000000 |
| 7 | 10×10 | 100 | 37.5% | 0.0184 | 0.00526 | 0.000000 |
| 8 | 13×13 | 169 | 62.4% | 0.0219 | 0.00685 | 0.000000 |
| 9 | 16×16 | 256 | 100% | 0.0226 | 0.00786 | 0.000001 |
| — | full 256 | 680 | 100% | 0.0934 | 0.03208 | 0.053706 |

**`hf_ratio` is identically zero across the whole ladder.** `spec_centroid` moves only
0.0157 → 0.0226 across si 1–9, against 0.0934 for the full image. `grad_energy` is the only
pixel metric with real range (7.4×).

**Consequence for every lane: a flat line across VAR's early band is a floor effect, not a
finding.** Any figure using a pixel-space metric over VAR's early scales needs that caveat.

### 6. New primary metric: `latent_rel_dist`

Finding 5 exposes a deeper problem. Every pixel metric runs on a **decoded** intermediate,
so it measures the model's commitment and the VQVAE decoder's behaviour together, and the
two cannot be separated afterwards.

`latent_rel_dist` — relative L2 from a stage's latent to the run's final latent — reads the
model's own state instead. 1.0 = nothing committed, 0.0 = fully determined. It needs no
decoder and works identically for VAR (`f_hat`) and DiT (`pred_xstart`). The two latent
spaces differ, so compare the **shape** of the curves, never the absolute values.

This is now the primary stage metric; the pixel metrics are supporting evidence.

### 7. Bug found and fixed: flat images return plausible nonsense

A constant image has a normalised power spectrum built entirely from floating-point noise.
`spectral_centroid` returns **0.556** on one — a value indistinguishable from a real
measurement. VAR scale 0 is a 1×1 token grid and decodes to a near-constant image, so this
case **is** reached in the real sweep, and it would have become a fabricated data point in
every figure.

`p3_metrics.py` now returns `NaN` below `FLAT_IMAGE_STD = 1e-6`. Any new spectral metric
needs the same guard.

### 8. Baseline contact sheets reviewed (breed check)

Both contact sheets from job 25665469 were checked image by image against the adjacent
ImageNet classes. **All 7 classes correct in both models; no off-by-one.** Details in
`P3_PLAN.md`.

Two observations with consequences for this lane:

- **A few baseline images are themselves generation failures** (notably VAR class 207
  image 4, a saturated-yellow frame, and VAR class 162 images 3 and 8, near-abstract).
  These must be identified by low baseline `cls_prob` and reported separately, not silently
  averaged into the stage curves.
- **VAR's output shows markedly heavier posterization and flat-colour patches than DiT's.**
  That is the signature of codebook quantization, and it is consistent with Finding 6: VAR's
  decoder imposes its own character on the output. A cheap confirmation is to measure the
  stage metrics on *final* baseline images from both models — a systematic offset there is a
  per-model signature that must be normalised out before any cross-model frequency claim.

## Decisions

| Metric | Status | Use |
|---|---|---|
| `latent_rel_dist` | **PRIMARY** | both models, decoder-independent |
| `grad_energy` | **PRIMARY pixel** | only pixel metric with range on the VAR ladder |
| `spectral_centroid` | **SUPPORTING** | graded but compressed on the VAR ladder |
| `hf_ratio` | **LATE STAGES ONLY** | ≈0 across the whole VAR ladder |
| `lap_var` | **LATE STAGES ONLY** | saturates by blur σ=2 |
| `lf_ratio` | **DEMOTED** | near-blind to noise (0.1%) |
| `pixel_entropy` | **DROPPED** | fails direction and sensitivity |

**Store the full 64-bin radial profile per stage**, not only the scalars: it lets the bands
be re-cut later without re-running the GPU sweep.

## Not yet calibrated

`semantic_similarity` (CLIP or classifier cosine similarity to the clean reference) needs
torch. Wire it up and re-run this harness. `incep_cos` in `p3_metrics.py` is the cheap
stand-in and needs no extra dependency.
