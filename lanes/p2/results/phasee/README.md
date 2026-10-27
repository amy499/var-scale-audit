# P2 Phase E: robustness checks of the corruption findings

Three checks of what Phase C found (`../phasec/README.md`), on the frozen manifest v1.1 (224 images,
7 dog classes), plus confidence intervals for every run. All 80 runs pass the pairing check.

| Job | Settings | What it tests |
|---|---|---|
| `26023658`, `26023660` | `phasee_dit_sigma_a/b.env` | E1: DiT with severity in units of the step's own noise level |
| `26023654`, `26023657` | `phasee_var_twostage.env`, `phasee_dit_twostage.env` | E2: corruption at two stages in one run |
| `26023656` | `phasee_var_d24.env` | E3: the VAR grid on VAR-d24 (1.0B parameters) |
| (CPU) | `python -m lanes.p2.analyze ci` | E4: 95 % bootstrap intervals over images, 1000 resamples (`p2_ci.csv`, also `../phasec/p2_ci.csv`) |

## E1: is DiT's late fragility real? (`P2_SCALE=sigma`)

Phase C measured severity relative to the latent's own spread (`rel`). Early on the latent is still
almost pure noise, so that is a small change there; late, the latent is nearly clean, so the same
`rel` severity is far more noise than the model expects at that step. `sigma` mode sets the noise to
severity × the step's own noise level σ instead: the same "surprise" at every step.

Fraction of images damaged at severity 1 (95 % interval for two cells):

| Timestep (p) | σ at that step | `rel` (Phase C) | `sigma` (E1) |
|---|---|---|---|
| 798 (0.20) | 1.00 | 0.05 | 0.06 |
| 598 (0.40) | 0.99 | 0.21 | 0.21 |
| 497 (0.50) | 0.96 | 0.58 | 0.55 |
| 397 (0.60) | 0.89 | 0.91 | 0.83 |
| 297 (0.70) | 0.77 | 1.00 | **0.88** [0.84, 0.92] |
| 197 (0.80) | 0.57 | 1.00 | 0.65 |
| 96 (0.90) | 0.30 | 1.00 | **0.00** [0.00, 0.02] |
| 36 (0.96) | 0.12 | 1.00 | **0.00** |

- **Early, the two definitions agree** (σ ≈ 1, so they inject nearly the same noise), and so do the
  results: a built-in consistency check.
- **Late, they split.** Under `sigma`, the last steps absorb the noise completely (grid
  `dit_sigma_sev1.jpg`: timesteps 96 and 36 look like the baseline). So "DiT's final steps cannot
  recover" in Phase C was largely the severity definition: there, `rel` noise is several times what
  the step expects.
- **What survives both definitions:** DiT's **middle-to-late steps (p ≈ 0.5-0.8) are its fragile
  region**, and its first steps repair almost anything. At severity 2 under `sigma`, damage is 85-100 %
  from 598 to 96, and only 1 % at 36.
- Report both: `rel` answers "same disturbance to the image state", `sigma` "same disturbance relative
  to what the sampler expects". They agree on where DiT is robust (early) and where it is fragile
  (middle-to-late), and disagree only on the final steps.

## E2: does damage add up across two stages?

Each stage of a two-stage run gets exactly the noise of the single-stage run at that stage (the noise is
seeded per image, stage and tensor), so the pair can be compared with its two singles directly
(VAR singles: Gate B, same noise seed and bit-identical baseline; DiT singles: Phase C). "If
independent" is 1 − (1 − a)(1 − b), the damage expected if the two corruptions acted separately.

| Pair | Severity | Damaged: single a, single b | If independent | **Pair** | KID: a + b | **KID pair** |
|---|---|---|---|---|---|---|
| VAR 3 + 6 | 2 | 0.25, 0.12 | 0.35 | **0.70** [0.64, 0.76] | 0.075 | **0.154** |
| VAR 1 + 8 | 2 | 0.29, 0.18 | 0.42 | **0.70** [0.63, 0.76] | 0.074 | **0.125** |
| VAR 0 + 5 | 2 | 0.42, 0.09 | 0.48 | **0.60** | 0.038 | **0.063** |
| VAR 0 + 9 | 2 | 0.42, 1.00 | 1.00 | 1.00 | 0.364 | 0.263 |
| VAR 5 + 9 | 2 | 0.09, 1.00 | 1.00 | 1.00 | 0.358 | 0.350 |
| VAR pairs | 1 | 0.00-0.12 each | 0.00-0.12 | 0.00-0.19 | ≤ 0.006 | 0.004-0.010 |
| DiT 497 + 197 | 0.5 | 0.05, 0.13 | 0.17 | **0.32** | 0.015 | **0.034** |
| DiT 798 + 197 | 0.5 | 0.00, 0.13 | 0.13 | 0.16 | 0.014 | 0.016 |
| DiT 798 + 497 | 1 | 0.05, 0.58 | 0.61 | 0.59 | 0.056 | 0.068 |
| DiT 497 + 197, 798 + 197 | 1 | 0.58 / 0.05, 1.00 | 1.00 | 0.99 | | |

- **Damage compounds.** Two corruptions that are each survivable do roughly twice the harm together that
  independence predicts (VAR 3 + 6 and 1 + 8: 70 % vs 35-42 %; KID about double the sum). A trajectory
  that has partly absorbed one corruption is less able to absorb the next.
- **DiT forgets early damage.** Any pair with timestep 798 behaves like its later stage alone (798 + 197:
  16 % vs 13 %): the early corruption is fully repaired before the second one arrives. VAR has no such
  stage: even its most robust scale (5) adds to scale 0's damage.
- **A fatal stage dominates.** With VAR scale 9 at severity 2 or DiT 197 at severity 1, the pair is as
  bad as that stage alone.
- At severity 1 every VAR pair stays near clean (KID ≤ 0.010) but tips several runs from "unchanged"
  to "degraded-in-place" (KID just above P3's 0.005 threshold).

## E3: does VAR's pattern hold on a bigger model? (VAR-d24)

d24 has its own baseline (class probability 0.82; d20: 0.81); the two models are never compared image
by image, only map by map. Fraction of images damaged:

| Scale | d24, sev 1 | d24, sev 2 | d20, sev 2 | d24, sev 4 | d20, sev 4 |
|---|---|---|---|---|---|
| 0 | 0.12 | **0.30** [0.24, 0.36] | 0.42 | 0.75 | 0.94 |
| 2 | 0.04 | 0.21 | 0.27 | 0.81 | 0.93 |
| 4 | 0.00 | 0.08 | 0.17 | 0.83 | 0.92 |
| 5 | 0.00 | **0.04** [0.02, 0.07] | 0.09 | 0.88 | 0.93 |
| 6 | 0.00 | 0.07 | 0.12 | 0.99 | 1.00 |
| 8 | 0.00 | **0.18** [0.13, 0.23] | 0.18 | 1.00 | 1.00 |
| 9 | 0.00 | 0.99 | 1.00 | 1.00 | 1.00 |

- **The same flat threshold:** every scale absorbs severity 1 (run outcome "unchanged" at all 10).
- **The same U-shape at severity 2:** most fragile at scale 0 (semantic drift) and scale 9 (structural
  damage), most robust at scale 5, as on d20; the intervals for 0, 5 and 8 do not overlap.
- **The bigger model is somewhat more robust at its early scales** (scale 0, severity 2: 30 % vs 42 %;
  severity 4: 75 % vs 94 %), and identical at the late ones.
- Cost: 0.15 s/image (d20: 0.11), peak GPU memory 23.2 GB (d20: 17.1).

## What Phase E changes in the P2 story

1. VAR's map is robust: to the noise draw (Phase C), to model size (E3), and to sampling of images
   (E4: non-overlapping intervals).
2. DiT's "early robust" is robust; its "late fragile" is true for the middle-to-late steps under both
   severity definitions, but the final steps' fragility depends on the definition (E1). The report
   should state both.
3. Corruptions compound, except that DiT forgets early corruption entirely (E2).

## Files

| File | Content |
|---|---|
| `p2_summary.csv`, `p2_summary_valid.csv` | One row per corrupted run (80): E1-E3, with and without broken baselines |
| `p2_ci.csv` | 95 % bootstrap intervals for every run (cls_prob, frac_damaged, incep_cos, KID) |
| `p2_curves.csv` | State gap to the baseline per run and traced stage |
| `p2_baselines.csv` | Every baseline image's classifier result and flag (DiT, d24) |
| `runs.csv` | Every run: config (d20 / d24), settings, pairing, timing, VRAM, code commit |
| `dit_sigma_sev1.jpg` | DiT, severity 1 in `sigma` mode, every timestep |

Full outputs are on NSCC in `outputs/p2/phasee/<JOBID>/`.
