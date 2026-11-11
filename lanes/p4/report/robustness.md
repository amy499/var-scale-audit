# P4 Phase E: robustness of the four-arm contrasts

Plan unit U8 (R9, R17). This note asks whether P4's contrasts hold up when the placement seed changes
and when VAR gets deeper (d20 to d24). Every number below comes from a CSV written by
`python -m lanes.p4.export compare`. The CSVs are in `lanes/p4/results/`, byte-identical copies of
the NSCC files in `outputs/p4/summary/2026-11-03/`. Where two runs disagree, the disagreement is
reported as a result; nothing was re-tuned to remove it.

## What is compared

Each arm is compared with the **control** at the same band and budget, as a paired contrast:
`<metric>_vs_control` = arm − control, per image, with a 95% bootstrap CI over the 224 images.
- `incep_cos`: lower is better, so a negative contrast means the arm beats the control.
- `cls_prob`: higher is better, so a positive contrast means the arm beats the control.

"Same sign" means the two runs agree on the direction. "CIs overlap" means their intervals overlap.
The images and class labels are the same in every run, so two runs that differ only in effect size
can agree in sign and still fail to overlap.

| Comparison | CSV (`lanes/p4/results/`) | Rows | Same sign | CIs overlap |
|---|---|---|---|---|
| VAR d20, seed 1 vs seed 2, single random control | `robustness_var.csv` (ref `seed1`, other `seed2`) | 36 | 21 | 4 |
| VAR d20 vs d24, single random control | `robustness_var.csv` (ref `seed1`, other `d24`) | 36 | 32 | 25 |
| VAR d20, seed 1 vs seed 2, **pooled control** | `robustness_var_seeds_pooled.csv` | 36 | 36 | 36 |
| VAR d20 vs d24, **pooled control** | `robustness_var_pooled.csv` | 36 | 34 | 19 |
| VAR shortlisted set {7,8,9}, d20 vs d24, pooled control | `robustness_var_s789_pooled.csv` | 12 | 11 | 3 |
| DiT, seed 1 vs seed 2 | `robustness_dit.csv` | 36 | 26 | 36 |

## 1. A single random control did not hold up across seeds, so VAR uses a pooled control

The protect and damage arms are deterministic: they were bit-identical across the two seeds on both
models (18 of 18 each). Only the control depends on the seed. With one random control placement per
condition, the VAR seed-1 vs seed-2 repeat agreed in sign on only 21 of 36 contrasts, and the CIs
overlapped on 4. VAR's 10 scales differ so much in importance that a single random cut mostly
measures *which* scales it happened to hit, and the per-image CIs do not include that luck.

The fix was made after that check and before reading any further results. VAR's control is now the
per-image mean over **every** placement with the damage arm's window structure (`arms.control_pool`;
pool roots `outputs/p4/pool/var/var_d20/26110495` and `.../var_d24/26124035`, 111 runs each). That
mean is the exact expectation of a random draw, so it no longer depends on the seed. Seed 1 and
seed 2 now agree on all 36 contrasts with identical values. This follows from the construction; it
shows the seed dependence is gone and is **not** an independent replication. The seed-2 root is kept
as the evidence for the change, not as a result.

DiT keeps its single seeded control. Its seed repeat overlaps on all 36 contrasts. The 10 sign
disagreements are on contrasts whose CIs include zero at both seeds: DiT's effects are small.

## 2. VAR d20 vs d24, pooled control

Source: `robustness_var_pooled.csv` (ref `d20` = `var_seed1_pooled_summary.csv`, other `d24` =
`var_d24_pooled_summary.csv`). Baseline cls_prob: d20 0.812, d24 0.823.

- **Direction:** 34 of 36 contrasts agree in sign between d20 and d24.
- **Magnitude:** where the CIs do not overlap (17 of 36), the direction is still the same in 15 cases.
  The size differs, usually smaller on d24. The deeper model is somewhat less sensitive to the same
  cut, but it ranks the arms the same way.
- **The two sign disagreements are both early-band damage at m = 3,** on both metrics. On d20,
  cutting all three early scales (0–2) is worse than the pooled control. On d24 it is no worse on
  incep_cos (CI includes 0) and better on cls_prob. Early-band pixel results also carry P3's floor
  caveat (the 2×2–16×16 grids have hf_ratio = 0), so the early band is the least settled result.

### Per-band reading

- **Middle (scales 3–5): holds on both models.** Protecting the middle beats the control at every
  budget on both metrics, on both d20 and d24 (12 of 12, every CI excludes 0). Damaging the middle is
  worse than the control at m = 2 and m = 3 on both metrics and both models. At m = 1 the damage
  contrast is weak: cls_prob is worse on d20 and has a CI including 0 on d24, and incep_cos has a CI
  including 0 on d20 and is slightly better on d24.
- **Late (scales 6–9): the same budget-dependent pattern on both models.**
  - m = 1: cutting one late scale is *better* than a random cut, and protecting late is worse.
  - m = 2: the reverse. Protecting late is better (cls_prob CI excludes 0 on both; incep_cos's CI
    includes 0 on d24), and damage is worse.
  - m = 3: both late arms are worse than the control on cls_prob.

  This replicates, but it is not a single rule. The late scales matter little one at a time and more
  in bulk.
- **Early (scales 0–2): direction agrees at m = 1 and m = 2, flips at m = 3 for damage** (above).

## 2b. The shortlisted set {7,8,9} on d20 and d24

The predeclared shortlist (`shortlist_var.json`, from snapshot `handoff_2026-11-10_SOURCES.json`)
picked scales {7,8,9}, which is not one of the three frozen bands. So the four-arm test was run on
that set (confirmation roots `outputs/p4/confirm/var/s7+8+9/26186463` on d20 and
`.../s7+8+9_d24/26186465` on d24), exported against each model's pooled control.

R17 also asks for a second-seed repeat of a new candidate set. For VAR that repeat would only change
the control. The pooled control has no seed (section 1), so the seed-2 repeat was not run. The d24
repeat was.

Source: `robustness_var_s789_pooled.csv`. 11 of 12 contrasts agree in sign between d20 and d24:

| m | arm | metric | d20 vs control [95% CI] | d24 vs control [95% CI] | same sign |
|---|---|---|---|---|---|
| 1 | damage | incep_cos | -0.113 [-0.119, -0.107] | -0.101 [-0.106, -0.095] | yes |
| 1 | damage | cls_prob | +0.104 [+0.078, +0.131] | +0.051 [+0.025, +0.076] | yes |
| 1 | protect | incep_cos | +0.143 [+0.126, +0.162] | +0.065 [+0.051, +0.079] | yes |
| 1 | protect | cls_prob | -0.306 [-0.350, -0.262] | -0.176 [-0.219, -0.135] | yes |
| 2 | damage | incep_cos | +0.001 [-0.011, +0.014] | +0.032 [+0.020, +0.043] | yes |
| 2 | damage | cls_prob | -0.135 [-0.172, -0.096] | -0.192 [-0.232, -0.148] | yes |
| 2 | protect | incep_cos | -0.066 [-0.079, -0.054] | -0.035 [-0.046, -0.024] | yes |
| 2 | protect | cls_prob | +0.260 [+0.220, +0.302] | +0.185 [+0.145, +0.226] | yes |
| 3 | damage | incep_cos | -0.069 [-0.076, -0.061] | -0.033 [-0.042, -0.024] | yes |
| 3 | damage | cls_prob | +0.025 [+0.005, +0.045] | -0.016 [-0.040, +0.011] | **no** |
| 3 | protect | incep_cos | +0.069 [+0.063, +0.075] | +0.090 [+0.081, +0.098] | yes |
| 3 | protect | cls_prob | -0.039 [-0.048, -0.030] | -0.094 [-0.111, -0.079] | yes |

On both depths, the set behaves like an important set only at m = 2. At m = 1, cutting one of
scales 7–9 is *better* than a random cut, and keeping all three intact is worse. At m = 3, cutting
all three is better than the control on incep_cos on both depths. On cls_prob it is better on d20
and has a CI including 0 on d24: this is the one sign disagreement. At m = 3 every arm has already
lost most of its class confidence (cls_prob 0.001–0.098 against baselines of 0.81–0.82), so that
budget carries little weight. The section on Figure 5 (`figure5_discussion.md`) reads this
against the lanes that ranked {7,8,9} highest.

## 3. What this supports

- The **protect-the-middle** result is the robust P4 finding on VAR. It survives the change of control,
  both seeds and both depths.
- The late band's pattern replicates across depths but depends on the budget, so it is reported as it
  is. It is not a usable "skip late" rule.
- The early band at m = 3 does not replicate across depths.
- The shortlisted set {7,8,9} is not confirmed as an important set: P4's direct test supports it only
  at m = 2, on both depths.
- VAR saves no compute: the intervention runs after the transformer pass. These are statements about
  which stages matter, not a speed-up.
- DiT's contrasts are small, and stable across seeds within their CIs.

The second seed reused the same images. It tests only the control placement, not new images.

## Files

All cited CSVs are in `lanes/p4/results/`. `FREEZE.md` lists the sha256 of each, together with the
run roots that produced them.

## Appendix: every VAR d20 vs d24 contrast (pooled control)

Generated from `robustness_var_pooled.csv`, not typed by hand.

| band | m | arm | metric | d20 vs control [95% CI] | d24 vs control [95% CI] | same sign | CIs overlap |
|---|---|---|---|---|---|---|---|
| early | 1 | protect | cls_prob | -0.037 [-0.071, -0.001] | -0.032 [-0.063, -0.002] | yes | yes |
| early | 1 | protect | incep_cos | -0.019 [-0.027, -0.010] | -0.017 [-0.025, -0.009] | yes | yes |
| early | 1 | damage | cls_prob | +0.167 [+0.129, +0.204] | +0.122 [+0.084, +0.161] | yes | yes |
| early | 1 | damage | incep_cos | +0.033 [+0.021, +0.044] | +0.052 [+0.042, +0.062] | yes | yes |
| early | 2 | protect | cls_prob | +0.045 [+0.008, +0.082] | +0.049 [+0.009, +0.090] | yes | yes |
| early | 2 | protect | incep_cos | -0.075 [-0.087, -0.063] | -0.065 [-0.076, -0.054] | yes | yes |
| early | 2 | damage | cls_prob | -0.096 [-0.148, -0.043] | -0.023 [-0.070, +0.026] | yes | yes |
| early | 2 | damage | incep_cos | +0.086 [+0.068, +0.103] | +0.046 [+0.032, +0.061] | yes | no |
| early | 3 | protect | cls_prob | +0.011 [-0.008, +0.029] | +0.074 [+0.045, +0.105] | yes | no |
| early | 3 | protect | incep_cos | -0.048 [-0.057, -0.040] | -0.079 [-0.091, -0.069] | yes | no |
| early | 3 | damage | cls_prob | -0.040 [-0.049, -0.032] | +0.062 [+0.026, +0.100] | **no** | no |
| early | 3 | damage | incep_cos | +0.070 [+0.064, +0.077] | -0.009 [-0.022, +0.004] | **no** | no |
| middle | 1 | protect | cls_prob | +0.131 [+0.104, +0.158] | +0.069 [+0.042, +0.095] | yes | no |
| middle | 1 | protect | incep_cos | -0.130 [-0.137, -0.123] | -0.109 [-0.115, -0.103] | yes | no |
| middle | 1 | damage | cls_prob | -0.059 [-0.097, -0.024] | -0.011 [-0.049, +0.025] | yes | yes |
| middle | 1 | damage | incep_cos | -0.007 [-0.017, +0.004] | -0.011 [-0.021, -0.001] | yes | yes |
| middle | 2 | protect | cls_prob | +0.327 [+0.288, +0.363] | +0.240 [+0.196, +0.281] | yes | no |
| middle | 2 | protect | incep_cos | -0.085 [-0.096, -0.074] | -0.047 [-0.058, -0.036] | yes | no |
| middle | 2 | damage | cls_prob | -0.231 [-0.265, -0.198] | -0.132 [-0.172, -0.093] | yes | no |
| middle | 2 | damage | incep_cos | +0.097 [+0.083, +0.111] | +0.025 [+0.010, +0.039] | yes | no |
| middle | 3 | protect | cls_prob | +0.155 [+0.124, +0.187] | +0.159 [+0.121, +0.197] | yes | yes |
| middle | 3 | protect | incep_cos | -0.152 [-0.162, -0.141] | -0.139 [-0.150, -0.128] | yes | yes |
| middle | 3 | damage | cls_prob | -0.020 [-0.025, -0.014] | -0.022 [-0.042, -0.002] | yes | yes |
| middle | 3 | damage | incep_cos | +0.019 [+0.014, +0.025] | +0.014 [+0.004, +0.024] | yes | yes |
| late | 1 | protect | cls_prob | -0.171 [-0.222, -0.123] | -0.115 [-0.162, -0.066] | yes | yes |
| late | 1 | protect | incep_cos | +0.121 [+0.104, +0.138] | +0.088 [+0.075, +0.102] | yes | no |
| late | 1 | damage | cls_prob | +0.119 [+0.092, +0.146] | +0.058 [+0.033, +0.082] | yes | no |
| late | 1 | damage | incep_cos | -0.116 [-0.122, -0.109] | -0.098 [-0.103, -0.093] | yes | no |
| late | 2 | protect | cls_prob | +0.185 [+0.142, +0.227] | +0.111 [+0.068, +0.157] | yes | yes |
| late | 2 | protect | incep_cos | -0.033 [-0.045, -0.021] | -0.011 [-0.024, +0.002] | yes | yes |
| late | 2 | damage | cls_prob | -0.193 [-0.225, -0.159] | -0.223 [-0.259, -0.182] | yes | yes |
| late | 2 | damage | incep_cos | +0.030 [+0.019, +0.041] | +0.047 [+0.036, +0.058] | yes | yes |
| late | 3 | protect | cls_prob | -0.139 [-0.155, -0.123] | -0.141 [-0.170, -0.111] | yes | yes |
| late | 3 | protect | incep_cos | +0.102 [+0.094, +0.110] | +0.076 [+0.062, +0.089] | yes | no |
| late | 3 | damage | cls_prob | -0.119 [-0.136, -0.101] | -0.193 [-0.212, -0.172] | yes | no |
| late | 3 | damage | incep_cos | +0.005 [-0.003, +0.012] | +0.047 [+0.040, +0.055] | yes | no |
