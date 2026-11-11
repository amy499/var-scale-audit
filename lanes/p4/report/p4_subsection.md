# P4: can stage importance be exploited?

Plan unit U9 (R19). Every number cites a file in `lanes/p4/results/`. `FREEZE.md` lists the run
roots and checksums behind each file.

## Question

The other lanes rank how much each generation stage matters:
- P1 removes stages.
- P2 corrupts them.
- P3 measures what each stage adds.

P4 asks the practical question. **If you must cut a fixed share of the stages, does it matter which
ones?** In other words: does protecting the stages the ranking calls important beat cutting at
random, and does cutting them do worse?

## Design

- **Models.**
  - VAR-d20, plus a d24 repeat: 10 scales, coarse to fine.
  - DiT-XL/2 at 256 px: 250 sampling steps.
  - Both run on the frozen 224-image manifest (`manifest/frozen_v1_1.csv`).
- **Bands.** Each model's stages are split into early, middle and late thirds:
  - VAR: scales 0–2, 3–5 and 6–9.
  - DiT: 83, 83 and 84 steps.
- **Budget.** In every arm, a fraction m/n of 0.1, 0.2 or 0.3 of the stages is cut:
  - VAR: m = 1, 2 or 3 scales.
  - DiT: m = 25, 50 or 75 steps.
- **Arms.** At each band and budget, all three cut the same number of stages:
  - **protect:** cut m stages outside the band.
  - **damage:** cut m stages inside the band.
  - **control:** cut m stages anywhere, with the same window structure.
  
  The baseline cuts nothing.
- **Interventions.**
  - VAR: a scale's contribution is dropped after the transformer pass (`lanes/p4/hooks.py`, λ = 0).
  - DiT: timesteps are skipped in the sampler (`--skip-timesteps`), which saves real sampling time.
- **Control.**
  - VAR: the per-image mean over every placement with damage's window structure (the pooled
    control; `robustness.md`, section 1 explains why).
  - DiT: one seeded draw (seed 20261006).
- **Metrics** (P3's, signed off with two conditions in `comparison_logic.md`):
  - `incep_cos`: distance from the uncut image; lower is better.
  - `cls_prob`: the classifier's probability of the intended class; higher is better.
- **Statistics.** Each arm is compared with its control per image, with a 95% paired bootstrap CI
  over the 224 images (2000 resamples, seed 20261102).
- **Shortlist.** A rule fixed in advance chooses, from P1–P3 only, a stage set for an extra
  confirmation test (`figure5_discussion.md`).

## Results: protect, damage and control

### VAR (`var_seed1_pooled_summary.csv`; d24 in `var_d24_pooled_summary.csv`)

The uncut baseline has cls_prob 0.812.

**Protecting the middle band (scales 3–5) is the clear, robust result.**

| m | protect cls_prob | control | protect − control [95% CI] | damage cls_prob | damage − control [95% CI] |
|---|---|---|---|---|---|
| 1 | 0.756 | 0.625 | +0.131 [+0.104, +0.158] | 0.566 | −0.059 [−0.097, −0.024] |
| 2 | 0.723 | 0.396 | +0.327 [+0.288, +0.363] | 0.165 | −0.231 [−0.265, −0.198] |
| 3 | 0.175 | 0.021 | +0.155 [+0.124, +0.187] | 0.001 | −0.020 [−0.025, −0.014] |

- Protect beats the control at every m, on both metrics, on d20 and d24 (12 of 12, every CI excludes
  0; `robustness_var_pooled.csv`).
- Damage is worse than the control at m = 2 and m = 3 on both models.
- P3's condition is met: every middle-band result is read against its control.

**The late band (scales 6–9) depends on the budget, and this replicates on d24:**

| m | Effect |
|---|---|
| 1 | Cutting one late scale is *better* than a random cut: cls_prob +0.119 [+0.092, +0.146], incep_cos −0.116 |
| 2 | Protecting late is better: cls_prob +0.185 [+0.142, +0.227]. Damage is worse: −0.193 [−0.225, −0.159] |
| 3 | Both late arms are worse than the control on cls_prob |

The late scales are cheap to lose one at a time, but costly in bulk.

**The early band (scales 0–2) is the least settled.**
- Protect is better on incep_cos at every m.
- cls_prob goes both ways.
- Damage at m = 3 flips sign between d20 and d24 (`robustness_var_pooled.csv`).
- P3's condition applies: pixel-space metrics floor on these coarse grids. incep_cos and cls_prob are
  not pixel-space, but early results should be read with that caveat.

**The shortlisted set {7,8,9} is not confirmed** (`var_s789_pooled_summary.csv`,
`var_s789_d24_pooled_summary.csv`):
- It behaves like an important set only at m = 2: protect +0.260 cls_prob over the control, damage
  −0.135.
- At m = 1, cutting one of 7–9 is better than the control (+0.104), and keeping all three is worse
  (−0.306).
- d20 and d24 agree in sign on 11 of 12 contrasts (`robustness_var_s789_pooled.csv`).

### DiT (`dit_seed1_summary.csv`; seed 2 in `dit_seed2_summary.csv`)

The uncut baseline has cls_prob 0.759.

- **The effects are small.** Every arm-vs-control contrast lies within ±0.09 on both metrics. The
  largest on cls_prob is 0.060.
- **Skipping late steps preserves fidelity best,** at every budget. incep_cos damage − control:

  | m | late damage − control |
  |---|---|
  | 25 | −0.076 [−0.089, −0.064] |
  | 50 | −0.081 [−0.092, −0.069] |
  | 75 | −0.088 [−0.103, −0.074] |

- **Class confidence barely moves.** Skipping inside any band never lowers cls_prob below the
  control beyond its CI. Skipping 30% of the early steps is slightly better: +0.052 [+0.004, +0.103].
- **The two metrics prefer different bands to skip:** late for fidelity, early for class
  confidence.
- **Seeds agree.** Seed 2 overlaps seed 1 on all 36 contrasts (`robustness_dit.csv`).

## Quality–compute

- **VAR saves no compute.** The intervention runs after the transformer pass, so VAR's curves show
  which scales matter at a given cut, not a speed-up. A VAR that skipped scales before the
  transformer would need different code; that is out of scope.
- **DiT's skip does save time.** The measured sampling-time saving matches the budget
  (`dit_seed1_summary.csv`, `saving`):

  | m/n | Measured saving |
  |---|---|
  | 0.1 | 9.7–10.0% |
  | 0.2 | 19.8–19.9% |
  | 0.3 | 29.7–29.9% |

- **What DiT's ~30% saving costs:**
  - Skipping late steps keeps the image closest to the original: incep_cos 0.159, against 0.247 for
    the control.
  - Skipping early steps keeps class confidence at baseline: cls_prob 0.771 [0.736, 0.803], against
    the baseline's 0.759 [0.720, 0.795].
- **Figure 4** (`figure4_incep_cos_pooled.png`, `figure4_cls_prob_pooled.png`) plots every arm
  against m/n, per model and band, with CI shading. Its titles name, at each m, whether protect and
  damage beat the control beyond the CI.

## VAR against DiT

**Where stages are cut matters far more in VAR than in DiT.**
- VAR's best placement beats a random one by up to 0.33 in cls_prob at the same budget (middle,
  m = 2).
- DiT's largest is 0.060.

A plausible reason, not tested here: VAR builds the image in only 10 coarse-to-fine steps, so each
one carries a lot of the result. DiT spreads the work over 250 small steps, so dropping up to 30% of
any band costs little.

**Practical reading:**
- For VAR, keep scales 3–5.
- For DiT, skip late steps if fidelity to the full-step image matters, and early steps if only the
  class matters. Either way, the saving is real and the cost is small.

## Figure 5 and the other lanes

`figure5_discussion.md` reads P4 against P1–P3, region by region:

- **VAR middle:** P1's necessity ranking agrees with P4.
- **VAR late:** P3's contribution measures rank 7–9 highest, but P1 and P4 show they are the
  cheapest to lose one at a time.
- **DiT late:** P1 and P2 rank the late steps most important, while P4 finds them the safest to skip
  for fidelity.
- **Cross-check:** P4's damage arm reproduces P1's whole-window removals for {0,1,2}, {3,4,5} and
  {7,8,9} (cls_prob 0.0011, 0.0011 and 0.064 in both lanes).

## Limits

- **VAR's seed-1 vs seed-2 agreement** is by construction of the pooled control. It is not an
  independent replication, and seed 2 reused the same 224 images.
- **The pooled control was adopted after the seed-2 check showed the single draw was unstable.** That
  was before the d24 pool or the confirmation runs were read.
- **The predeclared shortlist counts P3's two measures as two lanes.** They decided the {7,8,9}
  window.
- **P4's Figure 5 panel uses the largest budget,** where VAR arms have collapsed (see
  `figure5_discussion.md`).
- **The planned 5–6 Nov freeze moved to 12 Nov** (`FREEZE.md`).
