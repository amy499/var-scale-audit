# Figure 5: where the lanes agree on stage importance, and where they do not

Plan unit U9 (R15). Figure 5 (`lanes/p4/results/figure5_var.png`, `figure5_dit.png`) puts each
lane's per-stage importance on one progress axis, one panel per lane.
- Each panel is normalized within itself; higher means more important.
- Lanes are never averaged in the figure.

The inputs come from the snapshot of 10 Nov (`lanes/p4/results/handoff_2026-11-10_SOURCES.json`):

| Lane | Source | File |
|---|---|---|
| P1 | `origin/main` @ 14994c7 | `results_sweep.csv` |
| P2 | `origin/main` @ 14994c7 | `phasec/p2_summary.csv`, severity 1.0 |
| P3 | `origin/p3/dit-and-cross-lane` @ c03182e | `var_curves.csv`, `var_semantic.csv` and the DiT pair (P3's tables are not on main) |
| P4 | — | `var_seed1_pooled_summary.csv`, `dit_seed1_summary.csv` |

Each lane measures a different thing, which explains most of the disagreement below:

| Lane | Asks | Panel value |
|---|---|---|
| P1 necessity | What breaks when this stage is removed? | drop in cls_prob |
| P2 corruption | How badly does noise at this stage hurt? | cls_prob drop at severity 1.0 |
| P3 energy / identity | How much does this stage change the image / its class identity? | per-stage delta of the cumulative curves |
| P4 damage − control | Is cutting inside this band worse than cutting the same amount anywhere? | −(damage − control) on cls_prob at the band's largest budget |

## VAR (10 scales)

Lane rankings (top three, from the snapshot via `synthesis.lane_scores`):

| Lane | Most important scales | Least important |
|---|---|---|
| P1 necessity | 3, 2, 4 | 8, 1, 0 |
| P2 corruption | 2, 0, 1 | 7, 8, 4 |
| P3 energy | 9, 8, 7 | 1, 4, 2 |
| P3 identity | 8, 9, 7 | 0, 1, 2 |

No pair of *different* lanes agrees significantly. In `shortlist_var.json`, the Spearman rho over 10
scales is between −0.61 and −0.07, and every p is ≥ 0.06. P3's two measures agree with each other
(rho 0.95, p 2e-5).

### By stage region

- **Middle (scales 3–5): P1 and P4 agree; P2 and P3 do not.**
  - P1 ranks scale 3 the most necessary.
  - P4's strongest and most robust result is here: keeping the middle intact beats the control at
    every budget, on both metrics and both depths (`robustness.md`, section 2). At m = 2, protect is
    +0.327 cls_prob over the control, and damage is −0.231 (`var_seed1_pooled_summary.csv`, middle,
    m = 2).
  - P2 and P3 rank the middle low.
- **Early (scales 0–2): P2 ranks it highest; the others are mixed.**
  - Noise here hurts most (P2).
  - P1 finds scales 0 and 1 almost unnecessary one at a time (cls_prob drops of 0.023 and 0.026), but
    removing {0,1,2} together destroys the image.
  - P4's early results flip with the budget and do not replicate at m = 3 across depths.
  - P3's early-band caveat applies: pixel metrics floor at these 2×2–16×16 grids.
- **Late (scales 6–9): P3 ranks it highest; P1, P2 and P4 do not.**
  - Scales 7–9 hold 525 of VAR's 680 tokens (P1 `ranking.csv`). They are where the image changes most
    (P3 energy) and where its class identity is refined most (P3 identity).
  - But removing one of them costs little (P1: scale 8 is the least necessary scale, drop 0.056).
  - P4's direct test agrees with P1: cutting one late scale is *better* than a random cut on both
    depths.
  - A stage can do the most visible refining without being the most necessary for the result.

### The shortlist and its confirmation

The predeclared rule (KTD6) normalizes P1, P2, P3-energy and P3-identity and averages them per scale.
It picked {7,8,9} (score 0.416), ahead of the late band {6,7,8,9} (0.374) and {1,2,3} (0.334)
(`shortlist_var.json`).

The rule counts P3's two measures as two lanes. They agree with each other and both put 7–9 on top,
so P3 decided the window. That is how the rule was fixed before any result was seen, so it is
reported as it is, not re-weighted. Without P3 (P1 and P2 only), the same rule picks {1,2,3}.

P4 then ran the four-arm test on {7,8,9} on d20 and d24 (`robustness.md`, section 2b). The set
behaves like an important set only at m = 2. At m = 1 and m = 3, cutting inside it is no worse than
the control, and mostly better. **P4's intervention does not confirm the window the shortlist picked
from P3's contribution measures.** It sides with P1's necessity ranking.

### Cross-check with P1

Three windows were cut whole by both lanes. P4's damage arm at m = 3 matches P1's removal of the
same scales:

| Window | P1 cls_prob (`ranking.csv`) | P4 cls_prob (damage, m = 3) |
|---|---|---|
| {0,1,2} | 0.0011 (`remove_early`) | 0.0011 (`var_seed1_pooled_summary.csv`, early) |
| {3,4,5} | 0.0011 (`remove_mid`) | 0.0011 (`var_seed1_pooled_summary.csv`, middle) |
| {7,8,9} | 0.0639 (`remove_late`) | 0.064 (`var_s789_pooled_summary.csv`) |

Two different code paths agree on the same intervention, so P4's damage arm does what it says.

### A caveat on P4's own panel

P4's panel uses the band's largest budget (m = 3), as fixed in advance. At m = 3, every VAR arm has
already lost most of its class confidence. Middle damage and its control both sit near the floor
(0.0011 and 0.021), so the middle's contrast is compressed, and the panel ranks late > early > middle.

At m = 2, where nothing has collapsed, the same quantity ranks middle (−0.231) > late (−0.193) >
early (−0.096) (`var_seed1_pooled_summary.csv`, damage vs control, m = 2). The panel is left as
predeclared. This note is how it should be read.

## DiT (250 steps)

Mean score per band (snapshot, `synthesis.lane_scores`):

| Lane | Early (t 670–995) | Middle (t 337–666) | Late (t 0–333) |
|---|---|---|---|
| P1 necessity | 0.078 | 0.187 | **0.266** |
| P2 corruption | 0.018 | 0.439 | **0.757** |
| P3 energy | −0.041 | −0.048 | −0.047 |
| P3 identity | **−0.001** | −0.089 | −0.036 |

- **P1 and P2 agree:** the late steps matter most (rho 0.91, n = 8, p 0.0016, `shortlist_dit.json`).
  The shortlist's verdict is "covered by late", so the sweep's late band was the confirmation.
- **P3 differs.** Its energy measure is nearly flat across bands, and its identity measure peaks
  early. P3 shares fewer than 3 steps with P2, so that pair's agreement could not be computed.
- **P4 skips steps, a different intervention from P1's removal.**
  - Skipping inside any band (the damage arm, up to 30% of the band) never lowers cls_prob below the
    control beyond its CI.
  - Skipping late steps preserves fidelity best: incep_cos −0.076 to −0.088 against the control at
    every budget, every CI excluding 0 (`dit_seed1_summary.csv`, late, damage).
  - On cls_prob, skipping 30% of the early steps is slightly *better* than the control: +0.052
    [+0.004, +0.103].
  - So, for DiT, P4 finds the late steps the safest to skip for fidelity, which is the opposite of
    their importance in P1 and P2's removal and corruption tests.
- P2's own Phase E (E1) later revised its DiT late-step finding. The snapshot uses P2's Phase C
  tables, which the shortlist rule named in advance.

## Summary

| Region | Agree | Disagree |
|---|---|---|
| VAR middle | P1 necessity, P4 protect/damage (both depths) | P2, P3 rank it low |
| VAR early | P2 ranks it high | P1 singly low, jointly fatal; P4 unstable at m = 3 |
| VAR late | P3 energy and identity rank it highest | P1, P4 (one scale is cheap to lose), P2 |
| DiT late | P1, P2 rank it most important | P4: safest to skip for fidelity; P3 flat or early |

The lanes disagree mainly because they measure different things: necessity, sensitivity,
contribution. The stages that change the image most are not the ones it can least do without. For
"which stages can be cut", necessity (P1) and P4's matched-budget test are the direct evidence.
