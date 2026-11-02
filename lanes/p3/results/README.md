# P3 results — information lane, VAR and DiT

The small result tables behind P3's curves, the cross-lane agreement and the model-signature
check. They live here, like `lanes/p1/results/` and `lanes/p2/results/`, so that P4 and the
report can use them without re-running anything. Copied from `outputs/p3/` on 2 Nov 2026;
regenerate any of them with the commands in `../DIT_AND_AGREEMENT_HOWTO.md`.

## Where the inputs came from

| Input | Source |
|---|---|
| VAR stage states | P1's ride-along jobs 25891860 and 25985813 — 140 files each, byte-identical; manifest v1.1, 224 images |
| DiT stage states | P1's job 25985813 — 24 saved steps × 224 images |
| P1 | `lanes/p1/results/ranking.csv` (main, PR #7) |
| P2 | `lanes/p2/results/gateb/var_p2_summary.csv` and `phasec/p2_summary.csv` (main). Severities are P2's choice: **2 for VAR** (both noise draws averaged), **0.5 for DiT** |
| Baseline PNGs | P1's 20 VAR and 20 DiT baselines — the same 20 class/seed pairs |

## Files

| File | What it holds | Made by |
|---|---|---|
| `var_curves.csv` | VAR: rel_dist, norm ratio, cosine per scale (224 images) | `stage_curves.py --model var` |
| `var_semantic.csv` | VAR: chance-corrected identity_progress and classifier columns per scale (32 images, all 7 breeds) | `semantic_curve.py --model var` |
| `dit_curves.csv` | DiT: the same columns at 24 timesteps (224 images) | `stage_curves.py --model dit` |
| `dit_semantic.csv` | DiT: identity per timestep (the same 32 images as VAR) | `semantic_curve.py --model dit` |
| `agreement.csv` | VAR: pairwise Spearman, n = 10 — P1, P2 (validity drop and fraction damaged), P3 (energy, identity) | `prep_rankings.py` |
| `p1_remove.csv` | VAR: P1's validity drop per single-scale removal, with 95% interval | `prep_rankings.py` |
| `p2_corrupt_sev2.csv` | VAR: P2's validity drop and fraction damaged at severity 2 | `prep_rankings.py` |
| `dit_comparison.csv` | DiT: P2 at each of its 8 timesteps, next to how far P3's curves have got | `dit_agreement.py` |
| `dit_comparison_bands.csv` | DiT: P1, P2 and P3 by P1's three bands (early / mid / late) | `dit_agreement.py` |
| `model_signature.csv`, `_per_image.csv` | VAR vs DiT finished images on three pixel metrics, n = 20 (P1's baselines, paired) | `model_signature.py` |
| `model_signature_n32.csv`, `_per_image.csv` | The same, n = 32 (both models decoded on one machine) | `model_signature.py` |

## Headline numbers

- **VAR — the lanes disagree.** Removal (P1) costs most at scale 3: validity drop 0.466
  [0.413, 0.520]. Corruption at severity 2 (P2) costs most at scale 9 (0.806), then scale 0
  (0.346). Identity (P3) arrives last: scale 8 adds 36% of it. Across lanes |rho| ≤ 0.34, none
  significant (n = 10). P2's two yardsticks agree with each other (rho +0.78).
- **DiT — the lanes agree, at band resolution.** P1, P2, P3 identity and P3 closeness all rank
  the bands late > mid > early. P2's corruption stops being absorbed between t = 397 and t = 297;
  P3's identity first reaches 0.5 at t = 293.
- **Model signature — no detectable offset** between VAR's and DiT's finished images on
  spec_centroid, grad_energy or hf_ratio, at n = 20 or n = 32. The intervals are wide; see
  `calibration_report.md`, finding 11.

## Caveats

- Identity curves use 32 images; energy and closeness use all 224.
- DiT is compared at three bands, VAR at ten scales — a coarser comparison has fewer ways to
  disagree.
- DiT "closeness" (1 − rel_dist) is not called energy: DiT's early steps fail the orthogonality
  check (max |cos − ratio| = 0.37).
