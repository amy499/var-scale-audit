# What P1 and P2 should send me for the agreement analysis

**The short version:** one CSV each, one row per stage, two columns — `stage,score` — where
**higher = more important**, and the score is **per-stage, not cumulative**.

But the format is the easy part. The thing that decides whether the analysis means anything
is below.

---

## 1. The one decision we have to make together

**All three lanes must rank by the same underlying quantity, and vary only the
intervention.**

| Lane | Intervention | Scoring function |
|---|---|---|
| P1 | remove stage k | ← **must be the same** |
| P2 | corrupt stage k | ← **must be the same** |
| P3 | observe only | ← **must be the same** |

If P1 ranks by PSNR and P2 ranks by classifier probability, then a disagreement between
them could be a fact about the model *or* an artefact of two different yardsticks, and
nothing distinguishes the two. The whole point of the comparison is that only the
intervention differs.

### My proposal: rank by the drop in validity

```
score(k) = cls_prob_base − cls_prob(intervened at stage k)
```

Mean over images, higher = the stage matters more.

**Why validity and not pixel distance.** P2's own pilot settled this: VAR si0 sev1.0 has
PSNR 10.8 — the worst of any run — and is a perfectly clean image of a different dog.
DiT t197 sev1.0 has PSNR 12.6 and is destroyed. **Re-routing is not damage.** Ranking by
`img_psnr_db`, `img_mad255`, `gap_l2_ratio` or `gap_rel_l2_final` would rank VAR's early
scales as catastrophic when nothing is actually broken.

So: **do not rank by any pixel-distance or state-distance column.** They measure change,
not damage.

### Second column, if you can: KID

`cls_prob` catches loss of *meaning*; it misses oversaturation. P2's DiT t497 sev1.0 keeps
`cls_prob` 0.73 against a 0.82 baseline while looking visibly broken — only KID catches it
(+0.0185 against ≤0.003 for clean runs). If `kid_vs_base` is available per stage, send it as
a second ranking and we compare both. Where they disagree is itself informative.

---

## 2. The file

```
stage,score
0,0.0123
1,0.0098
...
9,0.4412
```

| Requirement | |
|---|---|
| Rows | one per stage — **10 for VAR** (si 0–9) |
| Column names | `stage` (or `si`/`step`) and any score column; tell me its name |
| Direction | **higher = more important.** If yours is inverted, say so — I pass `invert` |
| Cumulative? | **Say which.** If the number is "total damage up to stage k", I pass `delta` |
| Manifest | the frozen 224 (`manifest/frozen_v1_1.csv`) — or tell me what you used |
| Also send | `n_images` and a spread (sd or IQR) per stage, so I can caveat properly |

Extra columns are fine and ignored — `p2_summary.csv` filtered to one severity already
works as-is.

---

## 3. What each of you specifically needs to decide

### P2 — which severity

Your data is stage × severity. The agreement analysis needs **one number per stage**, so
either:

- **pick one severity and say which** — I'd suggest the one where stages separate best,
  probably 1.0, since 0.25 leaves VAR's late scales nearly untouched; or
- **average across severities** — defensible, but it mixes "barely touched" with "destroyed"
  and compresses the differences.

Either is fine. What is not fine is leaving it unstated, because the ranking changes with it.

From `p2_summary.csv` the score is then `cls_prob_base − cls_prob` at your chosen severity.

### P1 — what "removal" means

Your `score(k)` is the validity drop when stage k is removed. Two things to state:

- **which removal operation** — P4's doc describes VAR's as blending `f_hat` back toward its
  pre-stage value at λ = 0, the repo's tested `restore_all`. If you use the same, say so; if
  not, say what you use.
- **removal of stage k only**, not stages 0..k. One stage at a time, or the ranking is
  cumulative and I need `delta`.

### P4 — probably cannot join this table

Your unit is **bands** (early/middle/late), not stages, so there are 3 points rather than 10.
Spearman on n=3 says nothing. Better to compare qualitatively: does my per-stage ranking
agree with which band your arms found most protectable? Worth a paragraph in the report, not
a correlation.

---

## 4. How I will report it

```
python -m lanes.p3.agreement \
  --rank "p1_necessity=lanes/p1/necessity.csv:score" \
  --rank "p2_recovery=lanes/p2/recovery.csv:score" \
  --rank "p3_energy=outputs/p3/figure3/var_curves.csv:rel_dist:delta,invert" \
  --rank "p3_identity=outputs/p3/figure3/var_semantic.csv:identity_progress:delta"
```

Pairwise Spearman with p-values, plus each lane's ranked stage order.

**Read the p-values honestly.** n = 10 stages, so |rho| must exceed roughly **0.65** for
p < 0.05. A middling correlation means *this cannot resolve it*, not *the lanes disagree*.

**And disagreement is a result.** "The stages where most information appears are not the
stages that are hardest to remove" would be a genuine finding. Nobody should tune anything
toward agreement.

---

## 5. For reference — what P3 is contributing

Two rankings, both from the frozen 224-image run, already computed:

| stage | energy added | identity added |
|---|---|---|
| 0–5 | ~0.028 each | ~0.005 each |
| 6 | 0.053 | 0.082 |
| 7 | 0.078 | 0.245 |
| 8 | 0.101 | **0.362** |
| 9 | **0.597** | 0.279 |

`p3_energy` ranks `[9, 8, 7, 6, 5, 3, 0, 1, 4, 2]`, `p3_identity` ranks
`[8, 9, 7, 6, 5, 3, 4, 0, 1, 2]`, rho = +0.951.

They agree strongly but **swap the top two**: scale 9 adds the most energy, scale 8 the most
identity. The interesting question for P1 and P2 is which of those two your interventions
pick out — that is the comparison the whole four-lane design was for.

**A caveat on mine, stated up front:** the identity ranking is chance-corrected
(Inception cosine between unrelated images is ~0.33–0.50, so the raw curve mostly measures
the floor) and comes from 32 images across 7 classes, not all 224. The energy ranking uses
all 224.
