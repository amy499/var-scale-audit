# Section 6.5 — how to run the three items

Checked against what is actually on your disk. **One of the three is blocked, and it is not
the one you'd guess.**

| Item | Status | Why |
|---|---|---|
| 6.5a model-signature check | ❌ **blocked** | there is no DiT data on disk at all |
| 6.5b baseline-failure audit | ✅ ready | new tool: `quality.py` cannot read a flat folder |
| 6.5c agreement analysis | ✅ **already producing a result** | run it now |

Do **6.5c first** — it is on the critical path to the report and it already says something.

---

## 6.5c — Agreement analysis (do this first)

`lanes/p3/agreement.py`. Self-test first, so the statistics are checked before any data:

```
python -m lanes.p3.agreement --self-test
```

Then your own two rankings, which I have already run:

```
python -m lanes.p3.agreement --rank "p3_energy=outputs/p3/figure3/var_curves.csv:rel_dist:delta,invert" --rank "p3_identity=outputs/p3/figure3/var_semantic.csv:identity_progress:delta" --csv outputs/p3/figure3/agreement.csv
```

### The result, and why the headline is NOT the correlation

```
p3_energy   ranked most->least: [9, 8, 7, 6, 5, 3, 0, 1, 4, 2]
p3_identity ranked most->least: [8, 9, 7, 6, 5, 3, 4, 0, 1, 2]
rho +0.951  p 0.0000  significant: yes
```

The two agree strongly overall — but **the top two are swapped, and that swap is the
finding**:

| stage | energy added | identity added |
|---|---|---|
| 8 (13×13) | 0.101 | **0.362** ← identity's peak |
| 9 (16×16) | **0.597** ← energy's peak | 0.279 |

**Scale 9 contributes the most energy; scale 8 contributes the most identity.** That is the
precise, quotable form of "the last scale is texture, not content" — much stronger than the
loose version, because it comes from two independently-computed rankings.

### The trap this tool avoids

`rel_dist` and `identity_progress` are **cumulative**. Ranking them directly just recovers
the stage order — any monotone curve gives rho = 1 against any other, which means nothing.
What a lane claims about stage *k* is its **contribution**, so `:delta` differences the
column first, and `invert` handles `rel_dist` falling rather than rising. The self-test
demonstrates both (`cumulative vs stage index: rho +1.000 <- meaningless`).

### When P1 and P2 send their rankings

Any CSV with a stage column and a score column:

```
--rank "p1_necessity=lanes/p1/necessity.csv:score" --rank "p2_recovery=lanes/p2/recovery.csv:score"
```

Use `:delta` only if their column is cumulative too — ask them.

### Read the p-values honestly

n = 10 stages. |rho| must exceed roughly **0.65** for p < 0.05. So a middling correlation is
**"this cannot resolve it"**, not "the lanes disagree". The script prints that caveat every
run. And disagreement, where it is significant, is a result — do not tune anything toward
agreement.

---

## 6.5b — Baseline-failure audit

`lanes/p2/quality.py` **cannot do this**: it requires a `runner.generate` output directory
with a `run.json`, and your baseline PNGs are a flat folder. It fails at `load_run()` before
scoring anything — the same wall the semantic curve hit.

`lanes/p3/baseline_audit.py` reads the class out of the filename instead:

```
python -m lanes.p3.baseline_audit --images for_darien_20_clean_pngs --csv outputs/p3/baseline_audit.csv
```

It prints mean and median P(true class), top-1 accuracy, and the worst 10 images, flagging
anything under 0.10. The CSV is sorted worst-first.

It also works on the decoded finals, which is a useful cross-check on 32 images instead of
20:

```
python -m lanes.p3.baseline_audit --images p3_var_stages/decoded_var --stage 9
```

**Interpret with care.** A low score can mean the image is bad *or* that the classifier is
wrong — Inception confuses toy breeds routinely. **Look at the flagged images before
reporting them.** The output says so.

Your semantic curve already hints at what to expect: `top1_true` at stage 9 is 0.875, so
4 of 32 final images are not classified as their conditioned class. Some of those will be
genuine failures; some will be classifier error.

**Send the flagged list to P1, P2 and P4** — they pair against these same baselines, so a
broken baseline distorts their averages too, invisibly.

---

## 6.5a — Model-signature check: blocked

The point of this check is a **cross-model comparison**: compute the stage metrics on final
baseline images from VAR *and* DiT, and look for a systematic offset (VAR's codebook
quantization against DiT's continuous VAE) that would have to be normalised out before any
cross-model frequency claim.

**There is no DiT data on your disk.** No DiT baselines, no DiT stages, nothing. So this
cannot be done — it is not a tooling gap, it is missing data.

### What to ask P1 for

Two things, and the DiT run is the same conversation:

1. **~20 clean DiT baseline PNGs**, matching the VAR ones you already have. That alone
   unblocks 6.5a.
2. **The DiT stage extraction**, using the fixed `p3_observe.py`:
   `--hook lanes/p3/p3_observe.py:dit_stages`. The batch bug P1 found is fixed and
   reproduced (24 saves → 336); it needs to be merged before they re-run.

### When the DiT PNGs arrive

```
python -m lanes.p3.calibrate_metrics --baseline-dir <dit_pngs> --n-images 20
```

Compare the "FULL 256" row against VAR's, which I already measured on your 20 images:

| | spec_centroid | grad_energy | hf_ratio |
|---|---|---|---|
| **VAR** finals | 0.0700 | 0.03541 | 0.045089 |
| **DiT** finals | ? | ? | ? |

A systematic difference is the per-model signature. It is also the thing that would make a
naive cross-model frequency plot wrong, which is why it has to be measured before Figure 3
carries both models.

---

## Order of work

1. **6.5c** — run it, read the stage-8-vs-stage-9 swap, write it down
2. **6.5b** — run it, eyeball the flagged images, send the list to the team
3. **Ask P1** for the DiT baselines and the DiT re-run with the fixed hook
4. **6.5a** — when the DiT data lands

Items 1 and 2 are a laptop and ten minutes. Item 4 is someone else's reply.
