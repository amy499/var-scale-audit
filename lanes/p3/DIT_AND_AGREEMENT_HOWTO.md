# Steps 3–5: DiT identity curve, agreement analysis, model-signature check

Everything here runs on a MacBook. No GPU, no cluster, no new generation runs — only the data
P1 already captured in `25985813_p3_dit_stages/` and the result files P1 and P2 published.

| Step | What you get | Needs first | Rough time |
|---|---|---|---|
| **3** DiT identity curve | `dit_semantic.csv/.png` — the DiT half of Figure 3 | a ~335 MB download (3.1) | 15–45 min, mostly waiting |
| **4** Agreement analysis | VAR: rankings + Spearman table; DiT: side-by-side table by band | P1's and P2's CSVs (4.1); for DiT, step 3.5 | 5 min |
| **5** Model-signature check | DiT's row in `calibration_report.md` finding 11 | nothing — P1's `dit20/` folder is already here | 1 min |

Steps 4 and 5 do not depend on step 3's long decode, so you can do them while it runs
(in a second Terminal window — repeat "Before you start" there).

---

## Before you start (every time you open Terminal)

1. Open **Terminal**.
2. Go to the repo. Paste this whole line — the quotes matter, because the path has spaces:

   ```
   cd "/Users/darien/Documents/MSAI Course Notes/AI6132 Generative AI for Visual Synthesis/var-scale-audit"
   ```

3. Switch on the project's Python environment:

   ```
   conda activate var-dit
   ```

4. Check you are in the right place with the new files:

   ```
   ls lanes/p3/model_signature.py
   ```

   It should print `lanes/p3/model_signature.py`. If it says "No such file", you are in the
   wrong folder — redo step 2.

**Every command below is ONE line. Copy the whole line, paste, press Enter.**

---

## Step 3 — DiT identity curve

### 3.1 Download DiT's decoder (once only, ~335 MB)

DiT's images are decoded by a separate network (`sd-vae-ft-ema`). You need only that, not
the 2.7 GB DiT model. Run these three lines one at a time:

```
mkdir -p checkpoints/sd-vae-ft-ema
```
```
curl -L -o checkpoints/sd-vae-ft-ema/config.json https://huggingface.co/stabilityai/sd-vae-ft-ema/resolve/f04b2c4b98319346dad8c65879f680b1997b204a/config.json
```
```
curl -L -o checkpoints/sd-vae-ft-ema/diffusion_pytorch_model.safetensors https://huggingface.co/stabilityai/sd-vae-ft-ema/resolve/f04b2c4b98319346dad8c65879f680b1997b204a/diffusion_pytorch_model.safetensors
```

The second download shows a progress bar and takes a few minutes. You do not need to check
the files yourself: every time `decode_stages.py` runs, it checks both against the fingerprints
(sha256) pinned in `configs/dit_xl2_256.yaml` and stops with a clear message if either is
wrong. (`checkpoints/` is git-ignored, so none of this goes to GitHub.)

### 3.2 Decode 32 images × 24 steps

```
python -m lanes.p3.decode_stages --stages 25985813_p3_dit_stages --model dit --config configs/dit_xl2_256.yaml --vae-only --device mps --n 32 --sheet-rows 8
```

What it prints, in order:

```
decoding 32 images x 24 stages from 25985813_p3_dit_stages (7 classes)
DiT VAE: 2 files present in checkpoints/sd-vae-ft-ema, sha256 verified
building AutoencoderKL only on mps — DiT transformer not loaded
  stage 999 done
  stage 919 done
  ...
  stage 0 done
wrote 25985813_p3_dit_stages/decoded_dit/contact_dit.png (8 of 32 images; all 32 written as PNGs)
```

- **How long?** Time the gap between the first two `stage ... done` lines and multiply by 24.
- **If it fails with an error that mentions `mps`**, run the same line with `--device cpu`
  instead of `--device mps`. Slower, but always works.
- **Output:** `25985813_p3_dit_stages/decoded_dit/` — 768 PNGs plus `contact_dit.png`.

These are **the same 32 images your VAR identity curve used** (checked on your data), so the
VAR and DiT curves compare like with like.

### 3.3 Look at the contact sheet (this is check 3 for DiT)

Open `25985813_p3_dit_stages/decoded_dit/contact_dit.png`. One row per image, one column per
step, labelled 999 on the left to 0 on the right. Check three things:

1. **Each row gets clearer from left to right.** Every panel is the model's current guess at
   the *finished* image, so expect the left columns to be vague or blurry and the picture to
   firm up towards the right.
2. **The right-most column is a clean, finished dog.**
3. **Each row is a different dog.**

If a row gets *worse* from left to right, the right-most column is not a clean image, or every
row is the same dog, **stop** and send me the sheet.

### 3.4 Run the DiT identity curve

```
python -m lanes.p3.semantic_curve --decoded 25985813_p3_dit_stages/decoded_dit --model dit --csv outputs/p3/figure3/dit_semantic.csv --plot outputs/p3/figure3/dit_semantic.png
```

It starts with `32 images x 24 stages (1 non-matching file(s) skipped)` — the skipped file
is the contact sheet — and `final stage = 0`. Then one line per step, a table, and at the end:

```
identity_progress first reaches 0.50 at stage ...
identity_progress first reaches 0.90 at stage ...
identity_progress first reaches 0.95 at stage ...
```

**Write those three timesteps down.** For VAR the same lines read 0.50 at scale 8 and 0.90 at
scale 9. To compare "how far through generation", convert DiT's timestep to its step number
(out of 249) with this table:

| t | step | t | step | t | step | t | step |
|---|---|---|---|---|---|---|---|
| 999 | 0 | 562 | 109 | 249 | 187 | 64 | 233 |
| 919 | 20 | 502 | 124 | 209 | 197 | 44 | 238 |
| 839 | 40 | 445 | 138 | 173 | 206 | 28 | 242 |
| 766 | 58 | 389 | 152 | 140 | 214 | 16 | 245 |
| 694 | 76 | 341 | 164 | 112 | 221 | 8 | 247 |
| 626 | 93 | 293 | 176 | 84 | 228 | 0 | 249 |

Example: if DiT first reaches 0.50 at t = 249, that is step 187 of 249, i.e. 75% of the way
through generation.

It runs on CPU in a few minutes; add `--device mps` to the end to speed it up.

### 3.5 If you have not run it yet: the DiT energy curve (step 2)

```
python -m lanes.p3.stage_curves --stages 25985813_p3_dit_stages --model dit --csv outputs/p3/figure3/dit_curves.csv --plot outputs/p3/figure3/dit_curves.png
```

No decoding needed; runs in seconds. The plot title now states what the diagnostic found —
for DiT it reads `NOT orthogonal (max |cos - ratio| = 0.37): rel_dist carries direction too`.
That is correct and is the finding: unlike VAR, DiT's earliest guesses point partly the
wrong way and get revised.

---

## Step 4 — Agreement analysis

P2 has chosen the severities (2 Nov 2026): **2 for VAR** — at 1 every scale comes out
unchanged, at 4 every scale is over 90% damaged, so 2 is the only level that separates the
scales — and **0.5 for DiT**, where damage rises steadily instead of tying at 100%. Both use
P2's same severity rule (noise relative to the state's own spread).

Two things decide how P2's numbers are used, and both are already built into the scripts:

- **Yardstick.** `RANKING_SPEC.md` asks every lane to rank by the *validity drop*
  (classifier probability before minus after) — P1's yardstick too. P2's own summary measure
  is the *fraction of images damaged* (`recovery_ranking.csv`). The scripts rank P2 by the
  validity drop and show P2's damage measure next to it, so you can see whether the
  conclusion depends on the yardstick. (It does not: the two P2 rankings agree, rho +0.78.)
- **Both noise draws.** P2 ran VAR twice with different noise (`gateb/` = first draw,
  `phasec/` = second). P2 showed they agree; the script averages them, which halves the noise
  in P2's numbers. The first draw alone reproduces `recovery_ranking.csv` exactly.

The **preliminary runs** you made earlier (`outputs/p3/agreement/prelim_sev1/` and
`prelim_sev2/`) are superseded. They do no harm — `outputs/` never goes to GitHub — but you
can delete them: `rm -r outputs/p3/agreement/prelim_sev1 outputs/p3/agreement/prelim_sev2`

### 4.1 Get the latest files

P1's branch has been merged, so everything now comes from `main`. Run these one at a time:

```
git fetch origin
```
```
mkdir -p outputs/p3/agreement/inputs
```
```
git show origin/main:lanes/p1/results/ranking.csv > outputs/p3/agreement/inputs/p1_ranking.csv
```
```
git show origin/main:lanes/p2/results/gateb/var_p2_summary.csv > outputs/p3/agreement/inputs/p2_gateb_var_summary.csv
```
```
git show origin/main:lanes/p2/results/phasec/p2_summary.csv > outputs/p3/agreement/inputs/p2_phasec_summary.csv
```

(`p2_var_summary.csv` from your earlier fetch is the same file as `p2_gateb_var_summary.csv`
under its old name; it is no longer used.)

### 4.2 VAR: the final run

```
python -m lanes.p3.prep_rankings --p1 outputs/p3/agreement/inputs/p1_ranking.csv --p2 outputs/p3/agreement/inputs/p2_gateb_var_summary.csv outputs/p3/agreement/inputs/p2_phasec_summary.csv --p2-severity 2
```

Results go to `outputs/p3/agreement/`. Check the P2 block says
`noise draws (seeds) ['0', '1']; rows averaged per scale: 2`, then you should see:

```
p1_remove        | ranked most->least important: [3, 2, 4, 6, 9, 7, 5, 8, 1, 0]
p2_validity_sev2 | ranked most->least important: [9, 0, 8, 7, 2, 3, 1, 4, 6, 5]
p2_damaged_sev2  | ranked most->least important: [9, 0, 1, 2, 3, 8, 7, 4, 6, 5]
p3_energy        | ranked most->least important: [9, 8, 7, 6, 5, 3, 0, 1, 4, 2]
p3_identity      | ranked most->least important: [8, 9, 7, 6, 5, 3, 4, 0, 1, 2]

         p1_remove vs p2_validity_sev2  rho -0.248  p 0.4888  significant: no
         p1_remove vs p3_identity       rho -0.085  p 0.8152  significant: no
  p2_validity_sev2 vs p2_damaged_sev2   rho +0.782  p 0.0075  significant: yes
  p2_validity_sev2 vs p3_identity       rho +0.231  p 0.5208  significant: no
         p3_energy vs p3_identity       rho +0.951  p 0.0000  significant: yes
```

(plus a few more pairs). If your numbers differ, a teammate has updated a file since
2 Nov — use yours.

### 4.3 DiT: the side-by-side table

DiT is not ranked and correlated like VAR, for two reasons the script prints: P2's DiT ranking
rises steadily with progress, so a correlation with any of your (also steadily rising) curves
would be ±1 by construction — perfect-looking and meaningless; and P1 removed DiT in only three
bands. So it lines the three lanes up on DiT's timeline instead.

It needs your DiT energy curve from **step 3.5** (`outputs/p3/figure3/dit_curves.csv`) — run
that first if you have not (seconds). Then:

```
python -m lanes.p3.dit_agreement --p1 outputs/p3/agreement/inputs/p1_ranking.csv --p2 outputs/p3/agreement/inputs/p2_phasec_summary.csv --p2-severity 0.5
```

Before step 3.4 has run, the identity column shows `—` and the script says so. **After step
3.4, run the same line again** to fill it in. With your step-3 results and the files as of
2 Nov you should see, in the band table:

```
early  1-83     995-666   |  0.078 [0.027, 0.129]  |  0.016  (n = 1)  |  0.001  0.173
mid    84-166   662-333   |  0.187 [0.134, 0.240]  |  0.062  (n = 3)  |  0.473  0.278
late   167-249  329-0     |  0.266 [0.217, 0.315]  |  0.355  (n = 4)  |  0.519  0.609

    P1 removal       late > mid > early
    P2 corruption    late > mid > early
    P3 identity      late > mid > early
    P3 closeness     late > mid > early
```

and the milestones `P2: corruption at severity 0.5 first stops being absorbed at t = 297
(p = 0.70, outcome degraded-in-place)`, `P3: identity first reaches 0.5 at t = 293` and
`... 0.9 at t = 64`. Results go to `outputs/p3/agreement/dit_comparison.csv`
and `dit_comparison_bands.csv`.

("Closeness" is 1 − rel_dist: how close the predicted latent is to the final one. It is not
called energy for DiT, because the orthogonality check fails for DiT's early steps.)

### 4.4 How to read it, and what to write

**VAR — three lanes, three different answers.** Lead with the contrast, then the numbers:

- **Removal (P1)** hurts most at the mid-coarse scales: scale 3 drops validity by 0.466
  [0.413, 0.520], scale 2 by 0.346.
- **Corruption at severity 2 (P2)** hurts most at the two ends: scale 9 (0.806) and scale 0
  (0.346); the middle (scale 5, 0.138) is the most robust. P2's own README calls this
  U-shape semantic drift at the start and structural damage at the end.
- **Information (P3)** arrives last: scale 8 adds 36% of the identity, scale 9 28%.

None of the three pairs is correlated (P1–P2 −0.25, P1–P3 −0.09, P2–P3 +0.23, all n = 10, all
non-significant), and the P2 result holds under P2's own damage measure (the two P2 rankings
agree, +0.78). Suggested wording:

> On VAR, necessity, robustness and information peak at different scales. Removing a scale
> costs most at the mid-coarse scales (scale 3: validity drop 0.466 [0.413, 0.520]), corrupting
> it costs most at the two ends (scale 9: 0.806; scale 0: 0.346), and the instance identity
> arrives last (scale 8: 36% of it). No pair of the lanes' rankings is correlated (|rho| ≤ 0.34,
> all non-significant, n = 10). The scale that adds the most identity (scale 8) is among the least necessary to
> keep (0.056 [0.030, 0.083]). Containing information, being needed, and being repairable
> are different properties of a scale.

Rules that still apply: with n = 10, a non-significant rho means *the test cannot resolve it* —
write "the rankings do not agree", never "they are opposite"; and always state P2's severity.

**DiT — the lanes agree, at band resolution.** All four rankings put the late band
(steps 167–249) first and the early band last. Your identity is added almost entirely in the
middle and late bands, split nearly evenly (0.47 and 0.52; 0.001 in the early band). And it
becomes half-settled in the same interval in which P2's corruption stops being absorbed: P2 is
absorbed at t = 397 (identity 0.36 by then) and not at t = 297 (identity 0.54), and identity
first reaches 0.5 at t = 293. Suggested wording:

> On DiT, the lanes agree. Removing the late band (steps 167–249) costs the most validity
> (P1: 0.266 [0.217, 0.315], against 0.078 for the early band); corruption at severity 0.5 is
> absorbed up to t = 397 but not from t = 297 on (P2); and the instance identity is added
> almost entirely in the middle and late bands (P3: 0.47 and 0.52 of it, 0.001 in the early
> band), first reaching half at t = 293. Corruption stops being repairable in the same
> interval (p ≈ 0.6–0.7) in which the image's identity becomes half-settled.

That last sentence rests on P2's 8 timesteps: the switch happens somewhere between t = 397 and
297, not at an exact point. Say "the same interval", not "the same moment".

**The cross-model contrast is the headline:** VAR's lanes disagree, DiT's agree. Be honest
about one caveat when you write it — DiT is compared at three bands, VAR at ten scales, and a
coarser comparison has fewer ways to disagree.

---

## Step 5 — Model-signature check

P1's 20 DiT baselines are in `dit20/` (unzipped from `dit_baseline_20.zip`). They were checked
on 2 Nov 2026 and are fit for this:

- **Same 20 images as the VAR set** — identical class and seed for all 20, so the comparison is
  *paired*: each DiT image against its own VAR counterpart. 3 per breed (2 for samoyed).
- **From the same DiT run as your stage data.** A simple linear map from DiT's latent to a
  32×32 thumbnail picks out each PNG's *own* saved latent as the best match out of all 224,
  for all 20 (fit 0.86 vs at most 0.32 for any other latent).
- **Clean and valid:** 256×256 RGB, none blank or clipped, no duplicates, none accidentally a
  VAR image, correct breed in every one by eye.

So step 5 needs no download and no decoding.

### 5.1 Compare VAR and DiT

```
python -m lanes.p3.model_signature --var for_darien_20_clean_pngs --dit dit20 --csv outputs/p3/model_signature.csv
```

It takes about a minute. Check the first lines say:

```
same image in both: 20  ->  PAIRED comparison (DiT minus the same image in VAR)
```

and, under the table, `VAR side reproduces calibration_report.md finding 11 exactly` — proof
the VAR half is measured identically to the report. You should get:

```
        metric |  VAR mean  DiT mean |  DiT - VAR          95% interval  % of VAR | verdict
 spec_centroid |  0.070011  0.075864 |  +0.005853 [-0.014647, +0.027341]     +8.4% | no detectable offset
   grad_energy |  0.035407  0.041008 |  +0.005601 [-0.002655, +0.014710]    +15.8% | no detectable offset
      hf_ratio |  0.045089  0.046955 |  +0.001865 [-0.026390, +0.032413]     +4.1% | no detectable offset
```

**Reading the table.** The last column is the verdict:

| Verdict | Means | What you do |
|---|---|---|
| **OFFSET** | the 95% interval excludes 0: the two decoders leave different fingerprints on this metric | in any VAR-vs-DiT figure of that metric, plot each model **relative to its own final value**, never raw values on one axis |
| **no detectable offset** | the interval includes 0 at n = 20 | report it with n; do not write "the models are the same" |

**All three say "no detectable offset" — but read the intervals, not just the verdict.** They
are wide: as a share of VAR's value, spec_centroid runs from −21% to +39%, grad_energy −7% to
+42%, hf_ratio −59% to +72%. So the check rules out a *large* decoder fingerprint, but an
offset of a few tens of percent could still be hiding. Twenty images cannot settle it.

### 5.2 Update the report

The script prints a line starting `| **DiT** finals (n=20) |`. In
`lanes/p3/calibration_report.md`, finding 11, replace the line `| **DiT** finals | — | — | — |`
with it. Then replace the paragraph under that table (the one starting **DiT's half is still
missing**) with:

> **DiT's half is measured on P1's 20 DiT baselines, paired by class and seed with the VAR
> set.** DiT − VAR, mean of the paired differences with 95% bootstrap interval:
> spec_centroid +0.0059 [−0.0146, +0.0273], grad_energy +0.0056 [−0.0027, +0.0147],
> hf_ratio +0.0019 [−0.0264, +0.0324]. **No metric shows a detectable per-model offset at
> n = 20.** The intervals are wide, though — offsets of up to about +40% of the VAR value
> (spec_centroid, grad_energy) and ±70% (hf_ratio) are not ruled out — and none of the three
> metrics picks up the heavier posterization seen by eye in VAR's contact sheets (finding 8).
> Practical consequence: cross-model figures on a pixel metric still plot each model relative
> to its own final value. P3's primary metrics, latent `rel_dist` and `identity_progress`, are
> relative to each model's own final image by construction and are unaffected.

If any number looks odd, open `outputs/p3/model_signature_per_image.csv` — one row per image —
to see whether a single image is driving it.

### 5.3 Optional, after step 3: a larger sample (n = 32)

Once step 3.2 has decoded DiT, you have 32 finished images per model *from the same decoding
pipeline on your Mac* — the same 32 images in both models. That gives a second, slightly larger
check:

```
python -m lanes.p3.model_signature --var p3_var_stages/decoded_var --var-stage 9 --dit 25985813_p3_dit_stages/decoded_dit --dit-stage 0 --csv outputs/p3/model_signature_n32.csv
```

If its verdicts agree with 5.1, say "confirmed on a second set of 32 images" in the report. If
they disagree, report both and the larger n — that is a result, not a problem.

(`decode_stages.py --keys-like ... --final-only` from the earlier version of this guide is no
longer needed for step 5; the option stays in the script for anyone who lacks baseline PNGs.)

---

## If something goes wrong

| Message | Fix |
|---|---|
| `No module named 'lanes'` | You are not in the repo folder. Redo "Before you start". |
| `DiT VAE file(s) missing ...` | Do step 3.1. The message itself prints the exact lines to run. |
| `... is not the pinned file.` | That download is incomplete. Delete the file it names with `rm <that path>`, then rerun its `curl` line from 3.1. |
| An error mentioning `mps` | Run the same command with `--device cpu`. |
| `No module named 'diffusers'` | `pip install "diffusers==0.27.2" "numpy<2"` (the `numpy<2` keeps torch working). |
| `compiled using NumPy 1.x cannot be run in NumPy 2` | `pip install "numpy<2"` |
| `--p2 needs --p2-severity` | Add `--p2-severity` with a number, as in 4.2. |
| `no VAR rows ... at severity ...` | That severity does not exist; the message lists the ones that do. |
| `holds decoded stages [...]; pass --dit-stage` | Add `--dit-stage 0` (DiT) or `--var-stage 9` (VAR). |

---

## What changed in the code (for the PR)

| File | Change |
|---|---|
| `decode_stages.py` | `--vae-only` now supports DiT (loads only the AutoencoderKL, checks both files' sha256 first). DiT stages are laid out 999 → 0 (was 0 → 999, which made a correct extraction look reversed). New `--keys-like`, `--final-only`, `--sheet-rows`. VAR behaviour unchanged. |
| `stage_curves.py` | Plot title states what the orthogonality diagnostic actually found (was hard-coded "tracks accumulated energy", false for DiT); y-axis no longer clips DiT's 1.06; readable DiT axis labels. |
| `prep_rankings.py` | **New.** Extracts per-scale VAR scores from P1's and P2's tables (P2: several files averaged, validity drop and fraction damaged), runs `agreement.py`. |
| `dit_agreement.py` | **New.** DiT across the lanes on DiT's own timeline: P1's bands, P2's timesteps, P3's curves — no correlation, by design. |
| `model_signature.py` | **New.** VAR-vs-DiT pixel-metric offsets with bootstrap intervals. numpy + Pillow only — no OpenCV or scikit-image needed. |

`outputs/` and `checkpoints/` are git-ignored: results and downloads stay on your Mac.

Owner: P3.
