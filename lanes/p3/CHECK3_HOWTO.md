# Check 3 on your own MacBook — no GPU, no Colab

**You do not need Colab Pro, and you do not need the GPU.**

`fhat_to_img` is `decoder(post_quant_conv(f_hat))`. The VAR transformer is never touched by
check 3. So the only thing you need is the VQVAE:

| | |
|---|---|
| `vae_ch160v4096z32.pth` | **436 MB** ← the only download |
| `var_d20.pth` | 2.4 GB ← **not needed** |
| Compute | 5 images × 10 scales = 50 conv decodes, a couple of minutes on CPU |
| Peak memory | well under 2 GB |

`decode_stages.py` now has a `--vae-only` flag that builds just the VQVAE from the same
config the real runner uses, so the decoder is bit-for-bit the one that produced the images.

---

## Step 1 — Download the VAE (436 MB, once)

```
cd "/Users/darien/Documents/MSAI Course Notes/AI6132 Generative AI for Visual Synthesis/var-scale-audit"
```

```
mkdir -p checkpoints/var
```

```
curl -L -o checkpoints/var/vae_ch160v4096z32.pth https://huggingface.co/FoundationVision/var/resolve/6d0ee6598a42f75079aa79e58b012b57b284c91b/vae_ch160v4096z32.pth
```

**Verify it before using it** — a truncated download produces a confusing load error later:

```
shasum -a 256 checkpoints/var/vae_ch160v4096z32.pth
```

Must print:

```
7c3ec27ae28a3f87055e83211ea8cc8558bd1985d7b51742d074fb4c2fcf186c
```

If it does not match, delete and re-download. (`checkpoints/` is gitignored, so this never
gets committed.)

## Step 2 — Decode

```
conda activate var-dit
```

```
python -m lanes.p3.decode_stages --stages p3_var_stages --model var --config configs/var_d20.yaml --n 5 --vae-only
```

Expect:

```
decoding 5 images x 10 stages from p3_var_stages
building VQVAE only (ch=160, Cvae=32) on cpu — transformer not loaded
  stage 0 done
  ...
wrote p3_var_stages/decoded_var/contact_var.png
```

**On speed:** CPU is the safe default. You can try `--device mps` for the Apple GPU, but
PyTorch 2.1's MPS backend is patchy on some conv/attention paths — if it errors or produces
NaNs, drop back to CPU. 50 decodes is a couple of minutes either way; this is not worth
debugging.

## Step 3 — Open the contact sheet and actually look at it

`p3_var_stages/decoded_var/contact_var.png` — one row per image, one column per scale
(si 0 → 9, left to right).

**Three things to check:**

1. **Left to right goes coarse → fine.** si 0 should be a near-flat blob; si 9 a sharp dog.
2. **The last column matches that image's final baseline PNG.** If not, the hook fired at
   the wrong point.
3. **Rows are different images**, not the same one repeated. If identical, the batch
   indexing is wrong.

---

## Step 4 — The comparison that makes this worth doing

This is why check 3 matters more than it did a week ago. Compare what you see against the
ladder from the real-baseline calibration:

| si | grid | spec_centroid | grad_energy | hf_ratio |
|---|---|---|---|---|
| 1 | 2×2 | 0.0157 | 0.00103 | 0.000000 |
| 3 | 4×4 | 0.0157 | 0.00310 | 0.000000 |
| 5 | 6×6 | 0.0163 | 0.00475 | 0.000000 |
| 7 | 10×10 | 0.0192 | 0.00720 | 0.000000 |
| 9 | 16×16 | 0.0231 | 0.00990 | 0.000001 |
| — | full 256 | 0.0700 | 0.03541 | 0.045089 |

That ladder is the **best case** — a real photo downsampled to each scale grid and back. It
is an upper bound on what the model could have committed.

**If your decoded scales look like the ladder** (blurry early, sharp late): pixel metrics are
weak for VAR, as predicted, and the semantic curve has to carry the argument.

**If your decoded scales look sharp even at si = 2:** the VQVAE decoder is adding texture the
model has not committed to. Every pixel-space metric is then partly measuring the decoder
rather than the model — **that is a reportable finding**, and it is the second independent
piece of evidence (after the orthogonality result) that the latents alone cannot answer the
semantic question.

Either answer is useful. That is what makes this a real check rather than a formality.

---

## Step 5 — Then the semantic curve (the actual headline)

Once the decoder works, the same 436 MB gets you the result that replaces the energy curve.

Decode **all 10 stages** for a larger subset, then run the classifier on each stage:

```
python -m lanes.p3.decode_stages --stages p3_var_stages --model var --config configs/var_d20.yaml --n 32 --vae-only
```

```
python -m lanes.p2.quality <each stage's folder>
```

Then plot, per stage: `cls_prob` for the true class, whether top-1 matches the final image's
prediction, and `incep_cos` to the final image.

**State the prediction before you look.** From P2's pilot — perturbing scale 0 gave a
different but clean image *of the same class* — the class should be settled very early,
possibly by si 0–2, while the energy curve is still above 0.9. If that holds, the headline
is **"VAR decides *what* early and *how* late"**, and the two curves crossing is Figure 3.

Writing the prediction down first is what makes it a test rather than a story.

---

## If you do want Colab anyway

You do not need it for check 3, but for the record: `--vae-only` also works there, and Colab
free is enough since there is no transformer. The only reason to reach for Colab Pro on this
project is a full **generation** run (VAR peaks at 17.1 GB at batch 16), which is not what
check 3 does.

## Why this changed

The original `decode_stages.py` called `load_model(cfg)`, which builds *both* models and
calls `require_checkpoints`, so it demanded all 2.9 GB and refused to start without the
transformer. `--vae-only` builds `VQVAE(...)` directly with the config's own build kwargs and
loads only the 436 MB file.
