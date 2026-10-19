# P3 Step 2 — verifying the extraction, start to finish

**Run this on 16 images of your own BEFORE asking P1 to attach the hook to the baseline
run.** It takes a few minutes and it is the difference between a figure that means something
and a figure that is smooth, plausible and wrong.

> **Do not debug the hook inside someone else's job.** It cannot corrupt their run — an
> Observe hook gets `state.clone()` and its return value is discarded — but it *can* kill
> one, because hook functions are called with no try/except around them. `_save` never
> raises for exactly this reason, but a wrong `P3_OUT` or a full quota is still something to
> find on your own 16-image job.

Three checks. The first two are automated and run anywhere; the third needs the GPU and
needs your eyes, and it is the one that actually catches a bad extraction.

---

## Before you start: where to run these commands

**Every `python -m lanes.p3.…` command must be run from the REPO ROOT** — the folder that
contains `runner/`, `lanes/` and `manifest/`. That is how Python finds the `lanes` package.

```
cd /path/to/var-scale-audit
ls
```

You should see `runner  lanes  manifest  configs  jobs  docs`. If you do not, you are in the
wrong place.

Run from anywhere else and you get:

```
ModuleNotFoundError: No module named 'lanes'
```

which is confusing, because it is identical whether you are in the wrong directory or have
not cloned the repo at all. Two checks tell them apart:

| Command | Meaning |
|---|---|
| `pwd` | where you actually are |
| `ls manifest/frozen_v1_1.csv` | found → right place, just `cd` wrong earlier; "No such file" → not in a repo checkout |

If you have not cloned it yet:

```
git clone https://github.com/amy499/var-scale-audit.git
cd var-scale-audit
```

`make_pilot_manifest.py` is the exception — it finds the manifest itself, so
`python lanes/p3/make_pilot_manifest.py` works from any directory. The others
(`verify_extraction`, `decode_stages`, `calibrate_metrics`) import from the repo and do
need the repo root.

## Step 2.0 — Make a 16-image manifest

Do not run the pilot on the first 16 rows of the frozen manifest — those are all Chihuahuas,
so a bug that only shows on some breeds would hide. This takes an even spread across all 7
classes instead:

```
python -m lanes.p3.make_pilot_manifest
```

or, from any directory at all:

```
python lanes/p3/make_pilot_manifest.py
```

Expected output:

```
wrote manifest/pilot16.csv
  16 rows, 7 of the manifest's 7 classes
  per class: {'151': 3, '153': 3, '162': 2, '207': 2, '235': 2, '254': 2, '258': 2}
```

The rows are copied verbatim from `manifest/frozen_v1_1.csv`, so every pilot image is one
the full sweep will also produce — same `(class_id, seed)`, same image.

`--n 28` gives 4 per class; `--out` changes the destination. Keep `--n` a multiple of 16 or
the last batch is partial and `runner.generate` needs `--allow-partial-batch`.

> **Do not commit `pilot16.csv`.** It is scratch. A stray extra manifest in the repo is
> exactly the kind of thing someone accidentally runs the real sweep on. Add it to
> `.gitignore` if you like.

## Step 2.1 — Run the hook

**Where you run this decides what you get.** Two different jobs:

| | Where | Weights | Time | What it proves |
|---|---|---|---|---|
| **2.1a** | your Mac, CPU | tiny, random | minutes | the hook's MECHANICS are right |
| **2.1b** | NSCC, GPU | real | ~1 min | the extraction is CORRECT |

Do **2.1a first** — it catches almost every hook bug without a cluster queue or a 5.9 GB
download, and it is the fastest edit-test loop you will get.

> **Do not download the checkpoints to your laptop.** They are ~5.9 GB, the config is
> `device: cuda` so they will not run on a Mac anyway, and the repo README is explicit that
> **CPU outputs are never used for results** — they are not bit-identical to the GPU
> reference hashes. Real results come from NSCC only.

### 2.1a — tiny models on your Mac (do this first)

The repo ships random-weight models that keep the **real stage structure** — 10 VAR scales,
250 DiT steps — at a fraction of the size. Perfect for checking a hook.

```
conda activate var-dit
```

```
mkdir -p outputs/tiny
```

```
python scripts/phase3/check_hooks.py --make-tiny var outputs/tiny
```

That writes `outputs/tiny/var_tiny.yaml`. Now the hooked run and its baseline:

```
export P3_OUT=outputs/p3/tiny/stages
```

```
python -m runner.generate --config outputs/tiny/var_tiny.yaml --manifest manifest/pilot16.csv --batch-size 16 --out-dir outputs/p3/tiny/hooked --hook lanes/p3/p3_observe.py:var_stages
```

```
python -m runner.generate --config outputs/tiny/var_tiny.yaml --manifest manifest/pilot16.csv --batch-size 16 --out-dir outputs/p3/tiny/baseline
```

Then run **Step 2.2** against those two directories. Checks 1 and 2 are fully meaningful
here: check 1 proves the hook changes nothing, check 2 proves the right states come out.

**Check 3 is NOT meaningful on tiny models** — the weights are random, so the images are
noise. Skip it here and do it in 2.1b.

The tiny VAR model keeps `Cvae: 32` and the real `patch_nums`, so `f_hat` is `(B, 32, 16, 16)`
exactly as in the real model. Tiny **DiT** shrinks the image to 64 px, so its latent is
`8x8` rather than `32x32`; `verify_extraction.py` reads the expected size from `run.json`
and prints a note rather than failing.

### 2.1b — real weights on NSCC

On the **login node**, once:

```
python scripts/download_checkpoints.py --configs configs/var_d20.yaml
```

Then submit the pilot as a GPU job (copy `jobs/baseline.pbs` as a starting point — do not
run a GPU workload on the login node):

```
export P3_OUT=outputs/p3/pilot/stages
```

```
python -m runner.generate --config configs/var_d20.yaml --manifest manifest/pilot16.csv --batch-size 16 --out-dir outputs/p3/pilot/var_d20 --hook lanes/p3/p3_observe.py:var_stages
```

```
python -m runner.generate --config configs/var_d20.yaml --manifest manifest/pilot16.csv --batch-size 16 --out-dir outputs/p3/pilot/baseline_var
```

VAR ran at 0.11 s/image in the P2 pilot, so 16 images is seconds of compute — the queue wait
dominates. Run **Step 2.2 and Step 2.3** on this output; here all three checks mean
something.

For DiT, swap the config to `configs/dit_xl2_256.yaml` and the hook to `:dit_stages`.

> Running real weights on your Mac with `--device cpu` is possible but very slow, and the
> README says never for results. Not worth it: 2.1a answers the same question in minutes.

## Step 2.2 — Checks 1 and 2 (automated)

```bash
python -m lanes.p3.verify_extraction \
    --stages   outputs/p3/pilot/stages \
    --run      outputs/p3/pilot/var_d20 \
    --baseline outputs/p3/pilot/baseline_var \
    --model    var
```

Exit code 0 = passed, 1 = failed, so you can put this in a job script.

**What a pass looks like:**

```
CHECK 1 — hook changed nothing
  compared            : 16
  PASS  all 16 images byte-identical to the baseline

CHECK 2 — states written  (outputs/p3/pilot/stages/var)
  .npz files          : 10
  distinct stages     : 10  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
  latent shapes       : [(32, 16, 16)]
  total rows saved    : 160   (expect 10 stages x 16 images = 160)
  PASS  10 stages x 16 images, all finite and consistent
```

### What each failure means

| Message | What happened | Fix |
|---|---|---|
| `N of 16 images differ from the baseline` | The hook changed the run. It is not acting as an Observe hook. | Check it is `Observe(...)` not `Modify(...)`, and that nothing writes into a state tensor. **Discard the run.** |
| `consumed different randomness` | The two runs drew different random numbers. The comparison is invalid even if images match. | Same manifest, same batch size, same config in both runs. |
| `no .npz files at all` | The hook never fired, or wrote somewhere else. | Is `P3_OUT` exported in the *same shell* as the run? Does the `--hook` path resolve? |
| `9 VAR scales, expected 10` | A batch failed to save. | Look at the job log for that stage; disk quota is the usual cause. |
| `contains NaN or Inf` | The saved latent is not usable. | Likely a real numerical problem in the run — raise it, do not paper over it. |
| `latent is (8, 8), expected (16, 16)` | The wrong tensor was saved. | VAR `f_hat` is always 16×16 at every scale, not the scale's token grid. |
| `stage(s) do not have 16 rows` | Two runs wrote into one folder, or a batch was lost. | Delete `P3_OUT` and re-run; never merge two runs into one stages folder. |

---

## Step 2.3 — Check 3: decode five and LOOK (do not skip)

Checks 1 and 2 prove the plumbing. **They cannot tell you the saved tensor is the right
tensor.** A hook that saves the wrong thing, or the right thing at the wrong moment, passes
both and then produces a perfectly smooth, perfectly meaningless curve.

```bash
python -m lanes.p3.decode_stages \
    --stages outputs/p3/pilot/stages \
    --model  var \
    --config configs/var_d20.yaml \
    --n 5
```

Writes `outputs/p3/pilot/stages/decoded_var/contact_var.png` — one row per image, one column
per stage. **Open it.**

### What you are looking for

1. **Left to right goes coarse → fine** (VAR) or **noisy → clean** (DiT).
2. **The last column matches that image's final baseline PNG.** If the last column is not the
   finished image, the hook is firing at the wrong point.
3. **Rows are different images**, not the same one repeated. If they are identical, the batch
   indexing is wrong and every per-image number is garbage.

### Then compare it against the ladder

```bash
python -m lanes.p3.calibrate_metrics
```

prints the best case at each VAR scale on an ideal codec. Compare the look of your decoded
scales against it:

- **They match** (blurry early, sharp late) → the pixel metrics are weak for VAR, as the
  ladder predicts. Use `latent_rel_dist` as the primary metric and say so in the report.
- **Your scales look sharp even at si=2** → **the VQVAE decoder is adding texture the model
  has not committed to.** Every pixel-space metric is then partly measuring the decoder.
  That is a genuine finding, not a bug: record it, and lean on `latent_rel_dist`, which
  reads the model's state directly and does not touch the decoder.

Either way you learn which metric set is honest, which is the whole point of Gate B.

---

## Step 2.4 — Then, and only then, the ride-along

Now the hook is trusted, so ask P1 to attach it to the clean baseline run on the full
`manifest/frozen_v1_1.csv`, with `P3_OUT` exported to a fresh directory:

```
--hook lanes/p3/p3_observe.py:var_stages        # and :dit_stages for DiT
```

Expect ~70 MB (VAR, 10 scales × 224) and ~90 MB (DiT, 24 steps × 224). The baseline images
stay bit-identical, so this costs no extra GPU time.

**Then run `verify_extraction.py` once more on that output**, with `--baseline` pointing at
a clean 224-image baseline. Checks 1 and 2 are enough this time — you are no longer asking
*is the hook correct* (you know it is), you are asking *did this long job complete*. It
catches a batch lost halfway through, which is otherwise invisible until the curves look
subtly odd. Also check that `P3_OUT/_failures.log` does not exist.

---

## One caveat on the decode

`verify_extraction.py` is tested — I built fixtures reproducing each failure mode above and
confirmed it catches all of them.

`decode_stages.py` is **not** tested against the real models, because that needs the
checkpoints and a GPU. Its grouping, CLI and contact-sheet code are tested; the two decode
calls are written from the repo's own code:

- VAR: `vae.fhat_to_img(f_hat)` → `.add(1).mul(0.5)`, per `docs/hook_interface.md` line 372,
  which names `fhat_to_img` as "the only input to the decoder"
- DiT: `vae.decode(x / 0.18215).sample` → `127.5 * x + 128`, copied from
  `runner/dit_model.py`

If the decode throws on first run, that is where to look — and the fix is small.
