# P3 — information / semantics lane

Owner: P3. Measures **where information appears** during a normal generation. Nothing here
modifies a run: P3's only hook is an `Observe`, so its output is bit-identical to the clean
baseline.

## Files

| File | What it is |
|---|---|
| `p3_metrics.py` | the shared metric set — fidelity vs validity axes, and P3's stage metrics |
| `p3_observe.py` | the observe hook that saves intermediate states during a run |
| `calibrate_metrics.py` | metric calibration; runs on CPU with no checkpoints |
| `calibration_report.md` | calibration results, 1 Oct 2026 |

## For the other lanes: the metric set

**One number cannot measure recovery.** From the P2 pilot
(`lanes/p2/results/pilot_25656645`):

| Run | PSNR | `cls_prob` | KID | What it actually is |
|---|---|---|---|---|
| VAR si0 sev1.0 | 10.8 | 0.74 (base 0.75) | −0.003 | a clean image of a different dog |
| DiT t197 sev1.0 | 12.6 | **0.00** | **+0.187** | destroyed |

Nearly the same pixel distance, opposite meaning. So every run is placed on two axes:

```
                validity HIGH        validity LOW
fidelity HIGH   unchanged            degraded-in-place
fidelity LOW    RE-ROUTED            DAMAGED
```

Recovery means returning toward *unchanged*. **Re-routing is not damage** and must not be
scored as damage.

**Validity needs two metrics, not one.** DiT t=497 / sev 1.0 is visibly oversaturated but
keeps `cls_prob` 0.73 against a 0.82 baseline — on the classifier alone it is
indistinguishable from a clean re-route. KID catches it (+0.0185 against ≤0.003 for every
clean-looking run).

| Axis | Metric | Source | Status |
|---|---|---|---|
| fidelity | `incep_cos` | here | **required** — free, no new dependency |
| fidelity | `lpips` | here, optional | needs the `lpips` package |
| fidelity | `img_psnr_db`, `img_mad255` | `lanes/p2/analyze.py` | "how much changed", never "how damaged" |
| validity | `cls_prob`, `cls_top1`, `cls_top5` | `lanes/p2/quality.py` | **required** — loss of meaning |
| validity | `kid_vs_base` | `lanes/p2/analyze.py` | **required** — loss of visual quality |

### Why `incep_cos` and not LPIPS first

`lanes/p2/quality.py` already writes `inception_features.npz`: 2048-d Inception pool
features per image, keyed by `(class_id, seed)`. Cosine distance between a run's feature
and its baseline's is arithmetic on a file that already exists — **no new package, no second
forward pass**, which keeps `quality.py`'s "nothing to install" property. LPIPS and DINOv2
both break it.

```bash
# the classifier first (writes inception_features.npz), then fidelity
python -m lanes.p2.quality  RUN_DIR BASELINE_DIR
python -m lanes.p3.p3_metrics fidelity RUN_DIR --baseline BASELINE_DIR
python -m lanes.p3.p3_metrics fidelity RUN_DIR --baseline BASELINE_DIR --lpips   # optional
```

Appends to `RUN_DIR/metrics.csv` in P2's long format
(`class_id,seed,metric,value,stage`), replacing earlier values of the same metrics.
`--dry-run` prints without writing.

## Running the calibration

```bash
python -m lanes.p3.calibrate_metrics                    # sample photos, runs anywhere
python -m lanes.p3.calibrate_metrics --baseline-dir outputs/baseline/25665469/var_d20
```

Writes `calibration_curves.png` and `calibration_decisions.json`. Deterministic — the noise
draw is seeded from the project master seed. The pieces are importable, so other lanes can
reuse them instead of copying:

```python
from lanes.p3.calibrate_metrics import var_scale_ladder, DEGRADATIONS, load_images

for row in var_scale_ladder(load_images()):      # what can VAR scale si contain? (no GPU)
    print(row["si"], row["grid"], row["grad_energy"])

blurred = DEGRADATIONS["blur"][0](img, 2.0)      # apply a known degradation to your image
```

## P3's stage metrics

| Metric | Role |
|---|---|
| `latent_rel_dist` | **PRIMARY** — relative L2 from a stage's latent to the final latent. Needs no decoder. |
| `grad_energy` | primary pixel metric — the only one with real range on the VAR ladder |
| `spec_centroid` | supporting — graded but compressed on the VAR ladder |
| `hf_ratio` | **late stages only** — ≈0 across the whole VAR ladder |
| `pixel_entropy` | dropped 2026-10-01, failed calibration |

**Why the primary metric is latent-space.** Every pixel metric runs on a *decoded*
intermediate, so it measures the model's commitment and the decoder's behaviour together
and the two cannot be separated afterwards. `latent_rel_dist` reads the model's own state:
1.0 = nothing committed, 0.0 = fully determined. It works identically for VAR (`f_hat`) and
DiT (`pred_xstart`) — but the two latent spaces differ, so **compare the shape of the
curves, never the absolute values**.

## Two traps, both already hit

**Flat images return plausible nonsense.** A constant image has a normalised power spectrum
made entirely of floating-point noise, and `spec_centroid` returns `0.556` — a number
indistinguishable from a real measurement. VAR scale 0 is a 1×1 token grid and decodes to a
near-constant image, so this case *is* reached. `p3_metrics` returns `NaN` below
`FLAT_IMAGE_STD`. Any new spectral metric needs the same guard.

**Blur and noise move every frequency metric in opposite directions.** VAR's early scales
are blur-like (high frequency absent); DiT's early timesteps are noise-like (high frequency
in excess). A single "high-frequency energy" axis carrying a VAR curve and a DiT curve is
not a comparison — it is two unrelated quantities sharing a y-axis.

## Collecting intermediate states

`p3_observe.py` is an `Observe` hook, so it cannot change the run and the images come out
bit-identical to the baseline. **It can therefore ride along with the clean baseline run
instead of costing a separate GPU job.**

```bash
python -m runner.generate --config configs/var_d20.yaml \
    --manifest manifest/frozen_v1_1.csv --batch-size 16 \
    --out-dir outputs/p3/var_stages --hook lanes/p3/p3_observe.py:var_stages
```

Saves VAR `f_hat` at all 10 scales (~70 MB) and DiT `pred_xstart` at 24 steps (~90 MB).
The DiT steps are spaced **denser near the end of sampling**, because DDPM's schedule is
non-linear and uniform spacing would spend most of the budget on near-identical noisy
latents.

**Verify on 16 images before the full sweep:** final images byte-identical to the baseline
(`tensor_sha256`), file count = stages × images, and **decode five and look at them**.

## Known gap: VAR token entropy is not measurable

Two independent blockers, both worth stating in the report rather than omitting:

1. **Not reachable.** `docs/hook_interface.md` §89: `logits_BlV` is "only visible inside the
   sampler wrapper, before both hook points". The hook receives `idx_Bl` — *sampled token
   ids*, not the distribution they came from. A logits hook is listed as future work (§534).
2. **Confounded even if it were.** The frozen VAR cfg is a ramp,
   `t = cfg * si / (num_scales - 1)`: guidance is **0.0 at si=0** and **1.5 at si=9**.
   Guidance sharpens the output distribution by construction, so guidance strength is
   perfectly confounded with scale index and an entropy-vs-scale curve would be partly a
   plot of the cfg schedule.

`latent_rel_dist` answers the same question — "how much is still undetermined?" — with no
logits and no confound.
