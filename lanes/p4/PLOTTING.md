# The shared early / middle / late plot template

Owner: P4. For P1, P2 and P3 — **you should not need to read P4's code to use this.**

One import gives every lane the same early/middle/late figure and the same tidy table underneath it,
so four lanes' figures line up instead of drifting apart. It reads the `stages` table that
`runner/generate.py` already writes into every `run.json`, and the merged CSVs that
`python -m lanes.p2.schema collect` already produces. It needs no GPU, no checkpoint, and no change
to `runner/`.

```python
from lanes.p4 import plotting
```

Run from the repo root, with `PYTHONDONTWRITEBYTECODE=1` set, as for everything else in this repo.

---

## 1. Two layers, because the env has no plotting library

| Layer | Needs | What it does |
|---|---|---|
| **data** | standard library only | maps your run's stages to both progress axes, assigns bands, joins your metric values, emits one tidy table |
| **render** | matplotlib, else Pillow | turns a tidy table into a figure |

`matplotlib` is in **neither** `environment.yml` nor `environment-macos.yml`; Pillow is in both. So the
render layer imports its library at call time and `backend="auto"` uses matplotlib when it is there and
Pillow when it is not. **Your analysis works today either way** — only the picture needs a library.
Installing matplotlib into `var-dit` is P1's call (`docs/HANDOVER.md` forbids an unannounced
`pip install`); the Pillow output is a real plot, not a placeholder.

## 2. The tidy table

One row per (run, image, stage, metric). These are P2's merged columns (`lanes/p2/SCHEMA.md` §4–5) plus
the four this template adds, so you can hand over your table with no conversion.

| Column | Type | Meaning |
|---|---|---|
| `run_id` | str | P2's `<lane>/<experiment>/<arm>`, or whatever you pass |
| `lane` | str or None | `p1` … `p4` |
| `model` | str | `var` or `dit` |
| `arm` | str | the series name in the figure — your condition label (`baseline`, `protect`, `scale3_sev0.5`, …) |
| `class_id`, `seed` | int or None | the image (the manifest key) |
| `stage` | int | **native** stage: VAR `si` 0–9, DiT `timestep_in` 999…0 |
| `p_place` | float [0,1] | **added by P4.** The placement axis. Bands are cut on this; it is the figure's x axis |
| `p_func` | float [0,1] | **added by P4.** The functional axis — how much is actually committed at this stage (VAR `cum_tokens/total_tokens`, DiT `sqrt(alpha_bar_in)`) |
| `band` | str | **added by P4.** `early`, `middle` or `late` |
| `metric` | str | your metric's name, used verbatim |
| `label` | str | **added by P4**, from your `MetricSpec`: the axis text, defaulting to the name |
| `value` | float | the number |
| `direction` | str | **added by P4**, from your `MetricSpec`: `higher_is_better` or `lower_is_better` |
| `level` | str | **added by P4**, from your `MetricSpec`: `per_image` or `per_set` |

`plotting.TIDY_COLUMNS` is this list in order. `write_tidy(rows, path)` / `read_tidy(path)` round-trip
it as CSV.

**Answering P2's open question** (`lanes/p2/SCHEMA.md` §8): `stage` plus a join on `stages.csv` **is
enough** — this template does the join. Please do **not** copy `p` into `metrics.csv`; one stage key per
metric row keeps the long format additive.

## 3. Your metrics are data, not code

Describe each metric once; the template never hardcodes a name, so P3's final definitions drop straight
in.

```python
specs = [
    plotting.MetricSpec("lpips",    direction="lower_is_better",  level="per_image"),
    plotting.MetricSpec("clip_sim", direction="higher_is_better", level="per_image"),
    plotting.MetricSpec("fid",      direction="lower_is_better",  level="per_set"),
]
```

- `direction` is printed on the axis so a reader knows which way is good. Required values:
  `higher_is_better`, `lower_is_better`.
- `level` is `per_image` (several rows per stage, averaged for the figure) or `per_set` (exactly one
  value per stage — more than one is an error, not a silent average).
- `label=` overrides the axis text; the name is used otherwise. It travels in the tidy table, so it
  survives a CSV round trip and reaches the figure.

A metric row whose `stage` is empty is a number about the **final image**: it is placed at the last
stage, so it still lands in `late` rather than being dropped.

## 4. Entry points

| Call | Use when |
|---|---|
| `tidy_from_run(run, metric_rows, specs, arm=, lane=)` | you have one `run.json` and your own metric rows |
| `tidy_from_tables(tables_dir, specs, run_ids=None)` | you have `tables/` from `lanes.p2.schema collect` |
| `lane_plot(tidy, out_path, title=, kind=, backend=)` | tidy table → figure, one call |
| `lane_plot_spec(tidy, title=, kind=)` | the backend-neutral figure description, if you want to render it yourself |
| `render(spec, out_path, backend=)` | render a spec |
| `aggregate(tidy)` | `{(metric, arm): [(p_place, stage, value, p_func), …]}` — the numbers behind the figure |
| `write_tidy` / `read_tidy` | CSV round trip |
| `available_backends()` | `["matplotlib", "pillow"]`, whichever are installed |

`kind` is `"line"` (default) or `"bar"`. `backend` is `"auto"` (default), `"matplotlib"` or `"pillow"`;
naming one that is not installed raises `PlotBackendMissing` with a message saying what to install,
never a bare `ImportError` traceback.

A metric row naming a stage the run does not have raises `PlotError` listing those stages. Mismatches
are reported, never silently dropped — if a row vanished you would be reading a figure with a hole in
it.

## 5. Worked example — VAR

```python
import csv
from pathlib import Path
from lanes.p4 import plotting

run_dir = Path("outputs/p1/scale_sweep/baseline")          # holds run.json
with (run_dir / "metrics.csv").open(newline="") as f:      # lanes/p2/SCHEMA.md section 4
    metric_rows = list(csv.DictReader(f))

tidy = plotting.tidy_from_run(
    run_dir, metric_rows,
    [plotting.MetricSpec("lpips", direction="lower_is_better")],
    arm="baseline", lane="p1",
)
plotting.write_tidy(tidy, "outputs/p1/tidy_baseline.csv")   # works with no plotting library
plotting.lane_plot(tidy, "outputs/p1/fig_lanes.png", title="P1: VAR-d20 scale importance")
```

Several conditions in one figure: build one tidy table per run and concatenate them. The `arm` label is
what separates the series.

```python
tidy = []
for arm, d in (("baseline", base_dir), ("scale3", s3_dir), ("scale7", s7_dir)):
    tidy += plotting.tidy_from_run(d, rows_for(d), specs, arm=arm, lane="p1")
plotting.lane_plot(tidy, "outputs/p1/fig_lanes.png")
```

The x axis is `p_place`, ticked with the native `si`; the three bands are shaded and labelled with the
`p_func` range they cover ("committed 0.044–0.134" for VAR's middle band).

## 6. Worked example — DiT

Identical call shape; only the config and the native units differ. The template reads the model from
`run.json`, so you pass nothing extra.

```python
from lanes.p4 import plotting

tidy = plotting.tidy_from_run(
    "outputs/p2/dit_pilot/baseline", metric_rows,
    [plotting.MetricSpec("img_mse", direction="lower_is_better")],
    arm="baseline", lane="p2",
)
plotting.lane_plot(tidy, "outputs/p2/fig_dit.png", title="P2: DiT-XL/2 step importance")
```

Ticks are native timesteps (999 … 0) and a 250-step axis is thinned to at most 12 ticks so it stays
readable. A `--skip-timesteps` run works unchanged: its remaining steps keep the `p_place` of the
baseline schedule, so a skip run and its baseline line up on one axis.

## 7. From P2's merged tables

```python
specs = [plotting.MetricSpec("lpips", direction="lower_is_better")]
tidy = plotting.tidy_from_tables("outputs/p3/tables", specs)        # runs.csv + stages.csv + metrics.csv
plotting.lane_plot(tidy, "outputs/p3/fig_lanes.png", title="P3: quality metrics")
```

`arm` and `lane` come from `runs.csv`, so nothing extra is passed. `run_ids=[...]` limits it to the runs
you want in one figure.

## 8. Figure 5

`lanes/p4/figure5.py` places one panel per lane on the same progress axis, each with its own metric and
its own scale, with no averaging or ranking across lanes. Hand P4 a tidy table from this template —
`write_tidy(...)` output is enough — and it drops into the frame.

```python
from lanes.p4 import figure5
figure5.figure5([("P1 scales", p1_tidy), ("P2 corruption", p2_tidy)], "outputs/p4/figure5.png")
```

It renders with whichever lanes are present, so it is usable before all four hand over.

## 9. What the bands mean, and who can change them

Half-open `[lo, hi)` on `p_place`, cut at exactly 1/3 and 2/3, final band closed at 1.0.

| Model | early | middle | late |
|---|---|---|---|
| VAR-d20 | `si` 0–2 | `si` 3–5 | `si` 6–9 |
| DiT-XL/2 250-step | 83 steps, `t` 999…670 | 83 steps, `t` 666…337 | 84 steps, `t` 333…0 |

`python -m lanes.p4.bands --full` prints every native stage. The reasoning, the rejected alternatives
and the numbers behind them are in `lanes/p4/comparison_logic.md`, which P3 signs off on.

**These lists are frozen.** Once your figure is drawn against them, changing them invalidates it. If you
think a band is wrong, raise it with P4 and P1 — it is a group-level decision, not a lane edit.

## 10. What this template does not claim

VAR scales and DiT timesteps are **not** mechanically equivalent, and putting them on one axis does not
say they are. The shared axis is for matched intervention *intent and budget*, and every row keeps its
native stage so you can always report in the model's own units. Say "the middle band of generation",
never "VAR scale 4 equals DiT timestep 500".
