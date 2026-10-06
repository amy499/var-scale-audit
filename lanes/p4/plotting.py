"""The shared early / middle / late plot template for lanes P1-P4 (lanes/p4/PLOTTING.md).

Run from the repo root.

    from lanes.p4 import plotting
    rows = plotting.tidy_from_run("outputs/p1/.../run.json", metric_rows, [plotting.MetricSpec("lpips")],
                                  arm="baseline", lane="p1")
    plotting.lane_plot(rows, "outputs/p1/fig.png", title="P1: scale importance")

Two layers, so a lane that has no plotting library can still do the analysis:

  **data layer**  standard library only. Maps a run's stages to both progress axes, assigns bands,
                  joins the lane's metric values, and emits one tidy table. Importing this module and
                  building a table never imports a plotting library.
  **render layer** imports a plotting library at call time. matplotlib when it is installed, otherwise
                  Pillow, which both env files already carry. `backend="matplotlib"` asks for it by
                  name and raises one actionable message if it is absent.

Metric names, directions and aggregation levels are **data** (MetricSpec), never a hardcoded list, so
P3's final metric definitions drop in without a change here (lanes/p4/PLOTTING.md section 3).

The x axis is normalized progress with the native stage on the ticks, and the three bands are drawn at
their frozen extents, so a band with no rows renders empty instead of shifting the other two.
"""

import csv
import sys
from dataclasses import dataclass
from pathlib import Path

sys.dont_write_bytecode = True

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from lanes.p4 import bands, progress  # noqa: E402

BACKENDS = ("matplotlib", "pillow")      # render layer, in preference order
DIRECTIONS = ("higher_is_better", "lower_is_better")
LEVELS = ("per_image", "per_set")

# P2's merged columns (lanes/p2/SCHEMA.md) plus the four P4 adds: band, p_place, p_func, and the
# metric's direction/level. `arm` and `lane` already come from P2's runs.csv.
TIDY_COLUMNS = ("run_id", "lane", "model", "arm", "class_id", "seed", "stage",
                "p_place", "p_func", "band", "metric", "label", "value", "direction", "level")

BAND_EXTENTS = {"early": (0.0, bands.CUTS[0]), "middle": bands.CUTS, "late": (bands.CUTS[1], 1.0)}


class PlotError(ValueError):
    """A tidy table or metric spec this template cannot use, with the reason named."""


class PlotBackendMissing(RuntimeError):
    """No plotting library for the requested backend; the message names the package to install."""


@dataclass(frozen=True)
class MetricSpec:
    """One metric, described as data so the template needs no knowledge of it.

    name      the value of the `metric` column in the lane's metrics rows
    direction "higher_is_better" or "lower_is_better"; shown on the axis so a reader knows which way is good
    level     "per_image" (averaged over the images at a stage) or "per_set" (already one value per stage)
    label     axis label; defaults to the name
    """
    name: str
    direction: str = "higher_is_better"
    level: str = "per_image"
    label: str | None = None

    def __post_init__(self):
        if self.direction not in DIRECTIONS:
            raise PlotError(f"metric {self.name!r}: direction must be one of {DIRECTIONS}, got {self.direction!r}")
        if self.level not in LEVELS:
            raise PlotError(f"metric {self.name!r}: level must be one of {LEVELS}, got {self.level!r}")

    @property
    def axis_label(self) -> str:
        return f"{self.label or self.name}  ({direction_phrase(self.direction)})"


# ---------------------------------------------------------------- data layer (standard library only)

def direction_phrase(direction: str) -> str:
    """The words a reader sees on an axis, so a figure never has to spell this out itself."""
    return "higher is better" if direction == "higher_is_better" else "lower is better"


def native_label(models) -> str:
    """What the x ticks are, named for the model when a figure holds only one."""
    models = sorted(set(models))
    if models == ["var"]:
        return "VAR scale si"
    return "DiT timestep" if models == ["dit"] else "native stage"


def _specs(metrics) -> dict:
    out = {}
    for m in metrics:
        spec = m if isinstance(m, MetricSpec) else MetricSpec(**m) if isinstance(m, dict) else MetricSpec(m)
        if spec.name in out:
            raise PlotError(f"metric {spec.name!r} is given twice")
        out[spec.name] = spec
    if not out:
        raise PlotError("no metrics given; the template needs at least one MetricSpec")
    return out


def _int_or_none(value):
    if value is None or value == "":
        return None
    return int(value)


def tidy_from_run(run, metric_rows, metrics, *, arm: str, lane: str | None = None,
                  run_id: str | None = None) -> list[dict]:
    """One tidy table from a single `run.json` plus the lane's own metric rows.

    run          a run directory, a run.json path, or an already-loaded run record
    metric_rows  dicts with the columns of lanes/p2/SCHEMA.md section 4: class_id, seed, metric,
                 value, stage (stage empty/None = a number about the final image, which is placed at
                 the last stage so it still lands in a band)
    metrics      MetricSpecs (or names/dicts); rows whose metric is not listed are ignored
    """
    record = run if isinstance(run, dict) else progress.load_run(run)
    model = progress.model_of(record)
    mapped = bands.assign_bands(progress.map_run(record))
    by_stage = {r["stage"]: r for r in mapped}
    last_stage = mapped[-1]["stage"]
    specs = _specs(metrics)
    rid = run_id or (f"{lane}/{arm}" if lane else arm)

    unknown_stages, out = set(), []
    for i, row in enumerate(metric_rows):
        name = row.get("metric")
        if name not in specs:
            continue
        spec = specs[name]
        stage = _int_or_none(row.get("stage"))
        if stage is None:
            stage = last_stage          # a final-image number belongs to the end of generation
        if stage not in by_stage:
            unknown_stages.add(stage)
            continue
        if "value" not in row:
            raise PlotError(f"metric row {i} ({name!r}) has no 'value'")
        try:
            value = float(row["value"])
        except (TypeError, ValueError) as e:
            raise PlotError(f"metric row {i} ({name!r}) has value {row['value']!r}, which is not a number") from e
        stage_row = by_stage[stage]
        out.append({"run_id": rid, "lane": lane, "model": model, "arm": arm,
                    "class_id": _int_or_none(row.get("class_id")), "seed": _int_or_none(row.get("seed")),
                    "stage": stage, "p_place": stage_row["p_place"], "p_func": stage_row["p_func"],
                    "band": stage_row["band"], "metric": name, "label": spec.label or name,
                    "value": value, "direction": spec.direction, "level": spec.level})
    if unknown_stages:
        raise PlotError(
            f"{rid}: metric rows name stages {sorted(unknown_stages)} that this run's stages table does "
            f"not have (it has {len(by_stage)} stages, {mapped[0]['stage']}..{last_stage}). "
            "A mismatch is reported rather than dropped: join the metrics to the run they came from.")
    if not out:
        raise PlotError(f"{rid}: no metric row matched {sorted(specs)}")
    return out


def _read_csv(path: Path) -> list[dict]:
    try:
        with path.open(newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    except OSError as e:
        raise PlotError(f"cannot read {path}: {e}") from e


def tidy_from_tables(tables_dir, metrics, *, run_ids=None) -> list[dict]:
    """One tidy table from the merged CSVs of `python -m lanes.p2.schema collect`.

    Reads stages.csv, metrics.csv and runs.csv as lanes/p2/SCHEMA.md section 5 defines them, so a lane
    hands over its tables with no conversion. `run_ids` limits the result to those runs.
    """
    tables_dir = Path(tables_dir)
    stages = _read_csv(tables_dir / "stages.csv")
    metric_rows = _read_csv(tables_dir / "metrics.csv")
    runs = {r["run_id"]: r for r in _read_csv(tables_dir / "runs.csv")}
    if not stages:
        raise PlotError(f"{tables_dir / 'stages.csv'} has no rows")

    wanted = set(run_ids) if run_ids else set(runs)
    missing = sorted(wanted - set(runs))
    if missing:
        raise PlotError(f"runs.csv has no row for {missing}")

    by_run = {}
    for row in stages:
        by_run.setdefault(row["run_id"], []).append(row)
    out = []
    for run_id in sorted(wanted):
        meta = runs[run_id]
        if run_id not in by_run:
            raise PlotError(f"{run_id}: runs.csv lists it but stages.csv has no stage rows for it")
        record = {"model": meta["model"], "stages": _stage_rows(meta["model"], by_run[run_id], run_id)}
        out.extend(tidy_from_run(record, [r for r in metric_rows if r["run_id"] == run_id], metrics,
                                 arm=meta.get("arm") or run_id, lane=meta.get("lane") or None, run_id=run_id))
    return out


_NUMERIC_STAGE_FIELDS = ("stage", "p", "cum_tokens", "total_tokens", "alpha_bar_in", "alpha_bar_out",
                         "sigma_in", "sigma_out", "n_tokens", "pn", "step")


def _stage_rows(model: str, rows: list[dict], run_id: str) -> list[dict]:
    """stages.csv rows back into the `stages` table shape, with numbers as numbers."""
    out = []
    for row in rows:
        clean = {}
        for key, value in row.items():
            if key == "run_id" or value in (None, ""):
                continue
            clean[key] = float(value) if key in _NUMERIC_STAGE_FIELDS else value
        if "stage" not in clean:
            raise PlotError(f"{run_id}: a stages.csv row has no 'stage'")
        clean["stage"] = int(clean["stage"])
        out.append(clean)
    out.sort(key=lambda r: r.get("step", r["stage"]) if model == "dit" else r["stage"])
    return out


def write_tidy(rows: list[dict], path) -> Path:
    """The tidy table as CSV, with TIDY_COLUMNS in order."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=TIDY_COLUMNS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    return path


def read_tidy(path) -> list[dict]:
    out = []
    for row in _read_csv(Path(path)):
        out.append({**row, "stage": int(row["stage"]), "value": float(row["value"]),
                    "p_place": float(row["p_place"]), "p_func": float(row["p_func"]),
                    "class_id": _int_or_none(row.get("class_id")), "seed": _int_or_none(row.get("seed")),
                    "lane": row.get("lane") or None})
    return out


def mean(values):
    """The arithmetic mean, spelled one way across the lane (sum/len, not statistics.fmean)."""
    values = list(values)
    return sum(values) / len(values)


def aggregate(rows: list[dict]) -> dict:
    """{(metric, arm): [(p_place, stage, value, p_func), ...]} with per_image metrics averaged per stage."""
    buckets = {}
    for row in rows:
        key = (row["metric"], row["arm"], row["stage"])
        buckets.setdefault(key, []).append(row)
    series = {}
    for (metric, arm, stage), group in buckets.items():
        level = group[0]["level"]
        if level == "per_set" and len(group) > 1:
            raise PlotError(f"metric {metric!r} is per_set but has {len(group)} rows at stage {stage} "
                            f"of arm {arm!r}; a per_set metric is one value per stage")
        first = group[0]
        series.setdefault((metric, arm), []).append(
            (first["p_place"], stage, mean(r["value"] for r in group), first["p_func"]))
    for points in series.values():
        points.sort()
    return series


def band_annotation(rows: list[dict]) -> dict:
    """{band: "p_func a-b"} over the stages present, so the functional value travels with the figure."""
    out = {}
    for band in bands.BANDS:
        values = sorted({r["p_func"] for r in rows if r["band"] == band})
        out[band] = f"committed {values[0]:.3f}-{values[-1]:.3f}" if values else "no stages"
    return out


# ---------------------------------------------------------------- figure spec (backend neutral)

def lane_plot_spec(rows: list[dict], *, title: str | None = None, kind: str = "line") -> dict:
    """One panel per metric: normalized progress across, the metric up, one series per arm.

    The three bands are drawn at their frozen extents (BAND_EXTENTS), never at the extent of the data,
    so a band with no rows renders empty and the other two keep their place.
    """
    if not rows:
        raise PlotError("tidy table is empty")
    if kind not in ("line", "bar"):
        raise PlotError(f"kind must be 'line' or 'bar', got {kind!r}")
    models = sorted({r["model"] for r in rows})
    metrics = list(dict.fromkeys(r["metric"] for r in rows))      # input order, never a hardcoded list
    arms = list(dict.fromkeys(r["arm"] for r in rows))
    series = aggregate(rows)
    native = native_label(models)

    panels = []
    for metric in metrics:
        metric_rows = [r for r in rows if r["metric"] == metric]
        spec = MetricSpec(metric, metric_rows[0]["direction"], metric_rows[0]["level"],
                          label=metric_rows[0].get("label") or metric)
        panel_series = []
        for arm in arms:
            points = series.get((metric, arm))
            if not points:
                continue
            panel_series.append({"label": arm, "kind": kind,
                                 "points": [(p, v) for p, _s, v, _f in points]})
        ticks = sorted({(p, str(stage)) for arm in arms
                        for p, stage, _v, _f in series.get((metric, arm), [])})
        panels.append({"title": metric, "y_label": spec.axis_label, "series": panel_series,
                       "x_ticks": thin_ticks(ticks)})
    return {"title": title or f"early / middle / late: {', '.join(metrics)}",
            "x_label": f"normalized progress p_place  (ticks: {native})",
            "panels": panels, "bands": dict(BAND_EXTENTS), "band_notes": band_annotation(rows),
            "x_range": (0.0, 1.0),
            "footnote": "bands are cut on the placement axis at 1/3 and 2/3, half-open; "
                        "'committed' is the functional axis (VAR tokens, DiT sqrt(alpha_bar))"}


def thin_ticks(ticks: list, limit: int = 12) -> list:
    """Keep at most `limit` ticks, evenly spaced, so a 250-step DiT axis stays readable."""
    if len(ticks) <= limit:
        return list(ticks)
    step = (len(ticks) - 1) / (limit - 1)
    return [ticks[round(i * step)] for i in range(limit)]


# ---------------------------------------------------------------- render layer (lazy import)

def available_backends() -> list[str]:
    import importlib.util
    modules = {"matplotlib": "matplotlib", "pillow": "PIL"}
    return [name for name in BACKENDS if importlib.util.find_spec(modules[name]) is not None]


def render(spec: dict, out_path, *, backend: str = "auto", size=(1100, 420)) -> Path:
    """Render a figure spec. backend: "auto" (matplotlib, else Pillow), "matplotlib", or "pillow"."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    have = available_backends()
    if backend == "auto":
        if not have:
            raise PlotBackendMissing(
                "no plotting library is installed. Install matplotlib in the var-dit env (ask P1 first: "
                "docs/HANDOVER.md forbids an unannounced pip install), or install Pillow, which both "
                "env files already list. The data layer (tidy_from_run / write_tidy) needs neither.")
        backend = have[0]
    if backend not in BACKENDS:
        raise PlotError(f"backend must be 'auto' or one of {BACKENDS}, got {backend!r}")
    if backend not in have:
        package = "matplotlib" if backend == "matplotlib" else "Pillow"
        raise PlotBackendMissing(
            f"{package} is not installed in this environment, so backend={backend!r} cannot render. "
            f"Available: {have or 'none'}. Pass backend='auto' to use whatever is present, or add "
            f"{package} to the env (ask P1 first: docs/HANDOVER.md forbids an unannounced pip install).")
    return _render_matplotlib(spec, out_path, size) if backend == "matplotlib" \
        else _render_pillow(spec, out_path, size)


def lane_plot(rows: list[dict], out_path, *, title: str | None = None, kind: str = "line",
              backend: str = "auto") -> Path:
    """tidy table -> figure, in one call."""
    return render(lane_plot_spec(rows, title=title, kind=kind), out_path, backend=backend)


PALETTE = ("#1f4e79", "#c0504d", "#4f8a3b", "#8064a2", "#e08a1e", "#3f8fa8")


def band_label(band: str, note: str) -> str:
    """"middle (committed 0.044-0.134)", or just "middle" when there is no single range to quote."""
    return f"{band} ({note})" if note else band


def _series_range(panel: dict) -> tuple[float, float]:
    values = [v for s in panel["series"] for _x, v in s["points"]]
    if not values:
        return 0.0, 1.0
    lo, hi = min(values), max(values)
    if hi == lo:
        pad = abs(hi) * 0.1 or 0.5
        return lo - pad, hi + pad
    span = hi - lo
    return lo - span * 0.08, hi + span * 0.22   # extra headroom so the legend does not sit on the data


def _render_matplotlib(spec: dict, out_path: Path, size) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    panels = spec["panels"]
    fig, axes = plt.subplots(1, len(panels), figsize=(size[0] / 100, size[1] / 100), squeeze=False)
    for ax, panel in zip(axes[0], panels):
        for band, (lo, hi) in spec["bands"].items():
            ax.axvspan(lo, hi, color="#000000", alpha=0.04 if band == "middle" else 0.0)
            ax.axvline(lo, color="#999999", linewidth=0.6, linestyle=":")
            note = spec["band_notes"].get(band, "")
            ax.text((lo + hi) / 2, 1.01, f"{band}\n{note}" if note else band,
                    transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                    fontsize=6.5, color="#444444")
        for i, s in enumerate(panel["series"]):
            xs = [x for x, _v in s["points"]]
            ys = [v for _x, v in s["points"]]
            colour = PALETTE[i % len(PALETTE)]
            if s["kind"] == "bar":
                width = 0.8 / max(len(panel["series"]), 1) * (1.0 / max(len(xs), 1))
                ax.bar([x + (i - (len(panel["series"]) - 1) / 2) * width for x in xs], ys,
                       width=width, label=s["label"], color=colour)
            else:
                ax.plot(xs, ys, marker="o", markersize=3, linewidth=1.4, label=s["label"], color=colour)
        ax.set_xlim(*spec["x_range"])
        ax.set_ylim(*_series_range(panel))
        ax.set_xticks([p for p, _l in panel["x_ticks"]])
        ax.set_xticklabels([l for _p, l in panel["x_ticks"]], fontsize=6.5, rotation=45)
        ax.set_xlabel(spec["x_label"], fontsize=7.5)
        ax.set_ylabel(panel["y_label"], fontsize=7.5)
        ax.set_title(panel["title"], fontsize=9, pad=26)   # clears the band labels at y = 1.01
        ax.tick_params(labelsize=7)
        if panel["series"]:
            ax.legend(fontsize=7, loc="best", framealpha=0.85)
    fig.suptitle(spec["title"], fontsize=11)
    fig.text(0.5, 0.005, spec["footnote"], ha="center", fontsize=6.5, color="#555555")
    fig.tight_layout(rect=(0, 0.03, 1, 0.9))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def _font(size: int):
    from PIL import ImageFont
    try:
        return ImageFont.load_default(size=size)
    except TypeError:          # Pillow < 9.2: load_default takes no size
        return ImageFont.load_default()


def _render_pillow(spec: dict, out_path: Path, size) -> Path:
    """The fallback renderer: Pillow is in both env files, matplotlib is in neither."""
    from PIL import Image, ImageDraw
    W, H = size
    panels = spec["panels"]
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    f_small, f_mid, f_title = _font(10), _font(11), _font(14)
    d.text((10, 6), spec["title"], fill="black", font=f_title)
    d.text((10, H - 14), spec["footnote"], fill="#555555", font=f_small)

    pad_l, pad_r, pad_t, pad_b = 62, 14, 70, 64
    panel_w = (W - (len(panels) - 1) * 26) / len(panels)
    for pi, panel in enumerate(panels):
        x0 = pi * (panel_w + 26) + pad_l
        x1 = pi * (panel_w + 26) + panel_w - pad_r
        y0, y1 = pad_t, H - pad_b
        lo, hi = _series_range(panel)

        xlo, xhi = spec["x_range"]
        xspan = (xhi - xlo) or 1.0

        def px(p):
            return x0 + (p - xlo) / xspan * (x1 - x0)

        def py(v):
            return y1 - (v - lo) / (hi - lo) * (y1 - y0)

        for band, (blo, bhi) in spec["bands"].items():
            if band == "middle":
                d.rectangle([px(blo), y0, px(bhi), y1], fill="#f4f4f4")
            d.line([px(blo), y0, px(blo), y1], fill="#cccccc")
            label = band_label(band, spec["band_notes"].get(band, ""))
            d.text(((px(blo) + px(bhi)) / 2 - 3.0 * len(label) / 2, y0 - 30), label,
                   fill="#444444", font=f_small)
        d.rectangle([x0, y0, x1, y1], outline="#333333")
        d.text((x0, y0 - 50), panel["title"], fill="black", font=f_mid)
        d.text((x0, y1 + 32), spec["x_label"], fill="#333333", font=f_small)
        d.text((max(2, x0 - 56), y0 - 16), panel["y_label"], fill="#333333", font=f_small)
        for value in (lo, (lo + hi) / 2, hi):
            d.text((x0 - 58, py(value) - 5), f"{value:.3g}", fill="#333333", font=f_small)
            d.line([x0 - 3, py(value), x0, py(value)], fill="#333333")
        for p, label in panel["x_ticks"]:
            d.line([px(p), y1, px(p), y1 + 4], fill="#333333")
            d.text((px(p) - 3.0 * len(label) / 2, y1 + 8), label, fill="#333333", font=f_small)
        legend = []
        for i, s in enumerate(panel["series"]):
            colour = PALETTE[i % len(PALETTE)]
            pts = [(px(x), py(v)) for x, v in s["points"]]
            if s["kind"] == "bar":
                half = max(1.5, (x1 - x0) / max(len(pts), 1) / (2 * len(panel["series"])))
                off = (i - (len(panel["series"]) - 1) / 2) * 2 * half
                for cx, cy in pts:
                    d.rectangle([cx + off - half, cy, cx + off + half, y1], fill=colour)
            else:
                if len(pts) > 1:
                    d.line(pts, fill=colour, width=2)
                for cx, cy in pts:
                    d.ellipse([cx - 2, cy - 2, cx + 2, cy + 2], fill=colour)
            legend.append((colour, s["label"]))
        if legend:
            width = 22 + 6 * max(len(label) for _c, label in legend)
            height = 10 + 12 * len(legend)
            # put it in whichever top corner holds less data, so it never hides a series
            upper = y0 + 0.5 * (y1 - y0)
            mid_x = (x0 + x1) / 2
            crowd_left = sum(1 for s in panel["series"] for x, v in s["points"]
                             if py(v) < upper and px(x) < mid_x)
            crowd_right = sum(1 for s in panel["series"] for x, v in s["points"]
                              if py(v) < upper and px(x) >= mid_x)
            left = crowd_left < crowd_right
            box = ([x0 + 4, y0 + 4, x0 + width + 4, y0 + 4 + height] if left else
                   [x1 - width - 4, y0 + 4, x1 - 4, y0 + 4 + height])
            d.rectangle(box, fill="white", outline="#cccccc")
            for i, (colour, label) in enumerate(legend):
                d.rectangle([box[0] + 5, box[1] + 6 + i * 12, box[0] + 15, box[1] + 14 + i * 12],
                            fill=colour)
                d.text((box[0] + 19, box[1] + 5 + i * 12), label, fill="#333333", font=f_small)
    img.save(out_path)
    return out_path
