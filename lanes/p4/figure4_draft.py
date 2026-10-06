"""Figure 4 draft: the four arms against the reduction budget they share.

Run from the repo root. Reads only what the pilot wrote; the data layer needs no plotting library.

    python -m lanes.p4.figure4_draft outputs/p4/pilot_var/<JOBID> \
        outputs/p4/pilot_dit/<JOBID> --out outputs/p4/figure4_draft.png

One panel per model. The x axis is the **intervened fraction `m / n`**, which is the only reduction
axis the two models share: `--skip-timesteps` really removes DiT model evaluations, while the VAR
intervention runs *after* the transformer pass and therefore saves nothing. Measured wall-clock savings
are reported for DiT and their absence is stated on the VAR panel rather than left for a reader to
assume (lanes/p4/comparison_logic.md section 5).

The curve is bounded: damage must fit inside the candidate band, so `m / n` cannot exceed `k / n`
(0.3 for VAR-d20, 0.332 for the 250-step DiT schedule). This is a bounded contrast, not a general
reduction curve.

The metric is whatever the runs' `metrics.csv` holds, described by a MetricSpec, so P3's calibrated
metrics replace the placeholder without a change here.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from lanes.p4 import arms, plotting, progress  # noqa: E402
from lanes.p4.verify_pairing import find_runs  # noqa: E402

VAR_NO_SAVING = "VAR: no compute saving - the intervention runs after the transformer pass"


def _metric_values(run_dir: Path, metric: str) -> list[float]:
    path = run_dir / "metrics.csv"
    if not path.is_file():
        raise SystemExit(f"{path} does not exist; run lanes/p4/metric_placeholder.py first "
                         "(or P3's metric code)")
    with path.open(newline="", encoding="utf-8") as f:
        return [float(r["value"]) for r in csv.DictReader(f) if r["metric"] == metric]


def collect(pilot_root, metric: str) -> dict:
    """{model, points per arm as (fraction, mean value), measured savings, budgets} for one model."""
    pilot_root = Path(pilot_root)
    baseline, arm_dirs = find_runs(pilot_root)
    base_run = progress.load_run(baseline)
    model = base_run["model"]
    base_seconds = base_run["seconds"]["sample"]

    points, savings, seen_m = {}, {}, set()
    for d in [baseline, *arm_dirs]:
        info_path = d / "pilot.json"
        if not info_path.is_file():
            raise SystemExit(f"{info_path} does not exist; this run was not produced by lanes/p4/pilot.py")
        info = json.loads(info_path.read_text())
        values = _metric_values(d, metric)
        if not values:
            raise SystemExit(f"{d / 'metrics.csv'} has no rows for metric {metric!r}")
        fraction = info["fraction"]
        points.setdefault(info["arm"], []).append((fraction, plotting.mean(values)))
        seen_m.add(info["m"])
        # Only DiT actually removes model evaluations; a VAR "saving" would be timing noise, so it is
        # not computed at all rather than computed and quietly not shown.
        if info["arm"] != arms.BASELINE and model == "dit":
            seconds = progress.load_run(d)["seconds"]["sample"]
            savings.setdefault(fraction, []).append(1.0 - seconds / base_seconds if base_seconds else 0.0)

    # The baseline is the reference at every budget, so it is drawn flat across the axis.
    base_value = points[arms.BASELINE][0][1]
    widest = max(f for series in points.values() for f, _v in series)
    points[arms.BASELINE] = [(0.0, base_value), (widest, base_value)]
    for series in points.values():
        series.sort()
    return {"model": model, "points": points, "budgets": sorted(seen_m - {0}),
            "savings": {f: sum(v) / len(v) for f, v in savings.items()}, "widest": widest}


def figure4_spec(collected: list[dict], spec: "plotting.MetricSpec",
                 title: str = "Figure 4 (draft): four arms against the shared reduction budget") -> dict:
    panels, notes = [], []
    widest = max(c["widest"] for c in collected) or 1.0
    for c in collected:
        series = []
        for arm in arms.ARMS:
            if arm in c["points"]:
                series.append({"label": arm, "kind": "line", "points": c["points"][arm]})
        if c["model"] == "dit" and c["savings"]:
            saving = "  ".join(f"m/n {f:.2f}: {s * 100:.0f}%" for f, s in sorted(c["savings"].items()))
            note = f"DiT measured sampling-time saving - {saving}"
        else:
            note = VAR_NO_SAVING
        notes.append(note)
        panels.append({"title": f"{c['model'].upper()}   ({note})", "y_label": spec.axis_label,
                       "series": series,
                       "x_ticks": [(f, f"{f:.2f}") for f in
                                   sorted({p for s in series for p, _v in s["points"]})]})
    return {"title": title,
            "x_label": "intervened fraction m / n  (same budget in every arm; bounded by k / n)",
            "panels": panels, "bands": {}, "band_notes": {}, "x_range": (0.0, widest * 1.04),
            "footnote": "protect, damage and control cut the same number of stages and differ only in "
                        "where; " + "; ".join(dict.fromkeys(notes))}


def figure4(pilot_roots, out_path, *, metric: str = "p4_placeholder_rel_l2", backend: str = "auto",
            metric_spec=None) -> Path:
    if metric_spec is None:
        from lanes.p4.metric_placeholder import metric_specs
        metric_spec = next((s for s in metric_specs() if s.name == metric),
                           plotting.MetricSpec(metric))
    collected = [collect(Path(root), metric) for root in pilot_roots]
    spec = figure4_spec(collected, metric_spec)
    size = (520 * len(collected) + 140, 460)
    return plotting.render(spec, out_path, backend=backend, size=size)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pilot_roots", nargs="+", type=Path, help="one pilot root per model")
    ap.add_argument("--out", type=Path, default=Path("outputs/p4/figure4_draft.png"))
    ap.add_argument("--metric", default="p4_placeholder_rel_l2")
    ap.add_argument("--backend", default="auto", choices=("auto", *plotting.BACKENDS))
    args = ap.parse_args(argv)
    print(f"wrote {figure4(args.pilot_roots, args.out, metric=args.metric, backend=args.backend)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
