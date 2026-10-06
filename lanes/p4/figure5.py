"""Figure 5: all four lanes on one generation-progress axis, without forcing them to agree.

Run from the repo root. The data layer is standard library only; rendering follows
lanes/p4/plotting.py's lazy-import split (matplotlib if installed, else Pillow).

    from lanes.p4 import figure5
    figure5.figure5([("P1 scale importance", p1_tidy), ("P2 corruption", p2_tidy)],
                    "outputs/p4/figure5.png")

Each lane gets its own panel on the shared `p_place` axis, with its own metric name and its own value
range. Lanes are **never** averaged, ranked or reconciled into a single curve: the point of the figure
is to show where four different questions put their answer on one axis, including where they disagree.
Each panel is normalized within itself so that a lane measuring 0-1 and a lane measuring 0-300 are both
readable side by side; the panel label states the actual range so no number is lost.

It renders with whichever lanes are present, so it is usable before all four hand over.
"""

import argparse
import sys
from pathlib import Path

sys.dont_write_bytecode = True

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from lanes.p4 import bands, plotting  # noqa: E402


def _normalize(points, lo, hi):
    if hi == lo:
        return [(x, 0.5) for x, _v in points]     # a flat lane sits mid-panel rather than dividing by zero
    return [(x, (v - lo) / (hi - lo)) for x, v in points]


def figure5_spec(lane_tables, *, title: str = "Figure 5: stage importance across lanes",
                 kind: str = "line") -> dict:
    """One panel per lane, on the shared placement axis. lane_tables: [(lane label, tidy rows), ...]."""
    lane_tables = list(lane_tables)
    if not lane_tables:
        raise plotting.PlotError("no lanes given; Figure 5 needs at least one tidy table")

    panels, models = [], set()
    for label, rows in lane_tables:
        if not rows:
            raise plotting.PlotError(f"lane {label!r} has an empty tidy table")
        models.update(r["model"] for r in rows)
        series = plotting.aggregate(rows)
        metrics = list(dict.fromkeys(r["metric"] for r in rows))
        values = [v for points in series.values() for _p, _s, v, _f in points]
        lo, hi = min(values), max(values)

        panel_series, ticks = [], set()
        for (metric, arm), points in sorted(series.items()):
            xs = [(p, v) for p, _s, v, _f in points]
            panel_series.append({"label": arm if len(metrics) == 1 else f"{metric} / {arm}",
                                 "kind": kind, "points": _normalize(xs, lo, hi)})
            ticks.update((p, str(s)) for p, s, _v, _f in points)
        arrow = plotting.direction_phrase(rows[0]["direction"])
        panels.append({"title": label,
                       "y_label": f"{', '.join(metrics)} normalized in-panel "
                                  f"(actual {lo:.3g}-{hi:.3g}, {arrow})",
                       "series": panel_series,
                       "x_ticks": plotting.thin_ticks(sorted(ticks), limit=8),
                       "value_range": (lo, hi), "metrics": metrics})

    notes = plotting.band_annotation([r for _l, rows in lane_tables for r in rows]) if len(models) == 1 \
        else {band: "" for band in bands.BANDS}
    return {"title": title,
            "x_label": f"normalized progress p_place  (ticks: {plotting.native_label(models)})",
            "panels": panels, "bands": dict(plotting.BAND_EXTENTS), "band_notes": notes,
            "x_range": (0.0, 1.0),
            "footnote": "one panel per lane, each normalized within itself and labelled with its own "
                        "actual range; lanes are not averaged, ranked or reconciled"}


def figure5(lane_tables, out_path, *, title: str = "Figure 5: stage importance across lanes",
            kind: str = "line", backend: str = "auto", size=None) -> Path:
    """Build and render the frame. Panel width is fixed, so two lanes are not stretched to four."""
    spec = figure5_spec(lane_tables, title=title, kind=kind)
    size = size or (330 * len(spec["panels"]) + 120, 460)
    return plotting.render(spec, out_path, backend=backend, size=size)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", type=Path, help="output image path")
    ap.add_argument("--lane", action="append", default=[], metavar="LABEL=TIDY.csv",
                    help="a lane's label and its tidy table (repeatable; lanes/p4/PLOTTING.md)")
    ap.add_argument("--backend", default="auto", choices=("auto", *plotting.BACKENDS))
    args = ap.parse_args(argv)
    if not args.lane:
        ap.error("give at least one --lane LABEL=TIDY.csv")
    tables = []
    for item in args.lane:
        label, sep, path = item.partition("=")
        if not sep:
            ap.error(f"--lane {item!r}: expected LABEL=TIDY.csv")
        tables.append((label, plotting.read_tidy(path)))
    print(f"wrote {figure5(tables, args.out, backend=args.backend)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
