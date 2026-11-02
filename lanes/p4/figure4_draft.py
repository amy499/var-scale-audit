"""Figure 4: the four arms against the reduction budget they share, one panel per model and band.

Run from the repo root. Reads only what the sweep wrote; the data layer needs no plotting library.

    python -m lanes.p4.figure4_draft outputs/p4/full/var/seed20261006/<JOBID> \
        outputs/p4/full/dit/seed20261006/<JOBID> --metric incep_cos --out outputs/p4/figure4_incep_cos.png

One panel per (model, candidate band): a full sweep root gives three, a Gate B pilot root gives one.
Each arm is a line of its per-image mean against the **intervened fraction `m / n`**, the only reduction
axis the two models share, with a shaded 95% bootstrap CI around it (lanes/p4/export.py, the same
numbers as the summary table). Under each panel title, the **paired** protect − control and
damage − control contrasts are marked wherever their 95% CI excludes 0: that, not the overlap of two
shaded bands, is the test of whether placement matters.

`--skip-timesteps` really removes DiT model evaluations, so DiT panels state the measured sampling-time
saving. The VAR intervention runs *after* the transformer pass and saves nothing, which the VAR panels
state (lanes/p4/comparison_logic.md section 3). P3's two sign-off conditions are printed on the panels
they govern: the pixel-metric floor on VAR's early band, and the mandatory control on VAR's middle band.

The curve is bounded: damage must fit inside the candidate band, so `m / n` cannot exceed `k / n`.
Draw `incep_cos` and `cls_prob` as two figures and read them together (RUNBOOK section B4).
"""

import argparse
import sys
from pathlib import Path

sys.dont_write_bytecode = True

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from lanes.p4 import arms, export, plotting  # noqa: E402

VAR_NO_SAVING = "VAR: no compute saving - the intervention runs after the transformer pass"
VAR_EARLY_FLOOR = ("P3 condition, VAR early band: pixel-space metrics floor here (hf_ratio = 0 on the "
                   "2x2-16x16 grids), so a flat pixel curve is not a finding; incep_cos and cls_prob are "
                   "not pixel-space")
VAR_MIDDLE_CONTROL = ("P3 condition, VAR middle band: at small m protect lands in late (86.6% of tokens), "
                      "so protect/damage are read only against the control drawn here")
P3_METRIC_SPECS = plotting.P3_METRIC_SPECS


class FigureError(SystemExit):
    """A sweep Figure 4 refuses to draw, because the picture would look finished and not be."""


def _check_complete(model: str, band: str, conds: dict, where) -> list[int]:
    """The band's budgets, after refusing a missing arm or an arm missing a budget."""
    present = {arm for arm, _m in conds}
    missing = [a for a in arms.INTERVENED_ARMS if a not in present]
    if missing:
        raise FigureError(f"{where}: {model} {band} band has no runs for arm(s) {missing}; Figure 4 "
                          "reports four arms, so it will not render a partial sweep")
    budgets = sorted({m for _a, m in conds})
    uneven = {a: [m for m in budgets if (a, m) not in conds] for a in arms.INTERVENED_ARMS}
    uneven = {a: ms for a, ms in uneven.items() if ms}
    if uneven:
        raise FigureError(f"{where}: {model} {band} band: these arms are missing budgets {uneven}; the "
                          "arms would not be budget-matched, which is what makes the contrast "
                          "attributable to placement")
    return budgets


def collect(root, metric: str) -> list[dict]:
    """One panel's data per candidate band under `root`: points and CIs per arm, contrasts, savings."""
    root = Path(root)
    try:
        data = export.load_root(root)
        model = data["model"]
        planned = {}
        for run in data["arms"]:
            planned.setdefault(run["band"], set()).add((run["arm"], run["m"]))
        if not planned:
            raise FigureError(f"{root}: no arm runs beside the baseline")
        budgets_of = {band: _check_complete(model, band, conds, root) for band, conds in planned.items()}
        rows = export.summary_rows(data, metrics=(metric,))
    except export.ExportError as e:
        raise FigureError(f"{root}: {e}") from e
    base = rows[0]
    by_band = {}
    for row in rows[1:]:
        by_band.setdefault(row["band"], {})[(row["arm"], row["m"])] = row

    out = []
    for band in sorted(by_band, key=export._band_order):
        conds = by_band[band]
        budgets = budgets_of[band]
        widest = max(r["fraction"] for r in conds.values())
        points, ci, contrasts = {}, {}, {}
        points[arms.BASELINE] = [(0.0, base[metric]), (widest, base[metric])]
        ci[arms.BASELINE] = [(0.0, base[f"{metric}_lo"], base[f"{metric}_hi"]),
                             (widest, base[f"{metric}_lo"], base[f"{metric}_hi"])]
        for arm in arms.INTERVENED_ARMS:
            series = [conds[(arm, m)] for m in budgets]
            points[arm] = [(r["fraction"], r[metric]) for r in series]
            ci[arm] = [(r["fraction"], r[f"{metric}_lo"], r[f"{metric}_hi"]) for r in series]
            if arm in export.CONTRASTS:
                contrasts[arm] = [(r["fraction"], r[f"{metric}_vs_control"], r[f"{metric}_vs_control_lo"],
                                   r[f"{metric}_vs_control_hi"]) for r in series]
        by_fraction = {}
        for r in conds.values():
            if model == "dit" and isinstance(r["saving"], float):
                by_fraction.setdefault(r["fraction"], []).append(r["saving"])
        stages = sorted({s for run in data["arms"] if run["band"] == band and run["arm"] == "damage"
                         for s in run["stages"]})
        out.append({"model": model, "band": band, "points": points, "ci": ci, "contrasts": contrasts,
                    "budgets": budgets, "widest": widest, "damage_stages": stages,
                    "savings": {f: plotting.mean(v) for f, v in by_fraction.items()}})
    return out


def contrast_note(c: dict, spec: "plotting.MetricSpec") -> str:
    """Where each paired contrast's CI excludes 0, and whether that is better or worse than control."""
    better_if_positive = spec.direction == "higher_is_better"
    parts = []
    for arm in export.CONTRASTS:
        marks = []
        for fraction, est, lo, hi in c["contrasts"].get(arm, []):
            if lo > 0 or hi < 0:
                good = (est > 0) == better_if_positive
                marks.append(f"{fraction:.2f} {'better' if good else 'worse'}")
        parts.append(f"{arm} vs control: " + (", ".join(marks) if marks else "CI includes 0 at every m/n"))
    return "\n".join(parts)


def _stage_span(c: dict) -> str:
    stages = c["damage_stages"]
    if not stages:
        return ""
    unit = "si" if c["model"] == "var" else "t"
    return f"{unit} {min(stages)}-{max(stages)}" if len(stages) > 1 else f"{unit} {stages[0]}"


def figure4_spec(collected: list[dict], spec: "plotting.MetricSpec",
                 title: str = "Figure 4: four arms against the shared reduction budget") -> dict:
    panels, notes = [], []
    widest = max(c["widest"] for c in collected) or 1.0
    for c in collected:
        series = [{"label": arm, "kind": "line", "points": c["points"][arm], "ci": c["ci"][arm]}
                  for arm in arms.ARMS if arm in c["points"]]
        if c["model"] == "dit" and c["savings"]:
            saving = "  ".join(f"m/n {f:.2f}: {s * 100:.0f}%" for f, s in sorted(c["savings"].items()))
            note = f"DiT measured sampling-time saving - {saving}"
        else:
            note = VAR_NO_SAVING
        notes.append(note)
        conditions = []
        if c["model"] == "var" and c["band"] == "early":
            conditions.append(VAR_EARLY_FLOOR)
        if c["model"] == "var" and c["band"] == "middle":
            conditions.append(VAR_MIDDLE_CONTROL)
        span = _stage_span(c)
        panels.append({"title": f"{c['model'].upper()} {c['band']} band" + (f" ({span})" if span else "")
                                + f"\n{contrast_note(c, spec)}",
                       "y_label": spec.axis_label, "series": series, "conditions": conditions,
                       "x_ticks": [(f, f"{f:.2f}") for f in
                                   sorted({p for s in series for p, _v in s["points"]})]})
    # A full sweep is one row per model with a column per band; one band per model fits in one row.
    ncols = max(sum(1 for c in collected if c["model"] == model) for model in {c["model"] for c in collected})
    if ncols == 1:
        ncols = len(collected)
    conditions = list(dict.fromkeys(cond for p in panels for cond in p["conditions"]))
    return {"title": title,
            "x_label": "intervened fraction m / n  (same budget in every arm; bounded by k / n)",
            "panels": panels, "ncols": ncols, "bands": {}, "band_notes": {}, "x_range": (0.0, widest * 1.04),
            "footnote": "protect, damage and control cut the same number of stages and differ only in "
                        "where; shading = 95% bootstrap CI of the mean over images; contrasts are paired "
                        "per image. " + "; ".join(dict.fromkeys(notes))
                        + ("\n" + "\n".join(conditions) if conditions else "")}


def figure4(roots, out_path, *, metric: str = "p4_placeholder_rel_l2", backend: str = "auto",
            metric_spec=None) -> Path:
    if metric_spec is None:
        from lanes.p4.metric_placeholder import metric_specs
        known = {s.name: s for s in metric_specs()} | P3_METRIC_SPECS
        metric_spec = known.get(metric, plotting.MetricSpec(metric))
    collected = [panel for root in roots for panel in collect(Path(root), metric)]
    spec = figure4_spec(collected, metric_spec)
    ncols = spec["ncols"]
    nrows = -(-len(collected) // ncols)
    size = (520 * ncols + 140, 400 * nrows + 120)
    return plotting.render(spec, out_path, backend=backend, size=size)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("roots", nargs="+", type=Path, help="one sweep (or Gate B pilot) root per model")
    ap.add_argument("--out", type=Path, default=Path("outputs/p4/figure4_draft.png"))
    ap.add_argument("--metric", default="p4_placeholder_rel_l2")
    ap.add_argument("--backend", default="auto", choices=("auto", *plotting.BACKENDS))
    args = ap.parse_args(argv)
    print(f"wrote {figure4(args.roots, args.out, metric=args.metric, backend=args.backend)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
