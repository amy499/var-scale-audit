"""Figure 1c: per-class drop in top-1 after stage removal, VAR and DiT. numpy + csv + matplotlib; no GPU.

    python -m lanes.p1.plot_heatmap [--out lanes/p1/results/fig1c_heatmap.png]
                                    [--csv lanes/p1/results/heatmap_by_class.csv]

Rows: the 7 classes of manifest v1.1. Panels:
    A  VAR single scales 0-9 (s0, s9 from 25893211; s1-s8 from 25916920)
    B  VAR 2-scale windows {0,1} ... {8,9} (25916921)
    C  DiT early 1-83 / mid 84-166 / late 167-249 (25893213)
Cell = drop in top-1 for that class: mean over its images of baseline - arm (positive = worse), against the
same job's baseline, paired by (class_id, seed), n = 32 per class. One sequential scale, 0 to the largest
drop in any panel, shared by all three panels; a negative drop is drawn as 0 and annotated with its value.
CSV: one row per (model, arm, class) with n, baseline and arm top-1, the drop and its 95% paired bootstrap
CI (lanes.p1.summarize.paired_diff: 10000 resamples, seed 0). Stops if any class does not have exactly
N_PER_CLASS paired images in an arm. Removed scales / steps are read from runs.csv params.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from lanes.p1.summarize import job_dir, load_job, paired_diff  # noqa: E402

CLASSES = {151: "Chihuahua", 153: "Maltese", 162: "beagle", 207: "golden retriever",
           235: "German shepherd", 254: "pug", 258: "Samoyed"}
N_PER_CLASS = 32
PANELS = (   # (panel, model, title, [(job, ...)], columns from that job's removal arms)
    ("A", "var", "A  VAR: one scale removed", ("25893211", "25916920")),
    ("B", "var", "B  VAR: 2-scale window removed", ("25916921",)),
    ("C", "dit", "C  DiT: 83 steps skipped", ("25893213",)),
)
COLUMNS = ("model", "panel", "job", "arm", "removed", "class_id", "class_name", "n", "base_top1", "arm_top1",
           "drop_top1", "drop_top1_lo", "drop_top1_hi")
# Reference palette (dataviz skill): sequential blue ramp, steps 100..700, light -> dark.
RAMP = ("#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf",
        "#1c5cab", "#184f95", "#104281", "#0d366b")
SURFACE, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"


def _removed(params: dict) -> tuple:
    """(sort key, label) for an arm's params: VAR scales, or DiT step range."""
    if "scales" in params:
        s = params["scales"]
        return (s[0], len(s)), (str(s[0]) if len(s) == 1 else "{" + ",".join(map(str, s)) + "}")
    if "step_first" in params:
        return (params["step_first"], 0), f"{params['step_first']}-{params['step_last']}"
    raise SystemExit(f"ERROR: params do not say what was removed: {params}")


def panel_cells(jobs) -> list[dict]:
    """One dict per (arm, class) over the removal arms of `jobs`, in removed-stage order."""
    arms = []
    for j in jobs:
        job = job_dir(j)
        runs, metrics, _ = load_job(job)
        for r in runs:
            if not r["baseline_run_id"]:
                continue
            key, label = _removed(json.loads(r["params"] or "{}"))
            arms.append((key, label, job.name, r["arm"], r["model"],
                         metrics[r["baseline_run_id"]]["cls_top1"], metrics[r["run_id"]]["cls_top1"]))
    out = []
    for _, label, job, arm, model, base, run in sorted(arms):
        if set(base) != set(run):
            raise SystemExit(f"ERROR: {job} {arm}: images differ from its baseline; not a paired comparison")
        for c, name in CLASSES.items():
            keys = sorted(k for k in run if k[0] == c)
            if len(keys) != N_PER_CLASS:
                raise SystemExit(f"ERROR: {job} {arm}: class {c} ({name}) has {len(keys)} paired images, "
                                 f"expected {N_PER_CLASS}")
            d, lo, hi = paired_diff(base, run, keys)
            out.append({"model": model, "job": job, "arm": arm, "removed": label, "class_id": c, "class_name": name,
                        "n": len(keys), "base_top1": round(float(np.mean([base[k] for k in keys])), 4),
                        "arm_top1": round(float(np.mean([run[k] for k in keys])), 4),
                        "drop_top1": round(d, 4), "drop_top1_lo": round(lo, 4), "drop_top1_hi": round(hi, 4)})
        extra = sorted({k[0] for k in run} - set(CLASSES))
        if extra:
            raise SystemExit(f"ERROR: {job} {arm}: classes {extra} are not in CLASSES")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("lanes/p1/results/fig1c_heatmap.png"))
    ap.add_argument("--csv", type=Path, default=Path("lanes/p1/results/heatmap_by_class.csv"))
    args = ap.parse_args(argv)

    panels = []
    for panel, model, title, jobs in PANELS:
        cells = panel_cells(jobs)
        if {c["model"] for c in cells} != {model}:
            raise SystemExit(f"ERROR: panel {panel}: expected only {model} arms, got {sorted({c['model'] for c in cells})}")
        for c in cells:
            c["panel"] = panel
        cols = list(dict.fromkeys(c["removed"] for c in cells))
        grid = np.array([[next(c["drop_top1"] for c in cells if c["removed"] == col and c["class_id"] == cid)
                          for col in cols] for cid in CLASSES])
        panels.append((title, cols, grid, cells))
    if [len(p[1]) for p in panels] != [10, 9, 3]:
        raise SystemExit(f"ERROR: expected 10 / 9 / 3 columns, got {[len(p[1]) for p in panels]}")

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(c for p in panels for c in p[3])

    vmax = max(float(p[2].max()) for p in panels)
    cmap = LinearSegmentedColormap.from_list("p1_blue", RAMP)
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.4), gridspec_kw={"width_ratios": [len(p[1]) for p in panels],
                                                                    "wspace": 0.06})
    fig.patch.set_facecolor(SURFACE)
    for i, (ax, (title, cols, grid, _)) in enumerate(zip(axes, panels)):
        im = ax.imshow(np.clip(grid, 0, None), cmap=cmap, vmin=0, vmax=vmax, aspect="auto")
        for r in range(grid.shape[0]):
            for c in range(grid.shape[1]):
                v = grid[r, c]
                dark = max(v, 0) / vmax > 0.55   # light text on the darker half of the ramp
                ax.text(c, r, f"{v:.2f}", ha="center", va="center", fontsize=8,
                        color=SURFACE if dark else INK)
        ax.set_xticks(range(len(cols)), cols if i != 2 else ["early\n" + cols[0], "mid\n" + cols[1], "late\n" + cols[2]],
                      fontsize=9, color=INK2)
        ax.set_yticks(range(len(CLASSES)), [f"{cid} {name}" for cid, name in CLASSES.items()] if i == 0 else [],
                      fontsize=9, color=INK)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xticks(np.arange(-0.5, len(cols)), minor=True)
        ax.set_yticks(np.arange(-0.5, len(CLASSES)), minor=True)
        ax.grid(which="minor", color=SURFACE, linewidth=2)   # 2px surface gap between cells
        ax.tick_params(which="minor", length=0)
        ax.set_title(title, color=INK, fontsize=11, loc="left")
        ax.set_xlabel(("VAR scale removed", "VAR scales removed", "DiT step indices skipped")[i], color=INK2,
                      fontsize=9)
    cb = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.015)
    cb.set_label("drop in top-1 (baseline - arm)", color=INK2, fontsize=9)
    cb.ax.tick_params(colors=INK2, labelsize=8)
    cb.outline.set_visible(False)
    fig.text(0.01, -0.04, f"n = {N_PER_CLASS} images per class, so per-class CIs are wide (heatmap_by_class.csv). "
             "Each arm against its own job's clean baseline. Jobs 25893211, 25916920, 25916921 (VAR); 25893213 (DiT).",
             color=INK2, fontsize=8)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    print(f"wrote {args.out} and {args.csv} ({sum(len(p[3]) for p in panels)} rows); colour scale 0 to {vmax:.3f}")


if __name__ == "__main__":
    main()
