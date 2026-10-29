"""Figure 1: top-1 accuracy after stage removal, VAR (left) and DiT (right). Needs numpy + matplotlib; no GPU.

    python -m lanes.p1.plot_fig1 [--out lanes/p1/results/fig1.png]

Left: VAR top-1 vs position on the scale ladder. Single-scale arms (s0, s9 from 25893211; s1-s8 from
25916920) at the removed scale; 2-scale windows (25916921) and the three 3-scale groups (25893212,
labelled points) at the mean of their removed scales ({0,1} at 0.5, {3,4,5} at 4). Under each scale
index, its token-grid size (pn x pn, from the baseline run.json stages table).
Right: DiT baseline / early / mid / late (25893213) as bars.
Error bars: 95% percentile bootstrap CI of each arm's mean top-1 over its 224 images (10000 resamples,
seed 0, as lanes/p1/summarize.py). Dashed line: the model's clean baseline.
Removed scales / steps are read from each arm's params in JOB/tables/runs.csv, not from the arm name.
"""

import argparse
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from lanes.p1.summarize import N_BOOT, SEED, job_dir, load_job  # noqa: E402

SINGLE_JOBS, WINDOW2_JOB, WINDOW3_JOB, DIT_JOB = ("25893211", "25916920"), "25916921", "25893212", "25893213"
# Reference palette (dataviz skill, references/palette.md), categorical slots 1-3 on the light surface.
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, NEUTRAL = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"


def mean_ci(values: dict) -> tuple:
    """Mean of {(class_id, seed): value} and its 95% percentile bootstrap CI over images."""
    v = np.array([values[k] for k in sorted(values)])
    idx = np.random.default_rng(SEED).integers(0, len(v), size=(N_BOOT, len(v)))
    boot = v[idx].mean(axis=1)
    return float(v.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def arms(job: str) -> list[dict]:
    """Per run of a job: arm, role, params, and top-1 mean with CI."""
    runs, metrics, _ = load_job(job_dir(job))
    out = []
    for r in runs:
        m, lo, hi = mean_ci(metrics[r["run_id"]]["cls_top1"])
        out.append({"arm": r["arm"], "baseline": r["role"] == "baseline", "params": json.loads(r["params"] or "{}"),
                    "top1": m, "lo": lo, "hi": hi})
    return out


def var_baseline(*jobs) -> dict:
    """The VAR baseline top-1; every job's baseline must agree (they are byte-identical runs)."""
    bases = [a for j in jobs for a in arms(j) if a["baseline"]]
    if len({round(b["top1"], 9) for b in bases}) != 1:
        raise SystemExit(f"ERROR: VAR baselines differ across jobs {jobs}: {[b['top1'] for b in bases]}")
    return bases[0]


def _errbar(ax, xs, rows, color, marker, label, **kw):
    y = np.array([r["top1"] for r in rows])
    err = np.array([[r["top1"] - r["lo"] for r in rows], [r["hi"] - r["top1"] for r in rows]])
    ax.errorbar(xs, y, yerr=err, color=color, marker=marker, markersize=8, linewidth=2, capsize=3,
                elinewidth=1.2, markeredgecolor=SURFACE, markeredgewidth=1.5, label=label, **kw)


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK2)
    ax.tick_params(colors=INK2, labelcolor=INK2)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_ylim(-0.03, 1.05)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("lanes/p1/results/fig1.png"))
    args = ap.parse_args(argv)

    single = sorted((a for j in SINGLE_JOBS for a in arms(j) if not a["baseline"]),
                    key=lambda a: a["params"]["scales"])
    window2 = sorted((a for a in arms(WINDOW2_JOB) if not a["baseline"]), key=lambda a: a["params"]["scales"])
    window3 = sorted((a for a in arms(WINDOW3_JOB) if not a["baseline"]), key=lambda a: a["params"]["scales"])
    for rows, n in ((single, 1), (window2, 2), (window3, 3)):
        bad = [a["arm"] for a in rows if len(a["params"]["scales"]) != n]
        if bad:
            raise SystemExit(f"ERROR: arms {bad} do not remove {n} scale(s)")
    if [a["params"]["scales"][0] for a in single] != list(range(10)):
        raise SystemExit(f"ERROR: single-scale arms are not s0-s9: {[a['arm'] for a in single]}")
    vbase = var_baseline(*SINGLE_JOBS, WINDOW2_JOB, WINDOW3_JOB)
    stages = json.loads((job_dir(SINGLE_JOBS[0]) / "var" / "baseline" / "run.json").read_text())["stages"]
    grid = {s["stage"]: s["pn"] for s in stages}
    centre = lambda a: float(np.mean(a["params"]["scales"]))   # noqa: E731
    dit = {a["arm"]: a for a in arms(DIT_JOB)}
    dit_rows = [dit[k] for k in ("baseline", "remove_early", "remove_mid", "remove_late")]

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [2.2, 1]})
    fig.patch.set_facecolor(SURFACE)

    # Left: VAR
    _style(ax)
    ax.axhline(vbase["top1"], color=INK2, linestyle="--", linewidth=1.2, zorder=1)
    ax.text(10.35, vbase["top1"] + 0.015, f"baseline {vbase['top1']:.3f}", color=INK2, ha="right", va="bottom",
            fontsize=9)
    _errbar(ax, [a["params"]["scales"][0] for a in single], single, BLUE, "o", "1 scale removed (at that scale)")
    _errbar(ax, [centre(a) for a in window2], window2, ORANGE, "s", "2-scale window (at its centre)")
    _errbar(ax, [centre(a) for a in window3], window3, AQUA, "D", "3-scale group (at its centre)",
            linestyle="none")
    for a in window3:
        s = a["params"]["scales"]
        ax.annotate("{" + ",".join(map(str, s)) + "}", (centre(a), a["lo"]), textcoords="offset points",
                    xytext=(0, -6), ha="center", va="top", color=INK, fontsize=9)
    ax.set_xticks(range(10), [f"{i}\n{grid[i]}×{grid[i]}" for i in range(10)])
    ax.set_xlim(-0.5, 10.4)
    ax.set_xlabel("VAR scale removed (windows/groups at their centre)", color=INK)
    ax.set_ylabel("top-1 accuracy (own class)", color=INK)
    ax.set_title("VAR-d20: removing scales", color=INK, loc="left", fontsize=11, pad=26)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, frameon=False, fontsize=9, labelcolor=INK,
              borderaxespad=0.2, handletextpad=0.4, columnspacing=1.2)
    ax.set_ylim(-0.12, 1.05)

    # Right: DiT
    _style(bx)
    x = np.arange(len(dit_rows))
    y = np.array([r["top1"] for r in dit_rows])
    err = np.array([[r["top1"] - r["lo"] for r in dit_rows], [r["hi"] - r["top1"] for r in dit_rows]])
    bx.bar(x, y, width=0.6, color=[NEUTRAL, BLUE, BLUE, BLUE], edgecolor=SURFACE, linewidth=2, zorder=2)
    bx.errorbar(x, y, yerr=err, fmt="none", ecolor=INK, elinewidth=1.2, capsize=3, zorder=3)
    bx.axhline(dit["baseline"]["top1"], color=INK2, linestyle="--", linewidth=1.2, zorder=1)
    for xi, r in zip(x, dit_rows):   # inside the bar, clear of the error bar and the baseline line
        bx.text(xi, 0.04, f"{r['top1']:.3f}", ha="center", color=SURFACE, fontsize=9, fontweight="bold")
    bx.set_xticks(x, ["baseline", "early\n1-83", "mid\n84-166", "late\n167-249"])
    bx.set_xlabel("DiT steps skipped (83 of 250)", color=INK)
    bx.set_title("DiT-XL/2: skipping steps", color=INK, loc="left", fontsize=11, pad=26)

    fig.text(0.01, 0.005, "Error bars: 95% bootstrap CI of the mean over 224 images. Dashed: clean baseline. "
             "Jobs 25893211, 25916920, 25916921, 25893212 (VAR); 25893213 (DiT).", color=INK2, fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, facecolor=SURFACE)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
