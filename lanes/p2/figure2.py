"""Figure 2 (P2): where VAR and DiT recover from a corrupted intermediate state, and where damage sticks.

    python lanes/p2/figure2.py [--out lanes/p2/results/figure2]      # writes figure2.png and figure2.pdf

Reads only the result tables committed in lanes/p2/results/ (no GPU, no outputs/), so anyone can
regenerate it. Needs matplotlib, which is NOT in var-dit (P1's env); use any Python with matplotlib
and numpy, e.g. a separate env:  conda create -n p2-plot -c conda-forge python=3.11 matplotlib numpy

Panels
  A  VAR: fraction of images damaged, scale x severity (cell letter = P3's run-level outcome)
  B  DiT: the same, timestep x severity
  C  highest severity each stage tolerates (run still unchanged or re-routed), vs progress p
  D  fraction of images damaged at severity 2, vs progress p
  E  VAR recovery curves at severity 1: state gap to the baseline after the corruption,
     relative to the gap right after it (1 = no repair, < 1 = repaired, > 1 = diverged)
  F  DiT recovery curves, the same

VAR from Gate B (noise seed 0; Phase C seed 1 agrees in every cell), DiT from Phase C; the dashed
DiT (σ) lines in C and D from Phase E1 (P2_SCALE=sigma).
"""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

RES = Path(__file__).resolve().parent / "results"
SEVS = [0.25, 0.5, 1.0, 2.0, 4.0]
# Palette (dataviz reference instance): categorical slots 1-2, one sequential blue ramp, text inks.
VAR_C, DIT_C = "#2a78d6", "#eb6834"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BLUES = ["#f3f8fe", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SEQ = LinearSegmentedColormap.from_list("p2_blues", BLUES)
LETTER = {"unchanged": "U", "re-routed": "R", "degraded-in-place": "G", "damaged": "D"}


def read(path, model):
    return [r for r in csv.DictReader(open(path)) if r["model"] == model]


def cells(rows):
    return {(int(r["stage"]), float(r["severity"])): r for r in rows}


def heatmap(ax, c, stages, ylabels, title):
    m = np.array([[float(c[(s, v)]["frac_damaged"]) for v in SEVS] for s in stages])
    ax.imshow(m, cmap=SEQ, vmin=0, vmax=1, aspect="auto")
    for i, s in enumerate(stages):
        for j, v in enumerate(SEVS):
            ink = SURF if m[i, j] > 0.55 else INK
            ax.text(j, i, LETTER[c[(s, v)]["outcome"]], ha="center", va="center", fontsize=8, color=ink)
    ax.set_xticks(range(len(SEVS)), [f"{v:g}" for v in SEVS])
    ax.set_yticks(range(len(stages)), ylabels)
    ax.set_xlabel("noise severity (× the state's own std)")
    ax.set_title(title, loc="left", fontsize=10, color=INK)
    ax.tick_params(length=0, colors=INK2, labelsize=8)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xticks(np.arange(-.5, len(SEVS)), minor=True)
    ax.set_yticks(np.arange(-.5, len(stages)), minor=True)
    ax.grid(which="minor", color=SURF, linewidth=2)   # 2px surface gap between cells
    ax.tick_params(which="minor", length=0)


def style(ax, title, ylabel):
    ax.set_title(title, loc="left", fontsize=10, color=INK)
    ax.set_ylabel(ylabel, color=INK2, fontsize=9)
    ax.set_xlabel("progress p (0 = first stage, 1 = last)", color=INK2, fontsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)


def tolerated(c, stages, sevs=SEVS):
    out = []
    for s in stages:
        ok = [v for v in sevs if c[(s, v)]["outcome"] in ("unchanged", "re-routed")]
        out.append(max(ok) if ok else 0.125)
    return out


def curves(ax, path, model, metric, sev, stages_order, color_ramp, title):
    rows = [r for r in csv.DictReader(open(path))
            if r["model"] == model and r["metric"] == metric and float(r["severity"]) == sev and r["mean"]]
    by = {}   # injection stage -> {traced stage: (p, gap)}; runs repeated in two jobs are bit-identical
    for r in rows:
        by.setdefault(int(r["injection_stage"]), {})[int(r["stage"])] = (float(r["p"]), float(r["mean"]))
    inj = [s for s in stages_order if s in by]
    ramp = [color_ramp[int(round(k))] for k in np.linspace(2, len(color_ramp) - 1, len(inj))]
    ends = []
    for s, col in zip(inj, ramp):
        pts = sorted((p, st, g) for st, (p, g) in by[s].items())
        p_inj = next((p for p, st, _ in pts if st == s), None)
        g0 = next((g for p, st, g in pts if st == s), None)
        if not g0:
            continue   # injection step not traced (DiT timestep 36)
        after = [(p, g / g0) for p, st, g in pts if p >= p_inj]
        ax.plot([p for p, _ in after], [g for _, g in after], color=col, linewidth=2, marker="o", markersize=3)
        ends.append([after[-1][1], f"{s}"])
    # Spread the end labels so none overlap (minimum gap: 4.5% of the data range).
    ends.sort()
    lo, hi = ax.get_ylim()
    gap = 0.045 * (hi - lo)
    for i in range(1, len(ends)):
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + gap)
    for y, lab in ends:
        ax.text(1.015, y, lab, fontsize=7, color=INK2, va="center")
    ax.axhline(1, color=INK2, linewidth=0.8, linestyle="--")
    style(ax, title, "gap to baseline ÷ gap at injection")
    ax.set_xlim(-0.02, 1.08)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=RES / "figure2")
    args = ap.parse_args(argv)

    var = cells(read(RES / "gateb" / "var_p2_summary.csv", "var"))
    dit = cells(read(RES / "phasec" / "p2_summary.csv", "dit"))
    sig = cells(r for r in read(RES / "phasee" / "p2_summary.csv", "dit") if r["scale"] == "sigma")
    vs = list(range(10))
    ds = sorted({s for s, _ in dit}, reverse=True)   # 798 ... 36 (sampling order)
    vp = [float(var[(s, 1.0)]["p"]) for s in vs]
    dp = [float(dit[(s, 1.0)]["p"]) for s in ds]

    plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": SURF, "axes.facecolor": SURF})
    fig = plt.figure(figsize=(13, 11.5))
    gs = fig.add_gridspec(3, 2, hspace=0.45, wspace=0.28)

    heatmap(fig.add_subplot(gs[0, 0]), var, vs, [f"scale {s}  (p {p:.2f})" for s, p in zip(vs, vp)],
            "A  VAR-d20: fraction of images damaged")
    heatmap(fig.add_subplot(gs[0, 1]), dit, ds, [f"t {s}  (p {p:.2f})" for s, p in zip(ds, dp)],
            "B  DiT-XL/2: fraction of images damaged")

    ax = fig.add_subplot(gs[1, 0])
    for p, t, col, lab in ((vp, tolerated(var, vs), VAR_C, "VAR"), (dp, tolerated(dit, ds), DIT_C, "DiT")):
        ax.plot(p, t, color=col, linewidth=2, marker="o", markersize=6, drawstyle="steps-mid")
        ax.text(p[-1] + 0.02, t[-1], lab, color=INK, fontsize=9, va="center")
    ss = sorted({s for s, _ in sig}, reverse=True)
    sp, st = [float(sig[(s, 1.0)]["p"]) for s in ss], tolerated(sig, ss, (0.5, 1.0, 2.0))
    ax.plot(sp, st, color=DIT_C, linewidth=1.5, linestyle="--", marker="o", markersize=4, mfc=SURF,
            drawstyle="steps-mid")
    ax.text(sp[-1] - 0.02, st[-1] * 1.3, "DiT (σ)", color=INK, fontsize=9, va="center")   # above VAR's label
    ax.set_yscale("log", base=2)
    ax.set_yticks(SEVS, [f"{v:g}" for v in SEVS])
    ax.set_xlim(-0.02, 1.1)
    style(ax, "C  Highest severity still absorbed", "severity (unchanged or re-routed)")

    ax = fig.add_subplot(gs[1, 1])
    for p, c, st, col, lab in ((vp, var, vs, VAR_C, "VAR"), (dp, dit, ds, DIT_C, "DiT")):
        y = [float(c[(s, 2.0)]["frac_damaged"]) for s in st]
        ax.plot(p, y, color=col, linewidth=2, marker="o", markersize=6)
        lx, ly = (0.6, 1.05) if lab == "DiT" else (0.56, 0.02)   # beside each line's plateau / valley
        ax.text(lx, ly, lab, color=INK, fontsize=9, va="center")
    ax.plot(sp, [float(sig[(s, 2.0)]["frac_damaged"]) for s in ss], color=DIT_C, linewidth=1.5, linestyle="--",
            marker="o", markersize=4, mfc=SURF)
    ax.text(0.97, 0.1, "DiT (σ)", color=INK, fontsize=9, va="center")
    ax.set_ylim(-0.03, 1.08)
    ax.set_xlim(-0.02, 1.1)
    style(ax, "D  Damage at severity 2", "fraction of images damaged")

    blues = BLUES[2:]
    oranges = ["#fbe1d6", "#f6c3ad", "#f2a584", "#ee875c", "#eb6834", "#c4532a", "#9c4221", "#743118"]
    curves(fig.add_subplot(gs[2, 0]), RES / "gateb" / "var_p2_curves.csv", "var", "state_l2:f_hat", 1.0, vs, blues,
           "E  VAR, severity 1: gap grows (re-routes, no pull-back)")
    curves(fig.add_subplot(gs[2, 1]), RES / "phasec" / "p2_curves.csv", "dit", "state_l2:x", 1.0, ds, oranges,
           "F  DiT, severity 1: gap shrinks (repairs towards the original)")

    fig.suptitle("Figure 2. Corruption and recovery: noise injected at one stage of generation "
                 "(224 images, 7 dog classes)", x=0.06, ha="left", fontsize=12, color=INK)
    fig.text(0.06, 0.015, "Cell letters (P3's rule, run level): U unchanged, R re-routed, G degraded-in-place, "
             "D damaged. E/F: lines labelled by injection stage; DiT t36 is not traced. "
             "Severities are matched by rule, not mechanism: compare trends across models, not values.\n"
             "C/D dashed, DiT (σ): severity in units of the step's own noise level (Phase E1, severities 0.5-2), "
             "under which DiT's last steps absorb the noise.",
             fontsize=8, color=INK2)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(args.out.with_suffix(f".{ext}"), dpi=200, bbox_inches="tight")
    print(f"wrote {args.out.with_suffix('.png')} and .pdf")


if __name__ == "__main__":
    main()
