"""P1: P3's fidelity axis (incep_cos) for every ImageNet removal arm, read with P3's two-axis rule.

CPU only, from existing outputs: each run's inception_features.npz (written by lanes.p2.quality) and
metrics.csv (cls_prob). Nothing is written into the job folders.

    python3 -B -m lanes.p1.fidelity                 # sanity check, results/fidelity.csv, results/fig1f_fidelity.png
    python3 -B -m lanes.p1.fidelity --sanity-only

Per arm, against that job's own baseline, paired by (class_id, seed):
    incep_cos        mean per-image Inception cosine distance (lanes.p3.p3_metrics.fidelity_incep), with a
                     95% bootstrap CI over images (10000 resamples, seed 0, as summarize.py)
    validity_ratio   mean cls_prob of the arm / mean cls_prob of the baseline, CI from the same resamples
    frac_*           share of images in each of P3's four cells: lanes.p3.p3_metrics.classify(incep_cos,
                     cls_prob / max(baseline cls_prob of that image, 1e-6)), without KID, as lanes/p2/analyze.py
    outcome_run      classify(mean incep_cos, validity_ratio, kid_vs_base) for the whole arm, with KID
The thresholds are P3's PROVISIONAL values (FIDELITY_LOW, VALIDITY_LOW, KID_HIGH in lanes/p3/p3_metrics.py).
"""

import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from lanes.p1.summarize import N_BOOT, SEED, _removal, _write, job_dir, load_job  # noqa: E402
from lanes.p2.analyze import kid  # noqa: E402
from lanes.p3.p3_metrics import FIDELITY_LOW, KID_HIGH, VALIDITY_LOW, classify, fidelity_incep  # noqa: E402

JOBS = {"25893211": 1, "25893212": 1, "25893213": 1, "25916920": 1, "25916921": 1,
        "26203723": 2, "26203724": 2, "26203725": 2}
CELLS = ("unchanged", "re-routed", "degraded-in-place", "damaged")
THRESHOLDS = (f"P3 provisional: FIDELITY_LOW={FIDELITY_LOW}, VALIDITY_LOW={VALIDITY_LOW}, "
              f"KID_HIGH={KID_HIGH} (lanes/p3/p3_metrics.py)")
COLUMNS = ("model", "job", "seed_set", "arm", "removed", "n_removed", "unit", "tokens_removed", "n",
           "incep_cos", "incep_cos_lo", "incep_cos_hi", "validity_ratio", "validity_ratio_lo",
           "validity_ratio_hi", "kid_vs_base", "outcome_run",
           *(f"frac_{c.replace('-', '_')}" for c in CELLS), "thresholds")
OUT_CSV = REPO / "lanes/p1/results/fidelity.csv"
OUT_FIG = REPO / "lanes/p1/results/fig1f_fidelity.png"


def runs_of(job: Path):
    """[(run row, run_dir, baseline run_dir, {key: cls_prob}, {key: baseline cls_prob}, feats, base feats)]."""
    runs, metrics, feats = load_job(job)
    by_id = {r["run_id"]: r for r in runs}
    out = []
    for r in runs:
        b = r["baseline_run_id"]
        if not b:
            continue
        out.append((r, job / r["run_dir"], job / by_id[b]["run_dir"], metrics[r["run_id"]]["cls_prob"],
                    metrics[b]["cls_prob"], feats[r["run_id"]], feats[b]))
    return out


def sanity(jobs) -> None:
    """Each job's baseline against itself must give incep_cos 0 for every image."""
    print("sanity: baseline vs itself (lanes.p3.p3_metrics.fidelity_incep)")
    worst = 0.0
    for j in jobs:
        job = job_dir(j)
        base = sorted({b for _, _, b, *_ in runs_of(job)})
        for d in base:
            v = np.array([r["value"] for r in fidelity_incep(d, d)])
            worst = max(worst, float(np.max(np.abs(v))))
            print(f"  {j} {d.parent.name}/{d.name}: n {len(v)}, max |incep_cos| {np.max(np.abs(v)):.2e}, "
                  f"all exactly 0: {bool(np.all(v == 0))}")
    if worst > 1e-5:
        raise SystemExit(f"FAIL: a baseline is not at incep_cos 0 against itself (max {worst:.2e})")
    print(f"  PASS: max |incep_cos| over all baselines {worst:.2e}")


def arm_row(j: str, r, run_dir, base_dir, prob, prob_base, fa, fb) -> dict:
    cos = {(x["class_id"], x["seed"]): x["value"] for x in fidelity_incep(run_dir, base_dir)}
    keys = sorted(cos)
    if sorted(prob) != keys or sorted(prob_base) != keys:
        raise SystemExit(f"ERROR: {j} {r['arm']}: features and cls_prob cover different images")
    c = np.array([cos[k] for k in keys])
    p = np.array([prob[k] for k in keys])
    pb = np.array([prob_base[k] for k in keys])
    idx = np.random.default_rng(SEED).integers(0, len(keys), size=(N_BOOT, len(keys)))
    bc = c[idx].mean(axis=1)
    bv = p[idx].mean(axis=1) / pb[idx].mean(axis=1)
    ratio = float(p.mean() / pb.mean())
    k_run = round(kid(np.stack([fa[k] for k in keys]), np.stack([fb[k] for k in keys]), paired=True), 5) + 0.0
    cells = [classify(ci, pi / max(bi, 1e-6)) for ci, pi, bi in zip(c, p, pb)]
    row = {"model": r["model"], "job": j, "seed_set": JOBS[j], "arm": r["arm"],
           **_removal(job_dir(j), r["arm"]), "n": len(keys),
           "incep_cos": round(float(c.mean()), 4),
           "incep_cos_lo": round(float(np.percentile(bc, 2.5)), 4),
           "incep_cos_hi": round(float(np.percentile(bc, 97.5)), 4),
           "validity_ratio": round(ratio, 4),
           "validity_ratio_lo": round(float(np.percentile(bv, 2.5)), 4),
           "validity_ratio_hi": round(float(np.percentile(bv, 97.5)), 4),
           "kid_vs_base": k_run, "outcome_run": classify(float(c.mean()), ratio, k_run),
           "thresholds": THRESHOLDS}
    for cell in CELLS:
        row[f"frac_{cell.replace('-', '_')}"] = round(cells.count(cell) / len(cells), 4)
    return row


def label(row) -> str:
    if row["model"] == "dit":
        return row["arm"].replace("remove_", "")
    s = row["removed"].strip("{}").split(",")
    return f"s{s[0]}" if len(s) == 1 else "{" + ",".join(s) + "}"


def plot(rows, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted, grid = "#1f1f1e", "#6b6a63", "#e6e5e0"
    kinds = {  # categorical slots 1-3 (validated), plus a marker shape each
        "VAR single scale": ("#2a78d6", "o"),
        "VAR 2-3 scale window (named in fidelity.csv)": ("#1baf7a", "s"),
        "DiT band (83 steps)": ("#eb6834", "^"),
    }

    def kind(r):
        if r["model"] == "dit":
            return "DiT band (83 steps)"
        return "VAR single scale" if int(r["n_removed"]) == 1 else "VAR 2-3 scale window (named in fidelity.csv)"

    xmax = max(r["incep_cos_hi"] for r in rows) * 1.08
    ymax = max(1.12, max(r["validity_ratio_hi"] for r in rows) * 1.05)
    offsets = {"s0": (-20, 6), "s1": (-20, -13), "s7": (-18, 4), "s9": (6, -13), "early": (8, 4)}  # no overlaps
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.2), sharex=True, sharey=True)
    for ax, s in zip(axes, (1, 2)):
        ax.set_facecolor("#fcfcfb")
        for v in ("top", "right"):
            ax.spines[v].set_visible(False)
        for v in ("left", "bottom"):
            ax.spines[v].set_color(muted)
        ax.grid(True, color=grid, lw=0.8)
        ax.set_axisbelow(True)
        ax.axvline(FIDELITY_LOW, color=muted, ls="--", lw=1.2)
        ax.axhline(VALIDITY_LOW, color=muted, ls="--", lw=1.2)
        for txt, x, y, ha, va in (("unchanged", 0.01, ymax - 0.02, "left", "top"),
                                  ("re-routed", xmax - 0.005, ymax - 0.02, "right", "top"),
                                  ("degraded in place", 0.01, VALIDITY_LOW - 0.03, "left", "top"),
                                  ("damaged", xmax - 0.005, VALIDITY_LOW - 0.03, "right", "top")):
            ax.text(x, y, txt, ha=ha, va=va, fontsize=11, color=muted, style="italic")
        for r in (r for r in rows if r["seed_set"] == s):
            colour, marker = kinds[kind(r)]
            x, y = r["incep_cos"], r["validity_ratio"]
            ax.errorbar(x, y, xerr=[[x - r["incep_cos_lo"]], [r["incep_cos_hi"] - x]],
                        yerr=[[y - r["validity_ratio_lo"]], [r["validity_ratio_hi"] - y]],
                        fmt=marker, ms=8, color=colour, mec="#fcfcfb", mew=1.5, elinewidth=1.2, capsize=0,
                        zorder=3)
            if kind(r) != "VAR 2-3 scale window (named in fidelity.csv)":
                ax.annotate(label(r), (x, y), xytext=offsets.get(label(r), (6, 5)), textcoords="offset points",
                            fontsize=9, color=ink)
        ax.set_xlim(0, xmax)
        ax.set_ylim(-0.03, ymax)
        ax.set_title(f"Seed set {s}" + (" (manifest v1.1)" if s == 1 else " (seed2.csv)"),
                     loc="left", fontsize=12, color=ink)
        ax.set_xlabel("fidelity: incep_cos vs own baseline (higher = less like the baseline)", color=ink)
        ax.tick_params(colors=muted)
    axes[0].set_ylabel("validity: cls_prob / baseline cls_prob", color=ink)
    handles = [plt.Line2D([], [], color=c, marker=m, ls="", ms=8, label=k) for k, (c, m) in kinds.items()]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=10,
               bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Figure 1f. Validity vs fidelity per removal arm (95% bootstrap CIs; dashed: P3's provisional "
                 f"thresholds, incep_cos {FIDELITY_LOW}, validity {VALIDITY_LOW})", x=0.01, ha="left",
                 fontsize=12, color=ink)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    fig.savefig(path, dpi=150, facecolor="white")
    print(f"wrote {path.relative_to(REPO)}")


def main(argv=None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    for out in (OUT_CSV, OUT_FIG):
        if out.exists():
            raise SystemExit(f"ERROR: {out.relative_to(REPO)} exists; this script only writes new files")
    sanity(JOBS)
    if "--sanity-only" in argv:
        return
    rows = []
    for j in JOBS:
        for args in runs_of(job_dir(j)):
            rows.append(arm_row(j, *args))
    order = {"remove_early": 0, "remove_mid": 1, "remove_late": 2}
    rows.sort(key=lambda r: (r["seed_set"], r["model"] != "var",
                             (int(r["n_removed"]), int(r["removed"].strip("{}").split(",")[0]))
                             if r["model"] == "var" else (0, order[r["arm"]])))
    _write(OUT_CSV, COLUMNS, rows)
    print(f"wrote {OUT_CSV.relative_to(REPO)} ({len(rows)} arms); thresholds: {THRESHOLDS}")
    plot(rows, OUT_FIG)


if __name__ == "__main__":
    main()
