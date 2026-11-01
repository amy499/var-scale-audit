"""Figure 1b: example images per removal arm, VAR and DiT. Needs matplotlib + Pillow; no GPU.

    python -m lanes.p1.failure_grid [--out lanes/p1/results/fig1b_failures.png]

Rows (fixed rule, not picked by eye): the first manifest row of each of the first N_ROWS classes, in
manifest order (manifest/frozen_v1_1.csv: classes 151, 153, 162, 207).
Columns: VAR baseline, remove s0, s3, s9, {2,3}, {7,8}; then DiT baseline, early, mid, late (COLUMNS).
Each header gives the arm and its mean top-1 over all 224 images (lanes/p1/results/results_sweep.csv).
Images are the PNGs saved in each run directory; any missing image, run or result row is an error.
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
from PIL import Image  # noqa: E402

from lanes.p1.summarize import job_dir  # noqa: E402
from runner.manifest import load_manifest  # noqa: E402

MANIFEST = Path("manifest/frozen_v1_1.csv")
RESULTS = Path("lanes/p1/results/results_sweep.csv")
N_ROWS = 4
# (block, header, job, run directory under the job, arm name in results_sweep.csv)
COLUMNS = (
    ("VAR", "baseline", "25893211", "var/baseline", "baseline"),
    ("VAR", "remove s0", "25893211", "var/remove_s0", "remove_s0"),
    ("VAR", "remove s3", "25916920", "var/remove_s3", "remove_s3"),
    ("VAR", "remove s9", "25893211", "var/remove_s9", "remove_s9"),
    ("VAR", "remove {2,3}", "25916921", "var/remove_s23", "remove_s23"),
    ("VAR", "remove {7,8}", "25916921", "var/remove_s78", "remove_s78"),
    ("DiT", "baseline", "25893213", "dit/baseline", "baseline"),
    ("DiT", "early 1-83", "25893213", "dit/remove_early", "remove_early"),
    ("DiT", "mid 84-166", "25893213", "dit/remove_mid", "remove_mid"),
    ("DiT", "late 167-249", "25893213", "dit/remove_late", "remove_late"),
)
SURFACE, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"


def chosen_rows() -> list:
    """The first manifest row of each of the first N_ROWS classes, in manifest order."""
    rows, seen = [], set()
    for r in load_manifest(MANIFEST):
        if r.class_id not in seen:
            seen.add(r.class_id)
            rows.append(r)
    if len(rows) < N_ROWS:
        raise SystemExit(f"ERROR: {MANIFEST} has only {len(rows)} classes, need {N_ROWS}")
    return rows[:N_ROWS]


def top1_means() -> dict:
    """{(job, arm): mean top-1} from results_sweep.csv."""
    if not RESULTS.is_file():
        raise SystemExit(f"ERROR: {RESULTS} not found: run python -m lanes.p1.summarize first")
    with RESULTS.open(newline="", encoding="utf-8") as f:
        return {(r["job"], r["arm"]): float(r["cls_top1"]) for r in csv.DictReader(f)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("lanes/p1/results/fig1b_failures.png"))
    args = ap.parse_args(argv)

    rows, top1 = chosen_rows(), top1_means()
    missing, images = [], {}
    for _, _, job, run, arm in COLUMNS:
        d = job_dir(job) / run
        if (job, arm) not in top1:
            missing.append(f"{RESULTS}: no row for job {job} arm {arm}")
        if not (d / "run.json").is_file():
            missing.append(f"{d}/run.json")
            continue
        in_run = {(r["class_id"], r["seed"]) for r in json.loads((d / "run.json").read_text())["rows"]}
        for r in rows:
            png = d / f"{r.stem}.png"
            if (r.class_id, r.seed) not in in_run or not png.is_file():
                missing.append(str(png))
            else:
                images[(run, job, r.stem)] = png
    if missing:
        raise SystemExit("ERROR: missing inputs:\n  " + "\n  ".join(missing))

    n_var = sum(c[0] == "VAR" for c in COLUMNS)
    # Square cells of CELL inches, GAP between cells, one SPACER-wide empty column between VAR and DiT.
    CELL, GAP, SPACER, LEFT, RIGHT, TOP, BOTTOM = 1.5, 0.05, 0.4, 0.75, 0.1, 0.95, 0.35
    ratios = [1] * n_var + [SPACER / CELL] + [1] * (len(COLUMNS) - n_var)
    w = LEFT + RIGHT + CELL * sum(ratios) + GAP * (len(ratios) - 1)
    h = TOP + BOTTOM + CELL * len(rows) + GAP * (len(rows) - 1)
    fig, axes = plt.subplots(len(rows), len(ratios), figsize=(w, h),
                             gridspec_kw={"width_ratios": ratios, "wspace": GAP / CELL, "hspace": GAP / CELL})
    fig.subplots_adjust(left=LEFT / w, right=1 - RIGHT / w, top=1 - TOP / h, bottom=BOTTOM / h)
    fig.patch.set_facecolor(SURFACE)
    for i, r in enumerate(rows):
        for c, ax in enumerate(axes[i]):
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if c == n_var:
                ax.axis("off")   # spacer
                continue
            block, header, job, run, arm = COLUMNS[c if c < n_var else c - 1]
            ax.imshow(Image.open(images[(run, job, r.stem)]).convert("RGB"))
            if i == 0:
                ax.set_title(f"{header}\ntop-1 {top1[(job, arm)]:.3f}", color=INK, fontsize=9, pad=4)
            if c == 0:
                ax.set_ylabel(f"class {r.class_id}\nseed {r.seed}", color=INK2, fontsize=8)
    for block, col in (("VAR-d20 (scales removed)", 0), ("DiT-XL/2 (steps skipped, 83 of 250)", n_var + 1)):
        left = axes[0][col].get_position().x0
        fig.text(left, 1 - 0.15 / h, block, color=INK, fontsize=11, ha="left", va="top")
    fig.text(0.01, 0.08 / h, f"Rows: first manifest row of each of the first {N_ROWS} classes ({MANIFEST.as_posix()}). "
             "Headers: mean top-1 over all 224 images. Jobs 25893211, 25916920, 25916921 (VAR); 25893213 (DiT).",
             color=INK2, fontsize=8)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130, facecolor=SURFACE)
    print(f"wrote {args.out}  (rows: {', '.join(r.stem for r in rows)})")


if __name__ == "__main__":
    main()
