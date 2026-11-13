"""P1 Phase F: scale removal on real images (VAR), per dataset and removed scale. numpy (+ matplotlib for the
figure); no GPU.

    python -m lanes.p1.transfer_summary pair RUN_DIR --reference REF_DIR
        -> RUN_DIR/vs_reference.csv, per image: psnr_vs_ref and mad_vs_ref (mean |arm - reference| over pixels
           and channels, 0..255) between this run's PNG and the reference run's PNG of the same row.
           Run by jobs/transfer_var.pbs on NSCC after each arm; needs numpy + Pillow only.
    python -m lanes.p1.transfer_summary summary --crystal JOB --dog JOB
                                       [--out lanes/p1/results/transfer_var.csv]
                                       [--fig lanes/p1/results/fig1d_transfer.png]

JOB: outputs/p1/<id> or a bare id of a jobs/transfer_var.pbs run; its dataset.txt must name the dataset
given. Runs per job: recon (VQ codec only), check_k9 (real scales 0-9), reference (real scales 0-5, VAR-d20
generates 6-9) and remove_s0 .. remove_s9 (reference + scale j removed, lanes/p1/hooks.py).

CSV, one row per (dataset, run):
  psnr_real                mean PSNR to the real image (lanes/p2/transfer.py score), with 95% bootstrap CI
  psnr_drop                reference - arm, paired by image, 95% paired bootstrap CI (summarize.paired_diff)
  hf_ratio                 mean hf_ratio (nan for a flat input image is skipped)
  psnr_vs_ref, mad_vs_ref  change vs the reference run's output, mean with 95% bootstrap CI over images;
                           n_identical counts images equal to the reference (their PSNR is capped at 99 dB)
Bootstrap: 10000 resamples, seed 0 (lanes/p1/summarize.py). Figure: change vs removed scale (MAD, and PSNR
drop to the real image), crystals vs dogs, scales 0-5 real tokens, 6-9 generated.
"""

import argparse
import csv
import sys
from pathlib import Path

sys.dont_write_bytecode = True

import numpy as np  # noqa: E402

REAL_LAST = 5            # P2_REAL_STAGE: scales 0-5 are the real image's tokens
DATASETS = ("crystal", "dog")
LABEL = {"crystal": "crystals (OpenCrystalData)", "dog": "dogs (VAR's own outputs)"}
RUNS = ("recon", "check_k9", "reference", *(f"remove_s{j}" for j in range(10)))
COLUMNS = ("dataset", "job", "run", "removed_scale", "scale_kind", "n",
           "psnr_real", "psnr_real_lo", "psnr_real_hi", "psnr_drop", "psnr_drop_lo", "psnr_drop_hi", "hf_ratio",
           "psnr_vs_ref", "psnr_vs_ref_lo", "psnr_vs_ref_hi", "mad_vs_ref", "mad_vs_ref_lo", "mad_vs_ref_hi",
           "n_identical")


def _read(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------- pair (NSCC, per arm)

def pair(run_dir: Path, ref_dir: Path) -> Path:
    from PIL import Image

    from lanes.p2.transfer import psnr
    names = sorted(p.name for p in run_dir.glob("*.png"))
    ref = sorted(p.name for p in ref_dir.glob("*.png"))
    if not names or names != ref:
        raise SystemExit(f"ERROR: {run_dir} and {ref_dir} do not hold the same PNGs ({len(names)} vs {len(ref)})")
    rows = []
    for n in names:
        a = np.asarray(Image.open(run_dir / n).convert("RGB"))
        b = np.asarray(Image.open(ref_dir / n).convert("RGB"))
        rows.append({"file": Path(n).stem, "psnr_vs_ref": round(psnr(a, b), 3),
                     "mad_vs_ref": round(float(np.abs(a.astype(np.float64) - b).mean()), 4)})
    out = run_dir / "vs_reference.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{out}: {len(rows)} images, mean psnr_vs_ref {np.mean([r['psnr_vs_ref'] for r in rows]):.2f} dB, "
          f"mean mad_vs_ref {np.mean([r['mad_vs_ref'] for r in rows]):.3f}, "
          f"identical {sum(r['mad_vs_ref'] == 0 for r in rows)}")
    return out


# ---------------------------------------------------------------- summary (laptop)

def mean_ci(v) -> tuple:
    """Mean and 95% percentile bootstrap CI over images (N_BOOT resamples, seed SEED)."""
    from lanes.p1.summarize import N_BOOT, SEED
    v = np.asarray(v, dtype=float)
    boot = v[np.random.default_rng(SEED).integers(0, len(v), size=(N_BOOT, len(v)))].mean(axis=1)
    return float(v.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def load(job: Path, dataset: str) -> dict:
    """{run: {"real": {image: psnr}, "hf": {image: hf_ratio}, "ref": {file: (psnr, mad)}}} for one job."""
    info = dict(line.split("=", 1) for line in (job / "dataset.txt").read_text().split("\n") if "=" in line) \
        if (job / "dataset.txt").is_file() else {}
    if info.get("DATASET") != dataset:
        raise SystemExit(f"ERROR: {job}/dataset.txt names dataset {info.get('DATASET')!r}, expected {dataset!r}")
    out = {}
    for run in RUNS:
        d = job / run
        path = d / ("recon.csv" if run == "recon" else "transfer.csv")
        if not path.is_file():
            raise SystemExit(f"ERROR: {path} not found (did the job finish? see {job}/summary.txt)")
        rows = _read(path)
        r = {"real": {x["image"]: float(x["psnr"]) for x in rows},
             "hf": {x["image"]: float(x["hf_ratio"]) for x in rows}}
        if run.startswith("remove_s"):
            ref = d / "vs_reference.csv"
            if not ref.is_file():
                raise SystemExit(f"ERROR: {ref} not found")
            r["ref"] = {x["file"]: (float(x["psnr_vs_ref"]), float(x["mad_vs_ref"])) for x in _read(ref)}
        out[run] = r
    images = set(out["reference"]["real"])
    for run, r in out.items():
        if set(r["real"]) != images:
            raise SystemExit(f"ERROR: {job}/{run} scores other images than the reference")
        if "ref" in r and len(r["ref"]) != len(images):
            raise SystemExit(f"ERROR: {job}/{run}/vs_reference.csv has {len(r['ref'])} images, expected {len(images)}")
    return out


def rows_for(job: Path, dataset: str) -> list[dict]:
    from lanes.p1.summarize import paired_diff
    data = load(job, dataset)
    keys = sorted(data["reference"]["real"])
    out = []
    for run in RUNS:
        r = data[run]
        j = int(run[len("remove_s"):]) if run.startswith("remove_s") else None
        row = {"dataset": dataset, "job": job.name, "run": run, "removed_scale": "" if j is None else j,
               "scale_kind": "" if j is None else ("real" if j <= REAL_LAST else "generated"), "n": len(keys)}
        for s, v in zip(("", "_lo", "_hi"), mean_ci([r["real"][k] for k in keys])):
            row[f"psnr_real{s}"] = round(v, 3)
        hf = np.array([r["hf"][k] for k in keys])
        row["hf_ratio"] = round(float(np.nanmean(hf)), 4)
        if j is not None:
            for s, v in zip(("", "_lo", "_hi"), paired_diff(data["reference"]["real"], r["real"], keys)):
                row[f"psnr_drop{s}"] = round(v, 3) + 0.0
            ref = list(r["ref"].values())
            for name, col in (("psnr_vs_ref", 0), ("mad_vs_ref", 1)):
                for s, v in zip(("", "_lo", "_hi"), mean_ci([x[col] for x in ref])):
                    row[f"{name}{s}"] = round(v, 4)
            row["n_identical"] = sum(x[1] == 0 for x in ref)
        out.append(row)
    return out


def plot(rows: list[dict], path: Path, jobs: dict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    SURFACE, INK, INK2, GRID, BAND = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#f0efec"
    COLOR = {"crystal": "#2a78d6", "dog": "#eb6834"}
    MARK = {"crystal": "o", "dog": "s"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    fig.patch.set_facecolor(SURFACE)
    panels = (("mad_vs_ref", "A  Change vs the reference output", "mean |arm − reference| (0–255)"),
              ("psnr_drop", "B  PSNR lost to the real image", "PSNR drop vs reference (dB)"))
    for ax, (col, title, ylabel) in zip(axes, panels):
        ax.set_facecolor(SURFACE)
        ax.axvspan(-0.5, REAL_LAST + 0.5, color=BAND, zorder=0, linewidth=0)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(INK2)
        ax.tick_params(colors=INK2, labelcolor=INK2)
        ax.yaxis.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.axhline(0, color=INK2, linewidth=0.8)
        for k, ds in enumerate(DATASETS):
            rs = sorted((r for r in rows if r["dataset"] == ds and r["removed_scale"] != ""),
                        key=lambda r: r["removed_scale"])
            x = np.array([r["removed_scale"] for r in rs]) + (k - 0.5) * 0.24
            y = np.array([r[col] for r in rs])
            err = np.array([[r[col] - r[f"{col}_lo"] for r in rs], [r[f"{col}_hi"] - r[col] for r in rs]])
            ax.errorbar(x, y, yerr=err, color=COLOR[ds], marker=MARK[ds], markersize=7, linewidth=1.8, capsize=3,
                        elinewidth=1.2, markeredgecolor=SURFACE, markeredgewidth=1.2, label=LABEL[ds])
        ax.set_xticks(range(10))
        ax.set_xlim(-0.5, 9.5)
        ax.set_xlabel("VAR scale removed", color=INK)
        ax.set_ylabel(ylabel, color=INK)
        ax.set_title(title, color=INK, loc="left", fontsize=11, pad=24)
        top = ax.get_ylim()[1]
        ax.text(REAL_LAST / 2, top, "real tokens (0–5)", ha="center", va="bottom", color=INK2, fontsize=9)
        ax.text((REAL_LAST + 1 + 9) / 2, top, "generated (6–9)", ha="center", va="bottom", color=INK2, fontsize=9)
    axes[0].legend(loc="upper right", frameon=False, fontsize=9, labelcolor=INK)
    fig.text(0.01, 0.005, "VAR-d20, cfg 0, class 0 for both datasets; 95% bootstrap CIs over images (B: paired). "
             f"Jobs {jobs['crystal']} (crystals), {jobs['dog']} (dogs).", color=INK2, fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=SURFACE)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pair", help="per-image change of one run vs the reference run")
    p.add_argument("run_dir", type=Path)
    p.add_argument("--reference", required=True, type=Path)
    s = sub.add_parser("summary", help="transfer_var.csv and fig1d_transfer.png from the two jobs")
    s.add_argument("--crystal", required=True)
    s.add_argument("--dog", required=True)
    s.add_argument("--out", type=Path, default=Path("lanes/p1/results/transfer_var.csv"))
    s.add_argument("--fig", type=Path, default=Path("lanes/p1/results/fig1d_transfer.png"))
    args = ap.parse_args(argv)
    if args.cmd == "pair":
        pair(args.run_dir, args.reference)
        return
    from lanes.p1.summarize import _write, job_dir
    jobs = {"crystal": job_dir(args.crystal), "dog": job_dir(args.dog)}
    rows = [r for ds in DATASETS for r in rows_for(jobs[ds], ds)]
    _write(args.out, COLUMNS, rows)
    plot(rows, args.fig, {k: v.name for k, v in jobs.items()})
    print(f"wrote {args.out} ({len(rows)} rows) and {args.fig}")
    for ds in DATASETS:
        rs = {r["run"]: r for r in rows if r["dataset"] == ds}
        print(f"{ds}: codec {rs['recon']['psnr_real']:.2f} dB, real 0-9 {rs['check_k9']['psnr_real']:.2f} dB, "
              f"reference {rs['reference']['psnr_real']:.2f} dB; MAD vs reference by removed scale: "
              + " ".join(f"{j}:{rs[f'remove_s{j}']['mad_vs_ref']:.2f}" for j in range(10)))


if __name__ == "__main__":
    main()
