"""P1 Phase F, DiT version (exploratory): CSV and figure from the two jobs/transfer_dit.pbs runs. numpy (+
matplotlib for the figure); no GPU.

    python -m lanes.p1.transfer_dit_summary --crystal JOB --dog JOB
                                           [--out lanes/p1/results/transfer_dit.csv]
                                           [--fig lanes/p1/results/fig1g_transfer_dit.png]

JOB: outputs/p1/<id> or a bare id; its dataset.txt must name the dataset given. Runs per job: recon (DiT VAE
only), reference (real latent noised to the start timestep, denoised over every later step), skip_mid and
skip_late (the reference with 83 steps skipped). The per-image files are the job's own: transfer.csv
(lanes/p2/transfer.py score) and, for arms, vs_reference.csv (lanes/p1/transfer_summary.py pair).

CSV, one row per (dataset, run); the same columns and statistics as transfer_var.csv (transfer_summary.py):
  start_timestep, steps, timestep_first/last   from dataset.txt and the run's run.json (skip_timesteps)
  psnr_real                mean PSNR to the real image, with 95% bootstrap CI
  psnr_drop                reference - arm, paired by image, 95% paired bootstrap CI (summarize.paired_diff)
  hf_ratio                 mean hf_ratio (nan for a flat input image is skipped)
  psnr_vs_ref, mad_vs_ref  change vs the reference run's output, mean with 95% bootstrap CI over images;
                           n_identical counts images equal to the reference
Bootstrap: 10000 resamples, seed 0 (lanes/p1/summarize.py). Figure: MAD vs the reference and PSNR drop to the
real image for skip_mid and skip_late, crystals vs dogs.
"""

import argparse
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True

import numpy as np  # noqa: E402

from lanes.p1.transfer_summary import _read, mean_ci  # noqa: E402

DATASETS = ("crystal", "dog")
LABEL = {"crystal": "crystals (OpenCrystalData)", "dog": "dogs (VAR's own outputs)"}
ARMS = ("skip_mid", "skip_late")
RUNS = ("recon", "reference", *ARMS)
COLUMNS = ("dataset", "job", "run", "start_timestep", "steps", "timestep_first", "timestep_last", "n",
           "psnr_real", "psnr_real_lo", "psnr_real_hi", "psnr_drop", "psnr_drop_lo", "psnr_drop_hi", "hf_ratio",
           "psnr_vs_ref", "psnr_vs_ref_lo", "psnr_vs_ref_hi", "mad_vs_ref", "mad_vs_ref_lo", "mad_vs_ref_hi",
           "n_identical")


def load(job: Path, dataset: str) -> tuple:
    """(dataset.txt as a dict, {run: {"real": {image: psnr}, "hf": {...}, "ref": {file: (psnr, mad)},
    "skip": [timesteps]}}) for one job."""
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
        if run != "recon":
            r["skip"] = json.loads((d / "run.json").read_text())["skip_timesteps"]
        if run in ARMS:
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
    if out["reference"]["skip"]:
        raise SystemExit(f"ERROR: {job}/reference skips timesteps; it must be the full schedule")
    return info, out


def rows_for(job: Path, dataset: str) -> list[dict]:
    from lanes.p1.summarize import paired_diff
    info, data = load(job, dataset)
    keys = sorted(data["reference"]["real"])
    out = []
    for run in RUNS:
        r = data[run]
        skip = r.get("skip") or []
        row = {"dataset": dataset, "job": job.name, "run": run,
               "start_timestep": "" if run == "recon" else int(info["START_TIMESTEP"]),
               "steps": info.get(run, ""), "timestep_first": skip[0] if skip else "",
               "timestep_last": skip[-1] if skip else "", "n": len(keys)}
        for s, v in zip(("", "_lo", "_hi"), mean_ci([r["real"][k] for k in keys])):
            row[f"psnr_real{s}"] = round(v, 3)
        row["hf_ratio"] = round(float(np.nanmean([r["hf"][k] for k in keys])), 4)
        if run in ARMS:
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
    SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
    COLOR = {"crystal": "#2a78d6", "dog": "#eb6834"}   # as fig1d_transfer.png
    MARK = {"crystal": "o", "dog": "s"}
    start = {r["start_timestep"] for r in rows if r["start_timestep"] != ""}
    tick = {}
    for r in rows:
        if r["run"] in ARMS:
            tick[r["run"]] = (f"skip {r['run'][5:]}\nsteps {r['steps']}\n"
                              f"(t{r['timestep_first']}–t{r['timestep_last']})")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    fig.patch.set_facecolor(SURFACE)
    panels = (("mad_vs_ref", "A  Change vs the reference output", "mean |arm − reference| (0–255)"),
              ("psnr_drop", "B  PSNR change to the real image (negative = closer)", "PSNR change to real image (dB)"))
    for ax, (col, title, ylabel) in zip(axes, panels):
        ax.set_facecolor(SURFACE)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(INK2)
        ax.tick_params(colors=INK2, labelcolor=INK2)
        ax.yaxis.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.axhline(0, color=INK2, linewidth=0.8)
        for k, ds in enumerate(DATASETS):
            rs = [next(r for r in rows if r["dataset"] == ds and r["run"] == a) for a in ARMS]
            x = np.arange(len(ARMS)) + (k - 0.5) * 0.18
            y = np.array([r[col] for r in rs])
            err = np.array([[r[col] - r[f"{col}_lo"] for r in rs], [r[f"{col}_hi"] - r[col] for r in rs]])
            ax.errorbar(x, y, yerr=err, color=COLOR[ds], marker=MARK[ds], markersize=8, linestyle="none",
                        capsize=3, elinewidth=1.4, markeredgecolor=SURFACE, markeredgewidth=1.2, label=LABEL[ds])
            for xi, yi in zip(x, y):
                ax.annotate(f"{yi:.2f}", (xi, yi), xytext=(9 if k else -9, 0), textcoords="offset points",
                            ha="left" if k else "right", va="center", fontsize=8, color=INK2)
        ax.set_xticks(range(len(ARMS)), [tick.get(a, a) for a in ARMS])
        ax.set_xlim(-0.6, len(ARMS) - 0.4)
        ax.set_ylabel(ylabel, color=INK)
        ax.set_title(title, color=INK, loc="left", fontsize=11)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=2, frameon=False, fontsize=9,
               labelcolor=INK)
    fig.text(0.01, 0.005, f"DiT-XL/2, cfg 1, class 0 for both datasets; start: real latent noised to "
             f"t{'/'.join(map(str, sorted(start)))} (step 83). 95% bootstrap CIs over images (B: paired). "
             f"Jobs {jobs['crystal']} (crystals), {jobs['dog']} (dogs). Exploratory.", color=INK2, fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 0.94))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=SURFACE)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--crystal", required=True)
    ap.add_argument("--dog", required=True)
    ap.add_argument("--out", type=Path, default=Path("lanes/p1/results/transfer_dit.csv"))
    ap.add_argument("--fig", type=Path, default=Path("lanes/p1/results/fig1g_transfer_dit.png"))
    args = ap.parse_args(argv)
    from lanes.p1.summarize import _write, job_dir
    jobs = {"crystal": job_dir(args.crystal), "dog": job_dir(args.dog)}
    for p in (args.out, args.fig):
        if p.exists():
            raise SystemExit(f"ERROR: {p} exists; this script only writes new files (delete it to regenerate)")
    rows = [r for ds in DATASETS for r in rows_for(jobs[ds], ds)]
    _write(args.out, COLUMNS, rows)
    plot(rows, args.fig, {k: v.name for k, v in jobs.items()})
    print(f"wrote {args.out} ({len(rows)} rows) and {args.fig}")
    for ds in DATASETS:
        rs = {r["run"]: r for r in rows if r["dataset"] == ds}
        print(f"{ds}: codec {rs['recon']['psnr_real']:.2f} dB, reference (t{rs['reference']['start_timestep']}) "
              f"{rs['reference']['psnr_real']:.2f} dB; "
              + "; ".join(f"{a}: drop {rs[a]['psnr_drop']:+.2f} dB, MAD vs ref {rs[a]['mad_vs_ref']:.2f}" for a in ARMS))


if __name__ == "__main__":
    main()
