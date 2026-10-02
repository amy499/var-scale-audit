"""P2 analysis: image damage vs the baseline, and the tables behind Figure 2 (lanes/p2/README.md).

Run from the repo root. Needs numpy and Pillow only (both in var-dit); no GPU.

    python -m lanes.p2.analyze images RUN_DIR --baseline BASELINE_DIR
        -> appends to RUN_DIR/metrics.csv, per image:
           img_mse     mean squared pixel difference from the baseline image, pixels in [0, 1]
           img_mad255  mean absolute pixel difference, pixels in 0..255
           Computed on the saved PNGs, which are on the same scale for VAR and DiT (the raw .pt are not).

    python -m lanes.p2.analyze summary TABLES_DIR
        -> from the tables of `lanes.p2.schema collect`:
           p2_summary.csv  one row per corrupted run: stage, p, severity, mean image damage, and the
                           state gap right after injection and at the end (damage map + recovery numbers)
           p2_curves.csv   per run and traced stage, the mean of every state_* metric (recovery curves)

These are simple pixel/state distances for pilots; P3's calibrated metrics go into the same metrics.csv.
"""

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.dont_write_bytecode = True

from lanes.p2.schema import METRIC_COLUMNS, SchemaError, append_metrics, load_run  # noqa: E402

IMAGE_METRICS = ("img_mse", "img_mad255")
STATE = {"var": "f_hat", "dit": "x"}   # the tensor each model carries between stages
SUMMARY_COLUMNS = ("run_id", "experiment", "model", "hook", "scale", "stage", "p", "severity", "n_images",
                   "img_mse", "img_psnr_db", "img_mad255", "state", "gap_l2_injection", "gap_l2_final",
                   "gap_l2_ratio", "gap_rel_l2_final")
CURVE_COLUMNS = ("run_id", "model", "hook", "injection_stage", "severity", "stage", "p", "metric", "mean", "n_images")


def image_metrics(run_dir, baseline_dir) -> int:
    """Append img_mse / img_mad255 per image to RUN_DIR/metrics.csv (replacing earlier values). Returns the row count."""
    import numpy as np
    from PIL import Image
    run_dir, baseline_dir = Path(run_dir), Path(baseline_dir)
    run, base = load_run(run_dir), load_run(baseline_dir)
    keys = [(r["class_id"], r["seed"]) for r in run["rows"]]
    if keys != [(r["class_id"], r["seed"]) for r in base["rows"]]:
        raise SchemaError(f"{run_dir} and {baseline_dir} do not hold the same images in the same order")

    def pixels(d: Path, stem: str):
        return np.asarray(Image.open(d / f"{stem}.png").convert("RGB"), dtype=np.float64)

    rows = []
    for r in run["rows"]:
        diff = pixels(run_dir, r["file"]) - pixels(baseline_dir, r["file"])
        rows.append({"class_id": r["class_id"], "seed": r["seed"], "metric": "img_mse",
                     "value": float(np.mean((diff / 255.0) ** 2))})
        rows.append({"class_id": r["class_id"], "seed": r["seed"], "metric": "img_mad255",
                     "value": float(np.mean(np.abs(diff)))})
    path = run_dir / "metrics.csv"
    if path.is_file():   # keep everything else (e.g. the trace's state_* rows), drop earlier image metrics
        with path.open(newline="", encoding="utf-8") as f:
            kept = [row for row in csv.DictReader(f) if row["metric"] not in IMAGE_METRICS]
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=METRIC_COLUMNS)
            w.writeheader()
            w.writerows(kept)
    append_metrics(run_dir, rows)
    return len(rows)


def _read(path: Path) -> list[dict]:
    if not path.is_file():
        raise SchemaError(f"{path} not found: run `python -m lanes.p2.schema collect` first")
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _mean(values):
    return sum(values) / len(values) if values else None


def summary(tables) -> dict:
    """Write p2_summary.csv and p2_curves.csv into the tables directory. Means are over images."""
    tables = Path(tables)
    runs = [r for r in _read(tables / "runs.csv") if r["role"] == "intervention" and r["lane"] == "p2"]
    stage_p, last_stage = {}, {}
    for s in _read(tables / "stages.csv"):
        stage_p[(s["run_id"], s["stage"])] = s["p"]
        last_stage[s["run_id"]] = s["stage"]   # rows are in sampling order
    values = defaultdict(list)   # (run_id, metric, stage) -> per-image values
    for m in _read(tables / "metrics.csv"):
        values[(m["run_id"], m["metric"], m["stage"])].append(float(m["value"]))

    rows, curves = [], []
    for r in runs:
        rid, params = r["run_id"], json.loads(r["params"] or "{}")
        stage = str(params["stage"]) if isinstance(params.get("stage"), int) else ""   # single-stage runs only
        tensor = STATE[r["model"]]
        mse = _mean(values[(rid, "img_mse", "")])
        inj = _mean(values[(rid, f"state_l2:{tensor}", stage)]) if stage else None
        fin = _mean(values[(rid, f"state_l2:{tensor}", last_stage.get(rid, ""))])
        rows.append({
            "run_id": rid, "experiment": r["experiment"], "model": r["model"], "hook": params.get("hook"),
            "scale": params.get("scale"), "stage": stage, "p": stage_p.get((rid, stage), ""),
            "severity": params.get("severity"), "n_images": r["n_images"], "img_mse": mse,
            "img_psnr_db": None if mse is None else (math.inf if mse == 0 else 10 * math.log10(1.0 / mse)),
            "img_mad255": _mean(values[(rid, "img_mad255", "")]), "state": tensor,
            "gap_l2_injection": inj, "gap_l2_final": fin,
            "gap_l2_ratio": fin / inj if inj and fin is not None else None,
            "gap_rel_l2_final": _mean(values[(rid, f"state_rel_l2:{tensor}", last_stage.get(rid, ""))]),
        })
        for (run_id, metric, st), vals in values.items():
            if run_id == rid and metric.startswith("state_") and st != "":
                curves.append({"run_id": rid, "model": r["model"], "hook": params.get("hook"), "injection_stage": stage,
                               "severity": params.get("severity"), "stage": st, "p": stage_p.get((rid, st), ""),
                               "metric": metric, "mean": _mean(vals), "n_images": len(vals)})
    if not rows:
        raise SchemaError(f"{tables / 'runs.csv'} has no P2 intervention runs")
    for name, columns, data in (("p2_summary.csv", SUMMARY_COLUMNS, rows), ("p2_curves.csv", CURVE_COLUMNS, curves)):
        with (tables / name).open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=columns)
            w.writeheader()
            w.writerows(data)
    return {"summary": str(tables / "p2_summary.csv"), "runs": len(rows), "curve_points": len(curves)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    i = sub.add_parser("images", help="image distance of RUN_DIR from its baseline, into RUN_DIR/metrics.csv")
    i.add_argument("run_dir", type=Path)
    i.add_argument("--baseline", type=Path, required=True)
    s = sub.add_parser("summary", help="p2_summary.csv and p2_curves.csv from collected tables")
    s.add_argument("tables", type=Path)
    args = ap.parse_args(argv)
    try:
        if args.cmd == "images":
            print(f"wrote {image_metrics(args.run_dir, args.baseline)} image metric rows to {args.run_dir / 'metrics.csv'}")
        else:
            print(json.dumps(summary(args.tables)))
    except SchemaError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
