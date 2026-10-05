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

from lanes.p2.schema import SchemaError, load_run, replace_metrics  # noqa: E402

IMAGE_METRICS = ("img_mse", "img_mad255")
STATE = {"var": "f_hat", "dit": "x"}   # the tensor each model carries between stages
QUALITY = ("cls_prob", "cls_top1", "cls_top5")   # lanes/p2/quality.py; the run's value and its baseline's
SUMMARY_COLUMNS = ("run_id", "experiment", "model", "hook", "scale", "stage", "p", "severity", "n_images",
                   "img_mse", "img_psnr_db", "img_mad255", "state", "gap_l2_injection", "gap_l2_final",
                   "gap_l2_ratio", "gap_rel_l2_final",
                   *(c for q in QUALITY for c in (q, f"{q}_base")), "cls_agree_base", "kid_vs_base")
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
    replace_metrics(run_dir, IMAGE_METRICS, rows)   # keeps the trace's state_* rows
    return len(rows)


def _read(path: Path) -> list[dict]:
    if not path.is_file():
        raise SchemaError(f"{path} not found: run `python -m lanes.p2.schema collect` first")
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _mean(values):
    return sum(values) / len(values) if values else None


def kid(x, y, paired: bool = False) -> float:
    """Kernel Inception Distance: unbiased MMD^2 with the cubic polynomial kernel (x.y / d + 1)^3
    (Binkowski et al. 2018), between two sets of Inception features. Around 0 when both sets come from
    the same distribution; larger = further apart. Unbiased for small sets, but noisy with few images.

    paired: x[i] and y[i] are versions of the same image (a run and its baseline). The matching pairs
    are then left out of the cross term, as the within-set terms leave out i == j; otherwise their
    near-identity pushes the estimate below 0. A run identical to its baseline then gives exactly 0.
    """
    import numpy as np
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    m, n, d = len(x), len(y), x.shape[1]
    kxx, kyy, kxy = ((x @ x.T / d + 1) ** 3, (y @ y.T / d + 1) ** 3, (x @ y.T / d + 1) ** 3)
    if paired:
        if m != n:
            raise ValueError(f"paired KID needs equal set sizes, got {m} and {n}")
        cross = (kxy.sum() - np.trace(kxy)) / (m * (m - 1))
    else:
        cross = kxy.mean()
    return float((kxx.sum() - np.trace(kxx)) / (m * (m - 1)) + (kyy.sum() - np.trace(kyy)) / (n * (n - 1)) - 2 * cross)


def _features(run_dir: Path):
    import numpy as np
    path = run_dir / "inception_features.npz"
    return np.load(path)["feat"] if path.is_file() else None


def summary(tables, root=None) -> dict:
    """Write p2_summary.csv and p2_curves.csv into the tables directory. Means are over images.

    root: the folder `collect` was run on (run_dir in runs.csv is relative to it); default: the parent
    of the tables directory, which is where `collect` puts the tables by default.
    """
    tables = Path(tables)
    root = Path(root) if root else tables.parent
    all_runs = _read(tables / "runs.csv")
    run_dir = {r["run_id"]: root / r["run_dir"] for r in all_runs}
    runs = [r for r in all_runs if r["role"] == "intervention" and r["lane"] == "p2"]
    stage_p, last_stage = {}, {}
    for s in _read(tables / "stages.csv"):
        stage_p[(s["run_id"], s["stage"])] = s["p"]
        last_stage[s["run_id"]] = s["stage"]   # rows are in sampling order
    values = defaultdict(list)   # (run_id, metric, stage) -> per-image values
    pred = {}                    # (run_id, class_id, seed) -> classifier's top class
    for m in _read(tables / "metrics.csv"):
        values[(m["run_id"], m["metric"], m["stage"])].append(float(m["value"]))
        if m["metric"] == "cls_pred":
            pred[(m["run_id"], m["class_id"], m["seed"])] = m["value"]

    rows, curves = [], []
    for r in runs:
        rid, params = r["run_id"], json.loads(r["params"] or "{}")
        stage = str(params["stage"]) if isinstance(params.get("stage"), int) else ""   # single-stage runs only
        tensor = STATE[r["model"]]
        mse = _mean(values[(rid, "img_mse", "")])
        inj = _mean(values[(rid, f"state_l2:{tensor}", stage)]) if stage else None
        fin = _mean(values[(rid, f"state_l2:{tensor}", last_stage.get(rid, ""))])
        base = r["baseline_run_id"]
        quality = {}
        for q in QUALITY:
            quality[q] = _mean(values[(rid, q, "")])
            quality[f"{q}_base"] = _mean(values[(base, q, "")])
        same = [pred[k] == pred.get((base, *k[1:])) for k in pred if k[0] == rid and (base, *k[1:]) in pred]
        quality["cls_agree_base"] = _mean([float(s) for s in same])   # top class unchanged vs the baseline image
        fx, fy = _features(run_dir[rid]), _features(run_dir[base]) if base in run_dir else None
        quality["kid_vs_base"] = kid(fx, fy, paired=True) if fx is not None and fy is not None and min(len(fx), len(fy)) > 1 else None
        rows.append({
            "run_id": rid, "experiment": r["experiment"], "model": r["model"], "hook": params.get("hook"),
            "scale": params.get("scale"), "stage": stage, "p": stage_p.get((rid, stage), ""),
            "severity": params.get("severity"), "n_images": r["n_images"], "img_mse": mse,
            "img_psnr_db": None if mse is None else (math.inf if mse == 0 else 10 * math.log10(1.0 / mse)),
            "img_mad255": _mean(values[(rid, "img_mad255", "")]), "state": tensor,
            "gap_l2_injection": inj, "gap_l2_final": fin,
            "gap_l2_ratio": fin / inj if inj and fin is not None else None,
            "gap_rel_l2_final": _mean(values[(rid, f"state_rel_l2:{tensor}", last_stage.get(rid, ""))]),
            **quality,
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
    s.add_argument("--root", type=Path, help="the folder `collect` was run on (default: the tables' parent)")
    args = ap.parse_args(argv)
    try:
        if args.cmd == "images":
            print(f"wrote {image_metrics(args.run_dir, args.baseline)} image metric rows to {args.run_dir / 'metrics.csv'}")
        else:
            print(json.dumps(summary(args.tables, args.root)))
    except SchemaError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
