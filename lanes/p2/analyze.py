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

        With the quality step's features (lanes/p2/quality.py), also per run: incep_cos (fidelity, P3's
        definition), validity_ratio, kid_vs_base and P3's outcome label (unchanged / re-routed / damaged /
        degraded-in-place), plus the per-image fraction of each label.
        --exclude-broken-baselines: the same without images whose baseline is a generation failure
        -> p2_summary_valid.csv, p2_curves_valid.csv

    python -m lanes.p2.analyze ci TABLES_DIR [--n-boot 1000]
        -> p2_ci.csv: 95 % bootstrap intervals over images (cls_prob, frac_damaged, incep_cos, KID) per run

    python -m lanes.p2.analyze baselines TABLES_DIR
        -> p2_baselines.csv: every baseline image's classifier result, flagged ok / other-top1 / broken

Pixel and state distances say how much changed; the quality columns say whether it is still a good image.
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
SUMMARY_COLUMNS = ("run_id", "experiment", "model", "hook", "scale", "stage", "p", "severity", "noise_seed",
                   "n_images", "n_excluded",
                   "img_mse", "img_psnr_db", "img_mad255", "state", "gap_l2_injection", "gap_l2_final",
                   "gap_l2_ratio", "gap_rel_l2_final",
                   *(c for q in QUALITY for c in (q, f"{q}_base")), "cls_agree_base", "kid_vs_base",
                   "incep_cos", "validity_ratio", "outcome",
                   "frac_unchanged", "frac_re_routed", "frac_damaged", "frac_degraded_in_place")
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


def _features(run_dir: Path) -> dict | None:
    """{(class_id, seed): Inception pool feature} from RUN_DIR/inception_features.npz (lanes/p2/quality.py)."""
    import numpy as np
    path = run_dir / "inception_features.npz"
    if not path.is_file():
        return None
    z = np.load(path)
    return {(str(int(c)), str(int(s))): f for c, s, f in zip(z["class_id"], z["seed"], z["feat"])}


def cosine_distance(a, b) -> float:
    """1 - cosine similarity; the definition of P3's incep_cos (lanes/p3/p3_metrics.py)."""
    import numpy as np
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    return float("nan") if na == 0 or nb == 0 else float(1.0 - np.dot(a, b) / (na * nb))


# P3's two-axis rule (fidelity x validity, lanes/p3/p3_metrics.py classify). Used from there when lanes/p3
# is present, otherwise this copy with the same thresholds, so P2's outcome labels match P3's.
FIDELITY_LOW, VALIDITY_LOW, KID_HIGH = 0.30, 0.50, 0.005
OUTCOMES = ("unchanged", "re-routed", "damaged", "degraded-in-place")


def classify(fidelity, validity_ratio, kid_value=None) -> str:
    try:
        from lanes.p3.p3_metrics import classify as p3_classify
        return p3_classify(fidelity, validity_ratio, kid_value)
    except ImportError:
        far = fidelity > FIDELITY_LOW
        broken = validity_ratio < VALIDITY_LOW or (kid_value is not None and kid_value > KID_HIGH)
        return OUTCOMES[(2 if broken else 1) if far else (3 if broken else 0)]


def _final_values(tables: Path) -> dict:
    """(run_id, metric) -> {(class_id, seed): value} for numbers about the final image (empty stage)."""
    out = defaultdict(dict)
    for m in _read(tables / "metrics.csv"):
        if m["stage"] == "":
            out[(m["run_id"], m["metric"])][(m["class_id"], m["seed"])] = float(m["value"])
    return out


def broken_baselines(tables, rule: str = "top5") -> set:
    """(baseline run_id, class_id, seed) of baseline images that are generation failures by `rule`:
    top5 = the image's own class is not in the classifier's top 5. Paired comparisons against them say
    nothing about corruption, so `summary --exclude-broken-baselines` leaves them out."""
    final = _final_values(Path(tables))
    base_ids = {r["run_id"] for r in _read(Path(tables) / "runs.csv") if r["role"] == "baseline"}
    return {(rid, *k) for (rid, metric), vals in final.items() if metric == "cls_top5" and rid in base_ids
            for k, v in vals.items() if v == 0}


def summary(tables, root=None, exclude_broken_baselines: bool = False) -> dict:
    """Write p2_summary.csv and p2_curves.csv into the tables directory. Means are over images.

    root: the folder `collect` was run on (run_dir in runs.csv is relative to it); default: the parent
    of the tables directory, which is where `collect` puts the tables by default.
    exclude_broken_baselines: leave out images whose baseline is a generation failure (broken_baselines);
    the outputs are then p2_summary_valid.csv / p2_curves_valid.csv, next to the full ones.
    """
    import numpy as np
    tables = Path(tables)
    root = Path(root) if root else tables.parent
    all_runs = _read(tables / "runs.csv")
    run_dir = {r["run_id"]: root / r["run_dir"] for r in all_runs}
    runs = [r for r in all_runs if r["role"] == "intervention" and r["lane"] == "p2"]
    stage_p, last_stage = {}, {}
    for s in _read(tables / "stages.csv"):
        stage_p[(s["run_id"], s["stage"])] = s["p"]
        last_stage[s["run_id"]] = s["stage"]   # rows are in sampling order
    final = _final_values(tables)
    staged = defaultdict(dict)   # (run_id, metric, stage) -> {(class_id, seed): value}
    for m in _read(tables / "metrics.csv"):
        if m["stage"] != "":
            staged[(m["run_id"], m["metric"], m["stage"])][(m["class_id"], m["seed"])] = float(m["value"])
    broken = broken_baselines(tables) if exclude_broken_baselines else set()

    rows, curves = [], []
    for r in runs:
        rid, params = r["run_id"], json.loads(r["params"] or "{}")
        base = r["baseline_run_id"]
        skip = {(c, s) for (b, c, s) in broken if b == base}

        def mean(d):   # mean over the images kept for this run
            return _mean([v for k, v in d.items() if k not in skip])

        stage = str(params["stage"]) if isinstance(params.get("stage"), int) else ""   # single-stage runs only
        tensor = STATE[r["model"]]
        mse = mean(final[(rid, "img_mse")])
        inj = mean(staged[(rid, f"state_l2:{tensor}", stage)]) if stage else None
        fin = mean(staged[(rid, f"state_l2:{tensor}", last_stage.get(rid, ""))])
        quality = {}
        for q in QUALITY:
            quality[q] = mean(final[(rid, q)])
            quality[f"{q}_base"] = mean(final[(base, q)])
        run_pred, base_pred = final[(rid, "cls_pred")], final[(base, "cls_pred")]
        quality["cls_agree_base"] = _mean([float(v == base_pred[k]) for k, v in run_pred.items()
                                           if k in base_pred and k not in skip])   # top class unchanged

        # Fidelity (incep_cos), KID and P3's outcome labels, from the Inception features of both runs.
        fx, fy = _features(run_dir[rid]), _features(run_dir[base]) if base in run_dir else None
        keys = sorted(k for k in (fx or {}) if fy and k in fy and k not in skip)
        cos = {k: cosine_distance(fx[k], fy[k]) for k in keys}
        quality["incep_cos"] = _mean(list(cos.values()))
        quality["kid_vs_base"] = (kid(np.stack([fx[k] for k in keys]), np.stack([fy[k] for k in keys]), paired=True)
                                  if len(keys) > 1 else None)
        prob, prob_base = final[(rid, "cls_prob")], final[(base, "cls_prob")]
        if quality["incep_cos"] is not None and quality["cls_prob"] and quality["cls_prob_base"]:
            quality["validity_ratio"] = quality["cls_prob"] / quality["cls_prob_base"]
            quality["outcome"] = classify(quality["incep_cos"], quality["validity_ratio"], quality["kid_vs_base"])
        labels = [classify(cos[k], prob[k] / max(prob_base[k], 1e-6)) for k in keys if k in prob and k in prob_base]
        for o in OUTCOMES:   # per image, without KID (a set-level number): P3's "provisional" per-image label
            quality[f"frac_{o.replace('-', '_')}"] = labels.count(o) / len(labels) if labels else None

        rows.append({
            "run_id": rid, "experiment": r["experiment"], "model": r["model"], "hook": params.get("hook"),
            "scale": params.get("scale"), "stage": stage, "p": stage_p.get((rid, stage), ""),
            "severity": params.get("severity"), "noise_seed": params.get("noise_seed", 0),
            "n_images": len(keys) if keys else int(r["n_images"]) - len(skip), "n_excluded": len(skip),
            "img_mse": mse,
            "img_psnr_db": None if mse is None else (math.inf if mse == 0 else 10 * math.log10(1.0 / mse)),
            "img_mad255": mean(final[(rid, "img_mad255")]), "state": tensor,
            "gap_l2_injection": inj, "gap_l2_final": fin,
            "gap_l2_ratio": fin / inj if inj and fin is not None else None,
            "gap_rel_l2_final": mean(staged[(rid, f"state_rel_l2:{tensor}", last_stage.get(rid, ""))]),
            **quality,
        })
        for (run_id, metric, st), vals in staged.items():
            if run_id == rid and metric.startswith("state_"):
                kept = [v for k, v in vals.items() if k not in skip]
                curves.append({"run_id": rid, "model": r["model"], "hook": params.get("hook"), "injection_stage": stage,
                               "severity": params.get("severity"), "stage": st, "p": stage_p.get((rid, st), ""),
                               "metric": metric, "mean": _mean(kept), "n_images": len(kept)})
    if not rows:
        raise SchemaError(f"{tables / 'runs.csv'} has no P2 intervention runs")
    suffix = "_valid" if exclude_broken_baselines else ""
    for name, columns, data in ((f"p2_summary{suffix}.csv", SUMMARY_COLUMNS, rows),
                                (f"p2_curves{suffix}.csv", CURVE_COLUMNS, curves)):
        with (tables / name).open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
            w.writeheader()
            w.writerows(data)
    return {"summary": str(tables / f"p2_summary{suffix}.csv"), "runs": len(rows), "curve_points": len(curves),
            "excluded_images": len(broken)}


CI_COLUMNS = ("run_id", "model", "hook", "scale", "stage", "severity", "metric", "estimate", "ci_low", "ci_high",
              "n_images", "n_boot")


def ci(tables, root=None, n_boot: int = 1000, seed: int = 0, level: float = 0.95) -> dict:
    """p2_ci.csv: bootstrap confidence intervals over images for every corrupted run.

    Resamples the images with replacement n_boot times (paired: an image's run and baseline values move
    together) and reports the percentile interval of cls_prob, frac_damaged (per image, P3's rule without
    KID), incep_cos and paired KID. The estimate column is the value on all images, as in p2_summary.csv.
    """
    import numpy as np
    tables = Path(tables)
    root = Path(root) if root else tables.parent
    all_runs = _read(tables / "runs.csv")
    run_dir = {r["run_id"]: root / r["run_dir"] for r in all_runs}
    final = _final_values(tables)
    rng = np.random.default_rng(seed)
    lo_q, hi_q = 100 * (1 - level) / 2, 100 * (1 + level) / 2
    out = []
    for r in all_runs:
        if r["role"] != "intervention" or r["lane"] != "p2":
            continue
        rid, base, params = r["run_id"], r["baseline_run_id"], json.loads(r["params"] or "{}")
        fx, fy = _features(run_dir[rid]), _features(run_dir.get(base, Path("/nonexistent")))
        prob, prob_base = final[(rid, "cls_prob")], final[(base, "cls_prob")]
        keys = sorted(k for k in prob if k in prob_base and fx and fy and k in fx and k in fy)
        if len(keys) < 2:
            continue
        X = np.stack([fx[k] for k in keys]).astype(np.float64)
        Y = np.stack([fy[k] for k in keys]).astype(np.float64)
        d = X.shape[1]
        kxx, kyy, kxy = (X @ X.T / d + 1) ** 3, (Y @ Y.T / d + 1) ** 3, (X @ Y.T / d + 1) ** 3
        p = np.array([prob[k] for k in keys])
        cos = np.array([cosine_distance(fx[k], fy[k]) for k in keys])
        damaged = np.array([classify(c, pr / max(prob_base[k], 1e-6)) == "damaged" for c, pr, k in zip(cos, p, keys)],
                           dtype=float)

        def kid_of(idx):
            # Leave out every pair of positions holding the same original image, not only the diagonal:
            # a bootstrap resample repeats images, and a repeated image paired with itself would inflate
            # the within-set terms (and, being paired, the cross term) like a near-duplicate.
            other = idx[:, None] != idx[None, :]
            sub = lambda K: K[np.ix_(idx, idx)][other].mean()   # noqa: E731
            return sub(kxx) + sub(kyy) - 2 * sub(kxy)

        stats = {"cls_prob": lambda i: p[i].mean(), "frac_damaged": lambda i: damaged[i].mean(),
                 "incep_cos": lambda i: cos[i].mean(), "kid_vs_base": kid_of}
        allidx = np.arange(len(keys))
        boots = [rng.integers(0, len(keys), len(keys)) for _ in range(n_boot)]
        for name, f in stats.items():
            vals = np.array([f(b) for b in boots])
            out.append({"run_id": rid, "model": r["model"], "hook": params.get("hook"), "scale": params.get("scale"),
                        "stage": params.get("stage"), "severity": params.get("severity"), "metric": name,
                        "estimate": float(f(allidx)), "ci_low": float(np.percentile(vals, lo_q)),
                        "ci_high": float(np.percentile(vals, hi_q)), "n_images": len(keys), "n_boot": n_boot})
    if not out:
        raise SchemaError(f"no scored P2 runs with Inception features under {root}")
    with (tables / "p2_ci.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CI_COLUMNS)
        w.writeheader()
        w.writerows(out)
    return {"ci": str(tables / "p2_ci.csv"), "rows": len(out)}


def baselines(tables, manifest=None) -> dict:
    """p2_baselines.csv: every baseline image with its classifier result and a flag, so generation
    failures in the clean baselines can be reported separately (P3, PR #6).

    flag: ok (own class is top-1), other-top1 (own class in the top 5, not top-1: usually a neighbouring
    breed), broken (own class not in the top 5: excluded by `summary --exclude-broken-baselines`).
    image_no is the image's 1-based position within its class in the manifest (images.csv order).
    """
    tables = Path(tables)
    final = _final_values(tables)
    order = defaultdict(int)
    out = []
    base_ids = {r["run_id"] for r in _read(tables / "runs.csv") if r["role"] == "baseline"}
    for img in _read(tables / "images.csv"):
        if img["run_id"] not in base_ids:
            continue
        k = (img["class_id"], img["seed"])
        order[(img["run_id"], img["class_id"])] += 1
        get = lambda m: final[(img["run_id"], m)].get(k)   # noqa: E731
        if get("cls_prob") is None:
            continue
        flag = "ok" if get("cls_top1") == 1 else ("other-top1" if get("cls_top5") == 1 else "broken")
        out.append({"run_id": img["run_id"], "class_id": k[0], "seed": k[1],
                    "image_no": order[(img["run_id"], img["class_id"])], "file": img["file"],
                    "cls_prob": get("cls_prob"), "cls_pred": int(get("cls_pred")), "flag": flag})
    if not out:
        raise SchemaError(f"no scored baseline images in {tables}: run `python -m lanes.p2.quality` on the baselines")
    with (tables / "p2_baselines.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    counts = defaultdict(int)
    for o in out:
        counts[o["flag"]] += 1
    return {"baselines": str(tables / "p2_baselines.csv"), "images": len(out), **counts}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    i = sub.add_parser("images", help="image distance of RUN_DIR from its baseline, into RUN_DIR/metrics.csv")
    i.add_argument("run_dir", type=Path)
    i.add_argument("--baseline", type=Path, required=True)
    s = sub.add_parser("summary", help="p2_summary.csv and p2_curves.csv from collected tables")
    s.add_argument("tables", type=Path)
    s.add_argument("--root", type=Path, help="the folder `collect` was run on (default: the tables' parent)")
    s.add_argument("--exclude-broken-baselines", action="store_true",
                   help="leave out images whose baseline is a generation failure -> p2_summary_valid.csv")
    b = sub.add_parser("baselines", help="p2_baselines.csv: classifier result and flag for every baseline image")
    b.add_argument("tables", type=Path)
    c = sub.add_parser("ci", help="p2_ci.csv: bootstrap confidence intervals over images for every corrupted run")
    c.add_argument("tables", type=Path)
    c.add_argument("--root", type=Path, help="the folder `collect` was run on (default: the tables' parent)")
    c.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args(argv)
    try:
        if args.cmd == "images":
            print(f"wrote {image_metrics(args.run_dir, args.baseline)} image metric rows to {args.run_dir / 'metrics.csv'}")
        elif args.cmd == "baselines":
            print(json.dumps(baselines(args.tables)))
        elif args.cmd == "ci":
            print(json.dumps(ci(args.tables, args.root, args.n_boot)))
        else:
            print(json.dumps(summary(args.tables, args.root, args.exclude_broken_baselines)))
    except SchemaError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
