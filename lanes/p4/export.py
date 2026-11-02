"""P4's results as tables: a tidy table, a per-condition summary with paired CIs, and a cross-root comparison.

Run from the repo root. Standard library only; reads what lanes/p4/pilot.py and lanes/p4/score.py wrote.

    python -m lanes.p4.export tidy    ROOT --out outputs/p4/summary/<date>/<model>_<tag>_tidy.csv
    python -m lanes.p4.export summary ROOT --out outputs/p4/summary/<date>/<model>_<tag>_summary.csv
    python -m lanes.p4.export compare seed1=S1.csv seed2=S2.csv d24=D24.csv \
        --out outputs/p4/summary/<date>/robustness_compare.csv

**summary** writes one row per (model, band, arm, m): each metric's mean over the images with a 95%
bootstrap CI, the fraction m/n, and DiT's measured sampling-time saving. Protect and damage rows also
carry the **paired** contrast with the control at the same band and budget, `<metric>_vs_control`,
with its 95% CI. The contrast is computed per image -- the same (class_id, seed), so the same initial
noise -- and the images are then resampled. This is the comparison that decides whether placement
matters, and it is the one Figure 4 and the report quote.

The bootstrap draws BOOT_N resamples of the images, seeded from BOOT_SEED and the condition's name,
so re-running gives byte-identical output. Both are written into every row.

**compare** reads summary CSVs (the first is the reference) and writes, per contrast present in the
reference, both estimates, whether their signs agree and whether their CIs overlap. A disagreement is a
result to report, never a reason to re-tune.

An arm whose images do not match the control's or the baseline's (class_id, seed) set is refused: a
dropped image would quietly change what the contrast compares.
"""

import argparse
import csv
import json
import random
import sys
from pathlib import Path

sys.dont_write_bytecode = True

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from lanes.p4 import arms, bands, plotting, progress  # noqa: E402
from lanes.p4.verify_pairing import find_runs  # noqa: E402

METRICS = ("incep_cos", "cls_prob")
CONTRASTS = ("protect", "damage")           # each against the control at the same band and budget
BOOT_N = 2000
BOOT_SEED = 20261102
ALPHA = 0.05
BAND_ORDER = {"all": -1, **{band: i for i, band in enumerate(bands.BANDS)}}   # baseline's band is "all"

SUMMARY_COLUMNS = ("model", "band", "arm", "m", "fraction", "n_images", "saving",
                   *[c for metric in METRICS for c in (metric, f"{metric}_lo", f"{metric}_hi")],
                   *[c for metric in METRICS for c in (f"{metric}_vs_control", f"{metric}_vs_control_lo",
                                                       f"{metric}_vs_control_hi")],
                   "run_dir", "boot_n", "boot_seed")
COMPARE_COLUMNS = ("model", "band", "arm", "m", "metric", "ref", "ref_est", "ref_lo", "ref_hi",
                   "other", "est", "lo", "hi", "sign_agree", "ci_overlap")


class ExportError(ValueError):
    """A root whose runs cannot be summarized honestly, with the reason named."""


# ---------------------------------------------------------------- reading a root

def _metric_table(run_dir: Path) -> dict:
    """{metric: {(class_id, seed): value}} for the final-image metrics of one run."""
    path = run_dir / "metrics.csv"
    if not path.is_file():
        raise ExportError(f"{path} does not exist; score the root first (python -m lanes.p4.score)")
    out = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("stage"):
                continue
            out.setdefault(row["metric"], {})[(int(row["class_id"]), int(row["seed"]))] = float(row["value"])
    return out


def load_root(root) -> dict:
    """Every run under a root with its pilot.json metadata, metrics and sampling time."""
    root = Path(root)
    baseline, arm_dirs = find_runs(root)
    runs = []
    for d in [baseline, *arm_dirs]:
        info_path = d / "pilot.json"
        if not info_path.is_file():
            raise ExportError(f"{info_path} does not exist; this run was not produced by lanes/p4/pilot.py")
        info = json.loads(info_path.read_text())
        record = progress.load_run(d)
        runs.append({"dir": d, "rel": d.relative_to(root).as_posix(), "arm": info["arm"],
                     "band": info["band"], "m": int(info["m"]), "fraction": float(info["fraction"]),
                     "stages": list(info.get("stages", [])), "metrics": _metric_table(d),
                     "seconds": (record.get("seconds") or {}).get("sample")})
    model = progress.model_of(progress.load_run(baseline))
    return {"root": root, "model": model, "baseline": runs[0], "arms": runs[1:]}


# ---------------------------------------------------------------- statistics

def _rng(*parts) -> random.Random:
    return random.Random(":".join(str(p) for p in (BOOT_SEED, *parts)))


def bootstrap_ci(values: list[float], rng: random.Random, n: int = BOOT_N) -> tuple[float, float]:
    """Percentile 95% CI of the mean, resampling the values with replacement."""
    k = len(values)
    means = sorted(sum(rng.choices(values, k=k)) / k for _ in range(n))
    return means[int(n * ALPHA / 2)], means[int(n * (1 - ALPHA / 2)) - 1]


def _keys_equal(a: dict, b: dict, what: str):
    if set(a) != set(b):
        only_a, only_b = sorted(set(a) - set(b)), sorted(set(b) - set(a))
        raise ExportError(f"{what}: the image sets differ ({len(only_a)} only in the first, e.g. "
                          f"{only_a[:3]}; {len(only_b)} only in the second, e.g. {only_b[:3]}). "
                          "A paired contrast over a different image set is not the same contrast.")


def _values(run: dict, metric: str) -> dict:
    values = run["metrics"].get(metric)
    if not values:
        raise ExportError(f"{run['rel']}: metrics.csv has no rows for {metric!r}")
    return values


# ---------------------------------------------------------------- summary

def summary_rows(data: dict) -> list[dict]:
    model, base = data["model"], data["baseline"]
    base_seconds = base["seconds"]
    by_cond = {}
    for run in data["arms"]:
        key = (run["band"], run["m"], run["arm"])
        if key in by_cond:
            raise ExportError(f"two runs claim band {run['band']} m={run['m']} arm {run['arm']}: "
                              f"{by_cond[key]['rel']} and {run['rel']}")
        by_cond[key] = run

    def describe(run: dict) -> dict:
        row = {"model": model, "band": run["band"], "arm": run["arm"], "m": run["m"],
               "fraction": round(run["fraction"], 6), "run_dir": run["rel"],
               "boot_n": BOOT_N, "boot_seed": BOOT_SEED, "saving": ""}
        for metric in METRICS:
            values = _values(run, metric)
            _keys_equal(_values(base, metric), values, f"{run['rel']} vs baseline, {metric}")
            ordered = [values[k] for k in sorted(values)]
            lo, hi = bootstrap_ci(ordered, _rng(model, run["band"], run["m"], run["arm"], metric))
            row.update({metric: plotting.mean(ordered), f"{metric}_lo": lo, f"{metric}_hi": hi})
            row["n_images"] = len(ordered)
        if model == "dit" and run["arm"] != arms.BASELINE and base_seconds and run["seconds"]:
            row["saving"] = 1.0 - run["seconds"] / base_seconds
        return row

    rows = [describe(base)]
    for band, m, arm in sorted(by_cond, key=lambda k: (_band_order(k[0]), k[1], arms.ARMS.index(k[2]))):
        run = by_cond[(band, m, arm)]
        row = describe(run)
        if arm in CONTRASTS:
            control = by_cond.get((band, m, "control"))
            if control is None:
                raise ExportError(f"{run['rel']}: no control arm at band {band} m={m}; the contrast "
                                  "that decides whether placement matters cannot be computed")
            for metric in METRICS:
                a, c = _values(run, metric), _values(control, metric)
                _keys_equal(a, c, f"{run['rel']} vs {control['rel']}, {metric}")
                diffs = [a[k] - c[k] for k in sorted(a)]
                lo, hi = bootstrap_ci(diffs, _rng(model, band, m, arm, "vs_control", metric))
                row.update({f"{metric}_vs_control": plotting.mean(diffs),
                            f"{metric}_vs_control_lo": lo, f"{metric}_vs_control_hi": hi})
        rows.append(row)
    return rows


def _band_order(band: str) -> int:
    return BAND_ORDER.get(band, len(BAND_ORDER))


def _fmt(value) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return "" if value is None else str(value)


def write_rows(rows: list[dict], columns, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
        w.writeheader()
        for row in rows:
            w.writerow({k: _fmt(row.get(k)) for k in columns})
    return path


def read_summary(path) -> list[dict]:
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------- tidy

def tidy_rows(data: dict) -> list[dict]:
    """plotting.TIDY_COLUMNS rows for every run. run_id is the run's path in the root, which carries
    the candidate band and budget (`middle/protect_m3`); the `band` column is the stage band of the
    final-image metric, as the shared template defines it."""
    specs = [plotting.P3_METRIC_SPECS[m] for m in METRICS]
    out = []
    for run in [data["baseline"], *data["arms"]]:
        metric_rows = [{"class_id": c, "seed": s, "metric": metric, "value": v, "stage": None}
                       for metric in METRICS for (c, s), v in sorted(_values(run, metric).items())]
        out.extend(plotting.tidy_from_run(run["dir"], metric_rows, specs, arm=run["arm"], lane="p4",
                                          run_id=f"{data['model']}/{run['rel']}"))
    return out


# ---------------------------------------------------------------- compare

def _num(value):
    return float(value) if value not in (None, "") else None


def compare_rows(summaries: list[tuple[str, list[dict]]]) -> list[dict]:
    """Every contrast in the first summary, against the same contrast in each other summary."""
    (ref_label, ref), others = summaries[0], summaries[1:]
    if not others:
        raise ExportError("compare needs at least two summaries: a reference and one repeat")

    def index(rows):
        return {(r["model"], r["band"], r["arm"], r["m"]): r for r in rows if r["arm"] in CONTRASTS}

    ref_index = index(ref)
    out = []
    for label, rows in others:
        other_index = index(rows)
        for key in sorted(ref_index, key=lambda k: (k[0], _band_order(k[1]), int(k[3]), k[2])):
            r, o = ref_index[key], other_index.get(key)
            for metric in METRICS:
                col = f"{metric}_vs_control"
                ref_est, ref_lo, ref_hi = (_num(r[col]), _num(r[f"{col}_lo"]), _num(r[f"{col}_hi"]))
                row = {"model": key[0], "band": key[1], "arm": key[2], "m": key[3], "metric": metric,
                       "ref": ref_label, "ref_est": ref_est, "ref_lo": ref_lo, "ref_hi": ref_hi,
                       "other": label}
                if o is None or _num(o.get(col)) is None:
                    row.update({"sign_agree": "missing", "ci_overlap": "missing"})
                else:
                    est, lo, hi = _num(o[col]), _num(o[f"{col}_lo"]), _num(o[f"{col}_hi"])
                    row.update({"est": est, "lo": lo, "hi": hi,
                                "sign_agree": _sign(est) == _sign(ref_est),
                                "ci_overlap": max(lo, ref_lo) <= min(hi, ref_hi)})
                out.append(row)
    return out


def _sign(x: float) -> int:
    return (x > 0) - (x < 0)


# ---------------------------------------------------------------- CLI

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="action", required=True)
    for name in ("tidy", "summary"):
        p = sub.add_parser(name)
        p.add_argument("root", type=Path)
        p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("compare")
    p.add_argument("summaries", nargs="+", help="LABEL=SUMMARY.csv; the first is the reference")
    p.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    if args.action == "tidy":
        rows = tidy_rows(load_root(args.root))
        plotting.write_tidy(rows, args.out)
    elif args.action == "summary":
        rows = summary_rows(load_root(args.root))
        write_rows(rows, SUMMARY_COLUMNS, args.out)
    else:
        pairs = []
        for item in args.summaries:
            label, sep, path = item.partition("=")
            if not sep:
                ap.error(f"{item!r}: give LABEL=SUMMARY.csv")
            pairs.append((label, read_summary(path)))
        rows = compare_rows(pairs)
        write_rows(rows, COMPARE_COLUMNS, args.out)
    print(f"wrote {len(rows)} rows to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
