"""P4 placeholder quality measure: how far each arm row moved from its paired baseline row.

Run from the repo root, in the var-dit env (it loads the saved tensors).

    python -m lanes.p4.metric_placeholder RUN_DIR --baseline BASELINE_DIR
    python -m lanes.p4.metric_placeholder --pilot outputs/p4/pilot_var/<JOBID>    # every arm at once

**This is a stand-in, not a quality metric.** It measures distance from the baseline image, which says
an intervention changed something, not whether the result is good. P3 owns the calibrated
quality/semantic metrics; when they land they replace these through the same contract
(`lanes/p4/plotting.py` MetricSpec), with no change to the figure code.

Numbers are appended to RUN_DIR/metrics.csv in P2's long format (lanes/p2/SCHEMA.md section 4), with
`replace_metrics` so re-running overwrites rather than duplicates. Following lanes/p2/analyze.py.

    p4_placeholder_rel_l2   ||arm - baseline|| / ||baseline||, per image, on the raw output tensor
    p4_placeholder_mse      mean squared difference, per image, on the raw output tensor

Both are "lower is better" only in the sense that 0 means unchanged; a larger value is a larger
departure from the baseline, which for a protect arm is bad and for a damage arm is the point.
"""

import argparse
import sys
from pathlib import Path

sys.dont_write_bytecode = True

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from lanes.p2.schema import SchemaError, load_run, replace_metrics  # noqa: E402

REL_L2, MSE = "p4_placeholder_rel_l2", "p4_placeholder_mse"
METRIC_NAMES = (REL_L2, MSE)
METRIC_LABELS = {REL_L2: "departure from baseline (rel L2)", MSE: "departure from baseline (MSE)"}


def metric_specs():
    """The MetricSpecs for these numbers, so the figure code never hardcodes a metric name."""
    from lanes.p4.plotting import MetricSpec
    return [MetricSpec(name, direction="lower_is_better", level="per_image", label=METRIC_LABELS[name])
            for name in METRIC_NAMES]


def _tensor(path: Path, cache: dict | None):
    """The saved output tensor as float64, reused from `cache` when the same file comes round again."""
    import torch
    if cache is not None and path in cache:
        return cache[path]
    loaded = torch.load(path, map_location="cpu").double()
    if cache is not None:
        cache[path] = loaded
    return loaded


def row_metrics(run_dir, baseline_dir, cache: dict | None = None) -> list[dict]:
    """One pair of numbers per image. Rows are matched by (class_id, seed), not by file order.

    `cache` holds the **baseline's** tensors across calls, because every arm is compared with the same
    baseline and would otherwise reload those 16 files once per arm. An arm's own tensors are read
    once and are deliberately not cached: keeping them would grow the dict by every image in the
    sweep for no reuse.
    """
    run_dir, baseline_dir = Path(run_dir), Path(baseline_dir)
    run, base = load_run(run_dir), load_run(baseline_dir)
    base_by_key = {(r["class_id"], r["seed"]): r for r in base["rows"]}
    missing = [(r["class_id"], r["seed"]) for r in run["rows"]
               if (r["class_id"], r["seed"]) not in base_by_key]
    if missing:
        raise SchemaError(f"{run_dir} has images the baseline {baseline_dir} does not: {missing[:3]}")

    rows = []
    for r in run["rows"]:
        b = base_by_key[(r["class_id"], r["seed"])]
        a_t = _tensor(run_dir / f"{r['file']}.pt", None)
        b_t = _tensor(baseline_dir / f"{b['file']}.pt", cache)
        if a_t.shape != b_t.shape:
            raise SchemaError(f"{r['file']}: shapes differ, {tuple(a_t.shape)} vs {tuple(b_t.shape)}")
        diff = a_t - b_t
        denom = float(b_t.norm())
        rows.append({"class_id": r["class_id"], "seed": r["seed"], "metric": REL_L2,
                     "value": float(diff.norm()) / denom if denom else 0.0})
        rows.append({"class_id": r["class_id"], "seed": r["seed"], "metric": MSE,
                     "value": float((diff ** 2).mean())})
    return rows


def write_for(run_dir, baseline_dir, cache: dict | None = None) -> int:
    rows = row_metrics(run_dir, baseline_dir, cache)
    replace_metrics(run_dir, METRIC_NAMES, rows)
    return len(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", nargs="?", type=Path)
    ap.add_argument("--baseline", type=Path, help="the no-hook baseline run directory")
    ap.add_argument("--pilot", type=Path, help="a pilot root: do every arm against its baseline")
    args = ap.parse_args(argv)

    if args.pilot:
        from lanes.p4.verify_pairing import find_runs
        baseline, arm_dirs = find_runs(args.pilot)
        targets = [(d, baseline) for d in arm_dirs] + [(baseline, baseline)]
    elif args.run_dir and args.baseline:
        targets = [(args.run_dir, args.baseline)]
    else:
        ap.error("give RUN_DIR --baseline BASELINE_DIR, or --pilot PILOT_ROOT")

    cache: dict = {}
    for run_dir, baseline in targets:
        n = write_for(run_dir, baseline, cache)
        print(f"   wrote {n} rows to {run_dir / 'metrics.csv'}   (placeholder, replaced by P3's metrics)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
