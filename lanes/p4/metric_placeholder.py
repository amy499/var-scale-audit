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

METRIC_NAMES = ("p4_placeholder_rel_l2", "p4_placeholder_mse")


def metric_specs():
    """The MetricSpecs for these numbers, so the figure code never hardcodes a metric name."""
    from lanes.p4.plotting import MetricSpec
    return [MetricSpec("p4_placeholder_rel_l2", direction="lower_is_better", level="per_image",
                       label="departure from baseline (rel L2)"),
            MetricSpec("p4_placeholder_mse", direction="lower_is_better", level="per_image",
                       label="departure from baseline (MSE)")]


def row_metrics(run_dir, baseline_dir) -> list[dict]:
    """One pair of numbers per image. Rows are matched by (class_id, seed), not by file order."""
    import torch
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
        a_t = torch.load(run_dir / f"{r['file']}.pt", map_location="cpu").double()
        b_t = torch.load(baseline_dir / f"{b['file']}.pt", map_location="cpu").double()
        if a_t.shape != b_t.shape:
            raise SchemaError(f"{r['file']}: shapes differ, {tuple(a_t.shape)} vs {tuple(b_t.shape)}")
        diff = a_t - b_t
        denom = float(b_t.norm())
        rows.append({"class_id": r["class_id"], "seed": r["seed"], "metric": "p4_placeholder_rel_l2",
                     "value": float(diff.norm()) / denom if denom else 0.0})
        rows.append({"class_id": r["class_id"], "seed": r["seed"], "metric": "p4_placeholder_mse",
                     "value": float((diff ** 2).mean())})
    return rows


def write_for(run_dir, baseline_dir) -> int:
    rows = row_metrics(run_dir, baseline_dir)
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

    for run_dir, baseline in targets:
        n = write_for(run_dir, baseline)
        print(f"   wrote {n} rows to {run_dir / 'metrics.csv'}   (placeholder, replaced by P3's metrics)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
