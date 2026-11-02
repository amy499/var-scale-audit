"""Score every run under one P4 root on P3's metrics, then drop the raw tensors.

Run from the repo root, in the var-dit env, on a GPU node (CPU works, slowly).

    python -m lanes.p4.score outputs/p4/full/var/seed1/<JOBID>
    KEEP_RAW=1 python -m lanes.p4.score ROOT        # keep the .pt tensors

For each run this writes into its metrics.csv, in this order:

    cls_prob, cls_top1, cls_top5, cls_pred   lanes.p2.quality (+ inception_features.npz)
    incep_cos                                lanes.p4.p3_fidelity, against the root's baseline
    p4_placeholder_rel_l2, _mse              lanes.p4.metric_placeholder (plumbing control only)

The order is forced by the data:
- The baseline is scored first, because every arm's `incep_cos` reads the baseline's Inception features.
- An arm's own `.pt` files are dropped right after its placeholder ran, because the placeholder is the
  only reader of the raw tensors. The baseline's `.pt` files go last, because the placeholder reads
  them for every arm. If any arm failed, the baseline's are kept so the failed arm can be re-scored.

Raw tensors are ~0.8 MB per image and fill the 50 GB home quota (the P2 Phase C lesson,
lanes/p2/jobs/pilot.pbs drop_raw). Each tensor's sha256 stays in run.json, and pairing reads only
run.json, so dropping them loses no evidence.

A run that already has every metric is skipped, so a job killed by walltime resumes where it stopped.
Exit code 0 and the line `P4 SCORING COMPLETE` only when every run under the root is scored.
"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from lanes.p4 import metric_placeholder  # noqa: E402

QUALITY_METRICS = ("cls_prob", "cls_top1", "cls_top5", "cls_pred")
FIDELITY_METRIC = "incep_cos"
REQUIRED_METRICS = (*QUALITY_METRICS, FIDELITY_METRIC, *metric_placeholder.METRIC_NAMES)
FEATURES = "inception_features.npz"
COMPLETE = "P4 SCORING COMPLETE"


def metrics_present(run_dir: Path) -> set:
    path = Path(run_dir) / "metrics.csv"
    if not path.is_file():
        return set()
    with path.open(newline="", encoding="utf-8") as f:
        return {row["metric"] for row in csv.DictReader(f)}


def is_scored(run_dir: Path) -> bool:
    return set(REQUIRED_METRICS) <= metrics_present(run_dir) and (Path(run_dir) / FEATURES).is_file()


def raw_files(run_dir: Path) -> list[Path]:
    return sorted(Path(run_dir).glob("*.pt"))


def drop_raw(run_dir: Path) -> int:
    files = raw_files(run_dir)
    for f in files:
        f.unlink()
    return len(files)


class Scorers:
    """The three metric steps. The default ones load torch and the classifier lazily, once."""

    def __init__(self, device: str | None = None):
        self._device, self._model, self._cache = device, None, {}

    def quality(self, run_dir: Path):
        import torch
        from lanes.p2 import quality
        if self._model is None:
            self._device = torch.device(self._device or ("cuda" if torch.cuda.is_available() else "cpu"))
            self._model, _path = quality.load_classifier(self._device)
        quality.score_run(run_dir, self._model, self._device)

    def fidelity(self, run_dir: Path, baseline: Path):
        from lanes.p4 import p3_fidelity
        p3_fidelity.write_incep_cos(run_dir, baseline)

    def placeholder(self, run_dir: Path, baseline: Path):
        metric_placeholder.write_for(run_dir, baseline, self._cache)


def _score_one(run_dir: Path, baseline: Path, scorers) -> None:
    have = metrics_present(run_dir)
    if not set(QUALITY_METRICS) <= have or not (run_dir / FEATURES).is_file():
        scorers.quality(run_dir)
    if FIDELITY_METRIC not in have:
        scorers.fidelity(run_dir, baseline)
    if not set(metric_placeholder.METRIC_NAMES) <= have:
        if not raw_files(run_dir) or not raw_files(baseline):
            raise RuntimeError(f"the placeholder needs the raw .pt tensors of {run_dir.name} and the "
                               "baseline, and they are gone; re-generate this run")
        scorers.placeholder(run_dir, baseline)


def score_root(root, *, scorers=None, keep_raw: bool = False, log=print) -> int:
    """Score every run under `root`. Returns the number of runs that failed to score."""
    from lanes.p4.verify_pairing import find_runs
    root = Path(root)
    baseline, arm_dirs = find_runs(root)          # SystemExit when there is no single baseline
    scorers = scorers or Scorers()

    def label(d: Path) -> str:
        return d.relative_to(root).as_posix()

    log(f"== scoring {1 + len(arm_dirs)} runs under {root}  (keep raw: {keep_raw})")
    t0 = time.time()
    if is_scored(baseline):
        log(f"   skip  {label(baseline)}  already scored")
    else:
        try:
            _score_one(baseline, baseline, scorers)
        except Exception as e:   # noqa: BLE001 - without the baseline nothing else can be scored
            log(f"   FAIL  {label(baseline)}  {type(e).__name__}: {e}")
            log("BASELINE NOT SCORED - every arm's incep_cos needs it; stopping")
            return 1 + len(arm_dirs)
        log(f"   ok    {label(baseline)}")

    failures = 0
    for d in arm_dirs:
        if is_scored(d):
            log(f"   skip  {label(d)}  already scored")
            if not keep_raw:
                drop_raw(d)
            continue
        try:
            _score_one(d, baseline, scorers)
        except Exception as e:   # noqa: BLE001 - one bad arm must not cost the others their scores
            failures += 1
            log(f"   FAIL  {label(d)}  {type(e).__name__}: {e}   (raw tensors kept)")
            continue
        dropped = 0 if keep_raw else drop_raw(d)
        log(f"   ok    {label(d)}" + (f"  dropped {dropped} .pt" if dropped else ""))

    if failures:
        log(f"   baseline raw tensors kept: {failures} arm(s) still need them")
    elif not keep_raw:
        dropped = drop_raw(baseline)
        if dropped:
            log(f"   dropped {dropped} baseline .pt")
    log(f"{1 + len(arm_dirs) - failures}/{1 + len(arm_dirs)} runs scored in {time.time() - t0:.0f}s")
    log(COMPLETE if not failures else f"P4 SCORING FAILED ({failures} run(s) not scored)")
    return failures


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("roots", nargs="+", type=Path, help="one or more P4 roots, each with one baseline")
    ap.add_argument("--device", help="default: cuda if available, else cpu")
    ap.add_argument("--keep-raw", action="store_true", default=os.environ.get("KEEP_RAW") == "1",
                    help="keep the .pt tensors (default: KEEP_RAW env, else drop them once scored)")
    args = ap.parse_args(argv)
    scorers = Scorers(args.device)
    failed = 0
    for root in args.roots:
        failed += score_root(root, scorers=scorers, keep_raw=args.keep_raw)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
