"""scripts/phase1/timing.py, unchanged, plus clear records for out-of-memory and missing checkpoints.

    python scripts/phase3b/timing_oom.py --config configs/var_d24.yaml --batch-sizes 16 --reps 3 --out timing.json

Same arguments and output as scripts/phase1/timing.py, which this imports and runs as it is.
- CUDA out of memory: --out gets {"out_of_memory": true, ...} with the error, the peak memory reached
  and the GPU's total memory; exit code 3.
- A checkpoint file of the config is missing: --out gets {"checkpoint_not_downloaded": true, ...};
  exit code 4 (as runner.generate). Nothing is loaded.
"""

import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "phase1"))
import timing as phase1_timing  # noqa: E402  (scripts/phase1/timing.py)
import torch  # noqa: E402

from runner.config import CheckpointMissing, load_config, require_checkpoints  # noqa: E402

EXIT_OOM = 3
EXIT_CHECKPOINT_MISSING = 4


def is_oom(e: BaseException) -> bool:
    return isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in str(e).lower()


def main():
    # Only what this wrapper needs for its own records; timing.py parses (and validates) all arguments.
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--batch-sizes", type=int, nargs="+", required=True)
    ap.add_argument("--out", required=True, type=Path)
    known, _ = ap.parse_known_args()
    base = {"config": str(known.config), "batch_sizes": known.batch_sizes}

    try:
        require_checkpoints(load_config(known.config))
    except CheckpointMissing as e:
        known.out.write_text(json.dumps({"checkpoint_not_downloaded": True, **base, "error": str(e)}, indent=2))
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(EXIT_CHECKPOINT_MISSING)

    sys.argv = [str(REPO / "scripts" / "phase1" / "timing.py"), *sys.argv[1:]]
    try:
        phase1_timing.main()
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        if not is_oom(e):
            raise
        cuda = torch.cuda.is_available()
        rec = {
            "out_of_memory": True, **base,
            "error": str(e).splitlines()[0][:500],
            "gpu": torch.cuda.get_device_name() if cuda else None,
            "peak_alloc_gib_before_failure": round(torch.cuda.max_memory_allocated() / 2**30, 2) if cuda else None,
            "total_gib": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2) if cuda else None,
        }
        known.out.write_text(json.dumps(rec, indent=2))
        print(f"OUT OF MEMORY: {known.config} at batch sizes {known.batch_sizes}: {rec['error']}", flush=True)
        sys.exit(EXIT_OOM)


if __name__ == "__main__":
    main()
