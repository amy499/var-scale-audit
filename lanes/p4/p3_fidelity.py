"""Write P3's `incep_cos` fidelity metric into a run's metrics.csv.

Run from the repo root, after `python -m lanes.p2.quality RUN_DIR` has written inception_features.npz.

    python -m lanes.p4.p3_fidelity RUN_DIR --baseline BASELINE_DIR

Why this exists: `python -m lanes.p3.p3_metrics fidelity` emits rows with `"stage": ""`, and
`lanes.p2.schema.append_metrics` accepts `stage` only as a number or None, so it dies with
`ValueError: invalid literal for int()`. The numbers are P3's (`fidelity_incep` is called unchanged);
this only passes `stage=None`, which is what P2's schema means by "a number about the final image".
Neither lane is edited. Delete this file once P3 or P2 fixes the mismatch.
"""

import argparse
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from lanes.p2.schema import replace_metrics  # noqa: E402
from lanes.p3.p3_metrics import fidelity_incep  # noqa: E402


def write_incep_cos(run_dir, baseline_dir) -> list[dict]:
    rows = [{**r, "stage": None} for r in fidelity_incep(run_dir, baseline_dir)]
    replace_metrics(run_dir, ("incep_cos",), rows)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--baseline", type=Path, required=True)
    args = ap.parse_args(argv)
    rows = write_incep_cos(args.run_dir, args.baseline)
    vals = [r["value"] for r in rows]
    print(json.dumps({"run_dir": str(args.run_dir), "n_images": len(vals),
                      "incep_cos_mean": round(sum(vals) / len(vals), 4), "incep_cos_max": round(max(vals), 4)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
