"""P4: prove every pilot arm is paired with its baseline, one PASS/FAIL line per arm.

Run from the repo root. Standard library only.

    python lanes/p4/verify_pairing.py outputs/p4/pilot_var/<JOBID>     # baseline + every arm beside it
    python lanes/p4/verify_pairing.py --baseline BASE_DIR ARM_DIR [ARM_DIR ...]

The comparison itself is P2's merged checker (`lanes.p2.schema.check_pairing`), which already compares
every frozen setting and every row's `generator_state_sha256`. This is a thin per-arm wrapper, not a
second implementation: a pairing rule that lived in two places could disagree with itself.

What it adds over calling P2 directly: it finds the arms beside a baseline, matches rows by
(class_id, seed) so a different file order is not mistaken for a different manifest, and reports one
line per arm.

**A failure invalidates the comparison; it is never repaired.** `docs/HANDOVER.md` section 3: if a row's
generator state differs, something consumed random numbers differently and the two runs are not
comparable. Report it and re-run, do not adjust anything.

Exit code 0 only when every arm passes.
"""

import argparse
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from lanes.p2.schema import check_pairing, load_run  # noqa: E402
from lanes.p4.arms import BASELINE  # noqa: E402
from lanes.p4.pilot import PLAN_FILE  # noqa: E402


def _key(row: dict) -> tuple:
    return int(row["class_id"]), int(row["seed"])


def ordered_by_manifest_key(run: dict) -> dict:
    """A copy whose rows are sorted by (class_id, seed), so file order is not mistaken for a mismatch."""
    return {**run, "rows": sorted(run["rows"], key=_key)}


def compare(baseline_dir: Path, arm_dir: Path) -> dict:
    """One arm's report: P2's pairing result plus what P4 adds (row matching, missing/extra rows)."""
    baseline, arm = load_run(baseline_dir), load_run(arm_dir)
    base_keys = [_key(r) for r in baseline["rows"]]
    arm_keys = [_key(r) for r in arm["rows"]]
    missing = sorted(set(base_keys) - set(arm_keys))
    extra = sorted(set(arm_keys) - set(base_keys))
    reordered = base_keys != arm_keys and sorted(base_keys) == sorted(arm_keys)

    if missing or extra:
        # A different image set is not a pairing question: stop before any hash comparison.
        parts = []
        if missing:
            parts.append(f"{len(missing)} rows missing from the arm, e.g. {missing[:3]}")
        if extra:
            parts.append(f"{len(extra)} rows the baseline does not have, e.g. {extra[:3]}")
        return {"ok": False, "errors": [f"{arm_dir.name}: " + "; ".join(parts)], "warnings": [],
                "n_rows": len(arm["rows"]), "n_baseline_rows": len(baseline["rows"]),
                "rows_changed": 0, "rows_identical": 0, "reordered": reordered, "compared": False}

    report = check_pairing(ordered_by_manifest_key(baseline), ordered_by_manifest_key(arm))
    report["reordered"] = reordered
    report["compared"] = True
    report["n_baseline_rows"] = len(baseline["rows"])
    intervened = bool(arm.get("skip_timesteps")) or any(
        h.get("kind") == "modify" for h in arm.get("hooks", []))
    if intervened and report["rows_changed"] == 0:
        # Paired but identical: the hook or skip list was registered and still changed nothing, so the
        # arm is not the condition it claims to be. Silently plotting it would show a real-looking
        # "no effect" result produced by an intervention that never happened.
        report["ok"] = False
        report["errors"] = [*report["errors"],
                            "this arm intervenes but no image differs from the baseline; the "
                            "intervention did not take effect"]
    return report


def planned_arms(root: Path) -> list[str] | None:
    """The arm directory names lanes/p4/pilot.py intended, or None when no plan was written."""
    plan = Path(root) / PLAN_FILE
    if not plan.is_file():
        return None
    try:
        return [r["out_dir"] for r in json.loads(plan.read_text())["runs"]]
    except (OSError, ValueError, KeyError, TypeError):
        return None


def find_runs(root) -> tuple[Path, list[Path]]:
    """The baseline directory and every arm directory beneath `root`."""
    root = Path(root)
    run_dirs = sorted({p.parent for p in root.rglob("run.json")})
    if not run_dirs:
        raise SystemExit(f"no run.json under {root}")
    baselines = [d for d in run_dirs if d.name == BASELINE]
    if len(baselines) != 1:
        raise SystemExit(f"expected exactly one directory named {BASELINE!r} under {root}, "
                         f"found {[str(d) for d in baselines]}; pass --baseline explicitly")
    return baselines[0], [d for d in run_dirs if d != baselines[0]]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", type=Path,
                    help="one pilot root (baseline discovered beneath it), or the arm dirs with --baseline")
    ap.add_argument("--baseline", type=Path, help="the no-hook baseline run directory")
    args = ap.parse_args(argv)

    if args.baseline:
        baseline, arm_dirs = args.baseline, list(args.paths)
    elif len(args.paths) == 1 and args.paths[0].is_dir():
        baseline, arm_dirs = find_runs(args.paths[0])
    else:
        ap.error("give one pilot root, or --baseline BASE_DIR plus the arm directories")

    print(f"== pairing against {baseline}")
    ok = True

    if not arm_dirs:
        # Without this, a pilot that produced only its baseline -- or none of its arms -- printed
        # "ALL ARMS PAIRED" and exited 0, which is the exact silent pass this script exists to prevent.
        print("   FAIL  no arm directories were found to compare against the baseline")
        ok = False

    root = args.paths[0] if not args.baseline and len(args.paths) == 1 else baseline.parent
    planned = planned_arms(root)
    if planned is not None:
        present = {d.name for d in arm_dirs} | {baseline.name}
        missing = [name for name in planned if name not in present]
        if missing:
            print(f"   FAIL  the pilot planned {len(planned)} runs; these never produced a run.json: "
                  f"{missing}")
            ok = False
        else:
            print(f"   ok    all {len(planned)} planned runs are present")
    else:
        print(f"   note  no {PLAN_FILE} beside the baseline, so completeness cannot be checked here")

    for arm_dir in arm_dirs:
        try:
            r = compare(baseline, arm_dir)
        except Exception as e:   # noqa: BLE001 - an unreadable run is a failure, reported like one
            print(f"   FAIL  {arm_dir.name:<16} {type(e).__name__}: {e}")
            ok = False
            continue
        status = "PASS" if r["ok"] else "FAIL"
        detail = (f"{r['n_rows']}/{r['n_baseline_rows']} rows paired   "
                  f"{r['rows_changed']} of {r['n_rows']} images changed")
        print(f"   {status}  {arm_dir.name:<16} {detail}" + ("  (file order differs)" if r["reordered"] else ""))
        for message in r["errors"]:
            print(f"         ERROR   {message}")
        for message in r["warnings"]:
            print(f"         warning {message}")
        ok = ok and r["ok"]
    print("ALL ARMS PAIRED" if ok else "PAIRING FAILED - the comparison is invalid, re-run rather than repair")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
