"""P4 Gate B pilot driver: baseline plus the three arms, at a swept reduction budget.

Run from the repo root. Needs no torch itself -- it shells out to `runner.generate`, one process per
model, as everything that touches both models must (runner/upstream.py).

    python -m lanes.p4.pilot plan --model var --band middle
    python -m lanes.p4.pilot run  --model var --band middle --out-root outputs/p4/pilot/$JOB

`plan` prints what `run` would do and writes nothing but the plan; `run` executes it. The no-hook
baseline is produced and kept **first**, and every arm uses the same config, manifest and batch size 16
(docs/HANDOVER.md section 3).

Per model the pilot is 10 runs: one baseline plus three arms at each of three budgets. VAR arms are
hooked runs (lanes/p4/hooks.py, lambda = 0); DiT arms are `--skip-timesteps` runs over the arm's stage
set. `run.json` does not record this repo's branch, so each run directory also gets a `pilot.json`
carrying `git log -1 --oneline`, the arm record and the merged-window structure.

Nothing here computes a result. Pairing is checked by lanes/p4/verify_pairing.py afterwards, and a
failing pairing check invalidates the comparison rather than being repaired.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from lanes.p4 import arms, bands, fixtures  # noqa: E402

DEFAULT_MANIFEST = "manifest/provisional_4x4.csv"
BATCH_SIZE = 16                      # one batch size per experiment (docs/HANDOVER.md section 3)
PLAN_FILE = "pilot_plan.json"        # the arms this sweep intended, written before the first run
HOOK = "lanes/p4/hooks.py:var_degrade"


def git_describe() -> str:
    try:
        return subprocess.run(["git", "log", "-1", "--oneline"], cwd=REPO, capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def plan_runs(model: str, band: str, config: Path, manifest: Path, out_root: Path,
              fractions=arms.FRACTIONS, seed: int = arms.CONTROL_SEED) -> list[dict]:
    """Every run of one model's pilot, baseline first. Each entry carries its arm record and command."""
    cfg_model, stages = fixtures.stages_for_config(config)
    if cfg_model != model:
        raise ValueError(f"{config} is a {cfg_model} config but --model is {model}")
    budgets = arms.budget_grid(model, stages, band, fractions)

    plans = {m: arms.arm_plan(model, stages, band, m, seed=seed) for m in budgets}
    degenerate = [m for m, plan in plans.items() if plan["control"].get("degenerate")]
    if degenerate:
        raise ValueError(f"{model} {band} band: the control arm could not be separated from protect or "
                         f"damage at budget(s) {degenerate}. Running it would add a fourth arm that "
                         "tests nothing. Choose another budget, band or control seed "
                         "(lanes/p4/comparison_logic.md section 5).")
    runs = [{"arm": arms.BASELINE, "m": 0, "band": band, "out_dir": str(out_root / arms.BASELINE),
             "record": plans[budgets[0]][arms.BASELINE],
             "cmd": _cmd(config, manifest, out_root / arms.BASELINE), "env": {}}]
    for m in budgets:
        plan = plans[m]
        for arm in arms.INTERVENED_ARMS:
            record = plan[arm]
            out_dir = out_root / f"{arm}_m{m}"
            if model == "var":
                cmd = _cmd(config, manifest, out_dir, hook=HOOK)
                env = {arms.ENV_STAGES: ",".join(str(s) for s in record["stages"]),
                       arms.ENV_LAMBDA: f"{arms.VAR_GATE_B_LAMBDA:g}"}
            else:
                cmd = _cmd(config, manifest, out_dir, skip=record["stages"])
                env = {}
            runs.append({"arm": arm, "m": m, "band": band, "out_dir": str(out_dir),
                         "record": record, "cmd": cmd, "env": env})
    return runs


def clean_env(extra: dict | None = None) -> dict:
    """The environment a generate call runs in: this process's, minus any stale arm variables."""
    extra = extra or {}
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", **extra}
    for key in arms.ARM_ENV:
        if key not in extra:
            env.pop(key, None)
    return env


def _cmd(config: Path, manifest: Path, out_dir: Path, hook=None, skip=(),
         batch_size: int = BATCH_SIZE) -> list[str]:
    cmd = [sys.executable, "-m", "runner.generate", "--config", str(config),
           "--manifest", str(manifest), "--batch-size", str(batch_size), "--out-dir", str(out_dir)]
    if hook:
        cmd += ["--hook", hook]
    if skip:
        cmd += ["--skip-timesteps", *[str(s) for s in skip]]
    return cmd


def describe(runs: list[dict]) -> str:
    head = runs[0]["record"]
    lines = [f"{head['model']} {head['band']} band: n = {head['n']} stages, band holds k = {head['k']}, "
             f"batch size {BATCH_SIZE}", f"{len(runs)} runs, baseline first:"]
    for r in runs:
        rec = r["record"]
        lines.append(f"  {r['arm']:<9} m={r['m']:<3} fraction={rec['fraction']:.3f} "
                     f"windows={rec['n_windows']:<3} -> {Path(r['out_dir']).name}")
    return "\n".join(lines)


def run_all(runs: list[dict], out_root: Path, dry_run: bool = False) -> int:
    if not dry_run:
        out_root.mkdir(parents=True, exist_ok=True)
    commit = git_describe()
    if not dry_run:
        # Written before the first run so that a sweep killed by walltime, or an arm that failed, is
        # detectable afterwards: verify_pairing fails any planned arm whose directory has no run.json.
        # Without this a partial pilot is indistinguishable from a complete one.
        head = runs[0]["record"]
        (out_root / PLAN_FILE).write_text(json.dumps(
            {"model": head["model"], "band": head["band"], "batch_size": BATCH_SIZE,
             "code_commit": commit, "control_seed": runs[-1]["record"].get("control_seed"),
             "runs": [{"arm": r["arm"], "m": r["m"], "out_dir": Path(r["out_dir"]).name} for r in runs]},
            indent=2))
    failures = 0
    for i, r in enumerate(runs, 1):
        out_dir = Path(r["out_dir"])
        print(f"\n[{i}/{len(runs)}] {r['arm']} m={r['m']} -> {out_dir}", flush=True)
        print("   " + " ".join(r["cmd"]), flush=True)
        if dry_run:
            continue
        t0 = time.time()
        proc = subprocess.run(r["cmd"], cwd=REPO, env=clean_env(r["env"]))
        seconds = round(time.time() - t0, 1)
        if proc.returncode != 0:
            print(f"   FAILED (exit {proc.returncode}) after {seconds}s", flush=True)
            failures += 1
            if r["arm"] == arms.BASELINE:
                print("   BASELINE FAILED - no arm can be paired or measured against it; stopping "
                      "rather than spending the rest of the walltime", flush=True)
                return failures
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "pilot.json").write_text(json.dumps(
            {"arm": r["arm"], "m": r["m"], "band": r["band"], "batch_size": BATCH_SIZE,
             "severity_lambda": r["record"]["severity_lambda"], "stages": list(r["record"]["stages"]),
             "windows": list(r["record"]["windows"]), "n_windows": r["record"]["n_windows"],
             "fraction": r["record"]["fraction"], "control_seed": r["record"]["control_seed"],
             "degenerate": bool(r["record"].get("degenerate")),
             "code_commit": commit, "env": r["env"], "seconds": seconds,
             "cmd": r["cmd"], "baseline_dir": str(Path(runs[0]["out_dir"]))}, indent=2))
        print(f"   done in {seconds}s", flush=True)
    print(f"\n{len(runs) - failures}/{len(runs)} runs completed   commit {commit}")
    if failures:
        print("PILOT INCOMPLETE")
    return failures


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=("plan", "run"))
    ap.add_argument("--model", required=True, choices=("var", "dit"))
    ap.add_argument("--band", default="middle", choices=bands.BANDS)
    ap.add_argument("--config", type=Path, help="default: the frozen config for this model")
    ap.add_argument("--manifest", type=Path, default=Path(DEFAULT_MANIFEST))
    ap.add_argument("--out-root", type=Path, help="default: outputs/p4/pilot/<model>")
    ap.add_argument("--seed", type=int, default=arms.CONTROL_SEED)
    ap.add_argument("--fractions", type=float, nargs="+", default=list(arms.FRACTIONS))
    ap.add_argument("--dry-run", action="store_true", help="with 'run': print the commands only")
    args = ap.parse_args(argv)

    config = args.config or fixtures.CONFIGS[args.model]
    out_root = args.out_root or REPO / "outputs" / "p4" / "pilot" / args.model
    runs = plan_runs(args.model, args.band, config, args.manifest, out_root,
                     fractions=tuple(args.fractions), seed=args.seed)
    print(describe(runs))
    if args.action == "plan":
        return 0
    return 1 if run_all(runs, out_root, dry_run=args.dry_run) else 0


if __name__ == "__main__":
    sys.exit(main())
