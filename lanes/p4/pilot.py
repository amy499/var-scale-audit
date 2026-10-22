"""P4 four-arm sweep driver: one candidate band for Gate B, all bands for Phase C.

Run from the repo root. Needs no torch itself -- it shells out to `runner.generate`, one process per
model, as everything that touches both models must (runner/upstream.py).

    python -m lanes.p4.pilot plan --model var --band middle
    python -m lanes.p4.pilot run --model var --band middle --out-root outputs/p4/pilot/$JOB
    python -m lanes.p4.pilot plan-full --model var
    python -m lanes.p4.pilot run-full --model var --manifest manifest/<frozen>.csv \
        --out-root outputs/p4/full_var/$JOB

`plan` / `run` make the 10-run Gate B pilot for one band. `plan-full` / `run-full` make the
Phase C predeclared sweep: one no-hook baseline plus protect, damage and control across every frozen
band and budget. The baseline is always produced first; every arm shares its config, manifest and
batch size 16 (docs/HANDOVER.md section 3).

VAR arms are hooked runs (lanes/p4/hooks.py, lambda = 0); DiT arms are `--skip-timesteps` runs over
an arm's stage set. `run.json` does not record this repo's branch, so each run directory also gets a
`pilot.json` carrying `git log -1 --oneline`, the arm record and merged-window structure.

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


def _stages_for(model: str, config: Path):
    cfg_model, stages = fixtures.stages_for_config(config)
    if cfg_model != model:
        raise ValueError(f"{config} is a {cfg_model} config but --model is {model}")
    return stages


def _band_plans(model: str, stages, band: str, fractions, seed: int) -> tuple[list[int], dict]:
    budgets = arms.budget_grid(model, stages, band, fractions)
    plans = {m: arms.arm_plan(model, stages, band, m, seed=seed) for m in budgets}
    degenerate = [m for m, plan in plans.items() if plan["control"].get("degenerate")]
    if degenerate:
        raise ValueError(f"{model} {band} band: the control arm could not be separated from protect or "
                         f"damage at budget(s) {degenerate}. Running it would add a fourth arm that "
                         "tests nothing. Choose another budget, band or control seed "
                         "(lanes/p4/comparison_logic.md section 3).")
    return budgets, plans


def _intervention_run(model: str, config: Path, manifest: Path, out_dir: Path, band: str, m: int,
                      arm: str, record: dict) -> dict:
    if model == "var":
        cmd = _cmd(config, manifest, out_dir, hook=HOOK)
        env = {arms.ENV_STAGES: ",".join(str(s) for s in record["stages"]),
               arms.ENV_LAMBDA: f"{arms.VAR_GATE_B_LAMBDA:g}"}
    else:
        cmd, env = _cmd(config, manifest, out_dir, skip=record["stages"]), {}
    return {"arm": arm, "m": m, "band": band, "out_dir": str(out_dir), "record": record,
            "cmd": cmd, "env": env}


def _baseline_run(config: Path, manifest: Path, out_root: Path, record: dict) -> dict:
    return {"arm": arms.BASELINE, "m": 0, "band": record["band"],
            "out_dir": str(out_root / arms.BASELINE), "record": record,
            "cmd": _cmd(config, manifest, out_root / arms.BASELINE), "env": {}}


def plan_runs(model: str, band: str, config: Path, manifest: Path, out_root: Path,
              fractions=arms.FRACTIONS, seed: int = arms.CONTROL_SEED) -> list[dict]:
    """The 10-run Gate B pilot for one band, baseline first."""
    stages = _stages_for(model, config)
    budgets, plans = _band_plans(model, stages, band, fractions, seed)
    runs = [_baseline_run(config, manifest, out_root, plans[budgets[0]][arms.BASELINE])]
    for m in budgets:
        for arm in arms.INTERVENED_ARMS:
            runs.append(_intervention_run(model, config, manifest, out_root / f"{arm}_m{m}",
                                          band, m, arm, plans[m][arm]))
    return runs


def plan_full_runs(model: str, config: Path, manifest: Path, out_root: Path,
                   fractions=arms.FRACTIONS, seed: int = arms.CONTROL_SEED,
                   band_names=bands.BANDS) -> list[dict]:
    """The Phase C predeclared sweep: one baseline, then every band/arm/budget condition."""
    if not band_names:
        raise ValueError("the full sweep needs at least one candidate band")
    if any(band not in bands.BANDS for band in band_names):
        raise ValueError(f"full-sweep bands must be drawn from {bands.BANDS}, got {tuple(band_names)}")
    stages = _stages_for(model, config)
    per_band = {band: _band_plans(model, stages, band, fractions, seed) for band in band_names}
    first_band = band_names[0]
    first_budget = per_band[first_band][0][0]
    baseline = {**per_band[first_band][1][first_budget][arms.BASELINE], "band": "all",
                "k": len(stages)}
    runs = [_baseline_run(config, manifest, out_root, baseline)]
    for band in band_names:
        budgets, plans = per_band[band]
        for m in budgets:
            for arm in arms.INTERVENED_ARMS:
                runs.append(_intervention_run(model, config, manifest,
                                              out_root / band / f"{arm}_m{m}",
                                              band, m, arm, plans[m][arm]))
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
    band_names = tuple(dict.fromkeys(r["band"] for r in runs[1:]))
    if len(band_names) == 1:
        title = f"{head['model']} {band_names[0]} band: n = {head['n']} stages"
    else:
        title = f"{head['model']} predeclared bands {', '.join(band_names)}: n = {head['n']} stages"
    lines = [f"{title}, batch size {BATCH_SIZE}", f"{len(runs)} runs, baseline first:"]
    for r in runs:
        rec = r["record"]
        target = Path(r["out_dir"]).name if r["arm"] == arms.BASELINE else f"{r['band']}/{Path(r['out_dir']).name}"
        lines.append(f"  {r['arm']:<9} m={r['m']:<3} fraction={rec['fraction']:.3f} "
                     f"windows={rec['n_windows']:<3} -> {target}")
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
        band_names = tuple(dict.fromkeys(r["band"] for r in runs[1:]))
        (out_root / PLAN_FILE).write_text(json.dumps(
            {"kind": "full" if len(band_names) > 1 else "pilot", "model": head["model"],
             "band": band_names[0] if len(band_names) == 1 else None, "bands": list(band_names),
             "batch_size": BATCH_SIZE, "code_commit": commit,
             "control_seed": runs[-1]["record"].get("control_seed"),
             "runs": [{"arm": r["arm"], "m": r["m"],
                       "out_dir": Path(r["out_dir"]).relative_to(out_root).as_posix()} for r in runs]},
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
    ap.add_argument("action", choices=("plan", "run", "plan-full", "run-full"))
    ap.add_argument("--model", required=True, choices=("var", "dit"))
    ap.add_argument("--band", default="middle", choices=bands.BANDS,
                    help="candidate band for plan/run; plan-full/run-full use every frozen band")
    ap.add_argument("--config", type=Path, help="default: the frozen config for this model")
    ap.add_argument("--manifest", type=Path, default=Path(DEFAULT_MANIFEST))
    ap.add_argument("--out-root", type=Path, help="default: outputs/p4/pilot/<model>")
    ap.add_argument("--seed", type=int, default=arms.CONTROL_SEED)
    ap.add_argument("--fractions", type=float, nargs="+", default=list(arms.FRACTIONS))
    ap.add_argument("--dry-run", action="store_true", help="with 'run': print the commands only")
    args = ap.parse_args(argv)

    config = args.config or fixtures.CONFIGS[args.model]
    is_full = args.action.endswith("-full")
    out_root = args.out_root or REPO / "outputs" / "p4" / ("full" if is_full else "pilot") / args.model
    if is_full:
        runs = plan_full_runs(args.model, config, args.manifest, out_root,
                              fractions=tuple(args.fractions), seed=args.seed)
    else:
        runs = plan_runs(args.model, args.band, config, args.manifest, out_root,
                         fractions=tuple(args.fractions), seed=args.seed)
    print(describe(runs))
    if args.action.startswith("plan"):
        return 0
    return 1 if run_all(runs, out_root, dry_run=args.dry_run) else 0


if __name__ == "__main__":
    sys.exit(main())
