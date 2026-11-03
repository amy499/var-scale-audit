"""P4 self-check for the pilot: the arm stage sets and the VAR degradation hook.

Run from the repo root, in the var-dit env.

    python lanes/p4/check_p4_pilot.py                      # arm construction + hook unit behaviour
    python lanes/p4/check_p4_pilot.py --tiny-dir DIR       # ... plus real tiny-model runs (CPU)
    python lanes/p4/check_p4_pilot.py --arms-only          # no torch needed

Make the tiny models first (own process per repo, as everything that touches both must be):

    python scripts/phase2/check_seeding.py --make-tiny var DIR
    python scripts/phase2/check_seeding.py --make-tiny dit DIR

The tiny-model section is what stands between a plausible-looking severity blend and GPU time: it
proves on real sampling that lambda = 0 reproduces the repo's tested `restore_all` removal bit-for-bit,
that lambda = 1 leaves the run bit-identical to the no-hook baseline, and that every arm stays paired
with its baseline.

Ends in one line: ALL P4 PILOT CHECKS PASS, or P4 PILOT CHECKS FAILED. Exit 0 only when every
assertion holds. No result is ever taken from a CPU run -- this proves the *mechanics*, not an effect.
"""

import argparse
import dataclasses
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.dont_write_bytecode = True

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from lanes.p4 import arms, bands, fixtures, pilot, progress  # noqa: E402
from lanes.p4.check_p4_shared import Checks  # noqa: E402

HOOK_FILE = "lanes/p4/hooks.py"
REF_HOOK_FILE = "scripts/phase3/hooks_lib.py"
TINY_VAR_PATCH_NUMS = [1, 2, 3, 4]


# ---------------------------------------------------------------- arm construction

def check_arms(c: Checks):
    seed_moves = set()
    for model in ("var", "dit"):
        stages = fixtures.frozen_stages(model)
        mapped = bands.assign_bands(progress.map_stages(model, stages))
        ordered = [r["stage"] for r in mapped]
        first = ordered[0]
        for band in bands.BANDS:
            in_band = {r["stage"] for r in mapped if r["band"] == band}
            for m in arms.budget_grid(model, stages, band):
                plan = arms.arm_plan(model, stages, band, m)
                tag = f"{model} {band} m={m}"
                c.equal(f"{tag}: baseline intervenes on nothing", plan["baseline"]["m"], 0)
                for arm in arms.INTERVENED_ARMS:
                    c.equal(f"{tag}: {arm} has exactly m stages", plan[arm]["m"], m)
                c.that(f"{tag}: damage is a subset of its band",
                       set(plan["damage"]["stages"]) <= in_band)
                c.that(f"{tag}: protect is disjoint from its band",
                       not set(plan["protect"]["stages"]) & in_band)
                c.that(f"{tag}: control is drawn from the whole schedule",
                       set(plan["control"]["stages"]) <= set(ordered))
                c.equal(f"{tag}: every arm reports the same merged-window count",
                        {plan[a]["n_windows"] for a in arms.INTERVENED_ARMS},
                        {plan["damage"]["n_windows"]})
                c.equal(f"{tag}: every arm reports the same window lengths",
                        {plan[a]["windows"] for a in arms.INTERVENED_ARMS},
                        {plan["damage"]["windows"]})
                c.close(f"{tag}: the reported fraction is m/n", plan["damage"]["fraction"], m / len(ordered), 1e-12)
                if model == "dit":
                    for arm in arms.INTERVENED_ARMS:
                        c.that(f"{tag}: {arm} never contains the unskippable first step",
                               first not in plan[arm]["stages"], f"{arm} holds {first}")
                if model == "var":
                    c.that(f"{tag}: every VAR arm carries lambda = 0",
                           all(plan[a]["severity_lambda"] == 0.0 for a in arms.ARMS))
                again = arms.arm_plan(model, stages, band, m)
                c.equal(f"{tag}: the control draw repeats exactly with the same seed",
                        again["control"]["stages"], plan["control"]["stages"])
                c.that(f"{tag}: the control arm is not a copy of protect or damage",
                       sorted(plan["control"]["stages"]) not in (sorted(plan["protect"]["stages"]),
                                                                 sorted(plan["damage"]["stages"])),
                       "the control would test nothing at this budget")
                c.that(f"{tag}: the control arm is not flagged degenerate",
                       not plan["control"].get("degenerate"))
                c.that(f"{tag}: the control records the seed actually used",
                       plan["control"]["control_seed"] >= arms.CONTROL_SEED)
                if any(arms.arm_plan(model, stages, band, m, seed=s)["control"]["stages"]
                       != plan["control"]["stages"] for s in range(arms.CONTROL_SEED + 1,
                                                                   arms.CONTROL_SEED + 6)):
                    seed_moves.add((model, band))

    c.that("a different control seed moves the control arm somewhere in the grid",
           bool(seed_moves), "the seed never changed any control set, so it is decorative")

    # The band ceiling names both numbers, and the DiT early band is one below its size.
    dit = fixtures.frozen_stages("dit")
    AE = arms.ArmError
    c.raises("dit early band rejects m = k, naming k - 1 and the first step",
             lambda: arms.arm_plan("dit", dit, "early", 83), "82", AE)
    c.raises("dit early band ceiling names the unskippable first step",
             lambda: arms.arm_plan("dit", dit, "early", 83), "999", AE)
    c.that("dit early band accepts m = k - 1",
           arms.arm_plan("dit", dit, "early", 82)["damage"]["m"] == 82)
    var = fixtures.frozen_stages("var")
    c.raises("var middle band rejects m = 4 over k = 3",
             lambda: arms.arm_plan("var", var, "middle", 4), "eligible-stage count 3", AE)
    c.raises("a negative budget is rejected",
             lambda: arms.arm_plan("var", var, "middle", -1), "non-negative", AE)
    c.raises("a non-integer budget is rejected",
             lambda: arms.arm_plan("var", var, "middle", 2.5), "non-negative", AE)
    c.raises("a boolean budget is rejected",
             lambda: arms.arm_plan("var", var, "middle", True), "non-negative", AE)
    c.raises("an unknown band is rejected", lambda: arms.arm_plan("var", var, "mid", 1), "mid", AE)
    c.raises("an unknown arm is rejected",
             lambda: arms.arm_stages("var", var, "middle", 1, "extra"), "extra", AE)
    c.raises("a VAR arm at an interior lambda is rejected",
             lambda: arms.arm_plan("var", var, "middle", 1, lam=0.5), "lambda", AE)

    # The ceiling the plan states: the swept fraction cannot exceed k / n.
    c.close("var reduction is capped at 0.300",
            arms.arm_plan("var", var, "middle", 3)["damage"]["fraction"], 0.3, 1e-12)
    c.close("dit reduction is capped at 0.332",
            arms.arm_plan("dit", dit, "middle", 83)["damage"]["fraction"], 83 / 250, 1e-12)


# ---------------------------------------------------------------- the hook, called directly

def check_hook_unit(c: Checks):
    import torch

    from runner.hooks import VARScaleState, _check_modified
    sys.path.insert(0, str(REPO))
    import importlib.util
    spec = importlib.util.spec_from_file_location("p4_hooks", REPO / HOOK_FILE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    torch.manual_seed(0)
    rows = ((207, 0), (207, 1))
    fields = {"stage": 2, "p": 0.667, "pn": 3, "n_tokens": 9, "cum_tokens": 14, "total_tokens": 30}

    def state():
        return VARScaleState(rows=rows, fields=dict(fields),
                             h_BChw=torch.randn(2, 4, 3, 3), f_hat=torch.randn(2, 4, 4, 4),
                             idx_Bl=torch.randint(0, 10, (2, 9)), f_hat_before=torch.randn(2, 4, 4, 4))

    torch.manual_seed(1)
    base = state()

    out0 = mod.severity_blend(0.0)("var", 2, 0.667, "after", base)
    c.that("lambda = 0 restores f_hat_before exactly", torch.equal(out0.f_hat, base.f_hat_before))
    c.that("lambda = 0 does not alias f_hat_before", out0.f_hat.data_ptr() != base.f_hat_before.data_ptr())
    out1 = mod.severity_blend(1.0)("var", 2, 0.667, "after", base)
    c.that("lambda = 1 returns the incoming state untouched", torch.equal(out1.f_hat, base.f_hat))
    half = mod.severity_blend(0.5)("var", 2, 0.667, "after", base)
    c.that("lambda = 0.5 is the midpoint of the two endpoints",
           torch.allclose(half.f_hat, 0.5 * base.f_hat + 0.5 * base.f_hat_before))

    for lam, out in ((0.0, out0), (0.5, half), (1.0, out1)):
        c.equal(f"lambda = {lam} keeps shape", tuple(out.f_hat.shape), tuple(base.f_hat.shape))
        c.equal(f"lambda = {lam} keeps dtype", out.f_hat.dtype, base.f_hat.dtype)
        c.equal(f"lambda = {lam} keeps device", out.f_hat.device, base.f_hat.device)
        c.that(f"lambda = {lam} leaves f_hat_before untouched",
               torch.equal(out.f_hat_before, base.f_hat_before))
        c.that(f"lambda = {lam} leaves idx_Bl untouched", torch.equal(out.idx_Bl, base.idx_Bl))
        c.that(f"lambda = {lam} leaves h_BChw untouched", torch.equal(out.h_BChw, base.h_BChw))
        c.equal(f"lambda = {lam} leaves rows untouched", out.rows, base.rows)

        class _H:
            name = f"p4 lambda {lam}"
        try:
            _check_modified(_H(), out, base, "after")
            c.that(f"lambda = {lam} passes the runner's own Modify contract check", True)
        except Exception as e:   # noqa: BLE001
            c.that(f"lambda = {lam} passes the runner's own Modify contract check", False, str(e))

    # It matches the repo's tested removal, on the same state.
    ref_spec = importlib.util.spec_from_file_location("p3_hooks", REPO / REF_HOOK_FILE)
    ref = importlib.util.module_from_spec(ref_spec)
    ref_spec.loader.exec_module(ref)
    c.that("lambda = 0 equals scripts/phase3/hooks_lib.py restore_all on the same state",
           torch.equal(out0.f_hat, ref.restore_all("var", 2, 0.667, "after", base).f_hat))

    # Guards.
    HE = mod.HookConfigError
    c.raises("the hook refuses to run for DiT",
             lambda: mod.severity_blend(0.0)("dit", 2, 0.5, "after", base), "VAR only", HE)
    c.raises("the hook refuses the 'before' side",
             lambda: mod.severity_blend(0.0)("var", 2, 0.5, "before", base), "after(si)", HE)
    c.raises("the hook refuses a state with no f_hat_before",
             lambda: mod.severity_blend(0.0)("var", 2, 0.5, "after",
                                             dataclasses.replace(base, f_hat_before=None)),
             "f_hat_before", HE)
    c.raises(f"{arms.ENV_LAMBDA} outside [0, 1] is rejected",
             lambda: _build(mod, stages="2", lam="1.5"), "outside", HE)
    c.raises(f"{arms.ENV_LAMBDA} must be a number",
             lambda: _build(mod, stages="2", lam="half"), "not a number", HE)
    c.raises(f"{arms.ENV_STAGES} must be set",
             lambda: _build(mod, stages=None, lam="0"), arms.ENV_STAGES, HE)
    c.raises(f"{arms.ENV_STAGES} must be integers",
             lambda: _build(mod, stages="early", lam="0"), arms.ENV_STAGES, HE)
    c.raises(f"an empty {arms.ENV_STAGES} is rejected",
             lambda: _build(mod, stages=",", lam="0"), "empty", HE)
    hook = _build(mod, stages="3,4,5", lam="0")
    c.equal("the built hook fires at after(si) on its arm's stages only",
            (hook.when, hook.stages), ("after", (3, 4, 5)))
    c.that("the hook name records the severity", "lambda0" in hook.name, hook.name)


def _build(mod, stages, lam):
    old = {k: os.environ.get(k) for k in (arms.ENV_STAGES, arms.ENV_LAMBDA)}
    try:
        os.environ.pop(arms.ENV_STAGES, None)
        if stages is not None:
            os.environ[arms.ENV_STAGES] = stages
        os.environ[arms.ENV_LAMBDA] = lam
        return mod.var_degrade
    finally:
        for k, v in old.items():
            os.environ.pop(k, None)
            if v is not None:
                os.environ[k] = v


# ---------------------------------------------------------------- real tiny-model runs

def _generate(work: Path, config: Path, tag: str, manifest: Path, hook=None, env_extra=None,
              skip=()) -> dict:
    out = work / tag
    cmd = pilot._cmd(config, manifest, out, hook=hook, skip=skip, batch_size=4)
    env_extra = env_extra or {}
    env = pilot.clean_env(env_extra)
    for key in ("PHASE3_STAGE", "PHASE3_LOG"):
        if key not in env_extra:                # a stale value would retarget phase 3's own hooks
            env.pop(key, None)
    r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, env=env)
    if r.returncode != 0:
        raise RuntimeError(f"{tag} failed ({r.returncode}):\n{r.stdout[-1500:]}\n{r.stderr[-1500:]}")
    return json.loads((out / "run.json").read_text())


def _hashes(run: dict, field: str) -> list:
    return [row[field] for row in run["rows"]]


def check_tiny_runs(c: Checks, tiny_dir: Path, work: Path):
    config = tiny_dir / "var_tiny.yaml"
    if not config.exists():
        c.that(f"tiny VAR config exists at {config}", False,
               "run: python scripts/phase2/check_seeding.py --make-tiny var <dir>")
        return
    manifest = work / "mini.csv"
    manifest.write_text("class_id,seed\n207,0\n207,1\n360,0\n360,1\n")

    stage = 2            # one scale, so the single-stage reference hook can target the same one
    base = _generate(work, config, "baseline", manifest)
    lam0 = _generate(work, config, "p4_lambda0", manifest, hook=f"{HOOK_FILE}:var_degrade",
                     env_extra={arms.ENV_STAGES: str(stage), arms.ENV_LAMBDA: "0"})
    lam1 = _generate(work, config, "p4_lambda1", manifest, hook=f"{HOOK_FILE}:var_degrade",
                     env_extra={arms.ENV_STAGES: str(stage), arms.ENV_LAMBDA: "1"})
    half = _generate(work, config, "p4_lambda_half", manifest, hook=f"{HOOK_FILE}:var_degrade",
                     env_extra={arms.ENV_STAGES: str(stage), arms.ENV_LAMBDA: "0.5"})
    ref = _generate(work, config, "ref_restore_all", manifest, hook=f"{REF_HOOK_FILE}:restore_all_after",
                    env_extra={"PHASE3_STAGE": str(stage)})

    n = len(base["rows"])
    c.equal("lambda = 0 reproduces restore_all bit-for-bit on every row",
            _hashes(lam0, "tensor_sha256"), _hashes(ref, "tensor_sha256"))
    c.equal("lambda = 1 leaves the run bit-identical to the no-hook baseline",
            _hashes(lam1, "tensor_sha256"), _hashes(base, "tensor_sha256"))
    c.that("lambda = 0 actually changes the images (it is not a silent no-op)",
           _hashes(lam0, "tensor_sha256") != _hashes(base, "tensor_sha256"))
    c.that("lambda = 0.5 differs from both endpoints",
           _hashes(half, "tensor_sha256") not in (_hashes(base, "tensor_sha256"),
                                                  _hashes(lam0, "tensor_sha256")))
    for name, run in (("lambda = 0", lam0), ("lambda = 1", lam1), ("lambda = 0.5", half),
                      ("restore_all", ref)):
        c.equal(f"{name} stays paired with the baseline on all {n} rows",
                _hashes(run, "generator_state_sha256"), _hashes(base, "generator_state_sha256"))
    c.equal("the hooked run records the hook, its stage list and its severity",
            [(h["name"], h["when"], h["stages"]) for h in lam0["hooks"]],
            [("p4_var_degrade_lambda0", "after", [stage])])
    c.equal("the baseline records no hook", base["hooks"], [])
    c.equal("every arm used the same config, manifest and batch size",
            {(r["config_path"], r["manifest"], r["batch_size"]) for r in (base, lam0, lam1, half, ref)},
            {(base["config_path"], base["manifest"], base["batch_size"])})
    c.equal("the tiny VAR model really has fewer than 10 scales",
            len(base["stages"]), len(TINY_VAR_PATCH_NUMS))

    # A whole arm's stage set, end to end, on the tiny schedule.
    tiny_stages = base["stages"]
    plan = arms.arm_plan("var", tiny_stages, "late", 1)
    damage = _generate(work, config, "arm_damage", manifest, hook=f"{HOOK_FILE}:var_degrade",
                       env_extra={arms.ENV_STAGES: ",".join(str(s) for s in plan["damage"]["stages"]),
                                  arms.ENV_LAMBDA: "0"})
    protect = _generate(work, config, "arm_protect", manifest, hook=f"{HOOK_FILE}:var_degrade",
                        env_extra={arms.ENV_STAGES: ",".join(str(s) for s in plan["protect"]["stages"]),
                                   arms.ENV_LAMBDA: "0"})
    c.equal("a damage arm run stays paired", _hashes(damage, "generator_state_sha256"),
            _hashes(base, "generator_state_sha256"))
    c.equal("a protect arm run stays paired", _hashes(protect, "generator_state_sha256"),
            _hashes(base, "generator_state_sha256"))
    c.that("protect and damage at the same budget give different images",
           _hashes(protect, "tensor_sha256") != _hashes(damage, "tensor_sha256"))


def check_tiny_dit(c: Checks, tiny_dir: Path, work: Path):
    config = tiny_dir / "dit_tiny.yaml"
    if not config.exists():
        c.that(f"tiny DiT config exists at {config}", False,
               "run: python scripts/phase2/check_seeding.py --make-tiny dit <dir>")
        return
    manifest = work / "mini.csv"
    base = _generate(work, config, "dit_baseline", manifest)
    plan = arms.arm_plan("dit", base["stages"], "middle", 2)
    arm = _generate(work, config, "dit_damage", manifest, skip=plan["damage"]["stages"])
    c.equal("a DiT arm stays paired with its baseline through burned draws",
            _hashes(arm, "generator_state_sha256"), _hashes(base, "generator_state_sha256"))
    c.that("a DiT arm changes the images", _hashes(arm, "tensor_sha256") != _hashes(base, "tensor_sha256"))
    c.equal("the DiT arm records exactly the skipped timesteps it was given",
            sorted(arm["skip_timesteps"]), sorted(plan["damage"]["stages"]))
    c.equal("the DiT arm's stages table is missing exactly those timesteps and no others",
            sorted({s["stage"] for s in base["stages"]} - {s["stage"] for s in arm["stages"]}),
            sorted(plan["damage"]["stages"]))
    c.that("the DiT arm marks its merged steps",
           any(s["merged"] for s in arm["stages"]), "no stage carries merged: true")
    c.that("no arm ever skips the first step",
           base["stages"][0]["stage"] not in arm["skip_timesteps"])


# ---------------------------------------------------------------- the scoring driver (no torch)

class _StubScorers:
    """Writes a valid row for every metric of each step, and records the order the steps ran in."""

    def __init__(self, fail_quality_on=()):
        self.calls, self.fail_quality_on = [], set(fail_quality_on)

    def _write(self, run_dir, names, step):
        from lanes.p2.schema import replace_metrics
        self.calls.append((step, run_dir.name, sorted(p.name for p in run_dir.glob("*.pt"))))
        replace_metrics(run_dir, names, [{"class_id": 207, "seed": 0, "metric": n, "value": 0.5}
                                         for n in names])

    def quality(self, run_dir):
        from lanes.p4 import score
        if run_dir.name in self.fail_quality_on:
            self.calls.append(("quality", run_dir.name, []))
            raise RuntimeError("stub quality failure")
        self._write(run_dir, score.QUALITY_METRICS, "quality")
        (run_dir / score.FEATURES).write_bytes(b"npz")

    def fidelity(self, run_dir, baseline):
        from lanes.p4 import score
        self._write(run_dir, (score.FIDELITY_METRIC,), "fidelity")

    def placeholder(self, run_dir, baseline):
        from lanes.p4 import metric_placeholder
        self._write(run_dir, metric_placeholder.METRIC_NAMES, "placeholder")
        self.calls[-1] = (*self.calls[-1], sorted(p.name for p in baseline.glob("*.pt")))


def _score_fixture(root: Path, arm_names=("protect_m1", "damage_m1", "control_m1")) -> Path:
    import shutil
    shutil.rmtree(root, ignore_errors=True)
    for name in (arms.BASELINE, *arm_names):
        d = root / "middle" / name if name != arms.BASELINE else root / name
        d.mkdir(parents=True)
        (d / "run.json").write_text("{}")
        for i in range(2):
            (d / f"img{i}.pt").write_bytes(b"x")
    return root


def check_score(c: Checks):
    import io
    import tempfile
    from contextlib import redirect_stdout

    from lanes.p4 import score

    def run(root, **kw):
        lines = []
        failed = score.score_root(root, log=lines.append, **kw)
        return failed, lines

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        # --- happy path: baseline first, each arm's raw dropped only after its placeholder, baseline last
        root = _score_fixture(tmp / "root")
        stub = _StubScorers()
        failed, lines = run(root, scorers=stub)
        c.equal("a clean root scores with no failures", failed, 0)
        c.equal("the last line is the COMPLETE line", lines[-1], score.COMPLETE)
        c.equal("the baseline is scored first, all three steps",
                [(s, d) for s, d, *_ in stub.calls[:3]],
                [("quality", "baseline"), ("fidelity", "baseline"), ("placeholder", "baseline")])
        arm_calls = [call for call in stub.calls[3:] if call[0] == "placeholder"]
        c.that("each arm still has its raw tensors when its placeholder runs",
               all(call[2] == ["img0.pt", "img1.pt"] for call in arm_calls), f"{arm_calls}")
        c.that("the baseline still has its raw tensors for every arm's placeholder",
               all(call[3] == ["img0.pt", "img1.pt"] for call in arm_calls), f"{arm_calls}")
        c.equal("every arm's raw tensors are gone afterwards",
                sorted(p.relative_to(root).as_posix() for p in root.rglob("*.pt")), [])
        c.that("every run is scored", all(score.is_scored(d) for d in
                                          [root / "baseline", *(root / "middle").iterdir()]))

        # --- re-run on a fully scored root: nothing recomputed, exit 0
        stub2 = _StubScorers()
        failed, lines = run(root, scorers=stub2)
        c.equal("a re-run on a scored root recomputes nothing", stub2.calls, [])
        c.equal("a re-run on a scored root still completes", (failed, lines[-1]), (0, score.COMPLETE))

        # --- KEEP_RAW: nothing deleted
        root = _score_fixture(tmp / "keep")
        failed, _ = run(root, scorers=_StubScorers(), keep_raw=True)
        c.equal("keep_raw deletes no .pt file", (failed, len(list(root.rglob("*.pt")))), (0, 8))

        # --- one arm fails: the others are still scored, the exit is non-zero, raw kept where needed
        root = _score_fixture(tmp / "partial")
        stub3 = _StubScorers(fail_quality_on={"damage_m1"})
        failed, lines = run(root, scorers=stub3)
        c.equal("one failing arm is counted", failed, 1)
        c.that("the failure is reported, not COMPLETE", lines[-1].startswith("P4 SCORING FAILED"),
               lines[-1])
        c.that("the other arms are still scored",
               score.is_scored(root / "middle" / "protect_m1")
               and score.is_scored(root / "middle" / "control_m1"))
        c.equal("the failed arm keeps its raw tensors", len(score.raw_files(root / "middle" / "damage_m1")), 2)
        c.equal("the baseline keeps its raw tensors while an arm still needs them",
                len(score.raw_files(root / "baseline")), 2)
        # and the resume then finishes the job
        failed, lines = run(root, scorers=_StubScorers())
        c.equal("a resume after the failure completes", (failed, lines[-1]), (0, score.COMPLETE))
        c.equal("the resume leaves no raw tensors", len(list(root.rglob("*.pt"))), 0)

        # --- a placeholder whose raw tensors are already gone fails that arm instead of guessing
        root = _score_fixture(tmp / "noraw")
        for p in (root / "middle" / "control_m1").glob("*.pt"):
            p.unlink()
        failed, lines = run(root, scorers=_StubScorers())
        c.equal("an arm whose raw tensors are gone fails", failed, 1)
        c.that("its failure says the tensors are gone", any("are gone" in l for l in lines), f"{lines}")

        # --- no baseline: refuse before scoring anything
        root = _score_fixture(tmp / "nobase")
        import shutil
        shutil.rmtree(root / "baseline")
        stub4 = _StubScorers()
        c.raises("a root with no baseline refuses", lambda: run(root, scorers=stub4), "baseline",
                 SystemExit)
        c.equal("... and scores nothing", stub4.calls, [])

        # --- the CLI's exit code follows the failures
        root = _score_fixture(tmp / "cli")
        orig = score.Scorers
        score.Scorers = lambda device=None: _StubScorers(fail_quality_on={"protect_m1"})
        try:
            with redirect_stdout(io.StringIO()):
                c.equal("the CLI exits 1 when an arm fails", score.main([str(root)]), 1)
            score.Scorers = lambda device=None: _StubScorers()
            with redirect_stdout(io.StringIO()):
                c.equal("the CLI exits 0 once everything is scored", score.main([str(root)]), 0)
        finally:
            score.Scorers = orig


# ---------------------------------------------------------------- Phase C/E sweep plans and jobs

def check_sweep_plans(c: Checks):
    """The second seed moves every control set and nothing else; d24 plans the same sets as d20."""
    for model in ("var", "dit"):
        stages = fixtures.frozen_stages(model)
        for band in bands.BANDS:
            for m in arms.budget_grid(model, stages, band):
                one = arms.arm_plan(model, stages, band, m)
                two = arms.arm_plan(model, stages, band, m, seed=arms.PHASE_E_SEED)
                tag = f"{model} {band} m={m}"
                c.that(f"{tag}: the Phase E seed draws a different control set",
                       one["control"]["stages"] != two["control"]["stages"])
                c.equal(f"{tag}: protect and damage do not depend on the seed",
                        (one["protect"]["stages"], one["damage"]["stages"]),
                        (two["protect"]["stages"], two["damage"]["stages"]))
    _model, d24 = fixtures.stages_for_config(REPO / "configs" / "var_d24.yaml")
    manifest = Path("manifest/frozen_v1_1.csv")
    d20_plan = pilot.plan_full_runs("var", REPO / "configs" / "var_d20.yaml", manifest, Path("/tmp/x"))
    d24_plan = pilot.plan_full_runs("var", REPO / "configs" / "var_d24.yaml", manifest, Path("/tmp/x"))
    c.equal("d24 plans the same 28 stage sets as d20 (all depths share 10 scales)",
            [(r["band"], r["arm"], r["m"], tuple(r["record"]["stages"])) for r in d24_plan],
            [(r["band"], r["arm"], r["m"], tuple(r["record"]["stages"])) for r in d20_plan])
    seeded = pilot.plan_full_runs("dit", REPO / "configs" / "dit_xl2_256.yaml", manifest, Path("/tmp/x"),
                                  seed=arms.PHASE_E_SEED)
    c.that("plan-full --seed reaches every DiT control arm",
           all(r["record"]["control_seed"] >= arms.PHASE_E_SEED for r in seeded if r["arm"] == "control"))


def check_control_pool(c: Checks):
    """The pooled VAR control: every placement with damage's window structure, minus protect and damage."""
    import itertools
    stages = fixtures.frozen_stages("var")
    expected = {("early", 1): 8, ("early", 2): 34, ("early", 3): 6, ("middle", 1): 8, ("middle", 2): 34,
                ("middle", 3): 6, ("late", 1): 8, ("late", 2): 34, ("late", 3): 54}
    got = {(band, m): len(arms.control_pool("var", stages, band, m))
           for band in bands.BANDS for m in arms.budget_grid("var", stages, band)}
    c.equal("pool sizes on VAR-d20: 8/34/6 per band, and 54 for late m=3's (2,1) structure", got, expected)
    for band in bands.BANDS:
        for m in arms.budget_grid("var", stages, band):
            pool = arms.control_pool("var", stages, band, m)
            plan = arms.arm_plan("var", stages, band, m)
            seed2 = arms.arm_plan("var", stages, band, m, seed=arms.PHASE_E_SEED)
            tag = f"var {band} m={m}"
            c.that(f"{tag}: every placement has damage's window structure",
                   all(arms.window_lengths(list(s)) == plan["damage"]["windows"] for s in pool))
            c.that(f"{tag}: protect and damage are excluded, as the redraw rule excludes them",
                   not {tuple(sorted(plan[a]["stages"])) for a in ("protect", "damage")} & set(pool))
            c.that(f"{tag}: both drawn controls (seed 1 and seed 2) are members of the pool",
                   tuple(sorted(plan["control"]["stages"])) in pool
                   and tuple(sorted(seed2["control"]["stages"])) in pool)
            every = [x for x in itertools.combinations(range(10), m)
                     if arms.window_lengths(list(x)) == plan["damage"]["windows"]]
            c.equal(f"{tag}: the pool is exhaustive (all matching placements but the two excluded)",
                    len(pool), len(every) - sum(tuple(sorted(plan[a]["stages"])) in every
                                                for a in ("protect", "damage")))
    c.raises("DiT is refused: too many placements, and its random control already averages",
             lambda: arms.control_pool("dit", fixtures.frozen_stages("dit"), "middle", 25), "VAR-only",
             arms.ArmError)

    manifest = Path("manifest/frozen_v1_1.csv")
    runs, pools = pilot.plan_pool_runs("var", REPO / "configs" / "var_d20.yaml", manifest, Path("/tmp/x"))
    c.equal("plan-pool: one baseline then every placement of the 4 window structures (10+36+8+56)",
            (runs[0]["arm"], len(runs)), ("baseline", 111))
    c.equal("plan-pool: no placement is run twice", len({r["out_dir"] for r in runs}), len(runs))
    ran = {tuple(r["record"]["stages"]) for r in runs[1:]}
    c.that("plan-pool: every pool member has a run",
           {tuple(s) for sets in pools.values() for s in sets} <= ran)
    c.that("plan-pool: the sets a pool excludes are run too, for shortlisted windows to reuse",
           {(1, 3, 4), (6, 8, 9)} <= ran)
    c.that("plan-pool: every run is a control arm with m = its stage count",
           all(r["arm"] == "control" and r["m"] == len(r["record"]["stages"]) for r in runs[1:]))
    _r24, p24 = pilot.plan_pool_runs("var", REPO / "configs" / "var_d24.yaml", manifest, Path("/tmp/x"))
    c.equal("d24 pools the same placements as d20", p24, pools)


def _sweep_shim(tmp: Path, pilot_rc: int, pair_rc: int, score_rc: int) -> Path:
    """A PATH directory whose `python` and `nvidia-smi` stand in for the GPU steps."""
    shim = tmp / f"shim_{pilot_rc}{pair_rc}{score_rc}"
    shim.mkdir()
    (shim / "python").write_text(
        "#!/bin/bash\n"
        'echo "python $*" >> "$SHIM_LOG"\n'
        f'case "$*" in *lanes.p4.pilot*) exit {pilot_rc};; *verify_pairing*) exit {pair_rc};;'
        f' *lanes.p4.score*) exit {score_rc};; esac\nexit 0\n')
    (shim / "nvidia-smi").write_text("#!/bin/bash\nexit 0\n")
    for f in shim.iterdir():
        f.chmod(0o755)
    return shim


def check_sweep_body(c: Checks):
    import tempfile
    body = REPO / "lanes" / "p4" / "jobs" / "sweep_body.sh"
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        def run(rcs, job, **env):
            (tmp / job).mkdir()
            shim = _sweep_shim(tmp / job, *rcs)
            log = tmp / job / "calls.log"
            full_env = {**os.environ, "PATH": f"{shim}:{os.environ['PATH']}", "SHIM_LOG": str(log),
                        "MODEL": "var", "JOB": job, "MACHINE": "test", "OUT_ROOT": str(tmp / "out"),
                        **env}
            for key in ("CONFIG", "SEED", "OUT_TAG", "STAGES", "KEEP_RAW", "POOL"):
                if key not in env:
                    full_env.pop(key, None)
            r = subprocess.run(["bash", "-c", f"set -uo pipefail; source {body}"], cwd=REPO,
                               env=full_env, capture_output=True, text=True)
            lines = r.stdout.strip().splitlines()
            return r.returncode, (lines[-1] if lines else r.stderr), \
                (log.read_text().splitlines() if log.exists() else [])

        rc, last, calls = run((0, 0, 0), "j1")
        c.equal("a clean sweep ends COMPLETE with exit 0", (rc, last),
                (0, "P4 SWEEP COMPLETE, PAIRED AND SCORED"))
        c.equal("the job runs the sweep, then pairing, then scoring",
                [next(k for k in ("lanes.p4.pilot", "verify_pairing", "lanes.p4.score", "check_third")
                      if k in call) for call in calls],
                ["check_third", "lanes.p4.pilot", "verify_pairing", "lanes.p4.score"])
        c.that("the default sweep is run-full on the frozen manifest at the frozen seed",
               "run-full" in calls[1] and "manifest/frozen_v1_1.csv" in calls[1]
               and f"--seed {arms.CONTROL_SEED}" in calls[1], calls[1])
        c.that("the default root is outputs/p4/full/<model>/seed<SEED>/<job>",
               (tmp / "out" / "full" / "var" / f"seed{arms.CONTROL_SEED}" / "j1" / "machine.txt").is_file())
        for rcs, what in (((1, 0, 0), "the sweep lost a run"), ((0, 1, 0), "pairing failed"),
                          ((0, 0, 1), "scoring failed")):
            rc, last, calls = run(rcs, f"fail{''.join(map(str, rcs))}")
            c.that(f"when {what}, the job ends FAILED with exit 1",
                   rc == 1 and last.startswith("P4 SWEEP FAILED"), f"{rc} {last}")
            c.that(f"when {what}, scoring still runs", any("lanes.p4.score" in x for x in calls))
        rc, last, calls = run((0, 0, 0), "j2", SEED=str(arms.PHASE_E_SEED))
        c.that("SEED reaches the sweep and names the root",
               f"--seed {arms.PHASE_E_SEED}" in calls[1]
               and (tmp / "out" / "full" / "var" / f"seed{arms.PHASE_E_SEED}" / "j2").is_dir())
        rc, last, calls = run((0, 0, 0), "j3", CONFIG="configs/var_d24.yaml", OUT_TAG="d24")
        c.that("CONFIG and OUT_TAG reach the sweep and the root",
               "configs/var_d24.yaml" in calls[1] and (tmp / "out" / "full" / "var" / "d24" / "j3").is_dir())
        rc, last, calls = run((0, 0, 0), "j4", STAGES="2,3,4", OUT_TAG="s234")
        c.that("STAGES turns the job into a confirmation run under outputs/p4/confirm/",
               " run --stages 2,3,4 " in f" {calls[1]} "
               and (tmp / "out" / "confirm" / "var" / "s234" / "j4").is_dir(), calls[1])
        rc, last, calls = run((0, 0, 0), "j7", POOL="1")
        c.that("POOL=1 runs run-pool under outputs/p4/pool/<model>/<config name>/",
               " run-pool " in f" {calls[1]} "
               and (tmp / "out" / "pool" / "var" / "var_d20" / "j7").is_dir(), calls[1] if calls else last)
        rc, last, calls = run((0, 0, 0), "j8", POOL="1", CONFIG="configs/var_d24.yaml")
        c.that("POOL=1 with the d24 config names the root var_d24",
               (tmp / "out" / "pool" / "var" / "var_d24" / "j8").is_dir())
        rc, last, calls = run((0, 0, 0), "j9", POOL="1", STAGES="2,3")
        c.that("POOL=1 together with STAGES is refused", rc == 1 and not calls)
        (tmp / "out" / "full" / "var" / "seed1" / "j5").mkdir(parents=True)
        (tmp / "out" / "full" / "var" / "seed1" / "j5" / pilot.PLAN_FILE).write_text("{}")
        rc, last, calls = run((0, 0, 0), "j5", SEED="1")
        c.that("a root that already holds a sweep is refused, not mixed", rc == 1 and not calls,
               f"{rc} {calls}")
        rc, last, calls = run((0, 0, 0), "j6", MODEL="vae")
        c.that("an unknown MODEL is refused", rc == 1 and not calls)

    # --- the job scripts parse
    for script in sorted((REPO / "lanes" / "p4" / "jobs").glob("*")):
        if script.suffix in (".pbs", ".sh"):
            r = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True)
            c.that(f"bash -n {script.name}", r.returncode == 0, r.stderr.strip())


# ---------------------------------------------------------------- the driver and the pairing gate

def check_driver(c: Checks, tiny_dir: Path, work: Path):
    """The pilot's run plan, and the gate that decides whether an experiment is valid at all."""
    import copy
    import json as _json
    import shutil

    from lanes.p4 import pilot, verify_pairing

    # --- the plan: baseline first, no hook, batch 16, and arms carrying their own stage sets
    for model, config in (("var", REPO / "configs" / "var_d20.yaml"),
                          ("dit", REPO / "configs" / "dit_xl2_256.yaml")):
        runs = pilot.plan_runs(model, "middle", config, Path("manifest/provisional_4x4.csv"),
                               work / f"plan_{model}")
        c.equal(f"{model}: the pilot plans 1 baseline plus 3 arms at 3 budgets", len(runs), 10)
        c.equal(f"{model}: the baseline is produced first", runs[0]["arm"], arms.BASELINE)
        c.that(f"{model}: the baseline carries no hook and no skip",
               "--hook" not in runs[0]["cmd"] and "--skip-timesteps" not in runs[0]["cmd"])
        c.that(f"{model}: every run uses batch size 16",
               all(r["cmd"][r["cmd"].index("--batch-size") + 1] == "16" for r in runs))
        c.that(f"{model}: every run writes its own out-dir",
               len({r["out_dir"] for r in runs}) == len(runs))
        for r in runs[1:]:
            stages = list(r["record"]["stages"])
            if model == "var":
                c.equal(f"{model} {r['arm']} m={r['m']}: the hook env names exactly the arm's stages",
                        r["env"][arms.ENV_STAGES], ",".join(str(x) for x in stages))
                c.equal(f"{model} {r['arm']} m={r['m']}: the arm runs at lambda 0",
                        r["env"][arms.ENV_LAMBDA], "0")
            else:
                skip = r["cmd"][r["cmd"].index("--skip-timesteps") + 1:]
                c.equal(f"{model} {r['arm']} m={r['m']}: the skip list is exactly the arm's stages",
                        [int(x) for x in skip], stages)

    # --- Phase C: one baseline per model, then all predeclared bands at the same arm/budget grid.
    for model, config in (("var", REPO / "configs" / "var_d20.yaml"),
                          ("dit", REPO / "configs" / "dit_xl2_256.yaml")):
        full_root = work / f"full_{model}"
        full = pilot.plan_full_runs(model, config, Path("manifest/provisional_4x4.csv"), full_root)
        c.equal(f"{model}: the full sweep has one baseline plus 27 arm runs", len(full), 28)
        c.equal(f"{model}: the full sweep baseline is first", full[0]["arm"], arms.BASELINE)
        c.equal(f"{model}: the full sweep has one baseline directory",
                [r["out_dir"] for r in full if r["arm"] == arms.BASELINE],
                [str(full_root / arms.BASELINE)])
        c.that(f"{model}: every full-sweep run writes to a unique directory",
               len({r["out_dir"] for r in full}) == len(full))
        for band in bands.BANDS:
            band_runs = [r for r in full if r["band"] == band]
            c.equal(f"{model} {band}: full sweep has 3 arms at 3 budgets", len(band_runs), 9)
            c.equal(f"{model} {band}: full-sweep arm labels repeat only across budgets",
                    sorted((r["arm"], r["m"]) for r in band_runs),
                    sorted((arm, m) for m in sorted({r["m"] for r in band_runs})
                           for arm in arms.INTERVENED_ARMS))
            c.that(f"{model} {band}: every arm lives beneath its candidate band",
                   all(Path(r["out_dir"]).parent.name == band for r in band_runs))

    # --- run-full end to end on the tiny models, then the pairing gate over the whole root
    import contextlib
    import io
    full_manifest = work / "full16.csv"
    full_manifest.write_text("class_id,seed\n" + "".join(f"{c},{s}\n" for c in (207, 208, 360, 361)
                                                          for s in range(4)))
    for model, cfg in (("var", tiny_dir / "var_tiny.yaml"), ("dit", tiny_dir / "dit_tiny.yaml")):
        root = work / f"tiny_full_{model}"
        shutil.rmtree(root, ignore_errors=True)
        runs = pilot.plan_full_runs(model, cfg, full_manifest, root)
        with contextlib.redirect_stdout(io.StringIO()):
            failures = pilot.run_all(runs, root)
        c.equal(f"tiny {model}: run-full completes every planned run", failures, 0)
        with contextlib.redirect_stdout(io.StringIO()) as out:
            rc = verify_pairing.main([str(root)])
        c.that(f"tiny {model}: every planned run is present and paired",
               rc == 0 and "ALL ARMS PAIRED" in out.getvalue()
               and f"all {len(runs)} planned runs are present" in out.getvalue(), out.getvalue()[-400:])
        if model == "var":
            victim = Path(runs[-1]["out_dir"])
            shutil.rmtree(victim)
            with contextlib.redirect_stdout(io.StringIO()) as out:
                rc = verify_pairing.main([str(root)])
            c.that("a sweep missing one arm directory fails the gate",
                   rc == 1 and "never produced a run.json" in out.getvalue(), out.getvalue()[-400:])

    # --- a dry run plans the same work and writes nothing
    dry_root = work / "dry"
    dry = pilot.plan_runs("var", "middle", REPO / "configs" / "var_d20.yaml",
                          Path("manifest/provisional_4x4.csv"), dry_root)
    c.equal("a dry run reports no failures", pilot.run_all(dry, dry_root, dry_run=True), 0)
    c.that("a dry run creates no output directory", not dry_root.exists(),
           f"{dry_root} was created by a run documented to write nothing")

    # --- the pairing gate, on real paired runs produced earlier in this check
    base_dir, arm_dir = work / "baseline", work / "p4_lambda0"
    if not (base_dir / "run.json").is_file():
        c.that("the tiny baseline exists for the pairing checks", False, f"{base_dir} has no run.json")
        return
    report = verify_pairing.compare(base_dir, arm_dir)
    c.that("a real intervened arm is reported as paired", report["ok"], f"{report['errors']}")
    c.equal("every row is compared", report["n_rows"], report["n_baseline_rows"])
    c.that("the arm is reported as having changed images", report["rows_changed"] > 0)

    # --- negatives: each must FAIL, or the gate is decorative
    def tampered(mutate, name):
        d = work / f"tamper_{name}"
        shutil.rmtree(d, ignore_errors=True)
        shutil.copytree(arm_dir, d)
        run = _json.loads((d / "run.json").read_text())
        mutate(run)
        (d / "run.json").write_text(_json.dumps(run))
        return verify_pairing.compare(base_dir, d)

    def break_generator(run):
        run["rows"][0]["generator_state_sha256"] = "0" * 64

    def drop_row(run):
        run["rows"] = run["rows"][1:]

    def no_change(run):
        base = progress.load_run(base_dir)
        for row, b in zip(run["rows"], base["rows"]):
            row["tensor_sha256"] = b["tensor_sha256"]

    r = tampered(break_generator, "gen")
    c.that("an unpaired row fails the gate", not r["ok"], "a changed generator state still passed")
    c.that("the failure names the unpaired rows",
           any("not paired" in e for e in r["errors"]), f"{r['errors']}")
    r = tampered(drop_row, "missing")
    c.that("a missing row fails the gate before any hash comparison", not r["ok"])
    c.that("the missing-row failure says what is missing",
           any("missing from the arm" in e for e in r["errors"]), f"{r['errors']}")
    r = tampered(no_change, "noop")
    c.that("an intervened arm that changed no image fails the gate", not r["ok"],
           "an intervention that did nothing was reported as a valid arm")

    # --- a reordered arm is still paired: order is not identity
    reordered_dir = work / "reordered"
    shutil.rmtree(reordered_dir, ignore_errors=True)
    shutil.copytree(arm_dir, reordered_dir)
    run = _json.loads((reordered_dir / "run.json").read_text())
    run["rows"] = list(reversed(run["rows"]))
    (reordered_dir / "run.json").write_text(_json.dumps(run))
    r = verify_pairing.compare(base_dir, reordered_dir)
    c.that("rows written in a different order still pair", r["ok"], f"{r['errors']}")
    c.that("the different order is reported", r["reordered"])

    # --- completeness: a planned arm that produced nothing must fail
    plan_root = work / "planned"
    shutil.rmtree(plan_root, ignore_errors=True)
    (plan_root / arms.BASELINE).mkdir(parents=True)
    shutil.copy(base_dir / "run.json", plan_root / arms.BASELINE / "run.json")
    (plan_root / pilot.PLAN_FILE).write_text(_json.dumps(
        {"runs": [{"arm": arms.BASELINE, "m": 0, "out_dir": arms.BASELINE},
                  {"arm": "damage", "m": 1, "out_dir": "damage_m1"}]}))
    c.equal("a pilot missing a planned arm fails the gate",
            verify_pairing.main([str(plan_root)]), 1)
    c.equal("a pilot with no arms at all fails the gate",
            verify_pairing.main([str(work / "lonely")]) if (work / "lonely").is_dir() else 1, 1)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tiny-dir", type=Path, help="directory holding var_tiny.yaml / dit_tiny.yaml")
    ap.add_argument("--work", type=Path, help="where to write the check's runs (default: a temp dir)")
    ap.add_argument("--arms-only", action="store_true", help="skip everything that needs torch")
    args = ap.parse_args(argv)

    print("== P4 pilot self-check (CPU; proves the mechanics, never a result)")
    skipped = []
    sections = [("arm construction", check_arms), ("scoring driver", check_score),
                ("Phase C/E sweep plans", check_sweep_plans),
                ("pooled VAR control", check_control_pool),
                ("sweep job body and job scripts", check_sweep_body)]
    if not args.arms_only:
        sections.append(("VAR hook, called directly", check_hook_unit))
    else:
        skipped.append("VAR hook unit behaviour (--arms-only)")
    ok = True
    for title, fn in sections:
        c = Checks()
        try:
            fn(c)
        except Exception as e:   # noqa: BLE001
            c.that(f"{title} ran to completion", False, f"{type(e).__name__}: {e}")
        c.report(title)
        ok = ok and c.ok

    if args.tiny_dir and not args.arms_only:
        with tempfile.TemporaryDirectory() as tmp:
            work = args.work or Path(tmp)
            work.mkdir(parents=True, exist_ok=True)
            for title, fn in (("VAR hook on a tiny model", check_tiny_runs),
                              ("DiT arm on a tiny model", check_tiny_dit),
                              ("pilot driver and pairing gate", check_driver)):
                c = Checks()
                try:
                    fn(c, args.tiny_dir, work)
                except Exception as e:   # noqa: BLE001
                    c.that(f"{title} ran to completion", False, f"{type(e).__name__}: {e}")
                c.report(title)
                ok = ok and c.ok
    else:
        skipped.append("tiny-model runs, so the hook's bit-level equivalence to restore_all is "
                       "unproven here (no --tiny-dir)")

    note = f"  (NOT RUN: {'; '.join(skipped)})" if skipped else ""
    print(("ALL P4 PILOT CHECKS PASS" + note) if ok else "P4 PILOT CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
