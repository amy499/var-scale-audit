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
    c.raises("dit early band rejects m = k, naming k - 1 and the first step",
             lambda: arms.arm_plan("dit", dit, "early", 83), "82")
    c.raises("dit early band ceiling names the unskippable first step",
             lambda: arms.arm_plan("dit", dit, "early", 83), "999")
    c.that("dit early band accepts m = k - 1",
           arms.arm_plan("dit", dit, "early", 82)["damage"]["m"] == 82)
    var = fixtures.frozen_stages("var")
    c.raises("var middle band rejects m = 4 over k = 3", lambda: arms.arm_plan("var", var, "middle", 4), "3")
    c.raises("a negative budget is rejected", lambda: arms.arm_plan("var", var, "middle", -1), "non-negative")
    c.raises("an unknown band is rejected", lambda: arms.arm_plan("var", var, "mid", 1), "mid")
    c.raises("an unknown arm is rejected", lambda: arms.arm_stages("var", var, "middle", 1, "extra"), "extra")
    c.raises("a VAR arm at an interior lambda is rejected",
             lambda: arms.arm_plan("var", var, "middle", 1, lam=0.5), "lambda")

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
    c.raises("the hook refuses to run for DiT",
             lambda: mod.severity_blend(0.0)("dit", 2, 0.5, "after", base), "VAR only")
    c.raises("the hook refuses the 'before' side",
             lambda: mod.severity_blend(0.0)("var", 2, 0.5, "before", base), "after(si)")
    c.raises(f"{arms.ENV_LAMBDA} outside [0, 1] is rejected",
             lambda: _build(mod, stages="2", lam="1.5"), "outside")
    c.raises(f"{arms.ENV_STAGES} must be set", lambda: _build(mod, stages=None, lam="0"), arms.ENV_STAGES)
    c.raises(f"{arms.ENV_STAGES} must be integers",
             lambda: _build(mod, stages="early", lam="0"), arms.ENV_STAGES)
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
    c.equal("the baseline records no hook", lam0["batch_size"] == base["batch_size"] and base["hooks"], [])
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


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tiny-dir", type=Path, help="directory holding var_tiny.yaml / dit_tiny.yaml")
    ap.add_argument("--work", type=Path, help="where to write the check's runs (default: a temp dir)")
    ap.add_argument("--arms-only", action="store_true", help="skip everything that needs torch")
    args = ap.parse_args(argv)

    print("== P4 pilot self-check (CPU; proves the mechanics, never a result)")
    sections = [("U7 arm construction", check_arms)]
    if not args.arms_only:
        sections.append(("U7 VAR hook, called directly", check_hook_unit))
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
            for title, fn in (("U7 VAR hook on a tiny model", check_tiny_runs),
                              ("U7 DiT arm on a tiny model", check_tiny_dit)):
                c = Checks()
                try:
                    fn(c, args.tiny_dir, work)
                except Exception as e:   # noqa: BLE001
                    c.that(f"{title} ran to completion", False, f"{type(e).__name__}: {e}")
                c.report(title)
                ok = ok and c.ok
    elif not args.arms_only:
        print("-- tiny-model runs skipped (no --tiny-dir); the hook's bit-level equivalence is unproven here")

    print("ALL P4 PILOT CHECKS PASS" if ok else "P4 PILOT CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
