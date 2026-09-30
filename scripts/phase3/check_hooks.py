"""Hook tests (docs/hook_interface.md §6, Phase 3 list a-e). Every run is runner.generate in its own process.

    python scripts/phase3/check_hooks.py --configs tiny [--work DIR]    # CPU, random tiny weights
    python scripts/phase3/check_hooks.py --configs real [--work DIR]    # configs/*.yaml (GPU node)
    python scripts/phase3/check_hooks.py --config configs/var_d24.yaml [--config ...] --quick [--work DIR]

--config PATH (repeatable) runs those configs instead, with real weights. --quick runs only the
baseline and one run with every no-op/observe hook: test a (all together) and test b.
Expected stage counts come from each config: VAR len(patch_nums) scales (10 for every 256px depth,
680 tokens), DiT num_sampling_steps (250). A config whose checkpoint is missing is reported as
"checkpoint not downloaded"; a CUDA out-of-memory failure is reported as such.

tiny: builds random-weight VAR and DiT in checkpoints/tiny/ with the real stage structure (VAR-d20's
patch_nums, so 10 scales; DiT with the real config's 250 steps), then tests var_tiny, var_tiny_smooth,
dit_tiny (CFG) and dit_tiny_nocfg. real: configs/var_d20.yaml and configs/dit_xl2_256.yaml.
Every comparison is between runs at the same batch size (16) on the same manifest.

Asserted (exit 1 if any fails):
  a  no-op Modify (before, after), Observe (both sides), all together: every row's image hash and
     generator state equal the no-hook run; VAR also with token capture on (images and tokens)
  b  observers fire 10 (VAR) / 250 (DiT) times per side, before then after within a stage, stage
     sequence and p strictly increasing from 0 to 1, per-stage fields consistent and equal to run.json
  c  destructive Modify on row 0 only: row 0 changes, rows 1.. bit-identical, all generator states equal;
     VAR: sampled tokens identical to baseline for scales <= k
  d  VAR skip-scale (version b) at scales 2 and 8, DiT hold-latent, DiT skip-timestep: every row's
     image changes, generator states equal; a raising hook makes the run fail
  e  a hook registered for a stage the run does not have is an error (and DiT: a hook on a skipped
     timestep, skipping the first step)
  doc 4  before/after round trip; doc 5 hold details; doc 6 skip details, burn negative control,
         and the coefficients each step actually uses (bit-identical to baseline except the merged step)
Measured only: mean |pixel diff| (0-255) of each positive control vs baseline, dtypes the hooks see.
"""

import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import subprocess  # noqa: E402
import tempfile  # noqa: E402
from pathlib import Path  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

HOOKS = REPO / "scripts" / "phase3" / "hooks_lib.py"
TINY_DIR = REPO / "checkpoints" / "tiny"
REAL = {"var_d20": REPO / "configs" / "var_d20.yaml", "dit_xl2_256": REPO / "configs" / "dit_xl2_256.yaml"}
TINY = ("var_tiny", "var_tiny_smooth", "dit_tiny", "dit_tiny_nocfg")
BATCH = 16


def config_info(config: Path) -> dict:
    """model, expected stages per side (VAR scales, DiT steps), VAR depth and token count, from the config."""
    import yaml
    c = yaml.safe_load(config.read_text())
    if c["model"] == "var":
        pn = c["build"]["patch_nums"]
        return {"model": "var", "stages": len(pn), "depth": c["build"]["depth"], "total_tokens": sum(p * p for p in pn)}
    return {"model": "dit", "stages": int(c["sampler"]["num_sampling_steps"])}
VAR_K = 3                  # destructive / round-trip scale
VAR_SKIP_SCALES = (2, 8)
DIT_J = 125                # destructive / hold / skip step index j (baseline schedule)
COEF_J_ULP = 5             # a skip whose merged schedule object differs off the merged step (docs §11b)


# ---------------------------------------------------------------- tiny weights

def make_tiny(repo: str, out: Path):
    """Random-weight models (scripts/phase2/check_seeding.py) with the real configs' stage counts."""
    import yaml
    sys.path.insert(0, str(REPO / "scripts" / "phase2"))
    import check_seeding as cs
    if repo == "var":
        real = yaml.safe_load(REAL["var_d20"].read_text())
        cs.VAR_BUILD = {**cs.VAR_BUILD, "patch_nums": list(real["build"]["patch_nums"])}
    cs.make_tiny(repo, out)
    if repo == "dit":
        steps = yaml.safe_load(REAL["dit_xl2_256"].read_text())["sampler"]["num_sampling_steps"]
        for name in ("dit_tiny", "dit_tiny_nocfg"):
            p = out / f"{name}.yaml"
            c = yaml.safe_load(p.read_text())
            c["sampler"]["num_sampling_steps"] = steps
            p.write_text(yaml.safe_dump(c, sort_keys=False))


# ---------------------------------------------------------------- coefficients (doc test 6), in-process

def coefficients(config: Path, tau: int) -> dict:
    """Compare the schedule coefficients each executed step uses with and without skipping tau."""
    import numpy as np

    from runner.config import load_config
    from runner.dit_model import COEFFICIENT_ARRAYS, DiTModel
    cfg = load_config(config)
    cfg["device"] = "cpu"   # schedule arithmetic is numpy; the weights are not used
    model = DiTModel(cfg)

    def used(entry):
        return {a: float(getattr(entry["diffusion"], a)[entry["index"]]) for a in COEFFICIENT_ARRAYS}

    base = {e["step"]: e for e in model.plan()}
    skip = model.plan([tau])
    k = next(e["step"] for e in skip if e["skipped"])
    same_elsewhere, merged_differs, merged_steps = True, False, []
    for e in skip:
        if e["skipped"]:
            continue
        a, b = used(e), used(base[e["step"]])
        if e["fields"]["merged"]:
            merged_steps.append(e["step"])
            merged_differs = any(a[n] != b[n] for n in a)
        elif any(a[n] != b[n] for n in a) or e["diffusion"] is not base[e["step"]]["diffusion"]:
            same_elsewhere = False
    # The doc's original claim, for the record: the merged schedule object's own arrays at non-merged steps.
    m = next(e for e in skip if not e["skipped"] and e["fields"]["merged"])["diffusion"]
    bd = model.diffusion
    obj_diffs = sum(not np.array_equal(getattr(m, n)[m.timestep_map.index(t)], getattr(bd, n)[bd.timestep_map.index(t)])
                    for n in COEFFICIENT_ARRAYS for t in m.timestep_map if t != skip[k - 1]["fields"]["stage"])
    return {"skipped_step": k, "merged_steps": merged_steps, "used_identical_except_merged": same_elsewhere,
            "merged_step_differs": merged_differs, "merged_schedule_object_offmerge_diffs": int(obj_diffs)}


# ---------------------------------------------------------------- runs

class Runner:
    def __init__(self, work: Path, manifest: Path):
        self.work, self.manifest = work, manifest
        self.oom, self.missing = [], []   # runs that hit CUDA out of memory / a missing checkpoint

    def gen(self, cfg_name: str, config: Path, tag: str, hooks=(), extra=(), stage=None) -> dict:
        out = self.work / cfg_name / tag
        out.mkdir(parents=True, exist_ok=True)
        log = out / "hooks.jsonl"
        log.unlink(missing_ok=True)
        env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PHASE3_LOG": str(log)}
        env.pop("PHASE3_STAGE", None)
        if stage is not None:
            env["PHASE3_STAGE"] = str(stage)
        cmd = [sys.executable, "-m", "runner.generate", "--config", str(config), "--manifest", str(self.manifest),
               "--batch-size", str(BATCH), "--out-dir", str(out), *map(str, extra)]
        for h in hooks:
            cmd += ["--hook", f"{HOOKS}:{h}"]
        print(f"  [{cfg_name}] {tag}", flush=True)
        p = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True)
        (out / "stderr.txt").write_text(p.stderr, encoding="utf-8")
        if "OutOfMemoryError" in p.stderr or "CUDA out of memory" in p.stderr:
            self.oom.append(f"{cfg_name}/{tag}")
        if "checkpoint not downloaded" in p.stderr:
            self.missing.append(f"{cfg_name}/{tag}")
        run = json.loads((out / "run.json").read_text()) if p.returncode == 0 and (out / "run.json").exists() else None
        calls = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()] if log.exists() else []
        return {"rc": p.returncode, "stderr": p.stderr[-3000:], "run": run, "calls": calls, "dir": out}


def row_hashes(r, key="tensor_sha256"):
    return [x[key] for x in r["run"]["rows"]] if r["run"] else None


def mean_pixel_diff(a: dict, b: dict) -> float:
    import numpy as np
    from PIL import Image
    stems = [x["file"] for x in a["run"]["rows"]]
    return float(np.mean([np.abs(np.asarray(Image.open(a["dir"] / f"{s}.png"), dtype=np.float64)
                                 - np.asarray(Image.open(b["dir"] / f"{s}.png"), dtype=np.float64)).mean()
                          for s in stems]))


class Checks:
    def __init__(self):
        self.results, self.measured = {}, {}

    def check(self, name, ok, detail=None):
        self.results[name] = {"pass": bool(ok), **({"detail": detail} if detail is not None else {})}

    def same_as(self, name, run, base, rows=None):
        """Every row's image hash and generator state equal base's (rows: optional index subset)."""
        if run["rc"] != 0:
            return self.check(name, False, f"run failed: {run['stderr'][-400:]}")
        idx = range(len(base["run"]["rows"])) if rows is None else rows
        img = [i for i in idx if row_hashes(run)[i] != row_hashes(base)[i]]
        gen = [i for i in range(len(base["run"]["rows"]))
               if row_hashes(run, "generator_state_sha256")[i] != row_hashes(base, "generator_state_sha256")[i]]
        self.check(name, not img and not gen, {"rows_differing": img, "generator_states_differing": gen})

    def all_changed_paired(self, name, run, base):
        if run["rc"] != 0:
            return self.check(name, False, f"run failed: {run['stderr'][-400:]}")
        unchanged = [i for i, (a, b) in enumerate(zip(row_hashes(run), row_hashes(base))) if a == b]
        gen = row_hashes(run, "generator_state_sha256") == row_hashes(base, "generator_state_sha256")
        self.check(name, not unchanged and gen, {"rows_unchanged": unchanged, "generator_states_equal": gen})

    def fails_with(self, name, run, text):
        self.check(name, run["rc"] != 0 and text in run["stderr"],
                   {"exit_code": run["rc"], "stderr_tail": run["stderr"][-300:]})


def by_side(calls, when):
    return [c for c in calls if c["when"] == when]


def check_order_and_fields(c: Checks, model: str, run: dict, expected_n: int):
    calls, stages = run["calls"], run["run"]["stages"]
    before, after = by_side(calls, "before"), by_side(calls, "after")
    c.check("b fires per side", len(before) == len(after) == expected_n,
            {"before": len(before), "after": len(after), "expected": expected_n})
    seq = [(x["stage"], x["when"]) for x in calls]
    c.check("b before-then-after within each stage",
            seq == [(s["stage"], w) for s in stages for w in ("before", "after")])
    ps = [x["p"] for x in before]
    c.check("b p strictly increasing 0 -> 1", bool(ps) and ps[0] == 0 and ps[-1] == 1
            and all(a < b for a, b in zip(ps, ps[1:])) and ps == [x["p"] for x in after])
    c.check("b fields equal run.json stages table",
            [x["fields"] for x in before] == stages == [x["fields"] for x in after])
    c.check("b rows per call", all(x["n_rows"] == BATCH for x in calls))
    if model == "var":
        f = [x["fields"] for x in before]
        cum = [x["cum_tokens"] for x in f]
        c.check("b VAR stage = 0..SN-1", [x["stage"] for x in before] == list(range(len(before))))
        c.check("b VAR n_tokens = pn^2 = idx width",
                all(x["fields"]["n_tokens"] == x["fields"]["pn"] ** 2 == x["tensors"]["idx_Bl"]["shape"][1]
                    for x in calls))
        c.check("b VAR cum_tokens increasing, final = total_tokens",
                all(a < b for a, b in zip(cum, cum[1:])) and cum[-1] == f[-1]["total_tokens"]
                and all(x["total_tokens"] == sum(y["n_tokens"] for y in f) for x in f),
                {"final_cum_tokens": cum[-1], "total_tokens": f[-1]["total_tokens"]})
        c.check("b VAR shapes", all(
            x["tensors"]["h_BChw"]["shape"][2:] == [x["fields"]["pn"]] * 2 and x["tensors"]["f_hat"]["shape"][0] == BATCH
            and ("f_hat_before" in x["tensors"]) == (x["when"] == "after") for x in calls))
        c.measured["dtypes seen by hooks"] = sorted({f"{k}:{v['dtype']}" for x in calls for k, v in x["tensors"].items()})
    else:
        f = [x["fields"] for x in before]
        ab_in = [x["alpha_bar_in"] for x in f]
        c.check("b DiT stage = timestep_in, strictly decreasing",
                all(x["stage"] == x["timestep_in"] for x in f) and all(a["stage"] > b["stage"] for a, b in zip(f, f[1:])))
        c.check("b DiT alpha_bar_in strictly increasing", all(a < b for a, b in zip(ab_in, ab_in[1:])))
        c.check("b DiT alpha_bar_out(j) = alpha_bar_in(j+1), timestep_out(j) = timestep_in(j+1)",
                all(a["alpha_bar_out"] == b["alpha_bar_in"] and a["timestep_out"] == b["timestep_in"]
                    for a, b in zip(f, f[1:])))
        c.check("b DiT last step: alpha_bar_out = 1.0, timestep_out None",
                f[-1]["alpha_bar_out"] == 1.0 and f[-1]["timestep_out"] is None)
        c.check("b DiT sigma = sqrt(1 - alpha_bar)", all(
            abs(x["sigma_in"] - (1 - x["alpha_bar_in"]) ** 0.5) < 1e-15
            and abs(x["sigma_out"] - (1 - x["alpha_bar_out"]) ** 0.5) < 1e-15 for x in f))
        c.check("b DiT shapes", all(
            x["tensors"]["x"]["shape"][0] == BATCH
            and ({"x_before", "pred_xstart"} <= set(x["tensors"])) == (x["when"] == "after") for x in calls))
        c.measured["dtypes seen by hooks"] = sorted({f"{k}:{v['dtype']}" for x in calls for k, v in x["tensors"].items()})


def quick(c: Checks, R: Runner, name: str, config: Path, base: dict) -> Checks:
    """Test a (every no-op/observe hook together == no hooks) and test b (firing count, order, fields)."""
    info = config_info(config)
    run = R.gen(name, config, "all", ["noop_before", "noop_after", "observe"])
    c.same_as("a all together (noop Modify before+after, Observe before+after) == no hooks", run, base)
    if run["rc"] == 0:
        check_order_and_fields(c, info["model"], run, info["stages"])
    return c


def test_var(R: Runner, name: str, config: Path, quick_only: bool = False) -> Checks:
    c = Checks()
    base = R.gen(name, config, "base")
    if base["rc"]:
        c.check("baseline run", False, base["stderr"][-400:])
        return c
    if quick_only:
        return quick(c, R, name, config, base)
    n_stages = config_info(config)["stages"]
    # a
    obs = R.gen(name, config, "observe", ["observe"])
    c.same_as("a noop Modify before", R.gen(name, config, "noop_before", ["noop_before"]), base)
    c.same_as("a noop Modify after", R.gen(name, config, "noop_after", ["noop_after"]), base)
    c.same_as("a Observe before+after", obs, base)
    c.same_as("a all together", R.gen(name, config, "all", ["noop_before", "noop_after", "observe"]), base)
    cap_base = R.gen(name, config, "capture", extra=["--capture-tokens"])
    cap_hooks = R.gen(name, config, "capture_all", ["noop_before", "noop_after", "observe"], ["--capture-tokens"])
    c.same_as("a capture on, no hooks == baseline", cap_base, base)
    c.same_as("a capture on + all hooks == baseline", cap_hooks, base)
    c.check("a capture on: tokens identical with and without hooks", tokens_equal(cap_base, cap_hooks))
    # b
    check_order_and_fields(c, "var", obs, n_stages)
    # c
    row0 = R.gen(name, config, "row0", ["row0_after", "observe"], stage=VAR_K)
    c.check("c row 0 changed", row0["rc"] == 0 and row_hashes(row0)[0] != row_hashes(base)[0])
    c.same_as("c rows 1.. identical, all generator states equal", row0, base, rows=range(1, BATCH))
    idx = lambda r: [x["tensors"]["idx_Bl"]["sha"] for x in by_side(r["calls"], "after")]  # noqa: E731
    c.check(f"c tokens identical to baseline for scales <= {VAR_K}", idx(row0)[:VAR_K + 1] == idx(obs)[:VAR_K + 1])
    # doc test 4: round trip
    rt = R.gen(name, config, "roundtrip", ["zero_row0_before", "observe"], stage=VAR_K)
    a_k = next((x for x in by_side(rt["calls"], "after") if x["stage"] == VAR_K), None)
    b_k1 = next((x for x in by_side(rt["calls"], "before") if x["stage"] == VAR_K + 1), None)
    c.check("doc4 after(k).f_hat_before row 0 is zero", a_k is not None and a_k["tensors"]["f_hat_before"]["absmax_row0"] == 0)
    c.check("doc4 before(k+1).f_hat == after(k).f_hat",
            a_k is not None and b_k1 is not None and a_k["tensors"]["f_hat"]["sha"] == b_k1["tensors"]["f_hat"]["sha"])
    # d
    for k in VAR_SKIP_SCALES:
        s = R.gen(name, config, f"skip_scale_{k}", ["restore_all_after"], stage=k)
        c.all_changed_paired(f"d skip scale {k} (version b): every image changes, generator states equal", s, base)
        if s["rc"] == 0:
            c.measured[f"mean pixel diff, skip scale {k}"] = round(mean_pixel_diff(s, base), 3)
    c.fails_with("d raising hook fails the run", R.gen(name, config, "raises", ["raises_after"], stage=VAR_K + 2),
                 "phase3 deliberate hook failure")
    # e
    c.fails_with("e hook on a stage that does not exist", R.gen(name, config, "bad_stage", ["observe_at_stage"],
                                                                stage=n_stages),
                 "which this run does not have")
    return c


def tokens_equal(a: dict, b: dict) -> bool:
    import torch
    if a["rc"] or b["rc"]:
        return False
    for r in a["run"]["rows"]:
        ta = torch.load(a["dir"] / f"{r['file']}.tokens.pt")
        tb = torch.load(b["dir"] / f"{r['file']}.tokens.pt")
        if ta["patch_nums"] != tb["patch_nums"] or len(ta["scales"]) != len(tb["scales"]):
            return False
        for sa, sb in zip(ta["scales"], tb["scales"]):
            if sa.keys() != sb.keys() or not all(torch.equal(sa[k], sb[k]) for k in sa):
                return False
    return True


def test_dit(R: Runner, name: str, config: Path, quick_only: bool = False) -> Checks:
    c = Checks()
    base = R.gen(name, config, "base")
    if base["rc"]:
        c.check("baseline run", False, base["stderr"][-400:])
        return c
    if quick_only:
        return quick(c, R, name, config, base)
    stages = [s["stage"] for s in base["run"]["stages"]]
    tau_j, tau_0 = stages[DIT_J], stages[0]
    # a
    obs = R.gen(name, config, "observe", ["observe"])
    c.same_as("a noop Modify before", R.gen(name, config, "noop_before", ["noop_before"]), base)
    c.same_as("a noop Modify after", R.gen(name, config, "noop_after", ["noop_after"]), base)
    c.same_as("a Observe before+after", obs, base)
    c.same_as("a all together", R.gen(name, config, "all", ["noop_before", "noop_after", "observe"]), base)
    # b
    check_order_and_fields(c, "dit", obs, config_info(config)["stages"])
    # c
    row0 = R.gen(name, config, "row0", ["row0_after"], stage=tau_j)
    c.check("c row 0 changed", row0["rc"] == 0 and row_hashes(row0)[0] != row_hashes(base)[0])
    c.same_as("c rows 1.. identical, all generator states equal", row0, base, rows=range(1, BATCH))
    # doc test 4: round trip
    rt = R.gen(name, config, "roundtrip", ["zero_row0_before", "observe"], stage=tau_j)
    a_j = next((x for x in by_side(rt["calls"], "after") if x["stage"] == tau_j), None)
    c.check("doc4 after(j).x_before row 0 is zero", a_j is not None and a_j["tensors"]["x_before"]["absmax_row0"] == 0)
    # d + doc test 5: hold
    before_x = lambda r: [x["tensors"]["x"]["sha"] for x in by_side(r["calls"], "before")]  # noqa: E731
    hold = R.gen(name, config, "hold", ["restore_all_after", "observe"], stage=tau_j)
    c.all_changed_paired(f"d hold latent at step {DIT_J}: every image changes, generator states equal", hold, base)
    c.check(f"doc5 hold: before-latents identical to baseline for steps <= {DIT_J}",
            before_x(hold)[:DIT_J + 1] == before_x(obs)[:DIT_J + 1])
    c.check("doc5 hold: observer fires S times per side",
            len(by_side(hold["calls"], "before")) == len(by_side(hold["calls"], "after")) == len(stages))
    if hold["rc"] == 0:
        c.measured[f"mean pixel diff, hold at step {DIT_J}"] = round(mean_pixel_diff(hold, base), 3)
    # d + doc test 6: skip
    skip = R.gen(name, config, "skip", ["observe"], ["--skip-timesteps", tau_j])
    c.all_changed_paired(f"d skip timestep {tau_j} (step {DIT_J}): every image changes, generator states equal",
                         skip, base)
    c.check(f"doc6 skip: before-latents identical to baseline for steps < {DIT_J}",
            before_x(skip)[:DIT_J] == before_x(obs)[:DIT_J])
    c.check("doc6 skip: observer fires S-1 times per side, never at the skipped timestep",
            len(by_side(skip["calls"], "before")) == len(by_side(skip["calls"], "after")) == len(stages) - 1
            and all(x["stage"] != tau_j for x in skip["calls"]))
    if skip["rc"] == 0:
        c.measured[f"mean pixel diff, skip timestep {tau_j}"] = round(mean_pixel_diff(skip, base), 3)
        m = [s for s in skip["run"]["stages"] if s["merged"]]
        c.check("doc6 skip: one merged step, from the step before to the step after",
                len(m) == 1 and m[0]["timestep_in"] == stages[DIT_J - 1] and m[0]["timestep_out"] == stages[DIT_J + 1])
    noburn = R.gen(name, config, "skip_noburn", extra=["--skip-timesteps", tau_j, "--no-burn-skipped"])
    c.check("doc6 negative control: without the burn, generator states differ",
            noburn["rc"] == 0 and row_hashes(noburn, "generator_state_sha256") != row_hashes(base, "generator_state_sha256"))
    # Coefficients: at DIT_J, and at a step where the merged schedule object itself differs from the
    # baseline by 1 ULP away from the merged step (numpy-checked for the 250-step schedule), so the
    # check shows the runner does not use those values.
    for j in (DIT_J, COEF_J_ULP):
        coef = subprocess.run([sys.executable, str(Path(__file__)), "--coefficients", str(config), str(stages[j])],
                              cwd=REPO, capture_output=True, text=True)
        co = json.loads(coef.stdout.strip().splitlines()[-1]) if coef.returncode == 0 else {"error": coef.stderr[-400:]}
        c.check(f"doc6 coefficients used, skip step {j}: bit-identical to baseline except the merged step",
                co.get("used_identical_except_merged") is True and co.get("merged_step_differs") is True
                and co.get("merged_steps") == [j - 1], co)
        c.measured[f"merged schedule object: off-merge arrays differing from baseline, skip step {j}"] = \
            co.get("merged_schedule_object_offmerge_diffs")
    c.fails_with("d raising hook fails the run", R.gen(name, config, "raises", ["raises_after"], stage=tau_j),
                 "phase3 deliberate hook failure")
    # e
    c.fails_with("e hook on a stage that does not exist",
                 R.gen(name, config, "bad_stage", ["observe_at_stage"], stage=max(stages) + 1),
                 "which this run does not have")
    c.fails_with("e hook on a skipped timestep",
                 R.gen(name, config, "hook_on_skipped", ["observe_at_stage"], ["--skip-timesteps", tau_j], stage=tau_j),
                 "registered for skipped timesteps")
    c.fails_with("e skipping the first step", R.gen(name, config, "skip_first", extra=["--skip-timesteps", tau_0]),
                 "cannot skip the first step")
    return c


def label(config: Path, real: bool) -> str:
    import yaml
    device = yaml.safe_load(config.read_text())["device"]
    return f"{'GPU' if str(device).startswith('cuda') else 'CPU'}, {'real' if real else 'tiny'} weights"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--configs", choices=("tiny", "real"), default="tiny")
    ap.add_argument("--only", nargs="+", help="config names to run (default: all for --configs)")
    ap.add_argument("--config", action="append", type=Path, default=[],
                    help="run this config file instead (repeatable; real weights; overrides --configs)")
    ap.add_argument("--quick", action="store_true", help="only test a (all hooks together) and test b")
    ap.add_argument("--manifest", type=Path, default=REPO / "manifest" / "provisional_4x4.csv")
    ap.add_argument("--work", type=Path, help="output dir (default: a new temp dir)")
    ap.add_argument("--make-tiny", nargs=2, metavar=("REPO", "DIR"), help=argparse.SUPPRESS)
    ap.add_argument("--coefficients", nargs=2, metavar=("CONFIG", "TAU"), help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.make_tiny:
        return make_tiny(args.make_tiny[0], Path(args.make_tiny[1]))
    if args.coefficients:
        print(json.dumps(coefficients(Path(args.coefficients[0]), int(args.coefficients[1]))))
        return

    work = (args.work or Path(tempfile.mkdtemp(prefix="check_hooks_"))).resolve()
    work.mkdir(parents=True, exist_ok=True)
    real = args.configs == "real" or bool(args.config)
    if args.config:
        configs = {Path(p).stem: Path(p).resolve() for p in args.config}
    elif real:
        configs = dict(REAL)
    else:
        TINY_DIR.mkdir(parents=True, exist_ok=True)
        for repo in ("var", "dit"):
            subprocess.run([sys.executable, str(Path(__file__)), "--make-tiny", repo, str(TINY_DIR)], cwd=REPO, check=True,
                           env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
        configs = {n: TINY_DIR / f"{n}.yaml" for n in TINY}
    if args.only:
        configs = {n: configs[n] for n in args.only}

    R = Runner(work, args.manifest.resolve())
    report, ok = {}, True
    for name, config in configs.items():
        info = config_info(config)
        test = test_var if info["model"] == "var" else test_dit
        c = test(R, name, config, args.quick)
        mine = lambda runs: [r for r in runs if r.startswith(f"{name}/")]  # noqa: E731
        if info["model"] == "var":
            c.measured["total tokens per image (from config patch_nums)"] = info["total_tokens"]
        report[name] = {"label": label(config, real), "config": str(config), "model": info["model"],
                        **({"depth": info["depth"]} if "depth" in info else {}),
                        "batch_size": BATCH, "expected_stages_per_side": info["stages"], "quick": args.quick,
                        "checkpoint_not_downloaded": bool(mine(R.missing)), "out_of_memory": mine(R.oom),
                        "checks": c.results, "measured": c.measured}
        ok &= all(v["pass"] for v in c.results.values())
    (work / "report.json").write_text(json.dumps(report, indent=2))

    print()
    for name, rep in report.items():
        n_ok = sum(v["pass"] for v in rep["checks"].values())
        status = (" CHECKPOINT NOT DOWNLOADED" if rep["checkpoint_not_downloaded"] else "") + (
            f" CUDA OUT OF MEMORY at batch {BATCH}: {rep['out_of_memory']}" if rep["out_of_memory"] else "")
        print(f"{name}  [{rep['label']}{', depth ' + str(rep['depth']) if 'depth' in rep else ''}]  "
              f"{n_ok}/{len(rep['checks'])} checks pass{status}")
        for check, v in rep["checks"].items():
            print(f"    {'PASS' if v['pass'] else 'FAIL'}  {check}" + ("" if v["pass"] else f"   {v.get('detail')}"))
        for k, v in rep["measured"].items():
            print(f"    measure  {k}: {v}")
    print(f"\nreport: {work / 'report.json'}")
    print("ALL HOOK CHECKS PASS" if ok else "HOOK CHECKS FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
