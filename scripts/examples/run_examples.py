"""Run every example in scripts/examples/ on CPU with tiny random weights (no GPU, checkpoints or network).

    python scripts/examples/run_examples.py [--work DIR]

Each example is exactly the code shown in docs/hooks_quickstart.md or docs/HANDOVER.md (checked
verbatim; the HANDOVER.md checks are skipped if that file is not present), run against tiny VAR/DiT models built like scripts/phase3/check_hooks.py's (10 VAR scales,
250 DiT steps) at batch 16 on manifest/provisional_4x4.csv. The PBS jobs run under bash with the
config/hash overrides they accept; their real-weight hash checks can only pass on the GPU.
Exit 1 if any check fails.
"""

import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import tempfile  # noqa: E402
from pathlib import Path  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
EX = REPO / "scripts" / "examples"
MANIFEST = REPO / "manifest" / "provisional_4x4.csv"
HANDOVER = "docs/HANDOVER.md"
DOCS = {"fhat_norm.py": "docs/hooks_quickstart.md", "skip_scale_2.py": "docs/hooks_quickstart.md",
        "noise_hook.py": HANDOVER, "smoke.pbs": HANDOVER, "experiment.pbs": HANDOVER}
RESULTS = {}


def check(name, ok, detail=None):
    RESULTS[name] = {"pass": bool(ok), **({"detail": detail} if detail is not None else {})}
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + ("" if ok or detail is None else f"   {detail}"), flush=True)


def env(**extra):
    return {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", **{k: str(v) for k, v in extra.items()}}


def py(*args, **extra):
    return subprocess.run([sys.executable, *map(str, args)], cwd=REPO, env=env(**extra), capture_output=True, text=True)


def gen(config, out, *hook_specs, **extra):
    hooks = [a for h in hook_specs for a in ("--hook", h)]
    p = py("-m", "runner.generate", "--config", config, "--manifest", MANIFEST, "--batch-size", 16, "--out-dir", out,
           *hooks, **extra)
    run = json.loads((out / "run.json").read_text()) if p.returncode == 0 else None
    return p, run


def hashes(run, key="tensor_sha256"):
    return [r[key] for r in run["rows"]] if run else None


def text(p: Path) -> str:
    return p.read_bytes().decode("utf-8").replace("\r\n", "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", type=Path, help="scratch dir (default: a new temp dir)")
    args = ap.parse_args()
    work = (args.work or Path(tempfile.mkdtemp(prefix="examples_"))).resolve()
    work.mkdir(parents=True, exist_ok=True)

    # 0. The docs show exactly these files. docs/HANDOVER.md is kept out of git, so skip its checks if it's absent.
    handover_missing = not (REPO / HANDOVER).exists()
    if handover_missing:
        print("skipped: HANDOVER.md not present", flush=True)
    for name, doc in DOCS.items():
        if doc == HANDOVER and handover_missing:
            continue
        body = text(EX / name).strip("\n")
        check(f"docs: {doc} shows scripts/examples/{name} verbatim", body in text(REPO / doc))

    # Tiny models (random weights), as the phase 3 tests build them.
    tiny = work / "tiny"
    tiny.mkdir(exist_ok=True)
    for repo in ("var", "dit"):
        p = py(REPO / "scripts" / "phase3" / "check_hooks.py", "--make-tiny", repo, tiny)
        if p.returncode:
            check(f"build tiny {repo}", False, p.stderr[-600:])
            return finish(work)
    var_cfg, dit_cfg = tiny / "var_tiny.yaml", tiny / "dit_tiny.yaml"

    p, base = gen(var_cfg, work / "var_baseline")
    check("baseline (no hooks) runs", base is not None, p.stderr[-600:])
    if base is None:
        return finish(work)

    # 1. Observe example: runs, changes nothing, logs 10 scales x 16 rows.
    log = work / "fhat_norms.jsonl"
    p, run = gen(var_cfg, work / "fhat_norm", "scripts/examples/fhat_norm.py:fhat_norm", FHAT_NORM_LOG=log)
    lines = [json.loads(x) for x in log.read_text().splitlines()] if log.exists() else []
    check("fhat_norm.py (observe): runs, images and generator states identical to baseline, 10 x 16 norms logged",
          run is not None and hashes(run) == hashes(base)
          and hashes(run, "generator_state_sha256") == hashes(base, "generator_state_sha256")
          and [x["stage"] for x in lines] == list(range(10)) and all(len(x["norm"]) == 16 for x in lines),
          {"exit": p.returncode, "log_lines": len(lines), "stderr": p.stderr[-400:] if p.returncode else None})

    # 2. Modify example: every image changes, pairing kept.
    p, run = gen(var_cfg, work / "skip_scale_2", "scripts/examples/skip_scale_2.py:skip_scale_2")
    check("skip_scale_2.py (modify): runs, every image changes, generator states equal baseline",
          run is not None and all(a != b for a, b in zip(hashes(run), hashes(base)))
          and hashes(run, "generator_state_sha256") == hashes(base, "generator_state_sha256"),
          {"exit": p.returncode, "stderr": p.stderr[-400:] if p.returncode else None})

    # 3. The quickstart's "Running" command: both hooks together, recorded in run.json.
    log.unlink(missing_ok=True)
    p, run = gen(var_cfg, work / "quickstart_running", "scripts/examples/skip_scale_2.py:skip_scale_2",
                 "scripts/examples/fhat_norm.py:fhat_norm", FHAT_NORM_LOG=log)
    check("quickstart 'Running' command (both hooks): runs, both recorded in run.json",
          run is not None and [h["name"] for h in run["hooks"]] == ["skip_scale_2", "fhat_norm"],
          {"exit": p.returncode, "hooks": run and run["hooks"], "stderr": p.stderr[-400:] if p.returncode else None})

    # 4. Noise example: every image changes, pairing kept, deterministic.
    p, run = gen(var_cfg, work / "noise_a", "scripts/examples/noise_hook.py:noise_scale_3")
    p2, run2 = gen(var_cfg, work / "noise_b", "scripts/examples/noise_hook.py:noise_scale_3")
    check("noise_hook.py (illustration): runs, every image changes, generator states equal baseline, repeatable",
          run is not None and all(a != b for a, b in zip(hashes(run), hashes(base)))
          and hashes(run, "generator_state_sha256") == hashes(base, "generator_state_sha256")
          and run2 is not None and hashes(run2) == hashes(run),
          {"exit": [p.returncode, p2.returncode], "stderr": p.stderr[-400:] if p.returncode else None})

    # 5. Lane imports, as HANDOVER describes: a hook file in lanes/pN/ importing a helper from the same lane.
    lane = REPO / "lanes" / "p1"
    util, hooks = lane / "_example_check_util.py", lane / "_example_check_hooks.py"
    try:
        util.write_text("SEVERITY = 0.0\n")
        hooks.write_text("import dataclasses\n\nfrom lanes.p1._example_check_util import SEVERITY\n"
                         "from runner.hooks import Modify\n\n\n"
                         "def same(model, stage, p, when, state):\n"
                         "    return dataclasses.replace(state, f_hat=state.f_hat + SEVERITY)\n\n\n"
                         "noop = Modify(same, when=\"after\", name=\"lane_import_noop\")\n")
        q = py("-c", "import lanes.p1._example_check_util as m; print(m.__file__)")
        check("lanes import: `import lanes.p1.<module>` from the repo root", q.returncode == 0, q.stderr[-400:])
        p, run = gen(var_cfg, work / "lane_import", "lanes/p1/_example_check_hooks.py:noop")
        check("lanes import: --hook lanes/p1/<file>.py whose code does `from lanes.p1... import ...`",
              run is not None and hashes(run) == hashes(base),
              {"exit": p.returncode, "stderr": p.stderr[-600:] if p.returncode else None})
    finally:
        util.unlink(missing_ok=True)
        hooks.unlink(missing_ok=True)

    # 6. The PBS jobs, run under bash on CPU with tiny configs (module/conda/nvidia-smi lines just fail here).
    bash = shutil.which("bash")
    pbs_env = dict(PBS_JOBID="cpuexample.local", PBS_O_WORKDIR=REPO, VAR_CONFIG=var_cfg, DIT_CONFIG=dit_cfg,
                   OUT_ROOT=work / "lane")
    p = subprocess.run([bash, str(EX / "smoke.pbs")], cwd=REPO, capture_output=True, text=True,
                       env=env(**pbs_env, EXPECT_VAR="record", EXPECT_DIT="record"))
    reg = work / "lane" / "smoke" / "cpuexample" / "regression.json"
    r = json.loads(reg.read_text()) if reg.exists() else {}
    check("smoke.pbs on CPU (tiny configs, hashes recorded not checked): both generate steps run, regression passes",
          p.returncode == 0 and r.get("var", {}).get("pass") and r.get("dit", {}).get("pass"),
          {"exit": p.returncode, "regression": r, "tail": (p.stdout + p.stderr)[-600:]})

    p = subprocess.run([bash, str(EX / "experiment.pbs")], cwd=REPO, capture_output=True, text=True, env=env(**pbs_env))
    root = work / "lane" / "experiment" / "cpuexample"
    runs = {k: json.loads((root / k / "run.json").read_text()) if (root / k / "run.json").exists() else None
            for k in ("var/baseline", "var/intervention", "dit/baseline")}
    ok = all(runs.values())
    check("experiment.pbs on CPU (tiny configs): baseline, intervention and DiT baseline run; "
          "intervention changes every image with generator states paired",
          p.returncode == 0 and ok
          and all(a != b for a, b in zip(hashes(runs["var/intervention"]), hashes(runs["var/baseline"])))
          and hashes(runs["var/intervention"], "generator_state_sha256") == hashes(runs["var/baseline"], "generator_state_sha256")
          and hashes(runs["var/baseline"]) == hashes(base),
          {"exit": p.returncode, "missing": [k for k, v in runs.items() if v is None], "tail": (p.stdout + p.stderr)[-600:]})
    return finish(work)


def finish(work: Path):
    (work / "report.json").write_text(json.dumps({"label": "CPU, tiny weights", "checks": RESULTS}, indent=2))
    ok = all(v["pass"] for v in RESULTS.values())
    print(f"\n{sum(v['pass'] for v in RESULTS.values())}/{len(RESULTS)} checks pass  [CPU, tiny weights]")
    print(f"report: {work / 'report.json'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
