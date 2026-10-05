"""CPU checks of lanes/p2/hooks.py and lanes/p2/schema.py with tiny random weights (no GPU, checkpoints or network).

    python lanes/p2/check_p2.py [--work DIR] [--skip-dit]

Tiny VAR/DiT models with the real stage structure (10 scales, 250 steps), built like
scripts/phase3/check_hooks.py's, at batch 16 on manifest/provisional_4x4.csv. Tiny-model images are
noise: this tests the mechanics (hooks fire, change images, keep pairing, are repeatable; the schema
records and merges runs), not any result. Exit 1 if any check fails.
"""

import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import csv  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import tempfile  # noqa: E402
from pathlib import Path  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
MANIFEST = REPO / "manifest" / "provisional_4x4.csv"
HOOKS = "lanes/p2/hooks.py"
RESULTS = {}


def check(name, ok, detail=None):
    RESULTS[name] = {"pass": bool(ok), **({"detail": detail} if detail is not None else {})}
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + ("" if ok or detail is None else f"   {detail}"), flush=True)


def py(*args, **extra):
    env = {k: v for k, v in os.environ.items() if not k.startswith("P2_")}
    env.update({"PYTHONDONTWRITEBYTECODE": "1", **{k: str(v) for k, v in extra.items()}})
    return subprocess.run([sys.executable, *map(str, args)], cwd=REPO, env=env, capture_output=True, text=True)


def gen(config, out, *hook_names, batch_size=16, **extra):
    hooks = [a for h in hook_names for a in ("--hook", f"{HOOKS}:{h}")]
    p = py("-m", "runner.generate", "--config", config, "--manifest", MANIFEST, "--batch-size", batch_size,
           "--out-dir", out, *hooks, **extra)
    return p, (json.loads((out / "run.json").read_text()) if p.returncode == 0 else None)


def hashes(run, key="tensor_sha256"):
    return [r[key] for r in run["rows"]] if run else None


def metrics(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def by_stage(rows, metric) -> dict:
    """stage -> max value over images."""
    out = {}
    for r in rows:
        if r["metric"] == metric:
            out[int(r["stage"])] = max(out.get(int(r["stage"]), 0.0), float(r["value"]))
    return out


def err(p):
    return {"exit": p.returncode, "stderr": p.stderr[-500:]} if p.returncode else None


def check_noise_unit():
    import torch
    from lanes.p2.hooks import noise_like
    rows = [(207, 0), (207, 1), (360, 0)]
    t = torch.zeros(3, 4, 5, 5)
    a = noise_like(t, rows, 3, "f_hat")
    alone = noise_like(t[:1], [(207, 1)], 3, "f_hat")
    check("noise: an image's noise does not depend on its batch", torch.equal(a[1:2], alone))
    check("noise: differs between images (also same seed, different class), stages, tensors and noise seeds",
          not torch.equal(a[0], a[1]) and not torch.equal(a[0], a[2])
          and not torch.equal(a, noise_like(t, rows, 4, "f_hat")) and not torch.equal(a, noise_like(t, rows, 3, "x"))
          and not torch.equal(a, noise_like(t, rows, 3, "f_hat", noise_seed=1)))
    check("noise: repeatable, standard normal shape/dtype", torch.equal(a, noise_like(t, rows, 3, "f_hat"))
          and a.shape == t.shape and a.dtype == t.dtype and 0.8 < float(a.std()) < 1.2)


def check_model(model, cfg, work, hook, stage, trace_every):
    """Baseline (+ trace), one corruption (+ comparison), and the schema records for one model. Returns run dirs."""
    tag = f"{model} {hook}@{stage}"
    root = work / "runs" / model
    trace = {"P2_TRACE_DIR": root / "trace", "P2_TRACE_EVERY": trace_every}
    p0, plain = gen(cfg, root / "plain")
    p, base = gen(cfg, root / "baseline", "save_states", **trace)
    check(f"{model}: baseline with save_states is identical to a run with no hooks",
          base is not None and plain is not None and hashes(base) == hashes(plain), err(p) or err(p0))
    if base is None:
        return None

    corrupt = {"P2_STAGE": stage, "P2_SEVERITY": 0.5, "P2_TRACE_REF": root / "trace", "P2_TRACE_EVERY": trace_every}
    d = root / f"{hook}_s{stage}_sev0.5"
    p, run = gen(cfg, d, hook, "compare_states", P2_METRICS=d / "metrics.csv", **corrupt)
    check(f"{tag}: every image changes, generator states equal the baseline's",
          run is not None and all(a != b for a, b in zip(hashes(run), hashes(base)))
          and hashes(run, "generator_state_sha256") == hashes(base, "generator_state_sha256"), err(p))
    if run is None:
        return None
    name = run["hooks"][0]
    check(f"{tag}: run.json records the hook with its stage and severity",
          name["name"] == f"p2_{hook}_s{stage}_sev0.5_rel" and name["stages"] == [stage] and name["kind"] == "modify", name)

    p, again = gen(cfg, root / "repeat", hook, **corrupt)
    check(f"{tag}: repeatable (a second run gives identical images)", again is not None and hashes(again) == hashes(run), err(p))
    p, zero = gen(cfg, root / "sev0", hook, **{**corrupt, "P2_SEVERITY": 0})
    check(f"{tag}: severity 0 is bit-identical to the baseline", zero is not None and hashes(zero) == hashes(base), err(p))
    p, other = gen(cfg, root / "noise_seed1", hook, **{**corrupt, "P2_NOISE_SEED": 1})
    check(f"{tag}: P2_NOISE_SEED=1 gives different images", other is not None
          and all(a != b for a, b in zip(hashes(other), hashes(run))), err(p))

    # Trace: nothing differs before the injection stage, something does from it on.
    rows = metrics(d / "metrics.csv") if (d / "metrics.csv").is_file() else []
    field = "f_hat" if model == "var" else "x"
    gap = by_stage(rows, f"state_rel_l2:{field}")
    order = [s["stage"] for s in base["stages"]]
    traced = [s for s in order if s in gap]
    before = [s for s in traced if order.index(s) < order.index(stage)]
    after = [s for s in traced if order.index(s) >= order.index(stage)]
    check(f"{tag}: state gap is exactly 0 before the injection stage and > 0 from it on ({len(traced)} traced stages)",
          before and after and all(gap[s] == 0.0 for s in before) and all(gap[s] > 0.0 for s in after)
          and order[-1] in gap, {"before": len(before), "after": len(after), "n_rows": len(rows)})
    return {"baseline": root / "baseline", "run": d, "sev0": root / "sev0"}


def check_schema(var_cfg, dirs, work):
    base, run = dirs["baseline"], dirs["run"]
    p = py("-m", "lanes.p2.schema", "record", base, "--lane", "p2", "--experiment", "check", "--arm", "baseline")
    q = py("-m", "lanes.p2.schema", "record", run, "--lane", "p2", "--experiment", "check", "--arm", "h_s3_sev0.5",
           "--baseline", base, "--param", "hook=noise_h", "--param", "stage=3", "--param", "severity=0.5")
    exp = json.loads((run / "experiment.json").read_text()) if q.returncode == 0 else {}
    check("schema record: baseline and intervention get experiment.json; pairing OK, 16 changed, params typed",
          p.returncode == 0 and q.returncode == 0 and exp.get("role") == "intervention"
          and exp["baseline"]["run_id"] == "p2/check/baseline" and exp["pairing"]["ok"]
          and exp["pairing"]["rows_changed"] == 16 and exp["params"] == {"hook": "noise_h", "stage": 3, "severity": 0.5}
          and exp["resources"]["n_images"] == 16 and exp["resources"]["seconds_per_image"] is not None,
          {"exit": [p.returncode, q.returncode], "stderr": (p.stderr + q.stderr)[-500:]})

    p = py("-m", "lanes.p2.schema", "check", base, dirs["sev0"])
    check("schema check: a run identical to the baseline is paired, 0 images changed",
          p.returncode == 0 and "0/16 images changed" in p.stdout, p.stdout[-300:])
    q0, _ = gen(var_cfg, work / "runs" / "var" / "bs8", batch_size=8)
    p = py("-m", "lanes.p2.schema", "check", base, work / "runs" / "var" / "bs8")
    check("schema check: a run at another batch size is rejected", q0.returncode == 0 and p.returncode == 1
          and "batch_size differs" in p.stdout, p.stdout[-300:])
    p = py("-m", "lanes.p2.schema", "check", run, base)
    check("schema check: a corrupted run cannot serve as the baseline", p.returncode == 1 and "not clean" in p.stdout,
          p.stdout[-300:])
    p = py("-m", "lanes.p2.schema", "record", dirs["sev0"], "--lane", "p2", "--experiment", "check", "--arm", "x",
           "--baseline", run)
    check("schema record: refuses a --baseline that is not recorded as a baseline", p.returncode == 2, p.stderr[-300:])

    p = py("-m", "lanes.p2.analyze", "images", run, "--baseline", base)
    p = py("-m", "lanes.p2.analyze", "images", run, "--baseline", base) if p.returncode == 0 else p   # twice: no duplicates
    q = py("-m", "lanes.p2.analyze", "images", dirs["sev0"], "--baseline", base)
    mse = [float(r["value"]) for r in metrics(run / "metrics.csv") if r["metric"] == "img_mse"] if p.returncode == 0 else []
    mse0 = [float(r["value"]) for r in metrics(dirs["sev0"] / "metrics.csv")] if q.returncode == 0 else [1.0]
    check("analyze images: 16 img_mse > 0 for the corrupted run (state rows kept, no duplicates), all 0 at severity 0",
          len(mse) == 16 and all(v > 0 for v in mse) and len(mse0) == 32 and all(v == 0 for v in mse0)
          and any(r["metric"].startswith("state_") for r in metrics(run / "metrics.csv")),
          {"exit": [p.returncode, q.returncode], "n": len(mse), "stderr": (p.stderr + q.stderr)[-400:]})

    out = work / "tables"
    p = py("-m", "lanes.p2.schema", "collect", work / "runs", "--out", out)
    counts = json.loads(p.stdout) if p.returncode == 0 else {}
    runs = metrics(out / "runs.csv") if p.returncode == 0 else []
    rec = {r["run_id"]: r for r in runs}.get("p2/check/h_s3_sev0.5", {})
    imgs = [r for r in (metrics(out / "images.csv") if p.returncode == 0 else []) if r["run_id"] == "p2/check/h_s3_sev0.5"]
    check("schema collect: runs/images/stages/metrics tables; recorded run carries baseline, pairing and hook",
          p.returncode == 0 and counts["runs"] == len(runs) >= 6 and counts["images"] >= 16 * 6 and counts["metrics"] > 0
          and counts["stages"] >= 10 and rec.get("baseline_run_id") == "p2/check/baseline"
          and rec.get("pairing_ok") == "True" and rec.get("hooks", "").startswith("p2_noise_h_s3_sev0.5_rel")
          and len(imgs) == 16 and all(r["identical_to_baseline"] == "False" for r in imgs),
          {"exit": p.returncode, "counts": counts, "stderr": p.stderr[-400:]})

    p = py("-m", "lanes.p2.analyze", "summary", out)
    rows = metrics(out / "p2_summary.csv") if p.returncode == 0 else []
    curve = [r for r in (metrics(out / "p2_curves.csv") if p.returncode == 0 else []) if r["metric"] == "state_l2:f_hat"]
    s = rows[0] if rows else {}
    check("analyze summary: one row for the corrupted run (stage 3, p 1/3, severity 0.5, damage and gaps), 10-point curve",
          len(rows) == 1 and s["stage"] == "3" and abs(float(s["p"]) - 1 / 3) < 1e-9 and s["severity"] == "0.5"
          and s["hook"] == "noise_h" and float(s["img_mse"]) > 0 and float(s["gap_l2_injection"]) > 0
          and float(s["gap_l2_final"]) > 0 and float(s["gap_l2_ratio"]) > 0 and len(curve) == 10
          and sum(float(r["mean"]) == 0 for r in curve) == 3, {"exit": p.returncode, "row": s, "stderr": p.stderr[-400:]})


def check_kid():
    import numpy as np
    from lanes.p2.analyze import kid
    rng = np.random.default_rng(0)
    a = rng.normal(size=(16, 64))
    check("KID: a set paired with itself gives exactly 0; a shifted set gives > 0",
          kid(a, a, paired=True) == 0.0 and kid(a, a + 1.0, paired=True) > 0 and kid(a, a + 1.0) > 0)


def check_pilot_job(var_cfg, work):
    """lanes/p2/jobs/pilot.pbs under bash on CPU (VAR only, one stage, one severity), its variables given in a
    SETTINGS file as the Gate B jobs do; module/nvidia-smi just fail here. The quality step runs only if the
    classifier weights are present (python -m lanes.p2.quality --download)."""
    from lanes.p2.quality import WEIGHTS_DIR, WEIGHTS_URL
    with_quality = (WEIGHTS_DIR / Path(WEIGHTS_URL).name).is_file()
    settings = work / "check.env"
    settings.write_text(f'VAR_CONFIG={var_cfg}\nRUN_DIT=0\nVAR_STAGES="3"\nSEVERITIES="0.5"\nEXP_TAG=checktag\n'
                        f"QUALITY={int(with_quality)}\n")
    env = {k: v for k, v in os.environ.items() if not k.startswith("P2_")}
    env.update(PYTHONDONTWRITEBYTECODE="1", PBS_JOBID="cpucheck.local", PBS_O_WORKDIR=str(REPO),
               OUT_ROOT=str(work / "lane"), SETTINGS=str(settings))
    p = subprocess.run([shutil.which("bash"), str(REPO / "lanes" / "p2" / "jobs" / "pilot.pbs")], cwd=REPO, env=env,
                       capture_output=True, text=True)
    root = work / "lane" / "checktag" / "cpucheck"
    runs = metrics(root / "tables" / "runs.csv") if (root / "tables" / "runs.csv").is_file() else []
    arm = {r["arm"]: r for r in runs}.get("noise_h_s3_sev0.5", {})
    n_metrics = len(metrics(root / "tables" / "metrics.csv")) if (root / "tables" / "metrics.csv").is_file() else 0
    summary = metrics(root / "tables" / "p2_summary.csv") if (root / "tables" / "p2_summary.csv").is_file() else []
    s = summary[0] if summary else {}
    check("pilot.pbs on CPU (tiny VAR) with a SETTINGS file: baseline and one corruption run, recorded, paired, merged",
          p.returncode == 0 and "PILOT DONE" in p.stdout and len(runs) == 2 and arm.get("pairing_ok") == "True"
          and arm.get("experiment") == "var_checktag_cpucheck"
          and json.loads(arm.get("params", "{}")) == {"hook": "noise_h", "stage": 3, "severity": 0.5, "scale": "rel"}
          and n_metrics > 16 and len(summary) == 1,
          {"exit": p.returncode, "tail": (p.stdout + p.stderr)[-600:]})
    if with_quality:
        check("quality step: classifier scores and KID vs the baseline in the summary",
              s.get("cls_prob") not in (None, "") and s.get("cls_prob_base") not in (None, "")
              and s.get("cls_agree_base") not in (None, "") and s.get("kid_vs_base") not in (None, ""), s)
    else:
        print("SKIP  quality step (no classifier weights; python -m lanes.p2.quality --download)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", type=Path, help="scratch dir (default: a new temp dir)")
    ap.add_argument("--skip-dit", action="store_true", help="VAR and schema checks only (faster)")
    args = ap.parse_args()
    work = (args.work or Path(tempfile.mkdtemp(prefix="check_p2_"))).resolve()
    tiny = work / "tiny"
    tiny.mkdir(parents=True, exist_ok=True)

    check_noise_unit()
    check_kid()
    for repo in ("var",) if args.skip_dit else ("var", "dit"):
        p = py(REPO / "scripts" / "phase3" / "check_hooks.py", "--make-tiny", repo, tiny)
        if p.returncode:
            check(f"build tiny {repo}", False, p.stderr[-600:])
            return finish(work)
    var_cfg, dit_cfg = tiny / "var_tiny.yaml", tiny / "dit_tiny.yaml"

    dirs = check_model("var", var_cfg, work, "noise_h", 3, 1)
    _, fhat = gen(var_cfg, work / "runs" / "var" / "fhat", "noise_fhat", P2_STAGE=9, P2_SEVERITY=0.5)
    base = json.loads((work / "runs" / "var" / "baseline" / "run.json").read_text()) if dirs else None
    check("var noise_fhat@9 (last scale): every image changes, pairing kept", fhat is not None and base is not None
          and all(a != b for a, b in zip(hashes(fhat), hashes(base)))
          and hashes(fhat, "generator_state_sha256") == hashes(base, "generator_state_sha256"))
    p, _ = gen(var_cfg, work / "runs" / "var" / "wrong_model", "noise_x", P2_STAGE=3, P2_SEVERITY=0.5)
    check("var: the DiT hook noise_x on a VAR run fails instead of doing nothing", p.returncode != 0
          and "no such tensor" in p.stderr, p.stderr[-300:])
    p, _ = gen(var_cfg, work / "runs" / "var" / "no_env", "noise_h")
    check("a corruption hook without P2_STAGE / P2_SEVERITY fails to load", p.returncode != 0 and "P2_STAGE" in p.stderr,
          p.stderr[-300:])
    for d in ("wrong_model", "no_env"):   # failed runs leave no run.json; nothing to collect
        assert not (work / "runs" / "var" / d / "run.json").exists()
    if dirs:
        check_schema(var_cfg, dirs, work)
    check_pilot_job(var_cfg, work)

    if not args.skip_dit:
        plain = work / "runs" / "dit" / "plain"
        gen(dit_cfg, plain)
        stages = [s["stage"] for s in json.loads((plain / "run.json").read_text())["stages"]]
        check_model("dit", dit_cfg, work, "noise_x", stages[len(stages) // 2], 25)
    return finish(work)


def finish(work: Path):
    (work / "report.json").write_text(json.dumps({"label": "CPU, tiny weights", "checks": RESULTS}, indent=2))
    ok = all(v["pass"] for v in RESULTS.values())
    print(f"\n{sum(v['pass'] for v in RESULTS.values())}/{len(RESULTS)} checks pass  [CPU, tiny weights]")
    print(f"report: {work / 'report.json'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
