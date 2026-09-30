"""CPU checks for Phase 3b (VAR depths), with tiny random weights. No checkpoints, GPU or network needed.

    python scripts/phase3b/check_cpu.py [--work DIR]

Asserted (exit 1 if any fails):
  configs      configs/var_d{16,24,30}.yaml differ from var_d20.yaml only in build.depth and the VAR checkpoint
  downloads    download_checkpoints.py --dry-run: default = VAR-d20 + DiT; --configs / positional /
               --all-var-depths select the right files (nothing is downloaded)
  depth ok     tiny depth-2 checkpoint + depth-2 config loads; run.json records depth 2 and the checkpoint's
               blocks/width
  depth wrong  the same checkpoint with a depth-3 config fails with a message naming both depths (no traceback)
  missing      a config whose checkpoint file is absent: runner.generate and scripts/phase3b/timing_oom.py
               exit 4 with "checkpoint not downloaded" and no traceback; timing_oom writes a record
  hooks        scripts/phase3/check_hooks.py --config <tiny depth-2 config> --quick passes, with the
               expected counts taken from the config (10 scales per side, 680 tokens)
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
ENV = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
RESULTS = {}


def check(name, ok, detail=None):
    RESULTS[name] = {"pass": bool(ok), **({"detail": detail} if detail is not None else {})}
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + ("" if ok or detail is None else f"   {detail}"), flush=True)


def run(*args):
    return subprocess.run([sys.executable, *map(str, args)], cwd=REPO, env=ENV, capture_output=True, text=True)


def check_configs():
    import yaml
    base = yaml.safe_load((REPO / "configs" / "var_d20.yaml").read_text())
    rev = base["checkpoints"]["var"]["url"].split("/resolve/")[1].split("/")[0]
    for d in (16, 24, 30):
        c = yaml.safe_load((REPO / "configs" / f"var_d{d}.yaml").read_text())
        other = [k for k in base if k not in ("checkpoints", "build") and base[k] != c[k]]
        build = {k for k in base["build"] if base["build"][k] != c["build"][k]}
        v = c["checkpoints"]["var"]
        check(f"configs: var_d{d}.yaml differs from var_d20.yaml only in depth and VAR checkpoint",
              not other and build == {"depth"} and c["build"]["depth"] == d
              and c["checkpoints"]["vae"] == base["checkpoints"]["vae"] and set(c["checkpoints"]) == {"vae", "var"}
              and v["path"] == f"checkpoints/var/var_d{d}.pth" and f"/resolve/{rev}/var_d{d}.pth" in v["url"]
              and len(v.get("sha256") or "") == 64,
              {"other_sections": other, "build_keys": sorted(build), "var": v})


def check_downloads():
    def files(*args):
        p = run(REPO / "scripts" / "download_checkpoints.py", *args, "--dry-run")
        return sorted(line.split()[0] for line in p.stdout.splitlines() if line.startswith("  "))

    var = lambda *ds: [f"checkpoints/var/var_d{d}.pth" for d in ds]  # noqa: E731
    vae = "checkpoints/var/vae_ch160v4096z32.pth"
    dit = ["checkpoints/dit/DiT-XL-2-256x256.pt", "checkpoints/sd-vae-ft-ema/config.json",
           "checkpoints/sd-vae-ft-ema/diffusion_pytorch_model.safetensors"]
    cases = {
        "default": ((), sorted(dit + var(20) + [vae])),
        "--configs var_d24": (("--configs", "configs/var_d24.yaml"), sorted(var(24) + [vae])),
        "positional var_d24": (("configs/var_d24.yaml",), sorted(var(24) + [vae])),
        "--all-var-depths": (("--all-var-depths",), sorted(var(16, 20, 24, 30) + [vae])),
    }
    for label, (args, want) in cases.items():
        got = files(*args)
        check(f"downloads: {label} selects the right files", got == want, {"got": got, "want": want})


def check_depth(work: Path):
    import yaml
    tiny = work / "tiny"
    tiny.mkdir(parents=True, exist_ok=True)
    p = run(REPO / "scripts" / "phase3" / "check_hooks.py", "--make-tiny", "var", tiny)
    if p.returncode:
        check("depth: build tiny VAR", False, p.stderr[-800:])
        return None
    good = tiny / "var_tiny.yaml"
    c = yaml.safe_load(good.read_text())
    depth = c["build"]["depth"]

    out = work / "depth_ok"
    p = run("-m", "runner.generate", "--config", good, "--class-id", 207, "--seed", 0, "--out", out)
    rec = json.loads(out.with_suffix(".json").read_text()) if p.returncode == 0 else {}
    check(f"depth ok: depth-{depth} config + depth-{depth} checkpoint loads and records the depth",
          p.returncode == 0 and rec.get("depth") == depth
          and rec.get("depth_check") == {"config_depth": depth, "checkpoint_blocks": depth, "checkpoint_width": 64 * depth},
          {"exit": p.returncode, "depth": rec.get("depth"), "depth_check": rec.get("depth_check"),
           "stderr": p.stderr[-400:] if p.returncode else None})

    wrong = work / "var_tiny_wrong_depth.yaml"
    c_wrong = {**c, "build": {**c["build"], "depth": depth + 1}}
    wrong.write_text(yaml.safe_dump(c_wrong, sort_keys=False))
    p = run("-m", "runner.generate", "--config", wrong, "--class-id", 207, "--seed", 0, "--out", work / "depth_wrong")
    msg = next((line for line in p.stderr.splitlines() if "says VAR depth" in line), "")
    check(f"depth wrong: depth-{depth + 1} config + depth-{depth} checkpoint fails naming both depths, no traceback",
          p.returncode != 0 and f"says VAR depth {depth + 1} ({depth + 1} blocks, width {64 * (depth + 1)})" in msg
          and f"has {depth} blocks and width {64 * depth}" in msg and "Traceback" not in p.stderr,
          {"exit": p.returncode, "message": msg or p.stderr[-400:]})
    return good


def check_missing(work: Path, good: Path):
    import yaml
    c = yaml.safe_load(good.read_text())
    c["checkpoints"]["var"]["path"] = str(work / "no_such_dir" / "var_dXX.pth")
    cfg = work / "var_tiny_missing.yaml"
    cfg.write_text(yaml.safe_dump(c, sort_keys=False))

    p = run("-m", "runner.generate", "--config", cfg, "--class-id", 207, "--seed", 0, "--out", work / "missing")
    check("missing: runner.generate exits 4 with 'checkpoint not downloaded', no traceback",
          p.returncode == 4 and "checkpoint not downloaded" in p.stderr and "Traceback" not in p.stderr,
          {"exit": p.returncode, "stderr": p.stderr[-400:]})

    out = work / "missing_timing.json"
    p = run(REPO / "scripts" / "phase3b" / "timing_oom.py", "--config", cfg, "--batch-sizes", 16, "--reps", 1,
            "--device", "cpu", "--out", out)
    rec = json.loads(out.read_text()) if out.exists() else {}
    check("missing: timing_oom.py exits 4, records checkpoint_not_downloaded, no traceback",
          p.returncode == 4 and rec.get("checkpoint_not_downloaded") is True and "Traceback" not in p.stderr,
          {"exit": p.returncode, "record": rec, "stderr": p.stderr[-400:]})


def check_hooks_quick(work: Path, good: Path):
    hw = work / "hooks"
    p = run(REPO / "scripts" / "phase3" / "check_hooks.py", "--config", good, "--quick", "--work", hw)
    rep = json.loads((hw / "report.json").read_text()).get(good.stem, {}) if (hw / "report.json").exists() else {}
    checks = rep.get("checks", {})
    check("hooks: check_hooks --config <tiny> --quick passes (a all together, b counts from the config)",
          p.returncode == 0 and checks and all(v["pass"] for v in checks.values())
          and rep.get("expected_stages_per_side") == 10 and rep.get("depth") == 2
          and rep["measured"].get("total tokens per image (from config patch_nums)") == 680,
          {"exit": p.returncode, "failing": [k for k, v in checks.items() if not v["pass"]],
           "expected_stages_per_side": rep.get("expected_stages_per_side"), "depth": rep.get("depth"),
           "stderr": p.stderr[-400:] if p.returncode else None})
    return rep


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", type=Path, help="scratch dir (default: a new temp dir)")
    args = ap.parse_args()
    work = (args.work or Path(tempfile.mkdtemp(prefix="phase3b_cpu_"))).resolve()
    work.mkdir(parents=True, exist_ok=True)

    check_configs()
    check_downloads()
    good = check_depth(work)
    if good is not None:
        check_missing(work, good)
        check_hooks_quick(work, good)

    (work / "report.json").write_text(json.dumps({"label": "CPU, tiny weights", "checks": RESULTS}, indent=2))
    ok = all(v["pass"] for v in RESULTS.values())
    print(f"\n{sum(v['pass'] for v in RESULTS.values())}/{len(RESULTS)} checks pass  [CPU, tiny weights]")
    print(f"report: {work / 'report.json'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
