"""Build summary.txt from the files a phase3_check job wrote.

    python scripts/phase3/summarize.py outputs/phase3_check/<jobid>
"""

import json
import sys
from pathlib import Path

# Hook configurations the GPU job does not run (its real configs are VAR more_smooth=False and DiT
# with CFG). Their results exist only from scripts/phase3/check_hooks.py --configs tiny.
CPU_ONLY = (
    "VAR with more_smooth=True (Gumbel path): all hook tests - CPU, tiny weights only",
    "DiT without CFG (cfg_scale 1.0): all hook tests - CPU, tiny weights only",
)


def load(d: Path, name: str):
    try:
        return json.loads((d / name).read_text())
    except (OSError, ValueError):
        return None


def main():
    d = Path(sys.argv[1])
    L = []
    w = L.append

    w(f"== phase3_check  {d.name}")
    steps = [line.split("\t") for line in (d / "steps.tsv").read_text().splitlines()] if (d / "steps.tsv").exists() else []
    w("-- steps (exit code, seconds)")
    for name, code, secs in steps:
        w(f"  {'OK  ' if code == '0' else 'FAIL'} {name:<24} exit={code:<3} {secs}s")

    g = load(d, "gpu.json")
    w("-- gpu")
    if g:
        w(f"  host={g['host']} driver={g['driver']} torch={g['torch']} cuda_build={g['torch_cuda_build']} cudnn={g['cudnn']}")
        for dev in g["devices"]:
            w(f"  [{dev['index']}] {dev['name']} cc{dev['capability']} {dev['total_mem_gib']} GiB")
        for line in g.get("diagnosis", []):
            w(f"  DIAGNOSIS: {line}")
    else:
        w("  (missing gpu.json)")
    tp = d / "third_party.txt"
    w("-- third_party")
    w("  " + (tp.read_text().strip().replace("\n", "\n  ") if tp.exists() else "(missing)"))

    w("-- 1. timing: VAR-d20, batch 16, 3 warm reps (GPU, real weights)")
    for tag, label in (("fp32_strict", "strict fp32 (var_d20.yaml)"), ("fp16", "fp16 autocast (var_d20_fp16)")):
        t = load(d, f"timing_var_{tag}.json")
        if not t:
            w(f"  {label}: (missing timing_var_{tag}.json)")
            continue
        b = t["batches"][0]
        w(f"  {label:<30} {b['warm_mean_s']:.3f}s/batch (sd {b['warm_sd_s']:.3f})  "
          f"per 1,000 images {b['extrapolated']['sampling_s']:.1f}s  peak alloc {b['peak_alloc_gib']} GiB  "
          f"peak reserved {b['peak_reserved_gib']} GiB  [{t['gpu']}]")

    w("-- 2. baseline dry run: no hooks, batch 16, provisional 4x4 manifest")
    for m in ("var", "dit"):
        r = load(d / f"baseline_{m}", "run.json")
        w(f"  {m}: " + (f"{len(r['rows'])} rows, hooks={r['hooks']}, sample {r['seconds']['sample']}s -> baseline_{m}/"
                        if r else "(missing run.json)"))
    r = load(d, "baseline_repeat.json")
    if r:
        for name, v in r.items():
            if "error" in v:
                w(f"  repeat {name}: ERROR {v['error']}")
            else:
                w(f"  repeat {name} (baseline vs hook-suite baseline): "
                  f"{'PASS' if v['identical_rows'] == v['rows'] else 'FAIL'} {v['identical_rows']}/{v['rows']} identical")

    w("-- 2b. batch invariance: VAR strict fp32, batch 1 vs 16, manifest rows (measured, not pass/fail)")
    r = load(d, "batch.json")
    if r:
        for name, v in r.items():
            if "error" in v:
                w(f"  {name}: ERROR {v['error']}")
                continue
            for tag, t in v["tags"].items():
                present = [x for x in t["rows"].values() if not x.get("missing")]
                max_px = max((x["max_pixel_diff"] for x in present), default=None)
                w(f"  {name} {tag} vs {v['ref']}: {t['identical_rows']}/{len(t['rows'])} rows identical; "
                  f"max pixel diff {max_px} (0-255); max|diff| raw {t['max_abs_diff']}; "
                  f"visibly different (mean > {v['pixel_threshold']}): {len(t['visible_rows'])}"
                  + (f"; missing {t['missing']}" if t["missing"] else ""))
    else:
        w("  (missing batch.json)")

    w("-- 2c. VAR strict fp32: runner at batch 1 vs plain upstream at B=1, same runtime settings, all manifest rows")
    r = load(d, "upstream.json")
    if r:
        for name, v in r.items():
            if "error" in v:
                w(f"  {name}: ERROR {v['error']}")
                continue
            ok = v["rows"] > 0 and v["identical_rows"] == v["rows"]
            w(f"  {'PASS' if ok else 'FAIL'} {name}: {v['identical_rows']}/{v['rows']} rows identical"
              + (f"; differing {v['differing']} max|diff|={v.get('max_abs_diff')}" if v["differing"] else "")
              + (f"; missing {v['missing']}" if v["missing"] else ""))
    else:
        w("  (missing upstream.json)")

    w("-- 3a. regression: class 207 seed 0, batch 1 (GPU, real weights); hooked = no-op Modify + Observe, both sides")
    r = load(d, "regression.json")
    if r:
        for name, v in r.items():
            plain = (v["plain_sha256"] or "MISSING")[:12]
            ref = "recorded (no reference yet)" if v["mode"] == "record" else \
                f"expected {v['expected_prefix']} {'MATCH' if v['matches_expected'] else 'MISMATCH'}"
            w(f"  {'PASS' if v['pass'] else 'FAIL'} {name}: plain {plain} {ref}; "
              f"hooked {'==' if v['hooked_equals_plain'] else '!='} plain"
              + (f"; full plain sha256 {v['plain_sha256']}" if v["mode"] == "record" and v["plain_sha256"] else ""))
    else:
        w("  (missing regression.json)")

    w("-- 3b. hook tests, batch 16")
    rep = load(d / "hooks", "report.json")
    if rep:
        for name, v in rep.items():
            n_ok = sum(c["pass"] for c in v["checks"].values())
            w(f"  {name}  [{v['label']}]  {n_ok}/{len(v['checks'])} pass")
            for check, c in v["checks"].items():
                w(f"    {'PASS' if c['pass'] else 'FAIL'}  {check}" + ("" if c["pass"] else f"   {c.get('detail')}"))
            for k, m in v["measured"].items():
                w(f"    measure  {k}: {m}")
    else:
        w("  (missing hooks/report.json)")
    w("-- only CPU-tested (not covered by this job)")
    for line in CPU_ONLY:
        w(f"  {line}")

    text = "\n".join(L) + "\n"
    (d / "summary.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
