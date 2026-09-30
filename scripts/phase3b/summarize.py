"""Build summary.txt from the files a phase3b_depths job wrote.

    python scripts/phase3b/summarize.py outputs/phase3b_depths/<jobid>
"""

import json
import sys
from pathlib import Path

NOT_DOWNLOADED = "checkpoint not downloaded"
EXIT_OOM = 3   # scripts/phase3b/timing_oom.py


def load(d: Path, name: str):
    try:
        return json.loads((d / name).read_text())
    except (OSError, ValueError):
        return None


def read(p: Path) -> str | None:
    try:
        return p.read_text().strip()
    except OSError:
        return None


def main():
    d = Path(sys.argv[1])
    L = []
    w = L.append
    w(f"== phase3b_depths  {d.name}  (GPU, real weights; strict fp32; batch 16 except where noted)")

    steps = [line.split("\t") for line in (d / "steps.tsv").read_text().splitlines()] if (d / "steps.tsv").exists() else []
    w("-- steps (exit code, seconds)")
    for name, code, secs in steps:
        tag = ("OK  " if code == "0" else "SKIP" if code == NOT_DOWNLOADED
               else "OOM " if name.endswith("_timing") and code == str(EXIT_OOM) else "FAIL")
        w(f"  {tag} {name:<22} {'exit=' + code if code != NOT_DOWNLOADED else code:<26} {secs}s")

    g = load(d, "gpu.json")
    w("-- gpu")
    if g:
        w(f"  host={g['host']} driver={g['driver']} torch={g['torch']} cuda_build={g['torch_cuda_build']}")
        for dev in g["devices"]:
            w(f"  [{dev['index']}] {dev['name']} cc{dev['capability']} {dev['total_mem_gib']} GiB")
        for line in g.get("diagnosis", []):
            w(f"  DIAGNOSIS: {line}")
    else:
        w("  (missing gpu.json)")
    tp = read(d / "third_party.txt")
    w("-- third_party")
    w("  " + (tp.replace("\n", "\n  ") if tp else "(missing)"))

    w("-- depth check at load: configs/var_d16.yaml pointed at var_d20.pth must fail")
    guard = next((s for s in steps if s[0] == "depth_guard"), None)
    if read(d / "depth_guard" / "checkpoint_not_downloaded.txt"):
        w(f"  SKIP {read(d / 'depth_guard' / 'checkpoint_not_downloaded.txt')}")
    elif guard is None:
        w("  (not run)")
    else:
        msg = next((line for line in (read(d / "depth_guard" / "stderr.txt") or "").splitlines()
                    if "says VAR depth" in line), "(no depth message found)")
        w(f"  {'PASS' if guard[1] == '0' else 'FAIL'} rejected with: {msg.strip()}")

    depths = (read(d / "depths.txt") or "").split()
    for dep in depths:
        D = d / f"d{dep}"
        w(f"== VAR-d{dep}  (configs/var_d{dep}.yaml)")
        missing = read(D / "checkpoint_not_downloaded.txt")
        if missing:
            w(f"  CHECKPOINT NOT DOWNLOADED - all steps skipped. {missing}")
            continue
        if read(D / "no_config.txt"):
            w(f"  {read(D / 'no_config.txt')}")
            continue

        r = (load(D, "regression.json") or {}).get(f"var_d{dep}")
        if not r or not r["plain_sha256"]:
            w("  FAIL regression, class 207 seed 0, batch 1: (missing)")
        elif r["mode"] == "record":
            w(f"  regression, class 207 seed 0, batch 1: recorded sha256 {r['plain_sha256']}")
        else:
            w(f"  {'PASS' if r['pass'] else 'FAIL'} regression, class 207 seed 0, batch 1: sha256 {r['plain_sha256']}"
              f" {'==' if r['matches_expected'] else '!='} pinned {r['expected_prefix']}")

        u = (load(D, "upstream.json") or {}).get(f"var_d{dep}")
        if u and "error" not in u:
            ok = u["rows"] > 0 and u["identical_rows"] == u["rows"]
            w(f"  {'PASS' if ok else 'FAIL'} runner (batch 1) == plain upstream (B=1): {u['identical_rows']}/{u['rows']} rows"
              + (f"; differing {u['differing']}" if u["differing"] else "") + (f"; missing {u['missing']}" if u["missing"] else ""))
        else:
            w(f"  FAIL runner vs upstream: {u.get('error') if u else '(missing upstream.json)'}")

        h = (load(D / "hooks", "report.json") or {}).get(f"var_d{dep}")
        if h:
            n_ok = sum(c["pass"] for c in h["checks"].values())
            if h["out_of_memory"]:
                w(f"  DOES NOT FIT at batch 16 (CUDA out of memory) in hook runs {h['out_of_memory']}")
            w(f"  hooks at batch 16 (depth {h.get('depth')}, expecting {h['expected_stages_per_side']} calls per side): "
              f"{n_ok}/{len(h['checks'])} checks pass")
            for name, c in h["checks"].items():
                if not c["pass"]:
                    w(f"    FAIL {name}   {c.get('detail')}")
        else:
            w("  hooks: (missing hooks/report.json)")

        t = load(D, "timing.json")
        if not t:
            w("  timing: (missing timing.json)")
        elif t.get("out_of_memory"):
            w(f"  timing: DOES NOT FIT at batch {', '.join(map(str, t['batch_sizes']))} "
              f"(CUDA out of memory; peak {t['peak_alloc_gib_before_failure']}"
              f" GiB of {t['total_gib']} GiB on {t['gpu']})")
        elif t.get("checkpoint_not_downloaded"):
            w(f"  timing: {NOT_DOWNLOADED}")
        else:
            b = t["batches"][0]
            w(f"  timing, batch {b['batch_size']}: {b['warm_mean_s']:.3f}s/batch (sd {b['warm_sd_s']:.3f}), "
              f"per 1,000 images {b['extrapolated']['sampling_s']:.1f}s, peak alloc {b['peak_alloc_gib']} GiB, "
              f"peak reserved {b['peak_reserved_gib']} GiB, load {t['load_s']}s")

    text = "\n".join(L) + "\n"
    (d / "summary.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
