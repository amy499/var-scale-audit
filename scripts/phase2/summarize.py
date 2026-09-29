"""Build summary.txt from the files a phase2_check job wrote.

    python scripts/phase2/summarize.py outputs/phase2_check/<jobid>
"""

import json
import sys
from pathlib import Path


def load(d: Path, name: str):
    try:
        return json.loads((d / name).read_text())
    except (OSError, ValueError):
        return None


def fmt(x) -> str:
    return "n/a" if x is None else f"{x:.3e}"


def main():
    d = Path(sys.argv[1])
    L = []
    w = L.append

    w(f"== phase2_check  {d.name}")
    steps = [line.split("\t") for line in (d / "steps.tsv").read_text().splitlines()] if (d / "steps.tsv").exists() else []
    w("-- steps (exit code, seconds)")
    for name, code, secs in steps:
        w(f"  {'OK  ' if code == '0' else 'FAIL'} {name:<20} exit={code:<3} {secs}s")

    g = load(d, "gpu.json")
    w("-- gpu")
    if g:
        w(f"  host={g['host']} CUDA_VISIBLE_DEVICES={g['CUDA_VISIBLE_DEVICES']} driver={g['driver']}")
        w(f"  torch={g['torch']} cuda_build={g['torch_cuda_build']} cudnn={g['cudnn']} device_count={g['device_count']}")
        for dev in g["devices"]:
            w(f"  [{dev['index']}] {dev['name']} cc{dev['capability']} {dev['total_mem_gib']} GiB")
        for line in g.get("diagnosis", []):
            w(f"  DIAGNOSIS: {line}")
    else:
        w("  (missing gpu.json)")

    tp = d / "third_party.txt"
    w("-- third_party")
    w("  " + (tp.read_text().strip().replace("\n", "\n  ") if tp.exists() else "(missing)"))

    w("-- 1. regression: single-image runner, class 207 seed 0, vs Phase 1 hash")
    r = load(d, "regression.json")
    if r:
        for name, v in r.items():
            got = v.get("sha256", v.get("error", "?"))[:12]
            w(f"  {name}: {'MATCH   ' if v['match'] else 'MISMATCH'} got={got} expected={v['expected_prefix']}")
    else:
        w("  (missing regression.json)")

    w("-- 2. repeat: manifest at batch 16, run twice; every row's hash must be identical")
    r = load(d, "repeat.json")
    if r:
        for name, v in r.items():
            if "error" in v:
                w(f"  {name}: ERROR {v['error']}")
                continue
            ok = v["identical_rows"] == v["rows"]
            w(f"  {name}: {'PASS' if ok else 'FAIL'} {v['identical_rows']}/{v['rows']} identical"
              + (f"; differing {v['differing']} max|diff|={fmt(v.get('max_abs_diff'))}" if v["differing"] else "")
              + (f"; missing {v['missing']}" if v["missing"] else ""))
    else:
        w("  (missing repeat.json)")

    w("-- 3. batch sensitivity vs batch 16 (measured, not pass/fail)")
    r = load(d, "batch.json")
    if r:
        for name, v in r.items():
            if "error" in v:
                w(f"  {name}: ERROR {v['error']}")
                continue
            thr = v["pixel_threshold"]
            for tag, t in v["tags"].items():
                w(f"  {name} {tag}: {t['identical_rows']}/{len(t['rows'])} identical; max|diff| over rows {fmt(t['max_abs_diff'])}"
                  + (f"; visibly different (mean |pixel diff| > {thr} grey levels): {len(t['visible_rows'])}"
                     f" {t['visible_rows']}" if name == "var" else ""))
                for stem, row in t["rows"].items():
                    if row.get("missing"):
                        w(f"      {stem:<18} MISSING")
                        continue
                    w(f"      {stem:<18} max|diff|={fmt(row['max_abs_diff'])} mean|pixel diff|={row['mean_pixel_diff']:.3f}"
                      f" max pixel diff={row['max_pixel_diff']}")
            if "contact_sheet" in v:
                cs = v["contact_sheet"]
                w(f"  contact sheet: {cs['path']} (columns {' | '.join(cs['columns'])}; rows {cs['rows']})")
    else:
        w("  (missing batch.json)")

    w("-- 4. runner at batch 1 vs plain upstream at B=1 (same seed, same runtime settings), GPU")
    r = load(d, "upstream.json")
    if r:
        for name, v in r.items():
            if "error" in v:
                w(f"  {name}: ERROR {v['error']}")
                continue
            ok = v["identical_rows"] == v["rows"]
            w(f"  {name}: {'PASS' if ok else 'FAIL'} {v['identical_rows']}/{v['rows']} identical"
              + (f"; differing {v['differing']} max|diff|={fmt(v.get('max_abs_diff'))}" if v["differing"] else "")
              + (f"; missing {v['missing']}" if v["missing"] else ""))
    else:
        w("  (missing upstream.json)")

    w("-- 5. VAR divergence: notebook 'matched' vs 'asis', one setting changed at a time")
    r = load(d, "divergence.json")
    if r:
        w("  settings that differ between matched and asis:")
        for k, v in r["settings_differing"].items():
            w(f"    {k}: {v}")
        w("  identical in both: " + "; ".join(r["settings_identical"]))
        w(f"  baseline: {r['baseline']}; row {r['row']}")
        for label, v in r["single_row"].items():
            if "missing" in v:
                w(f"    {label:<68} MISSING")
                continue
            w(f"    {label:<68} {'IDENTICAL' if v['identical'] else 'DIFFERENT'} max|diff|={fmt(v['max_abs_diff'])}"
              f" mean|pixel diff|={v['mean_pixel_diff']:.3f} sha={v['sha256'][0]}/{v['sha256'][1]}")
        w("  all manifest rows, each kernel vs math (fresh processes):")
        for k, v in r["all_rows"].items():
            if v.get("missing"):
                w(f"    {k:<14} MISSING")
                continue
            w(f"    {k:<14} {v['identical_rows']}/{v['rows']} identical; max|diff|={fmt(v['max_abs_diff'])};"
              f" visibly different (mean |pixel diff| > {r['pixel_threshold']}): {len(v['visible_rows'])} {v['visible_rows']}")
        if "runner_bs1_vs_matched" in r:
            x = r["runner_bs1_vs_matched"]
            w(f"  runner batch 1 vs notebook matched: {x['identical_rows']}/{x['rows']} rows identical")
        v = r["verdict"]
        w(f"  VERDICT: divergence caused by {', '.join(v['cause'])}"
          f" (S1 diverges={v['S1_diverges']}, S2 diverges={v['S2_diverges']});"
          f" unrestricted SDPA reproduces kernel: {', '.join(v['unrestricted_sdpa_equals_kernel'])}")
    else:
        w("  (missing divergence.json)")

    text = "\n".join(L) + "\n"
    (d / "summary.txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
