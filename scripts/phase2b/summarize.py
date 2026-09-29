"""Build summary.txt from the files a phase2b_tokens job wrote.

    python scripts/phase2b/summarize.py outputs/phase2b_check/<jobid>
"""

import json
import sys
from pathlib import Path

PRECISIONS = (("fp16", "fp16 autocast (as configured)"), ("fp32", "fp32: autocast off, tf32 as configured"),
              ("fp32_notf32", "fp32 strict: autocast off, tf32 off"))


def load(d: Path, name: str):
    try:
        return json.loads((d / name).read_text())
    except (OSError, ValueError):
        return None


def fmt_corr(v) -> str:
    return "n/a (too few rows or no variation)" if v is None else f"{v:+.3f}"


def main():
    d = Path(sys.argv[1])
    L = []
    w = L.append

    w(f"== phase2b_tokens  {d.name}")
    steps = [line.split("\t") for line in (d / "steps.tsv").read_text().splitlines()] if (d / "steps.tsv").exists() else []
    w("-- steps (exit code, seconds)")
    for name, code, secs in steps:
        w(f"  {'OK  ' if code == '0' else 'FAIL'} {name:<22} exit={code:<3} {secs}s")

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
    tp = d / "third_party.txt"
    w("-- third_party")
    w("  " + (tp.read_text().strip().replace("\n", "\n  ") if tp.exists() else "(missing)"))

    w("-- 1. token capture changes nothing: class 207 seed 0 vs Phase 1 hashes")
    r = load(d, "regression.json")
    if r:
        for name, v in r.items():
            got = v.get("sha256", v.get("error", "?"))[:12]
            w(f"  {name:<16} {'MATCH   ' if v['match'] else 'MISMATCH'} got={got} expected={v['expected_prefix']}")
        w("  (DiT has no token capture; its run shows the DiT path is unchanged)")
    else:
        w("  (missing regression.json)")
    r = load(d, "vs_phase2.json")
    w("-- 1b. fp16 manifest runs with capture on vs Phase 2 runs (capture off), per-row hashes")
    if r:
        for name, v in r.items():
            if "error" in v:
                w(f"  {name}: ERROR {v['error']}")
                continue
            ok = v["identical_rows"] == v["rows"]
            w(f"  {name}: {'PASS' if ok else 'FAIL'} {v['identical_rows']}/{v['rows']} identical"
              + (f"; differing {v['differing']}" if v["differing"] else ""))
    else:
        w("  (skipped: no Phase 2 output dir given/found)")

    for key, label in PRECISIONS:
        t = load(d, f"tokens_{key}.json")
        w(f"-- 2. {key}: {label}; batch {t['batch_sizes'][0]} vs {t['batch_sizes'][1]}" if t else f"-- 2. {key}: (missing)")
        if not t:
            continue
        pns = t["patch_nums"]
        ps = t["per_scale"]
        w(f"  precision recorded: {t['precision_a']}")
        w("  per scale (pn):           " + " ".join(f"{p:>6}" for p in pns))
        w("  rows first diverging:     " + " ".join(f"{c:>6}" for c in ps["first_divergence_rows"])
          + f"   never: {ps['never_diverge_rows']}")
        w("  rows with any diff:       " + " ".join(f"{c:>6}" for c in ps["rows_with_any_diff"]))
        w("  mean frac tokens differ:  " + " ".join(f"{f:6.3f}" for f in ps["mean_frac_differing"]))
        w("  per row: first scale (pn), frac differing per scale, mean |pixel diff| (own; Phase 2)")
        for stem, v in t["rows"].items():
            first = "none" if v["first_scale"] is None else f"{v['first_scale']} ({v['first_pn']})"
            p2 = v["phase2_mean_pixel_diff"]
            w(f"    {stem:<18} first={first:<8} " + " ".join(f"{f:.2f}" for f in v["frac_differing"])
              + f"  pix={v['mean_pixel_diff']:.3f}" + ("" if p2 is None else f"; p2={p2:.3f}")
              + ("  (tensor identical)" if v["tensor_identical"] else ""))
        c = t["spearman_first_scale_vs_pixel_diff"]
        w(f"  spearman(first scale, mean |pixel diff|): own PNGs {fmt_corr(c.get('own_png'))}"
          + (f"; Phase 2 batch.json {fmt_corr(c['phase2_batch_json'])}" if "phase2_batch_json" in c else "")
          + f"   [{t['spearman_note']}]")
        if t["first_flips"]:
            w("  first flipped token (a = batch 1, b = batch 16; probs = post-CFG softmax before top-k/top-p):")
        for f in t["first_flips"]:
            a, b = f["a"], f["b"]
            w(f"    {f['row']:<18} scale {f['scale']} (pn {f['pn']}) pos {f['position']} {tuple(f['row_col'])}:"
              f" token a={a['token']} (p={a['p_chosen']:.4g}{', top1' if a['chosen_is_top1'] else ''})"
              f" b={b['token']} (p={b['p_chosen']:.4g}{', top1' if b['chosen_is_top1'] else ''})")
            w(f"        top2 a: p={a['top2_p'][0]:.4g}/{a['top2_p'][1]:.4g} ids={a['top2_i']}"
              f"   top2 b: p={b['top2_p'][0]:.4g}/{b['top2_p'][1]:.4g} ids={b['top2_i']}")
            w(f"        top1-top2 gap {f['gap_top1_top2_a']:.4g}; at this scale {f['gap_percentile_at_scale_a']:.0%}"
              f" of positions have a gap <= this (median gap {f['median_gap_at_scale_a']:.4g})")

    w("-- 3. does fp16 cause it?")
    fp16, fp32, strict = (load(d, f"tokens_{k}.json") for k, _ in PRECISIONS)

    def earliest(t):
        firsts = [v["first_scale"] for v in t["rows"].values() if v["first_scale"] is not None]
        return (len(firsts), min(firsts) if firsts else None,
                sorted(firsts)[len(firsts) // 2] if firsts else None)
    for key, t in (("fp16", fp16), ("fp32", fp32), ("fp32_notf32", strict)):
        if t:
            n, lo, med = earliest(t)
            w(f"  {key:<12} rows diverging {n}/{len(t['rows'])}; earliest first scale {lo}; median first scale {med}")
    def compare(base, other) -> str:
        n0, _, med0 = earliest(base)
        n1, _, med1 = earliest(other)
        if n1 == 0:
            return "divergence DISAPPEARS"
        if n1 < n0 or (med1 is not None and med0 is not None and med1 > med0):
            return "divergence is REDUCED / MOVES to later scales"
        return "divergence does NOT go away"
    if fp16:
        if earliest(fp16)[0] == 0:
            w("  VERDICT: no token divergence between batch 1 and 16 at fp16; nothing to attribute")
        else:
            if fp32:
                w(f"  VERDICT fp32 (autocast off, tf32 as configured) vs fp16: {compare(fp16, fp32)}")
            if strict:
                w(f"  VERDICT fp32 strict (autocast and tf32 off) vs fp16: {compare(fp16, strict)}")
    if (d / "divergence_per_scale.png").exists():
        w(f"  plot: {d / 'divergence_per_scale.png'}")

    text = "\n".join(L) + "\n"
    (d / "summary.txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
