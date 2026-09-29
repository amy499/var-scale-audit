"""Build summary.txt from the files a phase1_check job wrote.

    python scripts/phase1/summarize.py outputs/phase1_check/<jobid>
"""

import json
import sys
from pathlib import Path


def load(d: Path, name: str):
    p = d / name
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def main():
    d = Path(sys.argv[1])
    L = []
    w = L.append

    w(f"== phase1_check  {d.name}")
    steps = [line.split("\t") for line in (d / "steps.tsv").read_text().splitlines()] if (d / "steps.tsv").exists() else []
    w("-- steps (exit code, seconds)")
    for name, code, secs in steps:
        w(f"  {'OK  ' if code == '0' else 'FAIL'} {name:<16} exit={code:<3} {secs}s")

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

    w("-- runner (class/seed, attention actually used)")
    for m in ("var", "dit"):
        r = load(d, f"runner_{m}.json")
        if r:
            a = r["attention"]
            extra = (f"flash_blocks={a['blocks_using_flash']} xformers_blocks={a['blocks_using_xformers']} "
                     f"flash_attn_installed={a['flash_attn_installed']}" if m == "var"
                     else f"fused_attn_blocks={a['blocks_fused_attn']}/{a['blocks']}")
            w(f"  {m}: class={r['class_id']} seed={r['seed']} sampler={r['sampler']} load={r['seconds']['load']}s "
              f"sample={r['seconds']['sample']}s")
            w(f"       {a['impl']}; sdpa kernel={a['sdpa_kernel']['effective']}; {extra}; precision={r['precision']}")
        else:
            w(f"  {m}: (missing)")

    eq = load(d, "equivalence.json")
    w("-- equivalence: runner vs official notebook code path")
    w("   matched = notebook + runner's runtime flags (expect bit-identical);"
      " asis = notebook's own runtime (shows effect of those flags)")
    if eq:
        for name, r in eq.items():
            t, p = r["tensor"], r["pixels"]
            verdict = "IDENTICAL" if t["identical"] else "DIFFERENT"
            w(f"  {name:<12} tensor {verdict:<9} max|diff|={t.get('max_abs_diff')}  "
              f"pixels max_diff={p.get('max_diff')} differing={p.get('frac_differing', 0):.3%}  "
              f"sha={t['sha256'][0][:12]}/{t['sha256'][1][:12]}")
        w("   note: DiT uint8 conversion differs by design (runner: sample_ddp.py formula; notebook: save_image),"
          " so pixel max_diff<=1 with an identical tensor is expected")
    else:
        w("  (missing equivalence.json)")

    w("-- timing (warm mean over reps, one GPU)")
    for m in ("var", "dit"):
        t = load(d, f"timing_{m}.json")
        if not t:
            w(f"  {m}: (missing)")
            continue
        w(f"  {m}: {t['gpu']} load={t['load_s']}s weights={t['weights_gib']} GiB sampler={t['sampler']}")
        for b in t["batches"]:
            e = b["extrapolated"]
            w(f"    B={b['batch_size']:<3} cold={b['cold_s']}s warm={b['warm_mean_s']}s+/-{b['warm_sd_s']} "
              f"per_image={b['per_image_s']}s peak_alloc={b['peak_alloc_gib']} GiB reserved={b['peak_reserved_gib']} GiB")
            w(f"           {t['n_images']} images: {e['n_batches']} batches, {e['sampling_s'] / 60:.1f} min, "
              f"{e['gpu_hours_incl_load']} GPU-h, ~{e['su_incl_load']} SU (incl. one model load)")

    text = "\n".join(L) + "\n"
    (d / "summary.txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
