"""Which runtime setting makes VAR's notebook "asis" output diverge from "matched"?

scripts/phase1/ref_var_notebook.py runs the same notebook code (cell2) twice per process. Between
the two variants, seeds, tf32/cuDNN flags, fp16 autocast, g_seed, sampler settings and the model object
are identical. Exactly two runtime settings differ:
  S1  SDPA kernel: "matched" restricts torch SDPA to the config's kernel (sdp_kernel, math for
      var_d20); "asis" leaves all kernels enabled, so PyTorch chooses (flash / mem_efficient / math).
  S2  run order / process state: "asis" runs first in a fresh process; "matched" runs second, after
      "asis" has initialised CUDA, cuBLAS, the caching allocator etc. in the same process.

    var_divergence.py run --config configs/var_d20.yaml --sdpa KERNEL --manifest M --out-dir DIR
        Fresh process: the notebook path ("matched" branch of cell2, unmodified code) with the config's
        sdpa_backend set to KERNEL (auto = unrestricted, i.e. S1 changed), for every manifest row.
        Writes DIR/<stem>_matched.{pt,png}, DIR/hashes.json, DIR/config.yaml.
    var_divergence.py compare --dir D --runner RUNNER_BS1_DIR --out F [--pixel-threshold 1.0]
        D/{math,auto,flash,mem_efficient}/ from `run`, D/phase1_script/ref_var_{matched,asis}.pt from
        ref_var_notebook.py itself (its "matched" is matched with S2 changed). Changes one setting
        at a time from matched (math, fresh process) and reports which one diverges.
"""

import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
NOTEBOOK_SCRIPT = REPO / "scripts" / "phase1" / "ref_var_notebook.py"
# ref_var_notebook.py ends by running cell2 for both variants; everything before this is setup.
DRIVER_MARKER = "args.out.parent.mkdir(parents=True, exist_ok=True)\nfor variant in"
KERNELS = ("math", "auto", "flash", "mem_efficient")

SETTINGS = {
    "S1": "SDPA kernel: matched = sdp_kernel(config kernel only); asis = all kernels enabled (PyTorch chooses)",
    "S2": "run order: asis first in a fresh process; matched second, in the process asis already warmed",
}
SAME = ("seeds (torch/random/numpy manual_seed, g_seed)", "tf32 flags + set_float32_matmul_precision('high')",
        "cudnn.deterministic=True, cudnn.benchmark=False", "fp16 autocast (cache_enabled=True)",
        "inference_mode", "sampler cfg/top_k/top_p/more_smooth", "model object and weights")


def cmd_run(args):
    import yaml

    from runner.generate import tensor_sha256
    from runner.manifest import load_manifest

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(args.config.read_text())
    cfg["attention"]["sdpa_backend"] = args.sdpa
    for c in cfg["checkpoints"].values():   # keep relative checkpoint paths valid from the new location
        p = Path(c["path"])
        c["path"] = str(p if p.is_absolute() else REPO / p)
    variant_cfg = args.out_dir / "config.yaml"
    variant_cfg.write_text(yaml.safe_dump(cfg, sort_keys=False))

    rows = load_manifest(args.manifest)
    src = NOTEBOOK_SCRIPT.read_text(encoding="utf-8")
    if src.count(DRIVER_MARKER) != 1:
        raise SystemExit(f"{NOTEBOOK_SCRIPT}: driver marker not found exactly once; script changed")
    setup = src[:src.index(DRIVER_MARKER)]
    sys.argv = [str(NOTEBOOK_SCRIPT), "--config", str(variant_cfg), "--class-id", str(rows[0].class_id),
                "--seed", str(rows[0].seed), "--out", str(args.out_dir / rows[0].stem)]
    ns = {"__name__": "ref_var_notebook", "__file__": str(NOTEBOOK_SCRIPT)}
    exec(compile(setup, str(NOTEBOOK_SCRIPT), "exec"), ns)   # unmodified notebook setup: build + load

    import torch
    hashes = {}
    for r in rows:
        ns["args"].class_id, ns["args"].seed, ns["args"].out = r.class_id, r.seed, args.out_dir / r.stem
        ns["cell2"]("matched")   # "matched" branch: kernel from config; sdpa_backend auto -> no restriction
        hashes[r.stem] = tensor_sha256(torch.load(args.out_dir / f"{r.stem}_matched.pt"))
        print(f"{args.sdpa} {r.stem} {hashes[r.stem][:12]}", flush=True)
    (args.out_dir / "hashes.json").write_text(json.dumps(hashes, indent=2))


def diff(a_pt: Path, b_pt: Path) -> dict:
    import numpy as np
    import torch
    from PIL import Image

    from runner.generate import tensor_sha256
    ta, tb = torch.load(a_pt), torch.load(b_pt)
    pa = np.asarray(Image.open(a_pt.with_suffix(".png")).convert("RGB"), dtype=np.int16)
    pb = np.asarray(Image.open(b_pt.with_suffix(".png")).convert("RGB"), dtype=np.int16)
    d = np.abs(pa - pb)
    return {"identical": tensor_sha256(ta) == tensor_sha256(tb),
            "sha256": [tensor_sha256(ta)[:12], tensor_sha256(tb)[:12]],
            "max_abs_diff": float((ta.double() - tb.double()).abs().max()),
            "mean_pixel_diff": float(d.mean()), "max_pixel_diff": int(d.max())}


def cmd_compare(args):
    d = args.dir
    base_dir = d / "math"
    stems = list(json.loads((base_dir / "hashes.json").read_text()))
    first = stems[0]
    base = base_dir / f"{first}_matched.pt"
    single = {   # one setting changed from matched, all on the first manifest row
        "S1 changed: SDPA unrestricted (auto), fresh process": d / "auto" / f"{first}_matched.pt",
        "S2 changed: matched run after asis in one process (Phase 1 script)": d / "phase1_script" / "ref_var_matched.pt",
        "both (Phase 1 asis: unrestricted, first in process)": d / "phase1_script" / "ref_var_asis.pt",
        "diagnostic: flash only": d / "flash" / f"{first}_matched.pt",
        "diagnostic: mem_efficient only": d / "mem_efficient" / f"{first}_matched.pt",
        "runner batch 1 (same row)": args.runner / f"{first}.pt",
    }
    res = {"settings_differing": SETTINGS, "settings_identical": SAME, "baseline": "matched: math kernel, fresh process",
           "row": first, "pixel_threshold": args.pixel_threshold, "single_row": {}, "all_rows": {}}
    for label, p in single.items():
        res["single_row"][label] = diff(p, base) if p.exists() else {"missing": str(p)}

    for k in ("auto", "flash", "mem_efficient"):   # all manifest rows, each kernel vs math
        if not (d / k / "hashes.json").exists():
            res["all_rows"][k] = {"missing": True}
            continue
        per = {s: diff(d / k / f"{s}_matched.pt", base_dir / f"{s}_matched.pt") for s in stems}
        res["all_rows"][k] = {
            "identical_rows": sum(v["identical"] for v in per.values()), "rows": len(stems),
            "visible_rows": [s for s, v in per.items() if v["mean_pixel_diff"] > args.pixel_threshold],
            "max_abs_diff": max(v["max_abs_diff"] for v in per.values()), "per_row": per}
    if (args.runner / "run.json").exists():   # notebook matched (math) vs runner at batch 1, every row
        same = [s for s in stems if diff(args.runner / f"{s}.pt", base_dir / f"{s}_matched.pt")["identical"]]
        res["runner_bs1_vs_matched"] = {"identical_rows": len(same), "rows": len(stems)}

    sr = res["single_row"]
    changed = lambda label: not sr[label].get("identical", False) and "missing" not in sr[label]  # noqa: E731
    s1 = changed("S1 changed: SDPA unrestricted (auto), fresh process")
    s2 = changed("S2 changed: matched run after asis in one process (Phase 1 script)")
    # Which single kernel reproduces the unrestricted run bit-for-bit (i.e. what PyTorch picked)?
    auto_sha = sr["S1 changed: SDPA unrestricted (auto), fresh process"].get("sha256", [None])[0]
    kernel_sha = {"math": sr["S1 changed: SDPA unrestricted (auto), fresh process"].get("sha256", [None, None])[1],
                  "flash": sr["diagnostic: flash only"].get("sha256", [None])[0],
                  "mem_efficient": sr["diagnostic: mem_efficient only"].get("sha256", [None])[0]}
    picked = [k for k, sha in kernel_sha.items() if auto_sha and sha == auto_sha]
    res["verdict"] = {
        "S1_diverges": s1, "S2_diverges": s2,
        "cause": [s for s, bad in (("S1", s1), ("S2", s2)) if bad] or ["none (matched and asis agree on this row)"],
        "unrestricted_sdpa_equals_kernel": picked or ["none of math/flash/mem_efficient"],
    }
    args.out.write_text(json.dumps(res, indent=2))
    print(json.dumps(res["verdict"], indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)
    r = sub.add_parser("run")
    r.add_argument("--config", required=True, type=Path)
    r.add_argument("--sdpa", required=True, choices=KERNELS)
    r.add_argument("--manifest", required=True, type=Path)
    r.add_argument("--out-dir", required=True, type=Path)
    c = sub.add_parser("compare")
    c.add_argument("--dir", required=True, type=Path)
    c.add_argument("--runner", required=True, type=Path, help="runner batch-1 manifest run dir")
    c.add_argument("--out", required=True, type=Path)
    c.add_argument("--pixel-threshold", type=float, default=1.0)
    args = ap.parse_args()
    cmd_run(args) if args.mode == "run" else cmd_compare(args)


if __name__ == "__main__":
    main()
