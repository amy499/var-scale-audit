"""Generate images with exactly one model per process.

One image:
    python -m runner.generate --config configs/var_d20.yaml --class-id 207 --seed 0
    -> <out>.png, <out>.json (and <out>.pt with --save-raw)

Every row of a manifest (see runner/manifest.py), in batches:
    python -m runner.generate --config configs/var_d20.yaml --manifest manifest/provisional_4x4.csv --batch-size 16
    -> <out-dir>/class<c>_seed<s>.png and .pt per row, plus <out-dir>/run.json

Each image's randomness comes only from its own seed (one generator per row), so the
output for a row does not depend on --batch-size or on the other rows in the manifest.
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True  # before anything from third_party/ is imported

import torch  # noqa: E402
from PIL import Image  # noqa: E402

from runner import OUTPUTS_DIR  # noqa: E402
from runner.config import load_config  # noqa: E402
from runner.manifest import Row, load_manifest  # noqa: E402
from runner.runtime import apply_precision, effective_precision, environment_record  # noqa: E402


def load_model(cfg: dict):
    # Import only the wrapper for the configured repo.
    if cfg["model"] == "var":
        from runner.var_model import VARModel
        return VARModel(cfg)
    from runner.dit_model import DiTModel
    return DiTModel(cfg)


def tensor_sha256(t: torch.Tensor) -> str:
    """Hash of dtype, shape and raw bytes (same definition as scripts/phase1/compare.py)."""
    h = hashlib.sha256(f"{t.dtype}{tuple(t.shape)}".encode())
    h.update(t.contiguous().numpy().tobytes())
    return h.hexdigest()


def settings_record(cfg: dict, device: torch.device) -> dict:
    return {
        "model": cfg["model"],
        "config_path": cfg["_path"],
        "device": str(device),
        "sampler": cfg["sampler"],
        "build": cfg["build"],
        "checkpoints": {k: v["path"] for k, v in cfg["checkpoints"].items()},
        "precision": effective_precision(cfg["precision"], device),
        "env": environment_record(device),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--device", help="override config device (e.g. cpu for a smoke test)")
    one = ap.add_argument_group("one image")
    one.add_argument("--class-id", type=int)
    one.add_argument("--seed", type=int)
    one.add_argument("--out", type=Path, help="output path without extension (default: outputs/<config>/class<c>_seed<s>)")
    one.add_argument("--save-raw", action="store_true", help="also save the raw output tensor as <out>.pt")
    many = ap.add_argument_group("manifest")
    many.add_argument("--manifest", type=Path, help="generate every row of this manifest")
    many.add_argument("--batch-size", type=int, default=1, help="rows per model call (does not change the images)")
    many.add_argument("--out-dir", type=Path, help="default: outputs/<config>/<manifest name>")
    args = ap.parse_args(argv)

    if args.manifest:
        if args.class_id is not None or args.seed is not None or args.out or args.save_raw:
            ap.error("--manifest cannot be combined with --class-id/--seed/--out/--save-raw")
        if args.batch_size < 1:
            ap.error("--batch-size must be >= 1")
        rows = load_manifest(args.manifest)
    else:
        if args.class_id is None or args.seed is None:
            ap.error("give --class-id and --seed, or --manifest")
        if args.out_dir or args.batch_size != 1:
            ap.error("--out-dir/--batch-size only apply with --manifest")
        rows = [Row(args.class_id, args.seed)]

    cfg = load_config(args.config)
    if args.device:
        cfg["device"] = args.device
    num_classes = cfg["build"]["num_classes"]
    bad = [r for r in rows if not 0 <= r.class_id < num_classes]
    if bad:
        ap.error(f"class_id must be in [0, {num_classes}); got {bad[:5]}")

    device = torch.device(cfg["device"])
    apply_precision(cfg["precision"])
    t0 = time.time()
    model = load_model(cfg)
    t_load = time.time() - t0

    if not args.manifest:
        imgs, raw, attention = model.sample([args.class_id], [args.seed])
        t_sample = time.time() - t0 - t_load
        out = args.out or OUTPUTS_DIR / Path(cfg["_path"]).stem / rows[0].stem
        out.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(imgs[0]).save(out.with_suffix(".png"))
        if args.save_raw:
            torch.save(raw, out.with_suffix(".pt"))
        record = {
            **settings_record(cfg, device),
            "class_id": args.class_id,
            "seed": args.seed,
            "attention": attention,
            "seconds": {"load": round(t_load, 2), "sample": round(t_sample, 2)},
        }
        out.with_suffix(".json").write_text(json.dumps(record, indent=2))
        print(f"wrote {out.with_suffix('.png')}  (attention: {attention['sdpa_kernel']['effective']})")
        return

    out_dir = args.out_dir or OUTPUTS_DIR / Path(cfg["_path"]).stem / args.manifest.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    results, t_sample = [], 0.0
    for start in range(0, len(rows), args.batch_size):
        batch = rows[start:start + args.batch_size]
        t1 = time.time()
        imgs, raw, attention = model.sample([r.class_id for r in batch], [r.seed for r in batch])
        t_sample += time.time() - t1
        for i, r in enumerate(batch):
            one_raw = raw[i:i + 1].clone()  # 1xCxHxW, own storage (a view would save the whole batch)
            Image.fromarray(imgs[i]).save(out_dir / f"{r.stem}.png")
            torch.save(one_raw, out_dir / f"{r.stem}.pt")
            results.append({"class_id": r.class_id, "seed": r.seed, "file": r.stem,
                            "tensor_sha256": tensor_sha256(one_raw)})
        print(f"[{start + len(batch)}/{len(rows)}] rows done", flush=True)

    record = {
        **settings_record(cfg, device),
        "manifest": str(args.manifest.resolve()),
        "batch_size": args.batch_size,
        "attention": attention,
        "seconds": {"load": round(t_load, 2), "sample": round(t_sample, 2)},
        "rows": results,
    }
    (out_dir / "run.json").write_text(json.dumps(record, indent=2))
    print(f"wrote {len(rows)} images to {out_dir}  (attention: {attention['sdpa_kernel']['effective']})")


if __name__ == "__main__":
    main()
