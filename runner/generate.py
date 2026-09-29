"""Generate one image for one (class, seed) with exactly one model per process.

    python -m runner.generate --config configs/var_d20.yaml --class-id 207 --seed 0

Writes <out>.png and <out>.json (settings, attention backend, versions).
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True  # before anything from third_party/ is imported

import torch  # noqa: E402
from PIL import Image  # noqa: E402

from runner import OUTPUTS_DIR  # noqa: E402
from runner.config import load_config  # noqa: E402
from runner.runtime import apply_precision, effective_precision, environment_record  # noqa: E402


def load_model(cfg: dict):
    # Import only the wrapper for the configured repo.
    if cfg["model"] == "var":
        from runner.var_model import VARModel
        return VARModel(cfg)
    from runner.dit_model import DiTModel
    return DiTModel(cfg)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--class-id", required=True, type=int)
    ap.add_argument("--seed", required=True, type=int)
    ap.add_argument("--out", type=Path, help="output path without extension (default: outputs/<config>/class<c>_seed<s>)")
    ap.add_argument("--device", help="override config device (e.g. cpu for a smoke test)")
    ap.add_argument("--save-raw", action="store_true", help="also save the raw output tensor as <out>.pt")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    if args.device:
        cfg["device"] = args.device
    num_classes = cfg["build"]["num_classes"]
    if not 0 <= args.class_id < num_classes:
        ap.error(f"--class-id must be in [0, {num_classes})")

    device = torch.device(cfg["device"])
    apply_precision(cfg["precision"])
    t0 = time.time()
    model = load_model(cfg)
    t_load = time.time() - t0
    imgs, raw, attention = model.sample([args.class_id], args.seed)
    t_sample = time.time() - t0 - t_load

    out = args.out or OUTPUTS_DIR / Path(cfg["_path"]).stem / f"class{args.class_id:04d}_seed{args.seed}"
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(imgs[0]).save(out.with_suffix(".png"))
    if args.save_raw:
        torch.save(raw, out.with_suffix(".pt"))
    record = {
        "model": cfg["model"],
        "config_path": cfg["_path"],
        "class_id": args.class_id,
        "seed": args.seed,
        "device": str(device),
        "sampler": cfg["sampler"],
        "build": cfg["build"],
        "checkpoints": {k: v["path"] for k, v in cfg["checkpoints"].items()},
        "attention": attention,
        "precision": effective_precision(cfg["precision"], device),
        "env": environment_record(device),
        "seconds": {"load": round(t_load, 2), "sample": round(t_sample, 2)},
    }
    out.with_suffix(".json").write_text(json.dumps(record, indent=2))
    print(f"wrote {out.with_suffix('.png')}  (attention: {attention['sdpa_kernel']['effective']})")


if __name__ == "__main__":
    main()
