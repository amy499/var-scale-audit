"""Time one batch through the runner and extrapolate to N images.

    python scripts/phase1/timing.py --config configs/dit_xl2_256.yaml --batch-sizes 1 16 --reps 3 --out timing.json

Per batch size: 1 cold run + `reps` warm runs of runner model.sample() (includes decoding
and the copy to CPU; excludes writing PNGs). Peak VRAM covers all runs at that batch size.
"""

import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import statistics  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch  # noqa: E402

from runner.config import load_config  # noqa: E402
from runner.generate import load_model  # noqa: E402
from runner.runtime import apply_precision  # noqa: E402

SU_PER_GPU_HOUR = 64  # NSCC ASPIRE 2A FAQ: "GPU jobs: 64 SU for 1 ngpus per hour"


def timed(fn, device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    t = time.perf_counter()
    fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    return time.perf_counter() - t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--batch-sizes", type=int, nargs="+", required=True)
    ap.add_argument("--reps", type=int, required=True)
    ap.add_argument("--n-images", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.device:
        cfg["device"] = args.device
    device = torch.device(cfg["device"])
    cuda = device.type == "cuda"
    apply_precision(cfg["precision"])

    t0 = time.perf_counter()
    model = load_model(cfg)
    if cuda:
        torch.cuda.synchronize()
    load_s = time.perf_counter() - t0
    weights_gib = torch.cuda.memory_allocated(device) / 2**30 if cuda else None

    rec = {"model": cfg["model"], "config": cfg["_path"], "sampler": cfg["sampler"], "device": str(device),
           "gpu": torch.cuda.get_device_name(device) if cuda else None,
           "load_s": round(load_s, 2), "weights_gib": weights_gib and round(weights_gib, 2),
           "n_images": args.n_images, "batches": []}
    for bs in args.batch_sizes:
        class_ids = [(207 + i) % 1000 for i in range(bs)]
        if cuda:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
        seeds = [args.seed + i for i in range(bs)]
        run = lambda: model.sample(class_ids, seeds)  # noqa: E731
        cold = timed(run, device)
        warm = [timed(run, device) for _ in range(args.reps)]
        mean = statistics.mean(warm)
        n_batches = math.ceil(args.n_images / bs)
        est_s = n_batches * mean
        b = {
            "batch_size": bs,
            "cold_s": round(cold, 3),
            "warm_s": [round(w, 3) for w in warm],
            "warm_mean_s": round(mean, 3),
            "warm_sd_s": round(statistics.stdev(warm), 3) if len(warm) > 1 else 0.0,
            "per_image_s": round(mean / bs, 4),
            "peak_alloc_gib": round(torch.cuda.max_memory_allocated(device) / 2**30, 2) if cuda else None,
            "peak_reserved_gib": round(torch.cuda.max_memory_reserved(device) / 2**30, 2) if cuda else None,
            "extrapolated": {
                "n_batches": n_batches,
                "sampling_s": round(est_s, 1),
                "gpu_hours_incl_load": round((est_s + load_s) / 3600, 3),
                "su_incl_load": round((est_s + load_s) / 3600 * SU_PER_GPU_HOUR, 1),
            },
        }
        rec["batches"].append(b)
        print(json.dumps(b))

    args.out.write_text(json.dumps(rec, indent=2))


if __name__ == "__main__":
    main()
