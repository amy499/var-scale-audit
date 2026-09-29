"""Log GPU/torch/CUDA facts; exit 2 with a diagnosis if torch sees no GPU.

    python scripts/phase1/gpu_check.py --out gpu.json
"""

import argparse
import json
import os
import platform
import subprocess
import sys

import torch


def nvidia_smi(*args):
    try:
        return subprocess.run(["nvidia-smi", *args], capture_output=True, text=True, timeout=60).stdout.strip()
    except (OSError, subprocess.TimeoutExpired) as e:
        return f"<nvidia-smi failed: {e}>"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    count = torch.cuda.device_count()
    rec = {
        "host": platform.node(),
        "pbs_jobid": os.environ.get("PBS_JOBID"),
        "CUDA_VISIBLE_DEVICES": cvd,
        "nvidia_smi_L": nvidia_smi("-L"),
        "driver": nvidia_smi("--query-gpu=driver_version", "--format=csv,noheader"),
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "cuda_is_available": torch.cuda.is_available(),
        "device_count": count,
        "devices": [
            {"index": i, "name": torch.cuda.get_device_name(i),
             "capability": ".".join(map(str, torch.cuda.get_device_capability(i))),
             "total_mem_gib": round(torch.cuda.get_device_properties(i).total_memory / 2**30, 1)}
            for i in range(count)
        ],
    }

    if count == 0:
        why = []
        if rec["torch_cuda_build"] is None:
            why.append("torch is a CPU-only build (wrong conda env?)")
        if cvd is None:
            why.append("CUDA_VISIBLE_DEVICES is unset: did the job request ngpus (#PBS -l select=1:ngpus=1)?")
        elif cvd.strip() == "":
            why.append("CUDA_VISIBLE_DEVICES is empty: PBS assigned no GPU to this job")
        elif "GPU-" in cvd or "MIG-" in cvd:
            why.append(f"CUDA_VISIBLE_DEVICES holds UUIDs ({cvd}); check that torch/driver accept them")
        if "GPU " not in rec["nvidia_smi_L"]:
            why.append("nvidia-smi lists no GPU on this node")
        rec["diagnosis"] = why or ["unknown; see nvidia_smi_L and CUDA_VISIBLE_DEVICES"]

    with open(args.out, "w") as f:
        json.dump(rec, f, indent=2)
    print(json.dumps(rec, indent=2))

    if count == 0:
        print("\n" + "!" * 70 + "\nFATAL: torch.cuda.device_count() == 0 inside a GPU job.\n  - "
              + "\n  - ".join(rec["diagnosis"]) + "\n" + "!" * 70, file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
