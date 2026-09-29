"""Compare runner output against notebook-reference output.

    python scripts/phase1/compare.py --out equivalence.json NAME RUNNER_PREFIX REF_PREFIX [NAME ...]

Each PREFIX has .pt (raw output tensor) and .png (uint8 image) next to it.
Imports neither upstream repo.
"""

import argparse
import hashlib
import json

import numpy as np
import torch
from PIL import Image


def tensor_sha(t: torch.Tensor) -> str:
    h = hashlib.sha256(f"{t.dtype}{tuple(t.shape)}".encode())
    h.update(t.contiguous().numpy().tobytes())
    return h.hexdigest()


def compare(a_prefix: str, b_prefix: str) -> dict:
    ta, tb = torch.load(f"{a_prefix}.pt"), torch.load(f"{b_prefix}.pt")
    pa = np.asarray(Image.open(f"{a_prefix}.png").convert("RGB"), dtype=np.int16)
    pb = np.asarray(Image.open(f"{b_prefix}.png").convert("RGB"), dtype=np.int16)
    rec = {
        "tensor": {
            "dtype": [str(ta.dtype), str(tb.dtype)],
            "shape": [list(ta.shape), list(tb.shape)],
            "sha256": [tensor_sha(ta), tensor_sha(tb)],
        },
        "pixels": {"shape": [list(pa.shape), list(pb.shape)]},
    }
    rec["tensor"]["identical"] = rec["tensor"]["sha256"][0] == rec["tensor"]["sha256"][1]
    if ta.shape == tb.shape:
        rec["tensor"]["max_abs_diff"] = float((ta.double() - tb.double()).abs().max())
    if pa.shape == pb.shape:
        d = np.abs(pa - pb)
        rec["pixels"].update(identical=bool((d == 0).all()), max_diff=int(d.max()),
                             frac_differing=float((d.max(axis=-1) > 0).mean()))
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("triples", nargs="+", help="NAME RUNNER_PREFIX REF_PREFIX, repeated")
    args = ap.parse_args()
    if len(args.triples) % 3:
        ap.error("arguments must come in NAME RUNNER_PREFIX REF_PREFIX triples")
    results = {}
    for name, a, b in zip(*[iter(args.triples)] * 3):
        results[name] = compare(a, b)
        t, p = results[name]["tensor"], results[name]["pixels"]
        print(f"{name}: tensor identical={t['identical']} max|diff|={t.get('max_abs_diff')}  "
              f"pixels max diff={p.get('max_diff')} ({p.get('frac_differing', 0):.2%} of pixels)")
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
