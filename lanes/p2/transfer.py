"""P2 Phase F (optional, DiT only): does DiT's recovery transfer to real out-of-domain images (DIBaS)?

Follows the plan's DiT transfer recipe: check the VAE reconstruction first, then put a real image's latent
at a chosen diffusion point (forward-noised to that timestep) and let the released model denoise it from
there. Low starting timesteps test restoration, high ones how much of the image the model replaces.

    python -m lanes.p2.transfer manifest REAL_DIR --out M.csv [--class-id 0]
        -> one manifest row per image in REAL_DIR (sorted by name): row i <-> image i, seed i
    python -m lanes.p2.transfer recon REAL_DIR --out DIR
        -> DIR/<stem>.png: the 256 x 256 input and its VAE reconstruction side by side;
           DIR/recon.csv: PSNR of the reconstruction (the codec check: Gate 4, representation failure)
    python -m lanes.p2.transfer score RUN_DIR --real-dir REAL_DIR --manifest M.csv
        -> RUN_DIR/transfer.csv, per image: psnr to the real input, hf_ratio (high-frequency energy of the
           output / of the input: < 1 smoothed, > 1 invented texture), and the means printed

Hook (Modify, before the step at timestep P2_STAGE):  --hook lanes/p2/transfer.py:real_x
    P2_REAL_DIR       the images; P2_REAL_MANIFEST the manifest made by `manifest` (maps rows to images)
    P2_STAGE          the starting timestep (a timestep_in of run.json "stages"); P2_NOISE_SEED as in hooks.py
It replaces x_t with sqrt(alpha_bar) z0 + sqrt(1 - alpha_bar) eps, z0 = 0.18215 x the VAE encoder's mean;
the steps before it run as usual and are discarded. The model is class-conditional (the runner accepts
no null class), so every row uses one fixed ImageNet class; run with cfg_scale 1 (jobs/dit_cfg1.yaml) so
guidance does not pull towards that class. Needs a GPU for the hook; the CLI runs on CPU too.
"""

import argparse
import csv
import math
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True

REPO = Path(__file__).resolve().parents[2]
VAE_PATH = REPO / "checkpoints" / "sd-vae-ft-ema"
EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
_CACHE = {}


def images(real_dir) -> list[Path]:
    files = sorted(p for p in Path(real_dir).iterdir() if p.suffix.lower() in EXTS)
    if not files:
        raise SystemExit(f"ERROR: no images in {real_dir}")
    return files


def load_image(path, size: int = 256):
    """RGB in [-1, 1], (3, size, size): shorter side resized to size, then centre crop."""
    import numpy as np
    import torch
    from PIL import Image
    im = Image.open(path).convert("RGB")
    w, h = im.size
    s = size / min(w, h)
    im = im.resize((max(size, round(w * s)), max(size, round(h * s))), Image.BICUBIC)
    w, h = im.size
    im = im.crop(((w - size) // 2, (h - size) // 2, (w - size) // 2 + size, (h - size) // 2 + size))
    return torch.from_numpy(np.asarray(im, dtype=np.float32) / 127.5 - 1).permute(2, 0, 1)


def vae(device):
    key = f"vae:{device}"
    if key not in _CACHE:
        from diffusers.models import AutoencoderKL
        _CACHE[key] = AutoencoderKL.from_pretrained(str(VAE_PATH)).to(device).eval()
    return _CACHE[key]


def to_uint8(x):
    import torch
    return torch.clamp(127.5 * x + 128.0, 0, 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()


def psnr(a, b) -> float:   # uint8 arrays
    mse = float(((a.astype("float64") - b.astype("float64")) ** 2).mean())
    return 99.0 if mse == 0 else 10 * math.log10(255 ** 2 / mse)


def hf_energy(a) -> float:
    """Mean squared Laplacian of the grey image: how much fine texture / edge there is."""
    import numpy as np
    g = a.astype("float64").mean(axis=2)
    lap = 4 * g[1:-1, 1:-1] - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:]
    return float((lap ** 2).mean())


# ---------------------------------------------------------------- hook

def _row_images() -> dict:
    if "rows" not in _CACHE:
        files = images(os.environ["P2_REAL_DIR"])
        rows = [(int(r["class_id"]), int(r["seed"])) for r in csv.DictReader(open(os.environ["P2_REAL_MANIFEST"]))]
        if len(rows) != len(files):
            raise RuntimeError(f"P2_REAL_MANIFEST has {len(rows)} rows, P2_REAL_DIR {len(files)} images")
        _CACHE["rows"] = dict(zip(rows, files))
    return _CACHE["rows"]


def _real_x(model, stage, p, when, state):
    import dataclasses
    import torch
    from lanes.p2.hooks import noise_like
    if model != "dit":
        raise RuntimeError("real_x is a DiT hook")
    files = _row_images()
    x = state.x
    with torch.no_grad():
        img = torch.stack([load_image(files[tuple(r)]) for r in state.rows]).to(x.device)
        z0 = 0.18215 * vae(x.device).encode(img).latent_dist.mean
    ab = state.fields["alpha_bar_in"]
    eps = noise_like(x, state.rows, stage, "real", int(os.environ.get("P2_NOISE_SEED") or 0))
    return dataclasses.replace(state, x=(ab ** 0.5 * z0 + (1 - ab) ** 0.5 * eps).to(x.dtype))


def __getattr__(name):
    if name == "real_x":   # built on request, like the corruption hooks in hooks.py
        from runner.hooks import Modify
        stage = int(os.environ["P2_STAGE"])
        return Modify(_real_x, when="before", stages=[stage], name=f"p2_real_x_t{stage}")
    raise AttributeError(name)


# ---------------------------------------------------------------- CLI

def make_manifest(real_dir, out, class_id: int = 0):
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class_id", "seed"])
        for i, _ in enumerate(images(real_dir)):
            w.writerow([class_id, i])


def recon(real_dir, out, device):
    import numpy as np
    import torch
    from PIL import Image
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in images(real_dir):
        x = load_image(path)[None].to(device)
        with torch.no_grad():
            y = vae(device).decode(vae(device).encode(x).latent_dist.mean).sample
        a, b = to_uint8(x)[0], to_uint8(y)[0]
        Image.fromarray(np.concatenate([a, b], axis=1)).save(out / f"{path.stem}.png")
        rows.append({"image": path.name, "psnr": round(psnr(a, b), 3), "hf_ratio": round(hf_energy(b) / hf_energy(a), 4)})
    _write(out / "recon.csv", rows)


def score(run_dir, real_dir, manifest):
    import numpy as np
    from PIL import Image
    run_dir = Path(run_dir)
    files = images(real_dir)
    rows = []
    for r, path in zip(csv.DictReader(open(manifest)), files):
        stem = f"class{int(r['class_id']):04d}_seed{int(r['seed'])}"
        a = (load_image(path).permute(1, 2, 0).numpy() * 127.5 + 128).clip(0, 255).astype("uint8")
        b = np.asarray(Image.open(run_dir / f"{stem}.png").convert("RGB"))
        rows.append({"image": path.name, "file": stem, "psnr": round(psnr(a, b), 3),
                     "hf_ratio": round(hf_energy(b) / hf_energy(a), 4)})
    _write(run_dir / "transfer.csv", rows)


def _write(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    n = len(rows)
    print(f"{path}: {n} images, mean psnr {sum(r['psnr'] for r in rows) / n:.2f} dB, "
          f"mean hf_ratio {sum(r['hf_ratio'] for r in rows) / n:.3f}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("manifest", help="manifest with one row per real image")
    m.add_argument("real_dir")
    m.add_argument("--out", required=True)
    m.add_argument("--class-id", type=int, default=0)
    r = sub.add_parser("recon", help="VAE reconstruction check")
    r.add_argument("real_dir")
    r.add_argument("--out", required=True)
    r.add_argument("--device", default="cuda")
    s = sub.add_parser("score", help="PSNR and texture ratio of a run's outputs against the real inputs")
    s.add_argument("run_dir")
    s.add_argument("--real-dir", required=True)
    s.add_argument("--manifest", required=True)
    args = ap.parse_args(argv)
    if args.cmd == "manifest":
        make_manifest(args.real_dir, args.out, args.class_id)
    elif args.cmd == "recon":
        import torch
        recon(args.real_dir, args.out, args.device if torch.cuda.is_available() else "cpu")
    else:
        score(args.run_dir, args.real_dir, args.manifest)


if __name__ == "__main__":
    main()
