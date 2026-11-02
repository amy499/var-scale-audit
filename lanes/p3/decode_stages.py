"""Check 3 of P3 Step 2: decode saved intermediate states and LOOK at them.

This is the check that catches a wrong extraction. Checks 1 and 2
(`verify_extraction.py`) prove the plumbing works; only your eyes prove the saved states
mean what you think they mean. A hook that saves the wrong tensor, or saves it at the wrong
moment, passes both earlier checks and produces a perfectly smooth, perfectly meaningless
curve.

WHAT YOU SHOULD SEE
-------------------
    VAR   scale 0 -> 9     a blob resolving into a dog: coarse to fine
    DiT   step 0 -> 249    noise resolving into a dog: noisy to clean

If VAR's early scales already look sharp, the VQVAE decoder is adding texture that the model
has not committed to, and every pixel-space metric is partly measuring the decoder rather
than the model. That is a finding, not a bug — record it and lean on `latent_rel_dist`.

If the panels look random, or identical to each other, or the order is wrong, the extraction
is broken. Stop and fix it.

Compare what you see against the ladder table printed by
`python -m lanes.p3.calibrate_metrics`, which is the best case at each VAR scale on an ideal
codec.

VAE-ONLY MODE (--vae-only): run this on a laptop, no GPU
--------------------------------------------------------
`fhat_to_img` is `self.decoder(self.post_quant_conv(f_hat))`. **The VAR transformer is never
touched.** So check 3 needs only the VQVAE:

    checkpoints/var/vae_ch160v4096z32.pth     436 MB
    checkpoints/var/var_d20.pth               2.4 GB   <- NOT needed with --vae-only

With --vae-only the transformer is never built or loaded, `require_checkpoints` is not
called, and the decode runs on CPU in a couple of minutes for 5 images x 10 scales. That
turns check 3 from "needs cluster access" into "needs a 436 MB download".

**DiT works the same way.** Its decoder is a diffusers AutoencoderKL (sd-vae-ft-ema), two
files listed with pinned sha256 in configs/dit_xl2_256.yaml:

    checkpoints/sd-vae-ft-ema/config.json                          < 1 KB
    checkpoints/sd-vae-ft-ema/diffusion_pytorch_model.safetensors  ~335 MB
    checkpoints/dit/DiT-XL-2-256x256.pt                            2.7 GB  <- NOT needed

Both files are checked against their sha256 BEFORE torch is imported, so a missing or
half-finished download fails at once with the exact command to fix it. The load itself is
the same call runner/dit_model.py makes on the cluster.

Normal mode (no flag) still loads the full model, which is what you want if the checkpoints
are already there.

STAGE ORDER
-----------
VAR scales count UP (0 = coarsest). DiT timesteps count DOWN: 999 is pure noise, 0 is the
finished image. Stages are therefore decoded and laid out in GENERATION order — VAR 0 -> 9,
DiT 999 -> 0 — so the contact sheet always reads left to right as the model built the
image. (Sorting DiT's timesteps numerically would put the clean image first and make a
correct extraction look reversed.)

HOW IT DECODES
--------------
    VAR   vae.fhat_to_img(f_hat) -> [-1, 1] -> add 1, times 0.5 -> [0, 1]
          (f_hat is "the only input to the decoder", docs/hook_interface.md line 372)
    DiT   vae.decode(x / 0.18215).sample -> 127.5 * x + 128 -> [0, 255]
          (exactly as runner/dit_model.py does for the final image)

Only the VAE is used; the transformer is never called, so this is far cheaper than sampling.
It still loads the full model wrapper, because that is the one place the checkpoints, the
precision and the device are set up correctly.

RUN IT
------
    python -m lanes.p3.decode_stages --stages outputs/p3/pilot/stages \
        --model var --config configs/var_d20.yaml --n 5

    # DiT, laptop: the same 32 images the VAR semantic curve used, 8 rows on the sheet
    python -m lanes.p3.decode_stages --stages 25985813_p3_dit_stages --model dit \
        --config configs/dit_xl2_256.yaml --vae-only --device mps --n 32 --sheet-rows 8

    # DiT finished images only, for exactly the images in a folder of baseline PNGs
    python -m lanes.p3.decode_stages --stages 25985813_p3_dit_stages --model dit \
        --config configs/dit_xl2_256.yaml --vae-only --device mps \
        --keys-like for_darien_20_clean_pngs --final-only --out-dir outputs/p3/dit_finals

Writes `<out-dir>/contact_<model>.png`: one row per image, one column per stage, plus the
individual PNGs. Open the contact sheet and look at it. (With --final-only there is one
stage, so only the PNGs are written.)

Why DiT's final stage IS the finished image: at the last DDPM step (t = 0) the posterior
mean has coefficient 1 on pred_xstart and 0 on x_t, so the sample the runner returns is
exactly pred_xstart at t = 0. Decoding it here reproduces the baseline PNG up to the
VAE's float rounding on a different device (a level or so out of 255).

Owner: P3.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def load_vae_only(cfg, model_name, device=None):
    """Build and load ONLY the VQVAE. Returns an object with .vae and .device.

    `fhat_to_img` is `decoder(post_quant_conv(f_hat))`, so the transformer is dead weight
    for check 3 — 2.4 GB of it. Skipping it means the check needs a 436 MB download and no
    GPU, which is the difference between "ask someone with cluster access" and "run it on
    your laptop over lunch".

    The build kwargs are read from the same config the real runner uses, so the decoder is
    bit-for-bit the one that produced the images.

    DiT is handled by `_load_dit_vae` (a diffusers AutoencoderKL, loaded differently).
    """
    if model_name == "dit":
        return _load_dit_vae(cfg, device)

    import torch
    from runner import upstream
    from runner.config import resolve_path

    ckpt = resolve_path(cfg["checkpoints"]["vae"]["path"])
    if not ckpt.is_file():
        url = cfg["checkpoints"]["vae"].get("url", "")
        raise SystemExit(
            f"ERROR: VAE checkpoint not found: {ckpt}\n"
            f"       --vae-only needs ONLY this file (436 MB), not var_d20.pth (2.4 GB).\n"
            f"       mkdir -p {ckpt.parent}\n"
            f"       curl -L -o {ckpt} \\\n         {url}")

    upstream.activate("var")
    from models.vqvae import VQVAE

    b = cfg["build"]
    dev = torch.device(device or "cpu")
    print(f"building VQVAE only (ch={b['ch']}, Cvae={b['Cvae']}) on {dev} — "
          f"transformer not loaded")
    vae = VQVAE(vocab_size=b["V"], z_channels=b["Cvae"], ch=b["ch"], test_mode=True,
                share_quant_resi=b["share_quant_resi"],
                v_patch_nums=tuple(b["patch_nums"])).to(dev)
    vae.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=True)
    vae.eval().requires_grad_(False)

    class _VaeOnly:
        pass
    m = _VaeOnly()
    m.vae, m.device = vae, dev
    return m


def _sha256(path, chunk=8 << 20):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def _show(path):
    """A path as a user should type it: repo-relative when possible, and shell-quoted,
    because the repo may live under a folder whose name contains spaces."""
    import shlex
    path = Path(path)
    try:
        path = path.resolve().relative_to(REPO)
    except ValueError:
        pass
    return shlex.quote(str(path))


def _load_dit_vae(cfg, device=None):
    """Build and load ONLY DiT's decoder: the diffusers AutoencoderKL (sd-vae-ft-ema).

    Decoding `pred_xstart` never touches the DiT transformer (2.7 GB). The VAE is two files,
    ~335 MB, each pinned by sha256 in configs/dit_xl2_256.yaml. Both are checked BEFORE
    torch is imported, so a missing or corrupt download fails fast with the command that
    fixes it. The load is the same call runner/dit_model.py makes on the cluster.
    """
    import os
    import shlex
    from runner.config import resolve_path

    entry = cfg["checkpoints"]["vae"]
    root = resolve_path(entry["path"])
    files = entry.get("files") or {}
    if not files:
        raise SystemExit(f"ERROR: {cfg.get('_path', 'the config')} lists no files under "
                         f"checkpoints.vae — expected config.json and the .safetensors")

    missing = [f for f in files if not (root / f).is_file()]
    if missing:
        lines = [f"ERROR: DiT VAE file(s) missing in {_show(root)}: {', '.join(missing)}",
                 "       --vae-only needs ONLY these two files (~335 MB), not the 2.7 GB "
                 "DiT transformer.",
                 "       From the repo root, run:",
                 f"       mkdir -p {_show(root)}"]
        for fname in missing:
            lines.append(f"       curl -L -o {_show(root / fname)} "
                         f"{shlex.quote(files[fname]['url'])}")
        raise SystemExit("\n".join(lines))

    for fname, meta in files.items():
        want = (meta or {}).get("sha256")
        if not want:
            continue
        got = _sha256(root / fname)
        if got != want:
            raise SystemExit(
                f"ERROR: {_show(root / fname)} is not the pinned file.\n"
                f"       sha256   {got}\n"
                f"       expected {want}\n"
                f"       The download is incomplete or wrong: delete that file and download "
                f"it again.")
    print(f"DiT VAE: {len(files)} files present in {_show(root)}, sha256 verified")

    os.environ.setdefault("HF_HUB_OFFLINE", "1")   # load the local folder; never the network
    import torch
    # diffusers 0.27.2 reads the weights with `safetensors.torch.load_file` but never imports
    # `safetensors.torch` itself. On the cluster something else always has: timm (pulled in
    # by the DiT transformer) and accelerate (in environment.yml). The VAE-only path loads
    # neither, and environment-macos.yml leaves accelerate out, so without this line the load
    # fails with "module 'safetensors' has no attribute 'torch'".
    import safetensors.torch  # noqa: F401
    from diffusers.models import AutoencoderKL       # as runner/dit_model.py

    dev = torch.device(device or "cpu")
    print(f"building AutoencoderKL only on {dev} — DiT transformer not loaded")
    # low_cpu_mem_usage=False is what diffusers falls back to anyway without accelerate;
    # saying so explicitly just removes its "pip install accelerate" warning. Same weights.
    vae = AutoencoderKL.from_pretrained(str(root), low_cpu_mem_usage=False).to(dev)
    vae.eval().requires_grad_(False)

    class _VaeOnly:
        pass
    m = _VaeOnly()
    m.vae, m.device = vae, dev
    return m


def keys_from_pngs(folder):
    """{(class_id, seed)} named by the PNGs in `folder` (class<c>_seed<s>[...].png)."""
    import re
    pat = re.compile(r"^class(\d+)_seed(\d+)(?:_stage\d+)?\.png$")
    folder = Path(folder)
    if not folder.is_dir():
        raise SystemExit(f"ERROR: --keys-like folder not found: {folder}")
    keys = set()
    for p in folder.glob("*.png"):
        m = pat.match(p.name)
        if m:
            keys.add((int(m.group(1)), int(m.group(2))))
    if not keys:
        raise SystemExit(f"ERROR: no class<N>_seed<N>.png files in {folder}")
    return keys


def spread_across_classes(keys, n):
    """Pick n keys round-robin across classes, instead of the first n in sorted order.

    Sorting by (class_id, seed) groups every class together, so `[:n]` with n=32 returns
    32 images of ONE class. A semantic curve measured on 32 Chihuahuas is not a result
    about the model. Round-robin gives the widest class coverage n allows.
    """
    by_class = {}
    for k in keys:
        by_class.setdefault(k[0], []).append(k)
    picked, depth = [], 0
    while len(picked) < n:
        added = False
        for cls_keys in by_class.values():
            if depth < len(cls_keys):
                picked.append(cls_keys[depth]); added = True
                if len(picked) == n:
                    break
        if not added:
            break
        depth += 1
    return picked


def load_stage_files(stages_dir, model):
    """Group saved .npz by stage, keyed by (class_id, seed) so rows line up across stages."""
    d = Path(stages_dir) / model
    files = sorted(d.glob("stage*.npz"))
    if not files:
        raise SystemExit(f"ERROR: no stage*.npz in {d}")
    by_stage = {}
    for f in files:
        z = np.load(f)
        stage = int(z["stage"])
        entry = by_stage.setdefault(stage, {})
        for i, (c, s) in enumerate(zip(z["class_id"], z["seed"])):
            entry[(int(c), int(s))] = z["latent"][i]
    return by_stage


def decode_var(model, latents):
    """f_hat (N, Cvae, 16, 16) -> uint8 images (N, H, W, 3)."""
    import torch
    out = []
    with torch.inference_mode():
        for lat in latents:
            f_hat = torch.from_numpy(np.asarray(lat)).unsqueeze(0).to(model.device).float()
            img = model.vae.fhat_to_img(f_hat)          # [-1, 1]
            img = img.add(1).mul(0.5).clamp(0, 1)       # [0, 1], as upstream var.py:190 does
            out.append((img[0].permute(1, 2, 0).mul(255)
                        .cpu().numpy().astype(np.uint8)))
    return out


def decode_dit(model, latents):
    """pred_xstart (N, 4, 32, 32) -> uint8 images (N, H, W, 3)."""
    import torch
    out = []
    with torch.inference_mode():
        for lat in latents:
            x = torch.from_numpy(np.asarray(lat)).unsqueeze(0).to(model.device).float()
            img = model.vae.decode(x / 0.18215).sample          # as runner/dit_model.py
            img = torch.clamp(127.5 * img + 128.0, 0, 255)
            out.append(img[0].permute(1, 2, 0).to("cpu", dtype=torch.uint8).numpy())
    return out


def contact_sheet(rows, stages, keys, path, title):
    """One row per image, one column per stage."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_img, n_stage = len(keys), len(stages)
    fig, axes = plt.subplots(n_img, n_stage,
                             figsize=(1.5 * n_stage, 1.6 * n_img), squeeze=False)
    for i, key in enumerate(keys):
        for j, st in enumerate(stages):
            ax = axes[i][j]
            ax.imshow(rows[(key, st)])
            ax.axis("off")
            if i == 0:
                ax.set_title(str(st), fontsize=7)
        axes[i][0].set_ylabel(f"{key[0]}/{key[1]}", fontsize=6)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stages", required=True, type=Path, help="P3_OUT from the hook run")
    ap.add_argument("--model", required=True, choices=("var", "dit"))
    ap.add_argument("--config", required=True, type=Path,
                    help="configs/var_d20.yaml or configs/dit_xl2_256.yaml")
    ap.add_argument("--n", type=int, default=5, help="how many images to decode (default 5)")
    ap.add_argument("--vae-only", action="store_true",
                    help="load just the decoder, no GPU: VAR's VQVAE (436 MB) or DiT's "
                         "AutoencoderKL (~335 MB) — never the transformer")
    ap.add_argument("--device", help="cpu | mps | cuda (default: cpu with --vae-only)")
    ap.add_argument("--first-n", action="store_true",
                    help="take the first n keys in sorted order (one class) instead of "
                         "spreading across classes — almost never what you want")
    ap.add_argument("--keys-like", type=Path, metavar="DIR",
                    help="decode exactly the images named in this folder of PNGs "
                         "(class<c>_seed<s>.png), e.g. for_darien_20_clean_pngs; replaces --n")
    ap.add_argument("--final-only", action="store_true",
                    help="decode only the finished image (VAR: scale 9; DiT: timestep 0)")
    ap.add_argument("--sheet-rows", type=int, default=None, metavar="N",
                    help="show only the first N images on the contact sheet (every PNG is "
                         "still written). Useful for DiT: 24 columns x 32 rows is unwieldy")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="default: <stages>/decoded_<model>")
    args = ap.parse_args(argv)

    out = args.out_dir or (args.stages / f"decoded_{args.model}")
    out.mkdir(parents=True, exist_ok=True)

    by_stage = load_stage_files(args.stages, args.model)
    # Generation order: VAR scales count up (0 -> 9); DiT timesteps count DOWN (999 = noise,
    # 0 = the finished image). Either way the final stage ends up LAST.
    stages = sorted(by_stage, reverse=(args.model == "dit"))
    available = sorted(set.intersection(*(set(v) for v in by_stage.values())))
    if not available:
        raise SystemExit("ERROR: no (class_id, seed) present at every stage — "
                         "run verify_extraction.py first, a batch is probably missing.")
    if args.final_only:
        stages = stages[-1:]

    if args.keys_like:
        wanted = keys_from_pngs(args.keys_like)
        keys = [k for k in available if k in wanted]
        lost = sorted(wanted - set(available))
        if not keys:
            raise SystemExit(f"ERROR: none of the {len(wanted)} images named in "
                             f"{args.keys_like} are in {args.stages}")
        if lost:
            print(f"  WARNING: {len(lost)} of {len(wanted)} images in {args.keys_like} are not "
                  f"in the stage files, e.g. class {lost[0][0]} seed {lost[0][1]}")
    elif args.first_n:
        keys = available[:args.n]
    else:
        keys = spread_across_classes(available, args.n)
    ncls = len({k[0] for k in keys})
    print(f"decoding {len(keys)} images x {len(stages)} stages from {args.stages} "
          f"({ncls} class{'es' if ncls != 1 else ''})")
    if ncls == 1 and len({k[0] for k in available}) > 1 and not args.keys_like:
        print("  WARNING: all picked images are one class — drop --first-n for a spread")

    from runner.config import load_config
    cfg = load_config(args.config)
    if cfg["model"] != args.model:
        raise SystemExit(f"ERROR: --model {args.model} but {args.config} is for "
                         f"{cfg['model']!r}. They must match or the decode is meaningless.")

    if args.vae_only:
        model = load_vae_only(cfg, args.model, args.device)
    else:
        from runner.generate import load_model
        print(f"loading {args.model} (only the VAE is used; --vae-only skips the rest)…")
        model = load_model(cfg)

    decode = decode_var if args.model == "var" else decode_dit
    from PIL import Image
    rows = {}
    for st in stages:
        imgs = decode(model, [by_stage[st][k] for k in keys])
        for k, im in zip(keys, imgs):
            rows[(k, st)] = im
            Image.fromarray(im).save(out / f"class{k[0]:04d}_seed{k[1]}_stage{st:04d}.png")
        print(f"  stage {st} done", flush=True)

    if len(stages) == 1:
        print(f"\nwrote {len(keys)} final-stage PNGs to {out} (one stage, so no contact sheet)")
        return

    sheet = out / f"contact_{args.model}.png"
    expect = ("coarse -> fine" if args.model == "var" else "noisy -> clean")
    sheet_keys = keys[:args.sheet_rows] if args.sheet_rows else keys
    contact_sheet(rows, stages, sheet_keys, sheet,
                  f"P3 extraction check — {args.model}: expect {expect} left to right")

    shown = (f" ({len(sheet_keys)} of {len(keys)} images; all {len(keys)} written as PNGs)"
             if len(sheet_keys) < len(keys) else "")
    print(f"\nwrote {sheet}{shown}")
    print("\nNOW LOOK AT IT. You are checking three things:")
    print(f"  1. left to right goes {expect}")
    print("  2. the last column matches that image's final baseline PNG")
    print("  3. rows are different images, not the same one repeated")
    if args.model == "var":
        print("\n  If the early scales already look SHARP, the VQVAE decoder is adding texture")
        print("  the model has not committed to. That is a finding: record it, and use")
        print("  latent_rel_dist rather than the pixel metrics. Compare against the ladder")
        print("  table from `python -m lanes.p3.calibrate_metrics`.")


if __name__ == "__main__":
    main()
