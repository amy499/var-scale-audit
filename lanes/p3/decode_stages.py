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

Writes `<out-dir>/contact_<model>.png`: one row per image, one column per stage, plus the
individual PNGs. Open the contact sheet and look at it.

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
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="default: <stages>/decoded_<model>")
    args = ap.parse_args(argv)

    out = args.out_dir or (args.stages / f"decoded_{args.model}")
    out.mkdir(parents=True, exist_ok=True)

    by_stage = load_stage_files(args.stages, args.model)
    stages = sorted(by_stage)
    keys = sorted(set.intersection(*(set(v) for v in by_stage.values())))[:args.n]
    if not keys:
        raise SystemExit("ERROR: no (class_id, seed) present at every stage — "
                         "run verify_extraction.py first, a batch is probably missing.")
    print(f"decoding {len(keys)} images x {len(stages)} stages from {args.stages}")

    from runner.config import load_config
    from runner.generate import load_model
    cfg = load_config(args.config)
    if cfg["model"] != args.model:
        raise SystemExit(f"ERROR: --model {args.model} but {args.config} is for "
                         f"{cfg['model']!r}. They must match or the decode is meaningless.")
    print(f"loading {args.model} (VAE only is used, the transformer is never called)…")
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

    sheet = out / f"contact_{args.model}.png"
    expect = ("coarse -> fine" if args.model == "var" else "noisy -> clean")
    contact_sheet(rows, stages, keys, sheet,
                  f"P3 extraction check — {args.model}: expect {expect} left to right")

    print(f"\nwrote {sheet}")
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
