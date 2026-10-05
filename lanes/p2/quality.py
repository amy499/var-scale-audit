"""Interim quality / semantic metric for P2: what an ImageNet classifier sees in each generated image.

Until P3's calibrated metrics exist, this separates "the image changed but is still a good image of its
class" from "the image is broken", which pixel distance cannot (lanes/p2/results/pilot_25656645).

    python -m lanes.p2.quality --download                   # once, where there is internet (login node, laptop)
    python -m lanes.p2.quality RUN_DIR [RUN_DIR ...]        # GPU if available, else CPU
        -> appends per image to RUN_DIR/metrics.csv (replacing earlier values of these metrics):
           cls_prob   classifier probability of the image's own class (manifest class_id)
           cls_top1   1 if that class is the classifier's top prediction, else 0
           cls_top5   1 if it is among the top 5, else 0
           cls_pred   the top predicted class index (lanes.p2.analyze compares it with the baseline's)
        -> RUN_DIR/inception_features.npz: each image's 2048-d Inception pool features (class_id, seed, feat),
           from which lanes.p2.analyze computes KID between a run and its baseline (distribution-level quality)

The classifier measures meaning: it still recognises a dog in an oversaturated, artifact-heavy image.
KID measures whether the images still look like clean images as a set: a different but valid image
keeps it low, visible damage raises it. Use both.

Classifier: torchvision Inception-v3 with its ImageNet weights (the network family behind the IS / FID
scores both models report), weights pinned by URL and hash in checkpoints/classifier/. The 256 x 256
PNGs are resized to 299 x 299 (no crop) and normalised as for ImageNet. Uses torch/torchvision/Pillow
from var-dit only; nothing to install.
"""

import argparse
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True

REPO = Path(__file__).resolve().parents[2]
WEIGHTS_URL = "https://download.pytorch.org/models/inception_v3_google-0cc3c7bd.pth"   # torchvision IMAGENET1K_V1
WEIGHTS_DIR = REPO / "checkpoints" / "classifier"
METRICS = ("cls_prob", "cls_top1", "cls_top5", "cls_pred")
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def load_classifier(device, download: bool = False):
    import torch
    import torchvision
    path = WEIGHTS_DIR / Path(WEIGHTS_URL).name
    if not path.is_file() and not download:
        raise SystemExit(f"ERROR: classifier weights not downloaded: {path}\n"
                         "       run once where there is internet (login node): python -m lanes.p2.quality --download")
    # check_hash: the file name carries the first 8 hex digits of its sha256; torch verifies them on download.
    state = torch.hub.load_state_dict_from_url(WEIGHTS_URL, model_dir=str(WEIGHTS_DIR), map_location="cpu",
                                               check_hash=True, progress=False)
    # As torchvision builds it with these weights: transform_input=True maps ImageNet-normalised input to the
    # [-1, 1] range the original Google weights expect.
    model = torchvision.models.inception_v3(weights=None, aux_logits=True, transform_input=True, init_weights=False)
    model.load_state_dict(state)
    return model.eval().to(device), path


def score_run(run_dir, model, device, batch_size: int = 32) -> int:
    import torch
    import torch.nn.functional as F
    from PIL import Image
    import numpy as np
    from lanes.p2.schema import load_run, replace_metrics

    run_dir = Path(run_dir)
    rows = load_run(run_dir)["rows"]
    mean = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=device).view(1, 3, 1, 1)
    out, feats = [], []
    hook = model.avgpool.register_forward_hook(lambda m, i, o: feats.append(o.flatten(1).float().cpu()))
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            x = torch.stack([torch.from_numpy(np.asarray(Image.open(run_dir / f"{r['file']}.png").convert("RGB")).copy())
                             for r in batch]).permute(0, 3, 1, 2).float().div(255).to(device)
            x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False, antialias=True)
            probs = model((x - mean) / std).softmax(dim=1)
            top5 = probs.topk(5, dim=1).indices
            for i, r in enumerate(batch):
                c = r["class_id"]
                for name, value in (("cls_prob", probs[i, c].item()), ("cls_top1", float(top5[i, 0].item() == c)),
                                    ("cls_top5", float(c in top5[i].tolist())), ("cls_pred", float(top5[i, 0].item()))):
                    out.append({"class_id": c, "seed": r["seed"], "metric": name, "value": value})
    hook.remove()
    replace_metrics(run_dir, METRICS, out)
    np.savez(run_dir / "inception_features.npz", class_id=np.array([r["class_id"] for r in rows]),
             seed=np.array([r["seed"] for r in rows]), feat=torch.cat(feats).numpy())
    return len(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dirs", nargs="*", type=Path)
    ap.add_argument("--download", action="store_true", help="download and verify the classifier weights, then exit")
    ap.add_argument("--device", help="default: cuda if available, else cpu")
    args = ap.parse_args(argv)
    import torch
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if args.download:
        _, path = load_classifier("cpu", download=True)
        print(f"classifier weights OK: {path}")
        return
    if not args.run_dirs:
        ap.error("give one or more run directories, or --download")
    model, path = load_classifier(device)
    for d in args.run_dirs:
        n = score_run(d, model, device)
        print(json.dumps({"run_dir": str(d), "images": n, "classifier": path.name, "device": str(device)}))


if __name__ == "__main__":
    main()
