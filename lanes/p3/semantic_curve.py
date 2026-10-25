"""P3 semantic curve: WHEN is the image decided?

This is the metric the energy curve cannot provide. `stage_curves.py` showed that
`rel_dist` on VAR's `f_hat` is a function of accumulated energy (increments are
near-orthogonal, so cosine == norm ratio at every scale). That says how full the
accumulator is, not when the picture was settled. This script answers the second question,
from the decoded images.

    python -m lanes.p3.semantic_curve --decoded p3_var_stages/decoded_var --model var \
        --csv outputs/p3/figure3/var_semantic.csv --plot outputs/p3/figure3/var_semantic.png

Input: the flat folder `decode_stages.py` writes, named
`class<cccc>_seed<seed>_stage<ssss>.png`. (`lanes/p2/quality.py` CANNOT read this folder —
it expects a `runner.generate` output directory with a `run.json`, which decoded stages do
not have. That is why this script exists.)

WHAT IT MEASURES, AND WHICH NUMBER TO BELIEVE
---------------------------------------------
Per stage, over all images:

    cls_prob        mean P(true class)                      "does it look like the class yet"
    top1_true       fraction whose top-1 IS the true class   same question, harder threshold
    top1_final      fraction whose top-1 matches the FINAL   "has the identity stopped moving"
    incep_cos_final mean cosine to the FINAL image's pool features, per image
    incep_cos_chance the same, but paired against a DIFFERENT image's final features

**`cls_prob` and `top1_true` are the WEAKER measures here, and it matters.** VAR and DiT are
class-conditional: the class is an INPUT, not something the model discovers. A rising
`cls_prob` largely says "this now looks like a photo of a dog", which is about image quality
as much as about commitment.

**`incep_cos_final` and `top1_final` are the ones that answer the question**, because they
are instance-level: they ask whether this image has converged to the *particular* image it
was going to be, not whether it belongs to the conditioned class.

**`incep_cos_chance` is the control that makes `incep_cos_final` readable.** Inception pool
features are post-ReLU and therefore non-negative, so the cosine between two completely
unrelated images is NOT 0 — it is typically 0.2-0.4. Without the chance column, a stage-0
value of 0.33 looks like "a third of the way there" when it may be indistinguishable from
chance. The script prints a normalised column,

    progress = (cos_final - cos_chance) / (1 - cos_chance)

which is 0 at chance and 1 at the final image. **Quote the normalised number.**

Report all five, lead with the instance-level ones, and say why.

THE PREDICTION TO WRITE DOWN BEFORE LOOKING
-------------------------------------------
P2's pilot perturbed VAR at scale 0 and got a different but clean image **of the same
class** — so the class survives a perturbation at the very first scale, while the energy
curve still shows ~97% "remaining". If the semantic curve rises early while
`stage_curves.py`'s `rel_dist` is still high, the headline is **"VAR decides *what* early and
*how* late"**, and the crossing of the two curves is Figure 3.

Writing that down first makes this a test rather than a story.

REQUIREMENTS
------------
Needs **torchvision** (Inception-v3) — more than `decode_stages.py --vae-only` needed. The
classifier weights are the pinned, hash-verified ones from `lanes/p2/quality.py`, so these
numbers are comparable with P2's. Download once:

    python -m lanes.p2.quality --download

Owner: P3.
"""

import argparse
import collections
import re
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

NAME = re.compile(r"^class(\d+)_seed(\d+)_stage(\d+)\.png$")
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def index_decoded(folder):
    """{stage: {(class_id, seed): path}} from the flat decoded folder."""
    folder = Path(folder)
    by = collections.defaultdict(dict)
    skipped = 0
    for p in sorted(folder.glob("*.png")):
        m = NAME.match(p.name)
        if not m:
            skipped += 1          # the contact sheet lives here too
            continue
        c, s, st = int(m.group(1)), int(m.group(2)), int(m.group(3))
        by[st][(c, s)] = p
    if not by:
        raise SystemExit(
            f"ERROR: no files matching class<N>_seed<N>_stage<N>.png in {folder}\n"
            f"       Point --decoded at the folder decode_stages.py wrote "
            f"(usually <stages>/decoded_<model>).")
    return by, skipped


def score(paths, model, device, batch_size=16):
    """Return (probs [n,1000], pool features [n,2048]) for a list of image paths."""
    import torch
    from PIL import Image
    import torch.nn.functional as F

    mean = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=device).view(1, 3, 1, 1)
    feats, probs = [], []
    hook = model.avgpool.register_forward_hook(
        lambda m, i, o: feats.append(o.flatten(1).float().cpu()))
    with torch.inference_mode():
        for i in range(0, len(paths), batch_size):
            batch = paths[i:i + batch_size]
            x = torch.stack([
                torch.from_numpy(np.asarray(Image.open(p).convert("RGB")).copy())
                for p in batch]).permute(0, 3, 1, 2).float().div(255).to(device)
            # 299x299 bilinear+antialias, exactly as lanes/p2/quality.py does
            x = F.interpolate(x, size=(299, 299), mode="bilinear",
                              align_corners=False, antialias=True)
            probs.append(model((x - mean) / std).softmax(dim=1).float().cpu())
    hook.remove()
    return torch.cat(probs).numpy(), torch.cat(feats).numpy()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--decoded", required=True, type=Path)
    ap.add_argument("--model", default="var", choices=("var", "dit"))
    ap.add_argument("--device", help="cpu | mps | cuda (default: cuda if available else cpu)")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--csv", type=Path)
    ap.add_argument("--plot", type=Path)
    args = ap.parse_args(argv)

    by, skipped = index_decoded(args.decoded)
    stages = sorted(by)
    final_stage = stages[-1] if args.model == "var" else stages[0]
    common = sorted(set.intersection(*(set(v) for v in by.values())))
    if not common:
        raise SystemExit("ERROR: no image appears at every stage")
    print(f"{len(common)} images x {len(stages)} stages"
          f"{f' ({skipped} non-matching file(s) skipped)' if skipped else ''}")
    print(f"final stage = {final_stage}\n")

    import torch
    from lanes.p2.quality import load_classifier
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    clf, wpath = load_classifier(device)
    print(f"classifier: {wpath.name} on {device}\n")

    # the final stage first — everything else is measured against it
    fp, ff = score([by[final_stage][k] for k in common], clf, device, args.batch_size)
    final_top1 = fp.argmax(1)
    ffn = ff / np.linalg.norm(ff, axis=1, keepdims=True)
    true_cls = np.array([c for c, _ in common])

    rows = []
    order = stages if args.model == "var" else sorted(stages, reverse=True)
    for st in order:
        p, f = score([by[st][k] for k in common], clf, device, args.batch_size)
        fn = f / np.linalg.norm(f, axis=1, keepdims=True)
        cos_final = float((fn * ffn).sum(1).mean())
        # chance floor: same stage features, paired against SOMEONE ELSE'S final image.
        # Averaged over every non-identity shift so it does not depend on one permutation.
        shifts = [np.roll(np.arange(len(common)), k) for k in range(1, len(common))]
        cos_chance = float(np.mean([(fn * ffn[idx]).sum(1).mean() for idx in shifts]))
        denom = 1.0 - cos_chance
        rows.append({
            "stage": st,
            "n_images": len(common),
            "cls_prob": float(p[np.arange(len(common)), true_cls].mean()),
            "top1_true": float((p.argmax(1) == true_cls).mean()),
            "top1_final": float((p.argmax(1) == final_top1).mean()),
            "incep_cos_final": cos_final,
            "incep_cos_chance": cos_chance,
            "identity_progress": float((cos_final - cos_chance) / denom) if denom > 0 else float("nan"),
        })
        r = rows[-1]
        print(f"  stage {st:>4}  cls_prob {r['cls_prob']:.4f}  top1_true {r['top1_true']:.3f}"
              f"  top1_final {r['top1_final']:.3f}  cos {r['incep_cos_final']:.4f}"
              f"  chance {r['incep_cos_chance']:.4f}  progress {r['identity_progress']:.4f}")

    print(f"\n{'stage':>5} | {'cls_prob':>9} {'top1_true':>10} | {'top1_fin':>9} "
          f"{'cos':>7} {'chance':>7} {'PROGRESS':>9}")
    print(f"{'':>5} | {'weak: class is an INPUT':>20} | {'instance-level':>9}"
          f"{'':>16}{'<- quote this':>9}")
    print("-" * 74)
    for r in rows:
        print(f"{r['stage']:>5} | {r['cls_prob']:>9.4f} {r['top1_true']:>10.3f} | "
              f"{r['top1_final']:>9.3f} {r['incep_cos_final']:>7.4f} "
              f"{r['incep_cos_chance']:>7.4f} {r['identity_progress']:>9.4f}")

    print(f"\n  chance floor at the final stage: {rows[-1]['incep_cos_chance']:.4f} "
          f"— two UNRELATED images score this much by construction")
    print("  (Inception pool features are post-ReLU, so cosine between unrelated images is >> 0)")
    for thr in (0.50, 0.90, 0.95):
        hit = next((r["stage"] for r in rows if r["identity_progress"] >= thr), None)
        print(f"  identity_progress first reaches {thr:.2f} at stage {hit}")

    if args.csv:
        import csv
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        print(f"\nwrote {args.csv}")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        x = np.arange(len(rows))
        fig, ax = plt.subplots(figsize=(8.0, 5.0))
        ax.plot(x, [r["identity_progress"] for r in rows], "o-", lw=2.2,
                label="identity progress (chance-corrected)")
        ax.plot(x, [r["incep_cos_final"] for r in rows], "o:", lw=1.2, alpha=.6,
                label="raw incep_cos to final")
        ax.plot(x, [r["incep_cos_chance"] for r in rows], color="0.6", lw=1, ls="--",
                label="chance floor (unrelated images)")
        ax.plot(x, [r["top1_final"] for r in rows], "s-", lw=2,
                label="top-1 matches final")
        ax.plot(x, [r["cls_prob"] for r in rows], "^--", alpha=.75,
                label="P(true class) — weak: class is an input")
        ax.plot(x, [r["top1_true"] for r in rows], "v:", alpha=.6,
                label="top-1 is true class — weak")
        ax.set_xticks(x); ax.set_xticklabels([r["stage"] for r in rows], fontsize=8)
        ax.set_xlabel(f"{args.model.upper()} stage"); ax.set_ylim(-0.03, 1.12)
        ax.set_title(f"{args.model.upper()} semantic curve — when is the image decided?",
                     fontsize=11)
        # Upper left: every curve starts low at stage 0 and rises, so this is the one
        # corner that stays empty. A semi-transparent frame keeps it readable where the
        # flat chance-floor line passes behind it.
        ax.legend(fontsize=8, loc="upper left", frameon=True, framealpha=0.92,
                  facecolor="white", edgecolor="0.85", borderpad=0.6)
        ax.grid(alpha=.3)
        fig.tight_layout(); args.plot.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.plot, dpi=140); plt.close(fig)
        print(f"wrote {args.plot}")


if __name__ == "__main__":
    main()
