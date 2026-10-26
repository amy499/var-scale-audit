"""P3 baseline audit: find generated baseline images that are themselves failures.

Some clean baseline images are bad samples — the contact-sheet review on 20 Oct found VAR
class 207 image 4 is a saturated-yellow frame, and VAR 162 images 3 and 8 are near-abstract.
They are not labelling errors; the generator simply produced a poor image.

**This matters to every lane, not just P3.** P1, P2 and P4 all pair their interventions
against these baselines. An image that is already broken before any intervention drags down
whatever is averaged over it, and it does so invisibly.

This script scores each baseline image with the same pinned Inception-v3 that
`lanes/p2/quality.py` uses, and lists the worst ones so they can be reported separately
rather than silently averaged in.

    python -m lanes.p3.baseline_audit --images for_darien_20_clean_pngs
    python -m lanes.p3.baseline_audit --images <folder> --csv outputs/p3/baseline_audit.csv

WHY NOT `lanes/p2/quality.py`
-----------------------------
`quality.py` needs a `runner.generate` output directory with a `run.json`, and pairs images
by the manifest's `file` stem. A folder of baseline PNGs copied out of a run has neither, so
it fails at `load_run()` before scoring anything. This reads the class out of the filename
instead, so it works on any flat folder: `class<cccc>_seed<seed>.png`, with or without a
`_stage<nnnn>` suffix (so the final decoded stage works too).

Needs torchvision. Weights: `python -m lanes.p2.quality --download` (once).

Owner: P3.
"""

import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

NAME = re.compile(r"^class(\d+)_seed(\d+)(?:_stage(\d+))?\.png$")


def index_images(folder, stage=None):
    """[(class_id, seed, path)] from a flat folder. `stage` filters _stage<nnnn> files."""
    folder = Path(folder)
    out, skipped = [], 0
    for p in sorted(folder.glob("*.png")):
        m = NAME.match(p.name)
        if not m:
            skipped += 1
            continue
        st = m.group(3)
        if stage is not None and (st is None or int(st) != stage):
            continue
        if stage is None and st is not None:
            continue            # a stage folder: require --stage so the choice is explicit
        out.append((int(m.group(1)), int(m.group(2)), p))
    return out, skipped


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", required=True, type=Path)
    ap.add_argument("--stage", type=int,
                    help="for a decoded-stages folder, which stage counts as 'final'")
    ap.add_argument("--flag-below", type=float, default=0.10,
                    help="report images whose P(true class) is under this (default 0.10)")
    ap.add_argument("--device")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--csv", type=Path)
    args = ap.parse_args(argv)

    items, skipped = index_images(args.images, args.stage)
    if not items:
        raise SystemExit(
            f"ERROR: no class<N>_seed<N>[_stage<N>].png in {args.images}\n"
            f"       For a decoded-stages folder pass --stage 9 to pick the final stage.")
    print(f"{len(items)} images"
          f"{f' ({skipped} non-matching skipped)' if skipped else ''}"
          f"{f', stage {args.stage}' if args.stage is not None else ''}\n")

    import torch
    from lanes.p2.quality import load_classifier
    from lanes.p3.semantic_curve import score
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    clf, wpath = load_classifier(device)
    print(f"classifier: {wpath.name} on {device}\n")

    probs, _ = score([p for _, _, p in items], clf, device, args.batch_size)
    true = np.array([c for c, _, _ in items])
    p_true = probs[np.arange(len(items)), true]
    top1 = probs.argmax(1)

    rows = [{"class_id": c, "seed": s, "file": p.name,
             "cls_prob": round(float(pt), 6), "top1_pred": int(t),
             "top1_is_true": int(t == c)}
            for (c, s, p), pt, t in zip(items, p_true, top1)]
    rows.sort(key=lambda r: r["cls_prob"])

    print(f"mean P(true class) : {p_true.mean():.4f}")
    print(f"median             : {np.median(p_true):.4f}")
    print(f"top-1 accuracy     : {(top1 == true).mean():.3f}")
    print(f"\nWORST {min(10, len(rows))} — inspect these by eye before trusting them:")
    print(f"  {'cls_prob':>9} {'top1':>6} {'true':>6}  file")
    for r in rows[:10]:
        mark = "  <-- FLAG" if r["cls_prob"] < args.flag_below else ""
        print(f"  {r['cls_prob']:>9.5f} {r['top1_pred']:>6} {r['class_id']:>6}  "
              f"{r['file']}{mark}")

    flagged = [r for r in rows if r["cls_prob"] < args.flag_below]
    print(f"\n{len(flagged)} of {len(rows)} below {args.flag_below} "
          f"({100*len(flagged)/len(rows):.0f}%)")
    if flagged:
        print("  These are candidates to report separately. Look at them first — a low score")
        print("  can also mean the classifier is wrong, not that the image is bad.")
        print("  Share the list with P1/P2/P4: they pair against these same baselines.")
    else:
        print("  No image is below the threshold; the baseline looks clean on this sample.")

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        print(f"\nwrote {args.csv} (sorted worst first)")


if __name__ == "__main__":
    main()
