"""Contact sheet for a breed check: one row per class, the first N images of each, labelled.

    python scripts/baseline/contact_sheet.py <run dir> <out.png> [--per-class 8] [--title TEXT]

<run dir> is a runner.generate --manifest output directory. Images are found through its run.json
(each row's "file" stem + ".png", as runner/generate.py writes them), in manifest order.
Pillow only.
"""

import argparse
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# docs/manifest_frozen.md section 2 (ILSVRC-2012, 0-indexed).
CLASS_NAMES = {151: "Chihuahua", 153: "Maltese dog", 162: "beagle", 207: "golden retriever",
               235: "German shepherd", 254: "pug", 258: "Samoyed"}
LABEL_W = 220
TITLE_H = 40
GAP = 4


def font(size: int):
    try:
        return ImageFont.load_default(size=size)   # Pillow >= 10.1
    except TypeError:
        return ImageFont.load_default()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--per-class", type=int, default=8)
    ap.add_argument("--title", default="")
    args = ap.parse_args(argv)

    rows = json.loads((args.run_dir / "run.json").read_text())["rows"]
    by_class: dict[int, list[Path]] = {}
    for r in rows:   # dicts keep insertion order: classes appear in manifest order
        by_class.setdefault(r["class_id"], []).append(args.run_dir / f"{r['file']}.png")
    by_class = {c: files[:args.per_class] for c, files in by_class.items()}

    first = Image.open(next(iter(by_class.values()))[0])
    tw, th = first.size
    width = LABEL_W + args.per_class * (tw + GAP)
    height = TITLE_H + len(by_class) * (th + GAP)
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 10), args.title or str(args.run_dir), fill="black", font=font(18))

    missing = 0
    for y_i, (c, files) in enumerate(by_class.items()):
        y = TITLE_H + y_i * (th + GAP)
        draw.text((8, y + th // 2 - 24), str(c), fill="black", font=font(28))
        draw.text((8, y + th // 2 + 10), CLASS_NAMES.get(c, "?"), fill="black", font=font(20))
        for x_i, f in enumerate(files):
            x = LABEL_W + x_i * (tw + GAP)
            if f.is_file():
                sheet.paste(Image.open(f).convert("RGB").resize((tw, th)), (x, y))
            else:
                missing += 1
                draw.rectangle([x, y, x + tw - 1, y + th - 1], outline="red", width=3)
                draw.text((x + 8, y + 8), "missing", fill="red", font=font(18))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.out)
    print(f"wrote {args.out}: {len(by_class)} classes x up to {args.per_class} images"
          + (f", {missing} missing" if missing else ""))
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
