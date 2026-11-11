"""P1 Phase F: pick the OpenCrystalData images for the scale-removal transfer check.

    python lanes/p1/select_crystals.py SRC_DIR OUT_DIR [--per-mag 32]

SRC_DIR is the unzipped Kaggle dataset "In-line Image Characterization using EasyViewer"
(OpenCrystalData; CC-BY-SA-4.0; Barhate et al., Digital Chemical Engineering, 2024), with the
folders "EasyViewer 100" and "EasyViewer 400".

Rule (same idea as P2's Phase F, "evenly spaced over the sorted file list", done per magnification so
both get the same count): for each magnification, list every image file under it (all materials and
concentrations), sort by relative path, and take PER_MAG files evenly spaced over that list
(numpy.linspace, rounded). Copies them unchanged to OUT_DIR as ocd01_x100.<ext> ... and writes
OUT_DIR/sources.csv mapping each new name to the original file. Re-running gives the same selection.
"""

import argparse
import csv
import shutil
from pathlib import Path

import numpy as np

EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
MAGS = (("EasyViewer 100", "x100"), ("EasyViewer 400", "x400"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--per-mag", type=int, default=32)
    a = ap.parse_args()
    if a.out.exists() and any(a.out.iterdir()):
        raise SystemExit(f"ERROR: {a.out} is not empty")
    a.out.mkdir(parents=True, exist_ok=True)
    rows, n = [], 0
    for folder, tag in MAGS:
        root = a.src / folder
        files = sorted((p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in EXTS),
                       key=lambda p: p.relative_to(a.src).as_posix().lower())
        if len(files) < a.per_mag:
            raise SystemExit(f"ERROR: {root} has {len(files)} images, need {a.per_mag}")
        idx = np.round(np.linspace(0, len(files) - 1, a.per_mag)).astype(int)
        if len(set(idx.tolist())) != a.per_mag:
            raise SystemExit("ERROR: duplicate picks")
        for i in idx:
            n += 1
            src = files[i]
            rel = src.relative_to(a.src).as_posix()
            material = "Glutamic Acid" if "Glutamic" in rel else "Wollastonite" if "Wollastonite" in rel else "?"
            new = f"ocd{n:02d}_{tag}{src.suffix.lower()}"
            shutil.copy2(src, a.out / new)
            rows.append({"image": new, "magnification": tag, "material": material,
                         "index_in_sorted_list": int(i), "n_in_list": len(files), "source": rel})
    with open(a.out / "sources.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    counts = {}
    for r in rows:
        counts[(r["magnification"], r["material"])] = counts.get((r["magnification"], r["material"]), 0) + 1
    print(f"copied {len(rows)} images to {a.out}")
    for k, v in sorted(counts.items()):
        print(f"  {k[0]}  {k[1]:14s} {v}")


if __name__ == "__main__":
    main()
