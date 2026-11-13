"""P1 Phase F: the 64 dog images, VAR's own clean outputs, for the transfer check. Standard library only, so it
runs on the NSCC login node without a conda environment.

    python lanes/p1/make_dog64.py OUT_DIR        (e.g. ~/data_dog64_p1; must be outside the repo)

Rule: take the rows of manifest/frozen_v1_1.csv sorted by (class_id, seed), and pick the 64 at positions
round(i * 223 / 63), i = 0..63: evenly spaced, first and last row included, so all 7 classes are covered
(9 or 10 images each). For each, copy outputs/baseline/25665469/var_d20/class<c>_seed<s>.png unchanged to
OUT_DIR/dog<i>_class<c>_seed<s>.png (i = 01..64, so sorting by name keeps this order: lanes/p2/transfer.py
pairs manifest row i with the i-th image by name) and write OUT_DIR/dog64_list.csv (not an image, so
transfer.py ignores it). OUT_DIR must not exist yet or be empty. Re-running into a new folder gives the same
files.
"""

import csv
import hashlib
import shutil
import sys
from pathlib import Path

sys.dont_write_bytecode = True

REPO = Path(__file__).resolve().parents[2]
MANIFEST = REPO / "manifest" / "frozen_v1_1.csv"
SRC = REPO / "outputs" / "baseline" / "25665469" / "var_d20"
N = 64


def pick(rows: list) -> list[int]:
    return [round(i * (len(rows) - 1) / (N - 1)) for i in range(N)]


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        raise SystemExit(__doc__)
    out = Path(argv[0]).expanduser().resolve()
    if out == REPO or REPO in out.parents:
        raise SystemExit(f"ERROR: {out} is inside the repo; give a folder outside it (e.g. ~/data_dog64_p1)")
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"ERROR: {out} is not empty; give a new or empty folder")

    with MANIFEST.open(newline="") as f:
        rows = sorted((int(r["class_id"]), int(r["seed"])) for r in csv.DictReader(f))
    if len(rows) != 224 or len(set(rows)) != 224:
        raise SystemExit(f"ERROR: {MANIFEST} should have 224 distinct rows, has {len(rows)}")
    idx = pick(rows)
    if len(set(idx)) != N:
        raise SystemExit("ERROR: picked positions are not distinct")
    chosen = [rows[i] for i in idx]
    missing = [f"class{c:04d}_seed{s}.png" for c, s in chosen if not (SRC / f"class{c:04d}_seed{s}.png").is_file()]
    if missing:
        raise SystemExit(f"ERROR: {len(missing)} baseline PNGs missing from {SRC}, e.g. {missing[0]}")

    out.mkdir(parents=True, exist_ok=True)
    listing = []
    for n, ((c, s), i) in enumerate(zip(chosen, idx), start=1):
        src = SRC / f"class{c:04d}_seed{s}.png"
        name = f"dog{n:02d}_class{c:04d}_seed{s}.png"
        shutil.copyfile(src, out / name)
        listing.append({"image": name, "class_id": c, "seed": s, "sorted_row": i,
                        "source": src.relative_to(REPO).as_posix(),
                        "sha256": hashlib.sha256(src.read_bytes()).hexdigest()})
    with (out / "dog64_list.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(listing[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(listing)

    per_class = {}
    for c, _ in chosen:
        per_class[c] = per_class.get(c, 0) + 1
    print(f"{N} images -> {out}  (list: dog64_list.csv)")
    print("per class: " + ", ".join(f"{c}: {k}" for c, k in sorted(per_class.items())))


if __name__ == "__main__":
    main()
