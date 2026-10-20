"""Make a small manifest for the 16-image extraction check (P3 Step 2.0).

Takes an even spread across all 7 classes rather than the first N rows of the frozen
manifest — those are all Chihuahuas, so a bug that only shows on some breeds would hide.

The rows are copied verbatim from the frozen manifest, so every image is one the full
sweep will also produce: the same (class_id, seed) pairs, hence the same images.

    # from the repo root:
    python -m lanes.p3.make_pilot_manifest                    # 16 rows -> manifest/pilot16.csv

    # from ANYWHERE (finds the manifest itself):
    python lanes/p3/make_pilot_manifest.py
    python make_pilot_manifest.py --src /path/to/frozen_v1_1.csv

    --n 28 gives 4 per class; --out changes the destination.

NOTE: `python -m lanes.p3.X` only works from the REPO ROOT. Run it anywhere else and you
get "No module named 'lanes'", which says nothing about the real problem. The plain
`python <path>/make_pilot_manifest.py` form works from any directory.

DO NOT COMMIT the output. It is scratch, and a stray extra manifest in the repo is exactly
the kind of thing someone accidentally runs the real sweep on.

Owner: P3.
"""

import argparse
import csv
import sys
from collections import OrderedDict
from pathlib import Path

sys.dont_write_bytecode = True

REPO = Path(__file__).resolve().parents[2]      # correct when this file is at lanes/p3/


def find_manifest():
    """Locate frozen_v1_1.csv without assuming where this was run from.

    `python -m lanes.p3.make_pilot_manifest` only works from the repo root, and the error
    it gives otherwise ("No module named 'lanes'") says nothing about the real problem. So
    the standalone form `python make_pilot_manifest.py` has to work from anywhere, which
    means finding the manifest rather than assuming a working directory.
    """
    here = Path.cwd()
    candidates = [
        REPO / "manifest" / "frozen_v1_1.csv",      # this file is in a repo checkout
        here / "manifest" / "frozen_v1_1.csv",      # run from the repo root
        here / "frozen_v1_1.csv",                   # run from inside manifest/
        here.parent / "manifest" / "frozen_v1_1.csv",
    ]
    for parent in list(here.parents)[:4]:           # run from somewhere below the root
        candidates.append(parent / "manifest" / "frozen_v1_1.csv")
    for c in candidates:
        if c.is_file():
            return c
    return None


DEFAULT_OUT = REPO / "manifest" / "pilot16.csv"


def pick_spread(rows, n):
    """n rows spread as evenly as possible over the classes present, in file order.

    Takes them round-robin: one from each class, then a second from each, and so on, so any
    n gives the widest class coverage that n allows.
    """
    by_class = OrderedDict()
    for r in rows:
        by_class.setdefault(r[0], []).append(r)

    picked, depth = [], 0
    while len(picked) < n:
        added = False
        for cls_rows in by_class.values():
            if depth < len(cls_rows):
                picked.append(cls_rows[depth])
                added = True
                if len(picked) == n:
                    break
        if not added:                       # every class exhausted
            break
        depth += 1
    return picked, by_class


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=16,
                    help="how many rows (default 16: one batch at batch-size 16)")
    ap.add_argument("--src", type=Path, default=None,
                    help="the frozen manifest (found automatically if omitted)")
    ap.add_argument("--out", type=Path, default=None,
                    help="where to write (default: <repo>/manifest/pilot16.csv)")
    args = ap.parse_args(argv)

    if args.src is None:
        args.src = find_manifest()
        if args.src is None:
            raise SystemExit(
                "ERROR: could not find manifest/frozen_v1_1.csv.\n"
                f"       Looked relative to this script ({REPO}) and to the current\n"
                f"       directory ({Path.cwd()}).\n"
                "\n"
                "       Either cd to the repo root and run it again, or point at the file:\n"
                "           python make_pilot_manifest.py --src /path/to/frozen_v1_1.csv\n"
                "\n"
                "       If you have not cloned the repo yet:\n"
                "           git clone https://github.com/amy499/var-scale-audit.git")
        print(f"using manifest: {args.src}")
    elif not args.src.is_file():
        raise SystemExit(f"ERROR: {args.src} not found.")

    if args.out is None:
        args.out = args.src.parent / "pilot16.csv"

    with args.src.open(newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    if not rows:
        raise SystemExit(f"ERROR: {args.src} is empty")

    header, body = rows[0], [r for r in rows[1:] if r and any(c.strip() for c in r)]
    if [h.strip() for h in header] != ["class_id", "seed"]:
        raise SystemExit(f"ERROR: {args.src} header is {header}, expected ['class_id', 'seed']")
    if args.n > len(body):
        raise SystemExit(f"ERROR: asked for {args.n} rows but the manifest has {len(body)}")

    picked, by_class = pick_spread(body, args.n)
    if len(picked) < args.n:
        raise SystemExit(f"ERROR: could only pick {len(picked)} of {args.n} rows")
    if len({tuple(r) for r in picked}) != len(picked):
        raise SystemExit("ERROR: duplicate rows picked — the source manifest has duplicates")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(picked)

    counts = {}
    for r in picked:
        counts[r[0]] = counts.get(r[0], 0) + 1
    print(f"wrote {args.out}")
    print(f"  {len(picked)} rows, {len(counts)} of the manifest's {len(by_class)} classes")
    print(f"  per class: {dict(sorted(counts.items(), key=lambda kv: int(kv[0])))}")
    if args.n % 16:
        print(f"  note: {args.n} is not a multiple of 16, so the last batch will be partial;"
              f" runner.generate needs --allow-partial-batch for that")
    print("\n  Do NOT commit this file — it is scratch.")


if __name__ == "__main__":
    main()
