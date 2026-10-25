"""Compare two P3 VAR snapshot folders (lanes/p3/p3_observe.py:var_stages) file by file. numpy only.

    python -m lanes.p1.compare_p3_var outputs/p1/25891860/p3_var_stages outputs/p1/<JOBID>/p3_var_stages

PASS only if both folders hold the same .npz names (relative paths, e.g. var/stage0003_<seed>.npz), and in
every pair stage, class_id and seed are equal and latent is bit-identical (same dtype, shape and bytes).
Other fields (p, n_tokens, cum_tokens) are compared too and reported, but do not decide the result.
Prints the counts and the first differences; exit 0 on PASS, 1 on FAIL.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

REQUIRED = ("stage", "class_id", "seed", "latent")
N_SHOW = 10


def _same(a: np.ndarray, b: np.ndarray) -> bool:
    """Same dtype, shape and bytes (so NaN == NaN and -0.0 != 0.0)."""
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def compare_file(fa: Path, fb: Path) -> tuple[list, list]:
    """(errors, other-field differences) for one pair of .npz files."""
    errors, other = [], []
    with np.load(fa) as za, np.load(fb) as zb:
        for k in REQUIRED:
            if k not in za.files or k not in zb.files:
                errors.append(f"{k} missing in {'first' if k not in za.files else 'second'}")
            elif not _same(za[k], zb[k]):
                errors.append(f"{k} differs")
        for k in sorted((set(za.files) | set(zb.files)) - set(REQUIRED)):
            if k not in za.files or k not in zb.files or not _same(za[k], zb[k]):
                other.append(k)
    return errors, other


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a", type=Path, help="reference p3_var_stages folder")
    ap.add_argument("b", type=Path, help="folder to check against it")
    args = ap.parse_args(argv)
    for d in (args.a, args.b):
        if not d.is_dir():
            sys.exit(f"ERROR: not a directory: {d}")
    names = {d: {p.relative_to(d).as_posix() for p in d.rglob("*.npz")} for d in (args.a, args.b)}
    common = sorted(names[args.a] & names[args.b])
    only_a, only_b = sorted(names[args.a] - names[args.b]), sorted(names[args.b] - names[args.a])

    bad, other = [], []
    for n in common:
        errors, diff = compare_file(args.a / n, args.b / n)
        if errors:
            bad.append(f"{n}: {', '.join(errors)}")
        if diff:
            other.append(f"{n}: {', '.join(diff)}")

    ok = not only_a and not only_b and not bad and bool(common)
    print(f"{args.a}: {len(names[args.a])} .npz   {args.b}: {len(names[args.b])} .npz")
    print(f"common {len(common)}  identical {len(common) - len(bad)}  different {len(bad)}  "
          f"only in first {len(only_a)}  only in second {len(only_b)}")
    for label, items in (("only in first", only_a), ("only in second", only_b), ("different", bad),
                         ("other fields differ (not counted)", other)):
        for item in items[:N_SHOW]:
            print(f"  {label}: {item}")
        if len(items) > N_SHOW:
            print(f"  {label}: ... {len(items) - N_SHOW} more")
    for d in (args.a, args.b):
        if (d / "_failures.log").is_file():
            print(f"  warning: {d / '_failures.log'} exists")
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
