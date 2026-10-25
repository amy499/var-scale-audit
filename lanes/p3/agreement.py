"""P3 agreement analysis: do the four lanes rank the stages the same way?

Each lane produces a per-stage importance ranking by a different method. This compares them
with Spearman rank correlation.

    P1  necessity   how much does REMOVING stage k hurt?
    P2  recovery    how recoverable is damage AT stage k?
    P3  energy      how much latent energy does stage k add?
    P3  identity    how much instance identity does stage k add?

**Disagreement is a result, not a bug.** "The stages where most information appears are not
the stages that are hardest to remove" is a finding. Do not tune anything to make these
agree.

RANK ON CONTRIBUTIONS, NOT CUMULATIVE VALUES
--------------------------------------------
This is the easiest way to get the analysis wrong. `rel_dist` and `identity_progress` are
CUMULATIVE — they say how far along the run is by stage k, so they are monotone and ranking
them just recovers the stage order (rho = 1 against anything monotone, which is meaningless).

What a lane actually claims about stage k is its **contribution**: the change it causes.
Pass `--delta` to difference a cumulative column before ranking. For `rel_dist`, which falls
as the run progresses, also pass `--invert` so a larger drop counts as a larger contribution.

    # P3's own two rankings, from the files already produced
    python -m lanes.p3.agreement \\
        --rank "p3_energy=outputs/p3/figure3/var_curves.csv:rel_dist:delta,invert" \\
        --rank "p3_identity=outputs/p3/figure3/var_semantic.csv:identity_progress:delta"

    # add P1's and P2's when they arrive (any CSV with a stage column and a score column)
        --rank "p1_necessity=lanes/p1/necessity.csv:score" \\
        --rank "p2_recovery=lanes/p2/recovery.csv:score"

    python -m lanes.p3.agreement --self-test     # synthetic rankings, no data needed

A NOTE ON POWER
---------------
VAR has 10 stages. Spearman on n=10 is weak: |rho| must exceed about 0.65 to reach p<0.05.
So "no significant correlation" here means "this cannot resolve it", not "they disagree".
The script prints p-values and says so. Report rho with n, and do not over-read a middling
value.

Owner: P3.
"""

import argparse
import csv
import itertools
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True


def spearman(a, b):
    """Rank correlation and two-sided p-value, without requiring scipy.

    Falls back to scipy when available (exact for small n); otherwise uses the
    t-approximation, which is adequate for reporting alongside n.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = len(a)
    try:
        from scipy.stats import spearmanr
        r = spearmanr(a, b)
        return float(r.statistic), float(r.pvalue)
    except ImportError:
        pass
    ra, rb = _rankdata(a), _rankdata(b)
    rho = float(np.corrcoef(ra, rb)[0, 1])
    if n < 3 or abs(rho) >= 1.0:
        return rho, float("nan")
    t = rho * np.sqrt((n - 2) / (1 - rho ** 2))
    try:                                     # survival function of |t| with n-2 df
        from math import erf, sqrt
        p = 2 * (1 - 0.5 * (1 + erf(abs(t) / sqrt(2))))   # normal approximation
    except Exception:
        p = float("nan")
    return rho, float(p)


def _rankdata(x):
    """Average ranks, ties shared (the same convention scipy uses)."""
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), float)
    ranks[order] = np.arange(1, len(x) + 1)
    _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    for i, c in enumerate(counts):
        if c > 1:
            ranks[inv == i] = ranks[inv == i].mean()
    return ranks


def load_rank(spec):
    """'label=path:column[:delta,invert]' -> (label, stages, scores)."""
    if "=" not in spec:
        raise SystemExit(f"ERROR: --rank needs label=path:column, got {spec!r}")
    label, rest = spec.split("=", 1)
    parts = rest.split(":")
    if len(parts) < 2:
        raise SystemExit(f"ERROR: --rank needs label=path:column, got {spec!r}")
    path, column = parts[0], parts[1]
    opts = set(parts[2].split(",")) if len(parts) > 2 else set()

    rows = list(csv.DictReader(Path(path).open()))
    if not rows:
        raise SystemExit(f"ERROR: {path} is empty")
    stage_col = next((c for c in ("stage", "si", "step") if c in rows[0]), None)
    if stage_col is None:
        raise SystemExit(f"ERROR: {path} has no stage/si/step column (has {list(rows[0])})")
    if column not in rows[0]:
        raise SystemExit(f"ERROR: {path} has no column {column!r} (has {list(rows[0])})")

    pairs = sorted((int(float(r[stage_col])), float(r[column])) for r in rows)
    stages = [s for s, _ in pairs]
    vals = np.array([v for _, v in pairs], float)

    if "delta" in opts:
        # contribution OF each stage = change it produced. Stage 0's contribution is its
        # value measured from the run's starting point, which for a cumulative curve is
        # the first value itself.
        vals = np.diff(vals, prepend=vals[0] - (vals[1] - vals[0]) if len(vals) > 1 else 0.0)
    if "invert" in opts:
        vals = -vals
    return label, stages, vals


def self_test():
    """Synthetic rankings with known answers, so the maths is checked before real data."""
    print("SELF TEST — synthetic rankings, n=10\n")
    n = 10
    base = np.arange(n, dtype=float)
    cases = [
        ("identical", base, base.copy()),
        ("reversed", base, base[::-1].copy()),
        ("unrelated", base, np.array([3, 7, 1, 9, 0, 5, 2, 8, 4, 6], float)),
        ("agree except the ends", base, np.array([1, 0, 2, 3, 4, 5, 6, 7, 9, 8], float)),
    ]
    for name, a, b in cases:
        rho, p = spearman(a, b)
        print(f"  {name:>22}: rho {rho:+.3f}  p {p:.4f}")
    print("\n  Expected: +1.000, -1.000, something small, close to +1.")
    print("  With n=10, |rho| must exceed about 0.65 for p<0.05 — the 'unrelated' row")
    print("  should NOT be significant even though it is not exactly zero.\n")

    print("  Why --delta matters: ranking a CUMULATIVE curve against stage order")
    cum = np.array([0.03, 0.06, 0.08, 0.11, 0.14, 0.17, 0.23, 0.30, 0.40, 1.00])
    rho, _ = spearman(cum, base)
    print(f"    cumulative vs stage index : rho {rho:+.3f}  <- meaningless, it is monotone")
    contrib = np.diff(cum, prepend=0.0)
    rho, _ = spearman(contrib, base)
    print(f"    per-stage contribution    : rho {rho:+.3f}  <- this is the real claim")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rank", action="append", default=[],
                    metavar="LABEL=PATH:COLUMN[:delta,invert]")
    ap.add_argument("--csv", type=Path, help="write the pairwise table here")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)

    if args.self_test:
        self_test()
        return
    if len(args.rank) < 2:
        raise SystemExit("ERROR: give at least two --rank specs (or --self-test)")

    loaded = [load_rank(s) for s in args.rank]
    stages = loaded[0][1]
    for label, st, _ in loaded:
        if st != stages:
            raise SystemExit(f"ERROR: {label} has stages {st}, expected {stages} — "
                             "all rankings must cover the same stages")
    n = len(stages)

    print(f"stages: {stages}  (n={n})\n")
    w = max(len(l) for l, _, _ in loaded)
    print(f"{'':{w}} | " + " ".join(f"{s:>6}" for s in stages))
    print("-" * (w + 3 + 7 * n))
    for label, _, v in loaded:
        print(f"{label:{w}} | " + " ".join(f"{x:>6.3f}" for x in v))
        order = [stages[i] for i in np.argsort(-v)]
        print(f"{'':{w}} | ranked most->least important: {order}")

    print(f"\nPAIRWISE SPEARMAN (n={n})")
    rows = []
    for (la, _, va), (lb, _, vb) in itertools.combinations(loaded, 2):
        rho, p = spearman(va, vb)
        sig = "yes" if p == p and p < 0.05 else "no"
        rows.append({"a": la, "b": lb, "n": n, "rho": round(rho, 4),
                     "p": None if p != p else round(p, 4), "significant_p05": sig})
        print(f"  {la:>{w}} vs {lb:<{w}}  rho {rho:+.3f}  p {p:.4f}  significant: {sig}")

    print(f"\n  With n={n}, |rho| must exceed roughly 0.65 for p<0.05.")
    print("  A non-significant result means THIS CANNOT RESOLVE IT, not that the lanes")
    print("  disagree. Report rho together with n, and treat disagreement as a finding.")

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
