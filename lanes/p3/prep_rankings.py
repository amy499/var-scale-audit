"""P3: pull P1's and P2's per-scale VAR scores out of their result tables, then run the agreement.

`RANKING_SPEC.md` asks every lane to rank VAR's 10 scales by the SAME yardstick — the drop in
validity, cls_prob_base - cls_prob — and to vary only the intervention. P1 and P2 already
publish that number; it just sits inside larger tables. This script extracts it, writes one
CSV per lane, and hands everything to `lanes/p3/agreement.py`.

    P1  lanes/p1/results/ranking.csv
        the SINGLE-scale removal arms remove_s0 .. remove_s9   ->  score = drop_cls_prob
        (P1's own column, with its 95% interval drop_cls_prob_lo / _hi)

    P2  one or more run summaries, VAR rows, hook noise_h, ONE severity:
            lanes/p2/results/gateb/var_p2_summary.csv     (first noise draw,  seed 0)
            lanes/p2/results/phasec/p2_summary.csv        (second noise draw, seed 1; also DiT)
        rows for the same scale from several files are AVERAGED (P2 showed the two draws agree)
                                                 ->  p2_validity = cls_prob_base - cls_prob
                                                 ->  p2_damaged  = fraction of images damaged
        p2_validity is the shared yardstick. p2_damaged is P2's own measure — the one behind
        `recovery_ranking.csv` (which holds the first draw's values) — reported alongside so
        that a reader can see whether the conclusion depends on the yardstick.

    P3  outputs/p3/figure3/var_curves.csv     energy    (rel_dist, differenced, inverted)
        outputs/p3/figure3/var_semantic.csv   identity  (identity_progress, differenced)

RUN IT (from the repo root)
---------------------------
    python -m lanes.p3.prep_rankings --p1 outputs/p3/agreement/inputs/p1_ranking.csv --p2 outputs/p3/agreement/inputs/p2_gateb_var_summary.csv outputs/p3/agreement/inputs/p2_phasec_summary.csv --p2-severity 2

Writes outputs/p3/agreement/ p1_remove.csv, p2_corrupt_sev<S>.csv and agreement.csv, and
prints the pairwise Spearman table.

THE SEVERITY IS P2's CHOICE: 2 FOR VAR
--------------------------------------
P2's data is scale x severity, but a ranking needs one number per scale. P2 chose 2: at 1 every
scale comes out unchanged, at 4 every scale is over 90% damaged, so 2 is the only level that
separates the scales. Report the severity with any P2 result.

DiT is NOT handled here — see `dit_agreement.py`. (P2's DiT ranking rises monotonically with
progress, so a rank correlation against P3's curves would be trivially +-1.)

WHY ONLY SINGLE-SCALE ARMS FROM P1
----------------------------------
P1 also removes pairs (remove_s23) and bands (remove_early). Those answer a different
question — what a group of scales does together — and cannot sit in a per-scale ranking.

Owner: P3.
"""

import argparse
import csv
import sys
from pathlib import Path

sys.dont_write_bytecode = True

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

N_SCALES = 10


def _read(path):
    path = Path(path)
    if not path.is_file():
        raise SystemExit(f"ERROR: file not found: {path}")
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"ERROR: {path} has no rows")
    return rows


def _need(rows, cols, path):
    missing = [c for c in cols if c not in rows[0]]
    if missing:
        raise SystemExit(f"ERROR: {path} has no column(s) {missing}. Has: {list(rows[0])}")


def p1_single_scale(path):
    """[(stage, score, lo, hi, kid)] from P1's ranking.csv — single-scale VAR arms only."""
    rows = _read(path)
    _need(rows, ["model", "arm", "n_removed", "drop_cls_prob"], path)
    out = {}
    for r in rows:
        arm = r["arm"]
        if r["model"] != "var" or r["n_removed"] != "1" or not arm.startswith("remove_s"):
            continue
        try:
            st = int(arm[len("remove_s"):])
        except ValueError:
            continue
        if st in out:
            raise SystemExit(f"ERROR: {path} has two rows for {arm} — ask P1 which one is final")
        out[st] = (float(r["drop_cls_prob"]),
                   float(r.get("drop_cls_prob_lo") or "nan"),
                   float(r.get("drop_cls_prob_hi") or "nan"),
                   float(r.get("kid_vs_base") or "nan"))
    missing = sorted(set(range(N_SCALES)) - set(out))
    if missing:
        raise SystemExit(f"ERROR: {path} is missing single-scale arms for scales {missing}")
    return [(s, *out[s]) for s in range(N_SCALES)]


def p2_at_severity(paths, severity, hook):
    """[(stage, validity_drop, frac_damaged, kid, n_rows)] — VAR rows of one hook at one
    severity, pooled across files and averaged per scale. Duplicate run_ids are counted once."""
    rows, seen = [], set()
    for path in paths:
        rs = _read(path)
        _need(rs, ["model", "hook", "stage", "severity", "cls_prob", "cls_prob_base"], path)
        for r in rs:
            rid = r.get("run_id")
            if rid and rid in seen:
                continue
            seen.add(rid)
            rows.append(r)
    sel = [r for r in rows if r["model"] == "var" and r["hook"] == hook
           and abs(float(r["severity"]) - severity) < 1e-9]
    if not sel:
        sevs = sorted({float(r["severity"]) for r in rows if r["model"] == "var" and r["hook"] == hook})
        hooks = sorted({r["hook"] for r in rows if r["model"] == "var"})
        raise SystemExit(f"ERROR: no VAR rows for hook {hook!r} at severity {severity}.\n"
                         f"       VAR severities for {hook!r}: {sevs}\n       VAR hooks: {hooks}")
    by = {}
    for r in sel:
        by.setdefault(int(float(r["stage"])), []).append(r)
    missing = sorted(set(range(N_SCALES)) - set(by))
    if missing:
        raise SystemExit(f"ERROR: no {hook} rows at severity {severity} for scales {missing}")

    def mean(rs, col):
        vals = [float(r[col]) for r in rs if r.get(col) not in (None, "")]
        return sum(vals) / len(vals) if vals else float("nan")

    out = []
    for st in range(N_SCALES):
        rs = by[st]
        drop = sum(float(r["cls_prob_base"]) - float(r["cls_prob"]) for r in rs) / len(rs)
        out.append((st, drop, mean(rs, "frac_damaged"), mean(rs, "kid_vs_base"), len(rs)))
    seeds = sorted({r.get("noise_seed", "?") for r in sel})
    return out, seeds


def _write(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"wrote {path}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--p1", type=Path, help="P1's lanes/p1/results/ranking.csv")
    ap.add_argument("--p2", type=Path, nargs="+",
                    help="P2's run summaries (one or more; same-scale rows are averaged)")
    ap.add_argument("--p2-severity", type=float, help="the ONE severity P2 has chosen (VAR: 2)")
    ap.add_argument("--p2-hook", default="noise_h", help="P2 hook to use (default noise_h)")
    ap.add_argument("--p2-score", choices=("both", "validity", "damaged"), default="both",
                    help="validity = the shared yardstick; damaged = P2's own measure")
    ap.add_argument("--p3-energy", type=Path, default=Path("outputs/p3/figure3/var_curves.csv"))
    ap.add_argument("--p3-identity", type=Path,
                    default=Path("outputs/p3/figure3/var_semantic.csv"))
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/p3/agreement"))
    ap.add_argument("--no-run", action="store_true", help="write the CSVs, skip agreement.py")
    args = ap.parse_args(argv)

    if args.p2 and args.p2_severity is None:
        raise SystemExit("ERROR: --p2 needs --p2-severity. P2 chose 2 for VAR.")

    specs = []
    if args.p1:
        p1 = p1_single_scale(args.p1)
        f = args.out_dir / "p1_remove.csv"
        _write(f, ["stage", "score", "lo", "hi", "kid_vs_base"], p1)
        specs.append(f"p1_remove={f}:score")
        print("\nP1 — validity drop when ONE scale is removed (95% interval):")
        for st, sc, lo, hi, _ in p1:
            print(f"  scale {st}:  {sc:6.3f}  [{lo:6.3f}, {hi:6.3f}]"
                  f"{'   <- interval includes 0' if lo <= 0 <= hi else ''}")
        print()

    if args.p2:
        p2, seeds = p2_at_severity(args.p2, args.p2_severity, args.p2_hook)
        sev = f"{args.p2_severity:g}"
        tag = sev.replace(".", "_")
        f = args.out_dir / f"p2_corrupt_sev{sev}.csv"
        _write(f, ["stage", "score", "frac_damaged", "kid_vs_base", "n_rows"], p2)
        n = sorted({r[4] for r in p2})
        print(f"\nP2 — VAR, hook {args.p2_hook}, severity {sev}; noise draws (seeds) {seeds}; "
              f"rows averaged per scale: {n[0] if len(n) == 1 else n}")
        print("  scale | validity drop | fraction damaged")
        for st, drop, dam, _, _ in p2:
            print(f"    {st}   |    {drop:6.3f}     |     {dam:6.3f}")
        print()
        if args.p2_score in ("both", "validity"):
            specs.append(f"p2_validity_sev{tag}={f}:score")
        if args.p2_score in ("both", "damaged"):
            specs.append(f"p2_damaged_sev{tag}={f}:frac_damaged")

    for label, path, col, opts in (("p3_energy", args.p3_energy, "rel_dist", "delta,invert"),
                                   ("p3_identity", args.p3_identity, "identity_progress",
                                    "delta")):
        if path.is_file():
            specs.append(f"{label}={path}:{col}:{opts}")
        else:
            print(f"  skipped {label}: {path} not found")

    if len(specs) < 2:
        raise SystemExit("ERROR: need at least two rankings — give --p1 and/or --p2, and "
                         "check the P3 CSVs exist")
    if args.no_run:
        print("\n--no-run: CSVs written. To compare them:\n  python -m lanes.p3.agreement "
              + " ".join(f'--rank "{s}"' for s in specs))
        return

    print("=" * 78)
    from lanes.p3 import agreement
    argv2 = []
    for s in specs:
        argv2 += ["--rank", s]
    agreement.main(argv2 + ["--csv", str(args.out_dir / "agreement.csv")])


if __name__ == "__main__":
    main()
