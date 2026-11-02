"""P3: DiT across the lanes — one table on DiT's own timeline, no rank correlation.

VAR has 10 scales, so `prep_rankings.py` can rank them and correlate. DiT cannot be handled the
same way, for two reasons:

  1. P2's DiT ranking rises monotonically with progress (severity 0.5: validity drop 0.02 at
     timestep 798 up to 0.55 at 36). P3's curves are cumulative and also monotonic. A rank
     correlation between two monotonic sequences is +-1 BY CONSTRUCTION — it would look like
     perfect agreement and mean nothing (`agreement.py --self-test` shows the same trap).
  2. P1 removed DiT in three bands (early / middle / late), so P1 has n = 3: too few to rank.

So this script lines the three lanes up on DiT's timeline instead, two ways:

  BY BAND (P1's bands)  P1: validity drop when the band is removed (95% interval)
                        P2: mean validity drop for corruption at the P2 timesteps inside it
                        P3: how much identity / closeness to the final latent is ADDED inside it
  BY P2 TIMESTEP        P2's numbers at each of its 8 timesteps, next to how far P3's curves
                        have got by that timestep

    python -m lanes.p3.dit_agreement --p1 outputs/p3/agreement/inputs/p1_ranking.csv --p2 outputs/p3/agreement/inputs/p2_phasec_summary.csv --p2-severity 0.5

P3 inputs (from step 3): outputs/p3/figure3/dit_curves.csv (stage_curves.py) and, once the DiT
decode has run, outputs/p3/figure3/dit_semantic.csv (semantic_curve.py). Either may be absent;
its columns are then left blank.

DEFINITIONS
-----------
  identity     identity_progress (chance-corrected, 0 at chance, 1 at the final image)
  closeness    1 - rel_dist: closeness of the predicted latent to the final latent. NOT called
               "energy" for DiT: the orthogonality check fails for DiT's early steps
               (max |cos - ratio| = 0.37), so rel_dist there carries direction, not just
               accumulated magnitude. It starts slightly below 0 (rel_dist 1.06 at t = 999).
  band edges   a band removes steps a..b; "added inside the band" = value after step b minus
               value after step a-1 (after step 0 for the first band). P3 saved 24 of the 250
               steps, so values at band edges and at P2's timesteps are linearly interpolated
               between the two nearest saved timesteps.

Owner: P3.
"""

import argparse
import csv
import re
import sys
from pathlib import Path

sys.dont_write_bytecode = True

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _read(path):
    path = Path(path)
    if not path.is_file():
        raise SystemExit(f"ERROR: file not found: {path}")
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"ERROR: {path} has no rows")
    return rows


def p1_bands(path):
    """[{band, steps, t_start, t_end, drop, lo, hi}] from P1's DiT band-removal arms."""
    out = []
    for r in _read(path):
        if r.get("model") != "dit" or not r.get("arm", "").startswith("remove_"):
            continue
        m = re.search(r"steps\s+(\d+)-(\d+)\s+\(t\s+(\d+)-(\d+)\)", r.get("removed", ""))
        if not m:
            continue
        out.append({"band": r["arm"][len("remove_"):], "steps": f"{m.group(1)}-{m.group(2)}",
                    "t_start": int(m.group(3)), "t_end": int(m.group(4)),
                    "drop": float(r["drop_cls_prob"]),
                    "lo": float(r.get("drop_cls_prob_lo") or "nan"),
                    "hi": float(r.get("drop_cls_prob_hi") or "nan")})
    if not out:
        raise SystemExit(f"ERROR: no DiT band arms (remove_early/mid/late) in {path}")
    return sorted(out, key=lambda b: -b["t_start"])          # generation order: early first


def p2_timesteps(paths, severity, hook):
    """{t: {p, drop, damaged, kid, outcome, n}} for DiT rows at one severity, averaged."""
    rows, seen = [], set()
    for path in paths:
        for r in _read(path):
            rid = r.get("run_id")
            if rid and rid in seen:
                continue
            seen.add(rid)
            rows.append(r)
    sel = [r for r in rows if r.get("model") == "dit" and r.get("hook") == hook
           and abs(float(r["severity"]) - severity) < 1e-9]
    if not sel:
        sevs = sorted({float(r["severity"]) for r in rows if r.get("model") == "dit"})
        raise SystemExit(f"ERROR: no DiT rows for hook {hook!r} at severity {severity}. "
                         f"DiT severities in the file(s): {sevs}")
    by = {}
    for r in sel:
        by.setdefault(int(float(r["stage"])), []).append(r)
    out = {}
    for t, rs in by.items():
        mean = lambda c: sum(float(x[c]) for x in rs) / len(rs)
        out[t] = {"p": mean("p"), "drop": sum(float(x["cls_prob_base"]) - float(x["cls_prob"])
                                               for x in rs) / len(rs),
                  "damaged": mean("frac_damaged"), "kid": mean("kid_vs_base"),
                  "outcome": "/".join(sorted({x.get("outcome", "?") for x in rs})), "n": len(rs)}
    return out


def p3_curve(path, column, transform):
    """[(t, value)] sorted by t, from a stage_curves / semantic_curve CSV; None if absent."""
    if path is None or not Path(path).is_file():
        return None
    pts = sorted((int(float(r["stage"])), transform(float(r[column]))) for r in _read(path))
    return pts


def interp(curve, t):
    """Linear interpolation in timestep between the two nearest saved timesteps."""
    if curve is None:
        return float("nan")
    ts = [p[0] for p in curve]
    if t <= ts[0]:
        return curve[0][1]
    if t >= ts[-1]:
        return curve[-1][1]
    for (t0, v0), (t1, v1) in zip(curve, curve[1:]):
        if t0 <= t <= t1:
            return v0 if t1 == t0 else v0 + (t - t0) / (t1 - t0) * (v1 - v0)
    return float("nan")


def first_reaching(curve, thr):
    """Generation order (high t first): first saved timestep whose value >= thr."""
    if curve is None:
        return None
    for t, v in sorted(curve, reverse=True):
        if v >= thr:
            return t
    return None


def fmt(x, w=7, d=3):
    return f"{'—':>{w}}" if x != x else f"{x:>{w}.{d}f}"


def order(items, key):
    vals = [(b["band"], key(b)) for b in items]
    if any(v != v for _, v in vals):
        return "—"
    return " > ".join(n for n, _ in sorted(vals, key=lambda x: -x[1]))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--p1", required=True, type=Path, help="P1's lanes/p1/results/ranking.csv")
    ap.add_argument("--p2", required=True, type=Path, nargs="+",
                    help="P2's lanes/p2/results/phasec/p2_summary.csv")
    ap.add_argument("--p2-severity", required=True, type=float, help="P2 chose 0.5 for DiT")
    ap.add_argument("--p2-hook", default="noise_x")
    ap.add_argument("--p3-closeness", type=Path, default=Path("outputs/p3/figure3/dit_curves.csv"))
    ap.add_argument("--p3-identity", type=Path, default=Path("outputs/p3/figure3/dit_semantic.csv"))
    ap.add_argument("--csv", type=Path, default=Path("outputs/p3/agreement/dit_comparison.csv"))
    args = ap.parse_args(argv)

    bands = p1_bands(args.p1)
    p2 = p2_timesteps(args.p2, args.p2_severity, args.p2_hook)
    clo = p3_curve(args.p3_closeness, "rel_dist", lambda v: 1.0 - v)
    ide = p3_curve(args.p3_identity, "identity_progress", lambda v: v)
    for name, c, path in (("closeness", clo, args.p3_closeness), ("identity", ide, args.p3_identity)):
        print(f"P3 {name:<9}: {f'{len(c)} timesteps from {path}' if c else f'not found ({path}) — column left blank'}")
    t_first = max(t for t, _ in (clo or ide or [(999, 0)]))

    # ---- by band ---------------------------------------------------------------------------
    prev = t_first
    for b in bands:
        inside = [t for t in p2 if b["t_end"] <= t <= b["t_start"]]
        b["p2_drop"] = sum(p2[t]["drop"] for t in inside) / len(inside) if inside else float("nan")
        b["p2_n"] = len(inside)
        b["p3_clo"] = interp(clo, b["t_end"]) - interp(clo, prev)
        b["p3_id"] = interp(ide, b["t_end"]) - interp(ide, prev)
        prev = b["t_end"]

    sev = f"{args.p2_severity:g}"
    print(f"\nBY BAND — P1's three DiT bands (n = 3: compared side by side, not correlated)\n")
    print(f"{'band':<6} {'steps':<8} {'timesteps':<10}| {'P1 remove: drop [95%]':<25}| "
          f"{'P2 sev ' + sev + ': mean drop':<22}| {'P3 identity':>11} {'P3 closeness':>12}")
    print(f"{'':<26}| {'':<25}| {'(n = P2 timesteps)':<22}| {'added':>11} {'added':>12}")
    print("-" * 104)
    for b in bands:
        print(f"{b['band']:<6} {b['steps']:<8} {str(b['t_start']) + '-' + str(b['t_end']):<10}| "
              f"{b['drop']:6.3f} [{b['lo']:.3f}, {b['hi']:.3f}]{'':<5}| "
              f"{fmt(b['p2_drop'], 6)}  (n = {b['p2_n']}){'':<8}| "
              f"{fmt(b['p3_id'], 11)} {fmt(b['p3_clo'], 12)}")
    print("\n  most -> least, per lane:")
    print(f"    P1 removal       {order(bands, lambda b: b['drop'])}")
    print(f"    P2 corruption    {order(bands, lambda b: b['p2_drop'])}")
    print(f"    P3 identity      {order(bands, lambda b: b['p3_id'])}")
    print(f"    P3 closeness     {order(bands, lambda b: b['p3_clo'])}")

    # ---- by P2 timestep --------------------------------------------------------------------
    print(f"\nBY P2 TIMESTEP — severity {sev}, hook {args.p2_hook}; P3 values = how far its curves "
          f"have got by that timestep\n")
    print(f"{'t':>4} {'p':>6} {'band':<6}| {'P2 drop':>7} {'damaged':>8} {'KID':>7} {'outcome':<18}| "
          f"{'P3 identity':>11} {'P3 closeness':>12}")
    print("-" * 92)
    table = []
    for t in sorted(p2, reverse=True):
        band = next((b["band"] for b in bands if b["t_end"] <= t <= b["t_start"]), "?")
        d = p2[t]
        row = {"t": t, "p": round(d["p"], 3), "p1_band": band, "p2_validity_drop": d["drop"],
               "p2_frac_damaged": d["damaged"], "p2_kid": d["kid"], "p2_outcome": d["outcome"],
               "p3_identity_by_t": interp(ide, t), "p3_closeness_by_t": interp(clo, t)}
        table.append(row)
        print(f"{t:>4} {d['p']:>6.3f} {band:<6}| {d['drop']:>7.3f} {d['damaged']:>8.3f} "
              f"{d['kid']:>7.4f} {d['outcome']:<18}| {fmt(row['p3_identity_by_t'], 11)} "
              f"{fmt(row['p3_closeness_by_t'], 12)}")

    # ---- milestones ------------------------------------------------------------------------
    print("\nMILESTONES (generation order, noise -> image)")
    hit = next((t for t in sorted(p2, reverse=True) if "unchanged" not in p2[t]["outcome"]), None)
    if hit is None:
        print(f"  P2: corruption at severity {sev} is absorbed (unchanged) at every timestep")
    else:
        d = p2[hit]
        print(f"  P2: corruption at severity {sev} first stops being absorbed at t = {hit} "
              f"(p = {d['p']:.2f}, outcome {d['outcome']})")
    for thr in (0.5, 0.9):
        t = first_reaching(ide, thr)
        print(f"  P3: identity first reaches {thr:.1f} at t = {t if t is not None else '— (run step 3.4)'}")
    print("\n  Read the bands first. With n = 3 there is no test: say which band each lane puts")
    print("  first, and whether they agree. Do not compute a correlation on the timestep table —")
    print("  both P2's and P3's columns rise with progress, so any correlation is +-1 by design.")

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0])); w.writeheader(); w.writerows(table)
    bcsv = args.csv.with_name(args.csv.stem + "_bands.csv")
    with bcsv.open("w", newline="") as f:
        cols = ["band", "steps", "t_start", "t_end", "drop", "lo", "hi", "p2_drop", "p2_n",
                "p3_id", "p3_clo"]
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore"); w.writeheader(); w.writerows(bands)
    print(f"\nwrote {args.csv}\nwrote {bcsv}")


if __name__ == "__main__":
    main()
