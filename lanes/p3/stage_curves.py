"""P3 stage curves: what the saved latents say, and what they do NOT say.

Reads the .npz files written by p3_observe.py and computes, per stage:

    rel_dist    ||f_k - f_final|| / ||f_final||     distance to the final latent
    norm_ratio  ||f_k|| / ||f_final||               how much magnitude has accumulated
    cosine      cos(f_k, f_final)                   how well aligned the direction is

THE RESULT THAT CHANGES HOW THIS IS READ
----------------------------------------
On VAR's `f_hat`, measured on P1's 224-image run (23 Oct 2026):

    cosine - norm_ratio  is within +/-0.017 at every scale
    rel_dist             matches sqrt(1 - norm_ratio^2) within +/-0.017

Both identities hold if and only if **each scale's contribution is nearly orthogonal to the
sum accumulated so far** — which is what a residual pyramid produces by construction: every
scale encodes what the previous ones left behind.

The consequence is blunt: on VAR, `rel_dist` is **a function of accumulated energy**, not a
measure of semantic commitment. It is almost fully determined by `||f_k||`. A curve that
falls from 0.97 to 0.60 across si 0-8 says the accumulator is filling up; it does NOT say
"the image is 40% decided".

**So report it as accumulated latent energy, and nothing more.** The semantic question —
*when is the image decided?* — cannot be answered from `f_hat` norms. It needs the decoded
intermediates and a classifier (P3 Step 4 / check 3).

Independent support: P2's pilot (lanes/p2/results/pilot_25656645) perturbed VAR at scale 0
and got a different but clean image **of the same class** — semantic commitment was already
in place while, by this metric, 97% of the "distance" remained. Energy accumulates late;
meaning is fixed early. They are different quantities.

This script prints the orthogonality diagnostic every time so the caveat cannot be quietly
dropped.

    python -m lanes.p3.stage_curves --stages outputs/p3/stages --model var
    python -m lanes.p3.stage_curves --stages outputs/p3/stages --model var --csv out.csv --plot out.png

Owner: P3.
"""

import argparse
import collections
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True

VAR_PN = [1, 2, 3, 4, 5, 6, 8, 10, 13, 16]


def load_stages(stages_dir, model):
    """{stage: {(class_id, seed): latent}} from the hook's .npz files."""
    d = Path(stages_dir) / model
    files = sorted(d.glob("stage*.npz"))
    if not files:
        raise SystemExit(f"ERROR: no stage*.npz in {d}")
    by = collections.defaultdict(dict)
    extra = {}
    for f in files:
        z = np.load(f)
        st = int(z["stage"])
        extra.setdefault(st, {k: z[k] for k in z.files if k not in ("latent", "class_id", "seed")})
        for i, (c, s) in enumerate(zip(z["class_id"], z["seed"])):
            by[st][(int(c), int(s))] = z["latent"][i]
    return by, extra


def curves(by, model):
    """Per-stage rel_dist, norm_ratio and cosine, averaged over images present at every stage."""
    stages = sorted(by)
    # VAR counts up (si 0..9); DiT timesteps count DOWN (999..0), so the final stage is the min
    final_stage = stages[-1] if model == "var" else stages[0]
    common = sorted(set.intersection(*(set(v) for v in by.values())))
    if not common:
        raise SystemExit("ERROR: no image appears at every stage — run verify_extraction first")

    F = np.stack([by[final_stage][k] for k in common]).reshape(len(common), -1).astype(np.float64)
    nF = np.linalg.norm(F, axis=1)

    rows = []
    for st in (stages if model == "var" else sorted(stages, reverse=True)):
        A = np.stack([by[st][k] for k in common]).reshape(len(common), -1).astype(np.float64)
        nA = np.linalg.norm(A, axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            rel = np.linalg.norm(A - F, axis=1) / nF
            ratio = nA / nF
            cos = (A * F).sum(1) / (nA * nF)
        rows.append({"stage": st, "n_images": len(common),
                     "rel_dist": float(np.nanmean(rel)), "rel_dist_sd": float(np.nanstd(rel)),
                     "norm_ratio": float(np.nanmean(ratio)), "cosine": float(np.nanmean(cos))})
    return rows, final_stage, len(common)


def orthogonality_report(rows):
    """The diagnostic that decides how the curve may be described.

    Returns (max |cosine - norm_ratio|, max |rel_dist - sqrt(1 - norm_ratio^2)|).
    Both near zero => increments are near-orthogonal => rel_dist tracks accumulated energy.
    """
    dc = max(abs(r["cosine"] - r["norm_ratio"]) for r in rows)
    dr = max(abs(r["rel_dist"] - np.sqrt(max(0.0, 1 - r["norm_ratio"] ** 2))) for r in rows)
    return dc, dr


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stages", required=True, type=Path)
    ap.add_argument("--model", required=True, choices=("var", "dit"))
    ap.add_argument("--csv", type=Path, help="write the per-stage table here")
    ap.add_argument("--plot", type=Path, help="write a figure here")
    args = ap.parse_args(argv)

    by, _ = load_stages(args.stages, args.model)
    rows, final_stage, n = curves(by, args.model)

    tok = [p * p for p in VAR_PN]
    cum = np.cumsum(tok)
    total = sum(tok)

    print(f"{args.model.upper()}: {len(rows)} stages x {n} images "
          f"(final stage = {final_stage})\n")
    head = f"{'stage':>5} {'cum%':>7} | {'rel_dist':>9} {'sd':>6} {'|fk|/|fF|':>10} {'cosine':>7}"
    print(head); print("-" * len(head))
    for i, r in enumerate(rows):
        c = f"{100*cum[i]/total:>6.1f}%" if args.model == "var" and i < len(cum) else " " * 7
        print(f"{r['stage']:>5} {c} | {r['rel_dist']:>9.4f} {r['rel_dist_sd']:>6.4f} "
              f"{r['norm_ratio']:>10.4f} {r['cosine']:>7.4f}")

    dc, dr = orthogonality_report(rows)
    print(f"\nORTHOGONALITY DIAGNOSTIC")
    print(f"  max |cosine - norm_ratio|          = {dc:.4f}")
    print(f"  max |rel_dist - sqrt(1-ratio^2)|   = {dr:.4f}")
    if dc < 0.05 and dr < 0.05:
        print("\n  Both near zero: each stage's contribution is ~orthogonal to the sum so far.")
        print("  => rel_dist is a function of ACCUMULATED ENERGY, not semantic commitment.")
        print("  => Report it as accumulated latent energy. Do NOT write 'the image is")
        print("     X% decided by stage k'. That claim needs decoded images + a classifier.")
    else:
        print("\n  Not orthogonal — rel_dist carries directional information here, so it is")
        print("  more than an energy curve. Say which, and show the cosine column.")

    if args.csv:
        import csv
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)
        print(f"\nwrote {args.csv}")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        x = np.arange(len(rows))
        fig, ax = plt.subplots(figsize=(7, 4.2))
        ax.plot(x, [r["rel_dist"] for r in rows], "o-", label="rel_dist to final")
        ax.plot(x, [r["norm_ratio"] for r in rows], "s--", label="|f_k| / |f_final|")
        ax.plot(x, [r["cosine"] for r in rows], "^:", label="cosine to final")
        ax.plot(x, [np.sqrt(max(0, 1 - r["norm_ratio"] ** 2)) for r in rows],
                color="0.6", lw=1, ls="-.", label=r"$\sqrt{1-ratio^2}$ (energy prediction)")
        ax.set_xticks(x); ax.set_xticklabels([r["stage"] for r in rows], fontsize=8)
        ax.set_xlabel(f"{args.model.upper()} stage"); ax.set_ylim(-0.02, 1.05)
        ax.set_title(f"{args.model.upper()} stage curves — rel_dist tracks accumulated energy",
                     fontsize=10)
        ax.legend(fontsize=8, frameon=False); ax.grid(alpha=.3)
        fig.tight_layout(); args.plot.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.plot, dpi=140); plt.close(fig)
        print(f"wrote {args.plot}")


if __name__ == "__main__":
    main()
