"""P3 metric calibration — runnable by any lane, no GPU and no checkpoints needed.

WHY CALIBRATE
-------------
Later in the project a curve will claim "this much information appears at VAR scale 3".
Before anyone believes that curve, the measuring stick has to be shown to work. So: break
an image in a way where the answer is already known, and check the metric reacts correctly.

Three checks; a metric has to pass all three:

    DIRECTION      does it move the way physics says?        blur must lower sharpness
    MONOTONICITY   does it move consistently?                |Spearman rho| >= 0.90
    SENSITIVITY    does it move ENOUGH?                      >= 2% at the MILDEST damage

A metric that only notices catastrophic damage cannot resolve stages, so "it moved" is not
enough on its own.

The discipline that makes this a real test rather than looking at curves and finding reasons
they seem fine: EXPECTED below commits to a direction for every metric/degradation pair
BEFORE anything is computed.

RUN IT
------
    python -m lanes.p3.calibrate_metrics                      # sample photos, runs anywhere
    python -m lanes.p3.calibrate_metrics --baseline-dir outputs/baseline/25665469/var_d20
    python -m lanes.p3.calibrate_metrics --help

Writes `calibration_curves.png` and `calibration_decisions.json` to --out-dir (default: the
current directory). Deterministic: the noise draw is seeded from the project master seed.

IMPORT IT
---------
Every piece is importable, so other lanes can reuse the parts rather than copying them:

    from lanes.p3.calibrate_metrics import var_scale_ladder, DEGRADATIONS, load_images

    # what can VAR's scale si possibly contain? (no GPU)
    for row in var_scale_ladder(load_images()):
        print(row["si"], row["grid"], row["grad_energy"])

    # apply a known degradation to your own image
    blurred = DEGRADATIONS["blur"][0](img, sigma=2.0)

Owner: P3. Metric definitions live in `p3_metrics.py`; results in `calibration_report.md`.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import cv2
from scipy.stats import spearmanr

sys.dont_write_bytecode = True

try:                                    # works both as `python -m lanes.p3.…` and standalone
    from lanes.p3 import p3_metrics as P
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import p3_metrics as P

IMG_SIZE = 256                          # match the generation resolution
MASTER_SEED = 61320017                  # the project master seed, so runs are reproducible

# VAR-d20's scale schedule. 680 tokens in total.
PN = [1, 2, 3, 4, 5, 6, 8, 10, 13, 16]
TOK = [p * p for p in PN]
CUM = np.cumsum(TOK)
TOTAL = int(sum(TOK))

# Pass thresholds.
MONOTONICITY_MIN = 0.90
DIRECTION_MIN = 1.00
SENSITIVITY_MIN = 0.02


# ==========================================================================
# Images
# ==========================================================================

def to_gray256(a):
    """Any image array -> 256x256 grayscale float in [0, 1]."""
    from skimage import color
    if a.ndim == 3:
        a = color.rgb2gray(a)
    a = a.astype(np.float64)
    if a.max() > 1.5:
        a = a / 255.0
    return np.clip(cv2.resize(a, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA), 0, 1)


def load_images(baseline_dir=None, n=5):
    """Sample photos by default; real baseline PNGs when `baseline_dir` is given.

    Calibrating on the project's own dog images is strictly better than on stock photos,
    because the thresholds then come from the distribution the metrics will actually see.
    """
    if baseline_dir is None:
        from skimage import data
        names = ["chelsea", "astronaut", "coffee", "rocket", "camera"][:n]
        return {nm: to_gray256(getattr(data, nm)()) for nm in names}

    from PIL import Image
    pngs = sorted(Path(baseline_dir).glob("*.png"))[:n]
    if not pngs:
        raise SystemExit(f"ERROR: no .png files in {baseline_dir}")
    return {p.stem: to_gray256(np.asarray(Image.open(p).convert("RGB"))) for p in pngs}


# ==========================================================================
# Degradations — damage whose answer is already known
# ==========================================================================

def degrade_blur(img, sigma):
    """Gaussian blur. REMOVES high frequency. Simulates an early VAR scale."""
    if sigma <= 0:
        return img.copy()
    k = int(2 * round(3 * sigma) + 1)
    return cv2.GaussianBlur(img, (k, k), sigma)


def degrade_noise(img, sigma8, rng=None):
    """Additive Gaussian noise, sigma on a 0-255 scale.
    ADDS high frequency. Simulates an early DiT timestep."""
    if sigma8 <= 0:
        return img.copy()
    rng = rng or np.random.default_rng(MASTER_SEED)
    return np.clip(img + rng.normal(0, sigma8 / 255.0, img.shape), 0, 1)


def degrade_downsample(img, factor):
    """Downsample then upsample. REMOVES high frequency, blockier than blur.
    The closest analogue to VAR's coarse token grids."""
    if factor <= 1:
        return img.copy()
    small = cv2.resize(img, (IMG_SIZE // factor, IMG_SIZE // factor), interpolation=cv2.INTER_AREA)
    return np.clip(cv2.resize(small, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC), 0, 1)


# severities[0] is always the clean / identity case
DEGRADATIONS = {
    "blur":       (degrade_blur,       [0, 0.5, 1.0, 2.0, 4.0, 8.0]),
    "noise":      (degrade_noise,      [0, 2, 5, 10, 20, 40]),
    "downsample": (degrade_downsample, [1, 2, 4, 8, 16, 32]),
}

METRICS = {
    "spec_centroid": P.spec_centroid,
    "grad_energy":   P.grad_energy,
    "hf_ratio":      P.hf_ratio,
}

# +1 = the metric should RISE with severity, -1 = FALL. Committed before computing anything.
# Note blur and noise are OPPOSITE for every metric. That is the finding in §6 below.
EXPECTED = {
    "blur":       {"spec_centroid": -1, "grad_energy": -1, "hf_ratio": -1},
    "noise":      {"spec_centroid": +1, "grad_energy": +1, "hf_ratio": +1},
    "downsample": {"spec_centroid": -1, "grad_energy": -1, "hf_ratio": -1},
}


# ==========================================================================
# The VAR scale ladder — what each scale can possibly contain
# ==========================================================================

def var_scale_proxy(img, pn):
    """Downsample to pn x pn and back: the best case at that scale."""
    return degrade_downsample(img, max(1, IMG_SIZE // pn)) if pn < IMG_SIZE else img.copy()


def var_scale_ladder(images):
    """Metric values at each VAR scale, on an ideal codec.

    An UPPER BOUND on what the model could have committed by scale si, computed with no GPU.
    Useful to every lane: it says in advance which metrics have any room to move.
    """
    rows = []
    for si, pn in enumerate(PN):
        vals = {m: [] for m in METRICS}
        for img in images.values():
            small = cv2.resize(img, (pn, pn), interpolation=cv2.INTER_AREA)
            proxy = np.clip(cv2.resize(small, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC), 0, 1)
            for m, fn in METRICS.items():
                vals[m].append(fn(proxy))
        # all-NaN is the CORRECT answer at si=0: a 1x1 grid decodes to a flat image
        mean = lambda v: float(np.nanmean(v)) if not np.all(np.isnan(v)) else float("nan")
        rows.append({"si": si, "grid": f"{pn}x{pn}", "tokens": TOK[si],
                     "cum_pct": round(100 * float(CUM[si]) / TOTAL, 2),
                     **{m: mean(v) for m, v in vals.items()}})
    return rows


# ==========================================================================
# Calibration
# ==========================================================================

def run_calibration(images, rng=None):
    """Apply every degradation at every severity and compute every metric."""
    rng = rng or np.random.default_rng(MASTER_SEED)
    results = {}
    for dname, (fn, sevs) in DEGRADATIONS.items():
        values = {m: {} for m in METRICS}
        for iname, img in images.items():
            for s in sevs:
                dmg = fn(img, s, rng) if dname == "noise" else fn(img, s)
                for m, mfn in METRICS.items():
                    values[m].setdefault(iname, []).append(mfn(dmg))
        results[dname] = {"severities": sevs, "values": values}
    return results


def evaluate(results):
    """Score each (degradation, metric) pair against its committed prediction.

    Everything is measured against each image's OWN clean value, because between-image
    variance in natural images otherwise swamps the effect.
    """
    rows = []
    for dname, r in results.items():
        sevs = r["severities"]
        for mname, per_image in r["values"].items():
            exp = EXPECTED[dname][mname]
            rhos, dirs, sens = [], [], []
            for series in per_image.values():
                series = np.asarray(series, float)
                if np.isnan(series).any():          # the flat-image guard fired
                    continue
                rhos.append(spearmanr(sevs, series).statistic)
                base = series[0]
                if base != 0:
                    dirs.append(np.sign(series[-1] - base))
                    sens.append(abs((series[1] - base) / base))     # MILDEST damage
            rho = float(np.nanmean(rhos)) if rhos else float("nan")
            dfrac = float(np.mean([d == exp for d in dirs])) if dirs else float("nan")
            msens = float(np.nanmedian(sens)) if sens else float("nan")

            ok = (dfrac >= DIRECTION_MIN,
                  abs(rho) >= MONOTONICITY_MIN and np.sign(rho) == exp,
                  msens >= SENSITIVITY_MIN)
            verdict = "PASS" if all(ok) else "WEAK (insensitive)" if ok[0] and ok[1] else "FAIL"
            rows.append({"degradation": dname, "metric": mname, "expected": exp,
                         "rho": round(rho, 3), "direction": dfrac,
                         "sensitivity": round(msens, 4), "verdict": verdict})
    return rows


def plot(results, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ms = list(METRICS)
    fig, axes = plt.subplots(len(DEGRADATIONS), len(ms),
                             figsize=(3.4 * len(ms), 2.8 * len(DEGRADATIONS)))
    for i, (dname, r) in enumerate(results.items()):
        sevs = r["severities"]
        for j, mname in enumerate(ms):
            ax = axes[i, j]
            for iname, series in r["values"][mname].items():
                series = np.asarray(series, float)
                base = series[0] if (series[0] != 0 and not np.isnan(series[0])) else 1.0
                ax.plot(range(len(sevs)), series / base, marker="o", ms=3, lw=1.2,
                        label=iname, alpha=.85)
            ax.axhline(1.0, color="0.6", lw=.8, ls="--")
            ax.set_xticks(range(len(sevs)))
            ax.set_xticklabels([str(s) for s in sevs], fontsize=7)
            ax.set_title(f"{mname} — {dname}\nexpect "
                         f"{'up' if EXPECTED[dname][mname] > 0 else 'down'}", fontsize=8)
            ax.tick_params(labelsize=7)
            if j == 0:
                ax.set_ylabel("relative to clean", fontsize=8)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, fontsize=8, frameon=False,
               bbox_to_anchor=(.5, -.02))
    fig.suptitle("Metric calibration — each curve is one image", fontsize=12)
    fig.tight_layout(rect=[0, .03, 1, .97])
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


METRIC_ROLES = {
    "latent_rel_dist": "PRIMARY — decoder-independent, both models",
    "grad_energy":     "PRIMARY pixel metric — only one with real range on the VAR ladder",
    "spec_centroid":   "SUPPORTING — graded but compressed on the VAR ladder",
    "hf_ratio":        "LATE STAGES ONLY — ~0 across the whole VAR ladder",
    "pixel_entropy":   "DROPPED 2026-10-01 — failed direction and sensitivity",
}

WARNINGS = [
    "blur and noise move every frequency metric in OPPOSITE directions; VAR's early scales "
    "are blur-like, DiT's early timesteps are noise-like, so the same metric means opposite "
    "things in the two models",
    "flat images (VAR si=0) must return NaN: a constant image otherwise yields a plausible "
    "spec_centroid of 0.556 built entirely from floating-point noise",
    "pixel metrics run on DECODED intermediates, so they measure the model's commitment and "
    "the VQVAE decoder's behaviour together; latent_rel_dist avoids this",
]


# ==========================================================================
# CLI
# ==========================================================================

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline-dir", type=Path, default=None,
                    help="folder of clean baseline PNGs; omit to use sample photos")
    ap.add_argument("--n-images", type=int, default=5,
                    help="how many images to calibrate on (default 5; 20 for tighter numbers)")
    ap.add_argument("--out-dir", type=Path, default=Path("."),
                    help="where to write the figure and the decisions JSON")
    ap.add_argument("--no-plot", action="store_true", help="skip the figure")
    args = ap.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    images = load_images(args.baseline_dir, args.n_images)
    source = str(args.baseline_dir) if args.baseline_dir else "sample photos"
    print(f"calibrating on {len(images)} images from {source}\n")

    # ---- the VAR scale ladder -------------------------------------------------
    print("VAR scale ladder — best-case information at each scale (no GPU)")
    print(f"{'si':>2} {'grid':>7} {'tok':>5} {'cum%':>7} | "
          f"{'spec_centroid':>13} {'grad_energy':>12} {'hf_ratio':>10}")
    print("-" * 70)
    ladder = var_scale_ladder(images)
    for r in ladder:
        print(f"{r['si']:>2} {r['grid']:>7} {r['tokens']:>5} {r['cum_pct']:>6.1f}% | "
              f"{r['spec_centroid']:>13.4f} {r['grad_energy']:>12.5f} {r['hf_ratio']:>10.6f}")
    full = {m: float(np.nanmean([fn(i) for i in images.values()])) for m, fn in METRICS.items()}
    print("-" * 70)
    print(f"{'':>2} {'FULL 256':>7} {TOTAL:>5} {100.0:>6.1f}% | "
          f"{full['spec_centroid']:>13.4f} {full['grad_energy']:>12.5f} {full['hf_ratio']:>10.6f}")
    print("\n  si=0 is NaN and that is correct: a 1x1 grid decodes to a flat image.")
    print("  A flat line across VAR's early band is a FLOOR EFFECT, not a finding.\n")

    # ---- calibration ----------------------------------------------------------
    results = run_calibration(images)
    rows = evaluate(results)
    hdr = f"{'degradation':<12} {'metric':<15} {'exp':>4} {'rho':>7} {'dir':>6} {'sens':>8}  verdict"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        e = {1: "up", -1: "down"}[r["expected"]]
        print(f"{r['degradation']:<12} {r['metric']:<15} {e:>4} {r['rho']:>7.3f} "
              f"{r['direction']:>5.0%} {r['sensitivity']:>7.2%}  {r['verdict']}")

    # ---- outputs --------------------------------------------------------------
    if not args.no_plot:
        plot(results, args.out_dir / "calibration_curves.png")

    decisions = {"owner": "P3", "images": source, "n_images": len(images),
                 "thresholds": {"monotonicity": MONOTONICITY_MIN,
                                "direction": DIRECTION_MIN,
                                "sensitivity": SENSITIVITY_MIN},
                 "var_scale_ladder": ladder, "results": rows,
                 "metric_roles": METRIC_ROLES, "warnings": WARNINGS}
    (args.out_dir / "calibration_decisions.json").write_text(json.dumps(decisions, indent=2))

    print(f"\nwrote {args.out_dir / 'calibration_decisions.json'}"
          + ("" if args.no_plot else f" and {args.out_dir / 'calibration_curves.png'}"))
    print("\nmetric roles:")
    for k, v in METRIC_ROLES.items():
        print(f"  {k:18} {v}")


if __name__ == "__main__":
    main()
