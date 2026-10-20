"""P3 shared metrics: the set every lane reports, on two axes.

Built on lanes/p2/quality.py, which already scores VALIDITY (is this still a good image
of its class?). This module adds the missing axis, FIDELITY (is this still the SAME
image?), and the rule for reading the two together.

WHY TWO AXES
------------
The P2 pilot (lanes/p2/results/pilot_25656645) shows one number cannot work:

    VAR scale 0, sev 1.0   PSNR 10.8   cls_prob 0.74 (base 0.75)   KID -0.003
    DiT  t=197, sev 1.0    PSNR 12.6   cls_prob 0.00 (base 0.82)   KID +0.187

Nearly the same pixel distance. The VAR images are clean pictures of the right breed
(the model re-routed to a different valid sample); the DiT images are destroyed. Any
single "recovery" number collapses these two into one value and is therefore wrong.

So every run is placed on a plane:

                    validity HIGH          validity LOW
    fidelity HIGH   unchanged              degraded-in-place  (rare)
    fidelity LOW    RE-ROUTED              DAMAGED

"Recovery" means returning toward HIGH/HIGH. Re-routing is not damage and must not be
scored as damage.

THE METRIC SET
--------------
Fidelity, per image, paired against that image's own baseline:
    incep_cos   cosine distance in Inception-v3 pool space      FREE: no new package,
                                                                no forward pass. Reuses
                                                                inception_features.npz
                                                                that quality.py already
                                                                writes.
    lpips       LPIPS perceptual distance                       optional (needs `lpips`)
    dino_cos    DINOv2 cosine distance                          optional (needs timm)
    img_psnr_db, img_mad255                                     already computed by P2;
                                                                keep as "how much changed",
                                                                never as "how damaged"

Validity, per image, absolute:
    cls_prob, cls_top1, cls_top5, cls_pred     already in lanes/p2/quality.py
    kid_vs_base                                already in lanes/p2/analyze.py (set-level)

Stage trajectories, P3's own lane, on decoded intermediate images:
    spec_centroid, grad_energy, hf_ratio       calibrated 2026-10-01, see
                                               calibration_report.md
    sem_cos_final  cosine similarity of stage image to this run's FINAL image

USAGE
-----
    # free fidelity metric for runs quality.py has already scored
    python -m lanes.p3.p3_metrics fidelity RUN_DIR --baseline BASELINE_DIR

    # add LPIPS too, where the package is available
    python -m lanes.p3.p3_metrics fidelity RUN_DIR --baseline BASELINE_DIR --lpips

Both append to RUN_DIR/metrics.csv in P2's long format
(class_id, seed, metric, value, stage), replacing earlier values of the same metrics.

Owner: P3. Calibration: see lanes/p3/calibration_report.md.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True

FIDELITY_METRICS = ("incep_cos", "lpips", "dino_cos")
STAGE_METRICS = ("spec_centroid", "grad_energy", "hf_ratio", "sem_cos_final")

# Classification thresholds for the two-axis rule. Provisional until run on the real
# frozen baselines; see calibrate_on_baselines.ipynb, which re-derives them.
FIDELITY_LOW = 0.30     # incep_cos above this = "not the same image any more"
VALIDITY_LOW = 0.50     # cls_prob below this fraction OF ITS OWN BASELINE = "broken"
KID_HIGH = 0.005        # run-level KID above this = visible quality loss. From the P2
                        # pilot: every clean-looking run sits within +/-0.003; the two
                        # damaged DiT runs are +0.0185 and +0.1872.


# ==========================================================================
# Fidelity, computed from features quality.py already saved
# ==========================================================================

def load_features(run_dir):
    """Load inception_features.npz written by lanes.p2.quality.

    Returns {(class_id, seed): feature_vector}.
    """
    path = Path(run_dir) / "inception_features.npz"
    if not path.is_file():
        raise SystemExit(
            f"ERROR: {path} not found.\n"
            f"       Run the classifier first:  python -m lanes.p2.quality {run_dir}"
        )
    z = np.load(path)
    return {(int(c), int(s)): f for c, s, f in zip(z["class_id"], z["seed"], z["feat"])}


def cosine_distance(a, b):
    """1 - cosine similarity. 0 = identical direction, 1 = orthogonal, 2 = opposite.

    Inception pool features are post-ReLU and therefore non-negative, so in practice
    this lands in [0, 1].
    """
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return float("nan")
    return float(1.0 - np.dot(a, b) / (na * nb))


def fidelity_incep(run_dir, baseline_dir):
    """Per-image cosine distance to the paired baseline image, in Inception space.

    This is the free fidelity metric: no new dependency, no second forward pass.
    Pairing is on (class_id, seed), which is the manifest key.
    """
    run = load_features(run_dir)
    base = load_features(baseline_dir)
    missing = set(run) - set(base)
    if missing:
        raise SystemExit(
            f"ERROR: {len(missing)} images in the run have no baseline counterpart, "
            f"e.g. {sorted(missing)[:3]}.\n"
            f"       The run and its baseline must use the same manifest."
        )
    return [{"class_id": c, "seed": s, "metric": "incep_cos",
             "value": cosine_distance(run[(c, s)], base[(c, s)]), "stage": ""}
            for (c, s) in sorted(run)]


def fidelity_lpips(run_dir, baseline_dir, device=None):
    """Per-image LPIPS distance. Optional: needs the `lpips` package.

    LPIPS and incep_cos measure different things. LPIPS is sensitive to low-level
    perceptual change (texture, blur, colour); incep_cos is sensitive to semantic
    change. A re-routed VAR image scores HIGH on both; an oversaturated DiT image
    scores high on LPIPS and lower on incep_cos. Report both when available.
    """
    try:
        import torch
        import lpips as lpips_pkg
        from PIL import Image
    except ImportError as e:
        raise SystemExit(
            f"ERROR: LPIPS needs the `lpips` package (and torch): {e}\n"
            f"       pip install lpips      — or omit --lpips and use incep_cos alone."
        )
    from lanes.p2.schema import load_run

    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    net = lpips_pkg.LPIPS(net="alex").to(device).eval()

    run_dir, baseline_dir = Path(run_dir), Path(baseline_dir)
    rows = load_run(run_dir)["rows"]
    base_rows = {(r["class_id"], r["seed"]): r for r in load_run(baseline_dir)["rows"]}

    def load(d, stem):
        a = np.asarray(Image.open(d / f"{stem}.png").convert("RGB")).copy()
        t = torch.from_numpy(a).permute(2, 0, 1).float().div(127.5).sub(1.0)
        return t.unsqueeze(0).to(device)

    out = []
    with torch.inference_mode():
        for r in rows:
            key = (r["class_id"], r["seed"])
            if key not in base_rows:
                continue
            d = net(load(run_dir, r["file"]), load(baseline_dir, base_rows[key]["file"]))
            out.append({"class_id": r["class_id"], "seed": r["seed"],
                        "metric": "lpips", "value": float(d.item()), "stage": ""})
    return out


# ==========================================================================
# The two-axis reading rule
# ==========================================================================

def classify(fidelity, validity_ratio, kid=None,
             fid_low=FIDELITY_LOW, val_low=VALIDITY_LOW, kid_high=KID_HIGH):
    """Place one image (or run) on the fidelity/validity plane.

    fidelity         incep_cos (or lpips) vs the paired baseline. HIGHER = less faithful.
    validity_ratio   cls_prob / baseline cls_prob. LOWER = less valid.
    kid              run-level KID vs baseline, if known. See the warning below.

    Returns one of: 'unchanged', 're-routed', 'damaged', 'degraded-in-place'.

    The distinction that matters: 're-routed' is a DIFFERENT VALID IMAGE and must never
    be scored as damage. VAR does this constantly when perturbed early.

    WARNING — the classifier alone is not a sufficient validity axis.
    In the P2 pilot, DiT t=497 / sev 1.0 produces visibly oversaturated, artifact-heavy
    images, yet keeps cls_prob 0.73 against a 0.82 baseline (ratio 0.89). On cls_prob
    alone this run is indistinguishable from a clean re-route. KID catches it
    (+0.0185, against <=0.003 for every clean-looking run).

    So validity needs BOTH:
        cls_prob   loss of MEANING          (catches DiT t=197: cls_prob -> 0.00)
        KID        loss of VISUAL QUALITY   (catches DiT t=497: KID -> +0.0185)

    KID is a set-level statistic, so it cannot label a single image. Pass it when
    classifying a whole run; omit it per image and treat the per-image label as
    provisional until the run's KID is known.
    """
    far = fidelity > fid_low
    broken = validity_ratio < val_low
    if kid is not None and kid > kid_high:
        broken = True          # visual-quality failure the classifier did not see
    if not far and not broken:
        return "unchanged"
    if far and not broken:
        return "re-routed"
    if far and broken:
        return "damaged"
    return "degraded-in-place"


# ==========================================================================
# Stage-trajectory metrics (P3's own lane) — calibrated 2026-10-01
# ==========================================================================

# Below this standard deviation an image carries no signal and its normalised power
# spectrum is pure floating-point noise. VAR scale 0 is a 1x1 token grid and decodes
# to a near-constant image, so this case IS hit in the real sweep.
FLAT_IMAGE_STD = 1e-6


def _radial_power(img):
    """Radially-binned power spectrum of a grayscale float image in [0, 1].

    The Hann window is load-bearing: without it the image border acts as a hard edge
    and injects a bright cross into the spectrum that swamps real high-frequency
    content.

    Returns an all-NaN profile for a flat image. This matters: normalising a spectrum
    made of rounding error produces numbers that look entirely plausible — a constant
    image returns spec_centroid 0.556, indistinguishable from a real measurement —
    and would silently become VAR's scale-0 data point in every figure.
    """
    if float(np.std(img)) < FLAT_IMAGE_STD:
        return np.full(64, np.nan), np.linspace(0, 1.0, 65)[:-1]
    h, w = img.shape
    win = np.outer(np.hanning(h), np.hanning(w))
    F = np.fft.fftshift(np.fft.fft2((img - img.mean()) * win))
    power = np.abs(F) ** 2
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((y - cy) ** 2 + (x - cx) ** 2) / (min(h, w) / 2)
    nbins = 64
    bins = np.linspace(0, 1.0, nbins + 1)
    idx = np.digitize(r.ravel(), bins) - 1
    valid = (idx >= 0) & (idx < nbins)
    prof = np.bincount(idx[valid], weights=power.ravel()[valid], minlength=nbins)
    prof[0] = 0.0
    total = prof.sum()
    return (prof / total if total > 0 else prof), bins[:-1]


def spec_centroid(img):
    """Power-weighted mean radial frequency. PRIMARY: does not saturate."""
    prof, centres = _radial_power(img)
    return float((prof * centres).sum())


def grad_energy(img):
    """Mean gradient magnitude. PRIMARY: does not saturate."""
    gy, gx = np.gradient(img)
    return float(np.sqrt(gx ** 2 + gy ** 2).mean())


def hf_ratio(img):
    """Fraction of spectral power above 0.35 x Nyquist.

    SCOPED: saturates to ~0 below about 4x downsampling. VAR scales 0-2 are 85-256x
    downsampled, so this reads ~0 across the whole early band. A flat line there is a
    FLOOR EFFECT, not a finding. Use spec_centroid for early-stage analysis.
    """
    prof, centres = _radial_power(img)
    return float(prof[centres >= 0.35].sum())


def latent_rel_dist(stage_latent, final_latent):
    """Relative L2 distance from a stage's latent to the run's FINAL latent, per image.

    PRIMARY stage metric. 1.0 = nothing committed yet, 0.0 = fully determined.

    Why this and not the pixel metrics: it needs no decoder. Every pixel-space metric
    is computed on a DECODED intermediate, so it measures the model's commitment and
    the decoder's behaviour together, and the two cannot be separated after the fact.
    This one reads the model's own state directly.

    It also works identically for both models:
        VAR   f_hat at scale si        vs  f_hat at scale 9
        DiT   pred_xstart at step t    vs  pred_xstart at the last step
    The two latent spaces are different, so compare the SHAPE of the two curves, never
    the absolute values.

    Arrays are (B, C, H, W) or (B, ...); the norm is taken per batch row.
    """
    a = np.asarray(stage_latent, dtype=np.float64).reshape(len(stage_latent), -1)
    b = np.asarray(final_latent, dtype=np.float64).reshape(len(final_latent), -1)
    if a.shape != b.shape:
        raise ValueError(f"stage {a.shape} and final {b.shape} latents differ in shape")
    num = np.linalg.norm(a - b, axis=1)
    den = np.linalg.norm(b, axis=1)
    return np.where(den > 0, num / den, np.nan)


def radial_profile(img):
    """The full 64-bin profile. STORE THIS per stage, not only the scalars above:
    it lets the bands be re-cut later without re-running the GPU sweep."""
    prof, _ = _radial_power(img)
    return prof


# ==========================================================================
# CLI
# ==========================================================================

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fidelity", help="per-image distance to the paired baseline")
    f.add_argument("run_dir", type=Path)
    f.add_argument("--baseline", type=Path, required=True)
    f.add_argument("--lpips", action="store_true", help="also compute LPIPS (needs the package)")
    f.add_argument("--dry-run", action="store_true", help="print, do not write metrics.csv")
    args = ap.parse_args(argv)

    rows = fidelity_incep(args.run_dir, args.baseline)
    names = ["incep_cos"]
    if args.lpips:
        rows += fidelity_lpips(args.run_dir, args.baseline)
        names.append("lpips")

    vals = [r["value"] for r in rows if r["metric"] == "incep_cos"]
    summary = {"run_dir": str(args.run_dir), "baseline": str(args.baseline),
               "n_images": len(vals), "metrics": names,
               "incep_cos_mean": round(float(np.nanmean(vals)), 4),
               "incep_cos_max": round(float(np.nanmax(vals)), 4)}

    if args.dry_run:
        summary["dry_run"] = True
    else:
        from lanes.p2.schema import replace_metrics
        replace_metrics(args.run_dir, tuple(names), rows)
        summary["written"] = str(Path(args.run_dir) / "metrics.csv")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
