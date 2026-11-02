"""P3 model-signature check: do VAR's and DiT's FINISHED images differ on the pixel metrics?

VAR decodes through a quantised codebook (VQVAE); DiT through a continuous VAE. Two different
decoders can leave a systematic fingerprint on every image they produce — VAR's contact
sheets already show heavier posterization. If that fingerprint moves a pixel metric, then a
side-by-side VAR-vs-DiT curve of that metric shows the DECODER, not the model, and the
offset has to be measured and normalised out first. This is calibration_report.md
finding 11; VAR's half is already there (n=20):

    spec_centroid 0.070011   grad_energy 0.035407   hf_ratio 0.045089

This script fills DiT's half and says whether the gap is real.

    python -m lanes.p3.model_signature --var for_darien_20_clean_pngs --dit outputs/p3/dit_finals --dit-stage 0 --csv outputs/p3/model_signature.csv

INPUTS
------
Two flat folders of PNGs named class<cccc>_seed<seed>[_stage<ssss>].png. A folder written by
decode_stages.py has the _stage suffix: pass --dit-stage 0 (DiT's finished image) or
--var-stage 9. A folder of baseline PNGs has none: pass nothing.

When both folders hold the SAME images (same class and seed), the comparison is PAIRED —
each image's DiT value minus its own VAR value — which removes the class-to-class spread and
is much more sensitive. That is why the DiT finals are decoded with
`decode_stages.py --keys-like for_darien_20_clean_pngs`.

THE VERDICT
-----------
95% bootstrap interval of the mean difference (10,000 resamples, fixed seed). If it excludes
zero, the metric carries a per-model offset: in any cross-model figure, plot each model's
curve relative to its OWN final value, never raw values on one shared axis. If it includes
zero, no offset is detectable at this n — say so, with n.

PREPROCESSING
-------------
Identical to calibrate_metrics.to_gray256 (skimage's rgb2gray weights 0.2125/0.7154/0.0721
on [0,1] RGB, no resize at 256x256) — verified to 2e-16 on the 20 VAR baselines, which is
why VAR's numbers above reproduce exactly. Re-implemented here in numpy so this script needs
neither OpenCV nor scikit-image. numpy + Pillow only.

Owner: P3.
"""

import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from lanes.p3 import p3_metrics as P   # noqa: E402

NAME = re.compile(r"^class(\d+)_seed(\d+)(?:_stage(\d+))?\.png$")
RGB2GRAY = np.array([0.2125, 0.7154, 0.0721])          # skimage.color.rgb2gray
METRICS = ("spec_centroid", "grad_energy", "hf_ratio")
VAR_PUBLISHED = {"spec_centroid": 0.070011, "grad_energy": 0.035407, "hf_ratio": 0.045089}


def to_gray256(img):
    """RGB uint8 -> 256x256 grey in [0, 1], exactly as calibrate_metrics.to_gray256."""
    a = np.asarray(img)
    if a.ndim == 3:
        a = (a[..., :3].astype(np.float64) / 255.0) @ RGB2GRAY
    else:
        a = a.astype(np.float64)
        if a.max() > 1.5:
            a = a / 255.0
    if a.shape != (256, 256):
        from PIL import Image
        a = np.asarray(Image.fromarray(a.astype(np.float32))      # float32 2-D -> mode "F"
                       .resize((256, 256), Image.BOX), dtype=np.float64)
        print("  note: an image was not 256x256 and was resized; values may differ slightly "
              "from calibrate_metrics")
    return np.clip(a, 0, 1)


def index(folder, stage, flag):
    """{(class_id, seed): path} for one folder, using the stage rule described above."""
    folder = Path(folder)
    if not folder.is_dir():
        raise SystemExit(f"ERROR: folder not found: {folder}")
    plain, staged = {}, {}
    for p in sorted(folder.glob("*.png")):
        m = NAME.match(p.name)
        if not m:
            continue
        key = (int(m.group(1)), int(m.group(2)))
        if m.group(3) is None:
            plain[key] = p
        else:
            staged.setdefault(int(m.group(3)), {})[key] = p
    if stage is None:
        if plain:
            return plain
        if staged:
            raise SystemExit(f"ERROR: {folder} holds decoded stages {sorted(staged)}; pass "
                             f"{flag} to choose one (DiT's finished image is 0, VAR's is 9)")
        raise SystemExit(f"ERROR: no class<N>_seed<N>.png files in {folder}")
    if stage not in staged:
        raise SystemExit(f"ERROR: no _stage{stage:04d} files in {folder}; stages there: "
                         f"{sorted(staged) or 'none'}")
    return staged[stage]


def measure(paths):
    """{metric: np.array} over the images, in the order given."""
    from PIL import Image
    out = {m: [] for m in METRICS}
    for p in paths:
        g = to_gray256(np.asarray(Image.open(p).convert("RGB")))
        for m in METRICS:
            out[m].append(float(getattr(P, m)(g)))
    return {m: np.array(v) for m, v in out.items()}


def boot_mean(x, n_boot, rng):
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    return x[idx].mean(1)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--var", required=True, type=Path, help="folder of VAR finished images")
    ap.add_argument("--dit", required=True, type=Path, help="folder of DiT finished images")
    ap.add_argument("--var-stage", type=int, help="if --var is a decoded-stages folder")
    ap.add_argument("--dit-stage", type=int, help="if --dit is a decoded-stages folder (use 0)")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--csv", type=Path, help="summary table; per-image values go beside it")
    args = ap.parse_args(argv)

    var = index(args.var, args.var_stage, "--var-stage")
    dit = index(args.dit, args.dit_stage, "--dit-stage")
    common = sorted(set(var) & set(dit))
    paired = len(common) >= 5
    print(f"VAR {len(var)} images from {args.var}\nDiT {len(dit)} images from {args.dit}")
    print(f"same image in both: {len(common)}  ->  "
          f"{'PAIRED comparison (DiT minus the same image in VAR)' if paired else 'UNPAIRED comparison (too few shared images to pair)'}\n")

    vk, dk = (common, common) if paired else (sorted(var), sorted(dit))
    V, D = measure([var[k] for k in vk]), measure([dit[k] for k in dk])
    rng = np.random.default_rng(0)

    rows, per_image = [], []
    identical = paired
    for m in METRICS:
        v, d = V[m], D[m]
        if paired:
            diff = d - v
            identical &= bool(np.all(diff == 0))
            ok = np.isfinite(diff)
            dm, boot = float(diff[ok].mean()), boot_mean(diff[ok], args.n_boot, rng)
            med = float(np.median(diff[ok]))
        else:
            dm = float(np.nanmean(d) - np.nanmean(v))
            boot = (boot_mean(d[np.isfinite(d)], args.n_boot, rng)
                    - boot_mean(v[np.isfinite(v)], args.n_boot, rng))
            med = float(np.nanmedian(d) - np.nanmedian(v))
        lo, hi = (float(x) for x in np.percentile(boot, [2.5, 97.5]))
        offset = lo > 0 or hi < 0
        rows.append({"metric": m, "n_var": len(v), "n_dit": len(d), "paired": int(paired),
                     "var_mean": float(np.nanmean(v)), "var_sd": float(np.nanstd(v, ddof=1)),
                     "dit_mean": float(np.nanmean(d)), "dit_sd": float(np.nanstd(d, ddof=1)),
                     "diff_mean": dm, "diff_median": med, "ci_lo": lo, "ci_hi": hi,
                     "pct_of_var": 100 * dm / float(np.nanmean(v)),
                     "verdict": "OFFSET" if offset else "no detectable offset"})
    if paired:
        for i, k in enumerate(common):
            per_image.append({"class_id": k[0], "seed": k[1],
                              **{f"var_{m}": V[m][i] for m in METRICS},
                              **{f"dit_{m}": D[m][i] for m in METRICS}})

    print(f"{'metric':>14} | {'VAR mean':>9} {'DiT mean':>9} | {'DiT - VAR':>10} "
          f"{'95% interval':>21} {'% of VAR':>9} | verdict")
    print("-" * 96)
    for r in rows:
        print(f"{r['metric']:>14} | {r['var_mean']:>9.6f} {r['dit_mean']:>9.6f} | "
              f"{r['diff_mean']:>+10.6f} [{r['ci_lo']:>+9.6f}, {r['ci_hi']:>+9.6f}] "
              f"{r['pct_of_var']:>+8.1f}% | {r['verdict']}")

    if identical:
        print("\n  WARNING: every paired difference is exactly 0 — the two folders hold the SAME "
              "images. Check you pointed --dit at DiT output.")
    vs = {r["metric"]: r["var_mean"] for r in rows}
    if paired and len(common) == 20 and all(abs(vs[m] - VAR_PUBLISHED[m]) < 5e-7 for m in METRICS):
        print("\n  VAR side reproduces calibration_report.md finding 11 exactly — same images, "
              "same preprocessing.")

    print("\nHOW TO READ IT")
    print("  OFFSET               the interval excludes 0: the decoders leave different fingerprints")
    print("                       on this metric. In cross-model figures, plot each model relative")
    print("                       to its own final value — never raw values on one shared axis.")
    print("  no detectable offset the interval includes 0 at this n. Report it with n, not as")
    print("                       'the models are the same'.")

    print("\nFOR calibration_report.md, finding 11 (paste over the empty DiT row):")
    dm = {r["metric"]: r["dit_mean"] for r in rows}
    print(f"| **DiT** finals (n={rows[0]['n_dit']}) | {dm['spec_centroid']:.6f} | "
          f"{dm['grad_energy']:.6f} | {dm['hf_ratio']:.6f} |")

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        print(f"\nwrote {args.csv}")
        if per_image:
            pi = args.csv.with_name(args.csv.stem + "_per_image.csv")
            with pi.open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(per_image[0]))
                w.writeheader(); w.writerows(per_image)
            print(f"wrote {pi}  (one row per image: look here if one image drives the result)")


if __name__ == "__main__":
    main()
