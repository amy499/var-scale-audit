"""P1: Fourier Ring Correlation (FRC) between image pairs, by spatial frequency. CPU only: numpy + Pillow
(+ matplotlib for the figure), from the PNGs already on disk.

    python -m lanes.p1.frc [--crystal-dir ../crystal_64_p1] [--sanity-only]
        -> lanes/p1/results/frc.csv, frc_bands.csv, fig1e_frc.png

Method, per image pair:
  - load as 256 x 256 RGB (real images: shorter side resized to 256, centre crop, as lanes/p2/transfer.py
    load_image, so the same pixels `transfer score` compared), luminance Y = 0.299 R + 0.587 G + 0.114 B;
  - subtract the image's mean, multiply by a 2D Hann window, 2D FFT;
  - FRC(r) = Re(sum F1 conj(F2)) / sqrt(sum |F1|^2 * sum |F2|^2) over the ring of frequencies with
    round(|f|) = r cycles/image, r = 1..128 (r = 0 is removed by the mean subtraction).
  Band summaries per pair: mean FRC over r = 1..8 ("latent band": up to 8 cycles, what VAR's 16 x 16 token
  grid can carry) and over r = 9..128 ("decoder band"). VAR scale k (grid n_k = 1 2 3 4 5 6 8 10 13 16) can
  carry up to n_k / 2 cycles/image.

Comparisons (per arm: mean FRC curve over images; band means with 95% bootstrap CIs over images, 10000
resamples, seed 0):
  phasef_vs_reference  Phase F jobs 26205157 (crystal), 26205158 (dog): remove_s0..s9 vs the reference run
  phasef_vs_real       the same arms and the reference run vs the real input image
  imagenet_vs_baseline seed 1 free generation: remove_s0..s9 (25893211: s0, s9; 25916920: s1-s8) vs their own
                       job's baseline, paired by (class_id, seed), 224 images
Real images: crystals from --crystal-dir; dogs re-derived from outputs/baseline/25665469/var_d20 by
make_dog64.py's rule. Each job's transfer.csv names the real image of every output file; the script stops
unless those names match the images found locally (64 per dataset).
Sanity check, printed first: an image with itself gives FRC = 1 at every r; baseline image i vs image i + 32
(another class and seed) gives about 0.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

SIZE, R_MAX, LATENT = 256, 128, 8
GRID = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
PHASEF = {"crystal": "26205157", "dog": "26205158"}
IMAGENET = {"25893211": (0, 9), "25916920": tuple(range(1, 9))}
OUT = Path("lanes/p1/results")
CURVE_COLUMNS = ("comparison", "dataset", "arm", "removed_scale", "r", "frc", "n")
BAND_COLUMNS = ("comparison", "dataset", "arm", "removed_scale", "n",
                "latent_1_8", "latent_1_8_lo", "latent_1_8_hi", "decoder_9_128", "decoder_9_128_lo",
                "decoder_9_128_hi")

_r = np.hypot(*np.meshgrid(np.fft.fftfreq(SIZE) * SIZE, np.fft.fftfreq(SIZE) * SIZE, indexing="ij"))
RING = np.rint(_r).astype(int)                       # ring index of every FFT bin
WINDOW = np.outer(np.hanning(SIZE), np.hanning(SIZE))


def load_rgb(path: Path) -> np.ndarray:
    """uint8 (256, 256, 3): lanes/p2/transfer.py load_image without torch (16-bit stretch, resize, crop)."""
    im = Image.open(path)
    if im.mode in ("I", "I;16", "I;16B", "I;16L", "F"):
        a = np.asarray(im, dtype=np.float32)
        lo, hi = np.percentile(a, (0.5, 99.5))
        im = Image.fromarray((np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8))
    im = im.convert("RGB")
    w, h = im.size
    if (w, h) != (SIZE, SIZE):
        s = SIZE / min(w, h)
        im = im.resize((max(SIZE, round(w * s)), max(SIZE, round(h * s))), Image.BICUBIC)
        w, h = im.size
        im = im.crop(((w - SIZE) // 2, (h - SIZE) // 2, (w - SIZE) // 2 + SIZE, (h - SIZE) // 2 + SIZE))
    return np.asarray(im)


_FFT = {}


def spectrum(path: Path) -> np.ndarray:
    """FFT of the mean-subtracted, Hann-windowed luminance (cached per file)."""
    key = str(Path(path).resolve())
    if key not in _FFT:
        y = load_rgb(path).astype(np.float64) @ np.array([0.299, 0.587, 0.114])
        _FFT[key] = np.fft.fft2((y - y.mean()) * WINDOW)
    return _FFT[key]


def frc(f1: np.ndarray, f2: np.ndarray) -> np.ndarray:
    """FRC at r = 1..R_MAX for two spectra."""
    ring = RING.ravel()
    num = np.bincount(ring, (f1 * np.conj(f2)).real.ravel(), minlength=R_MAX + 1)
    d1 = np.bincount(ring, (np.abs(f1) ** 2).ravel(), minlength=R_MAX + 1)
    d2 = np.bincount(ring, (np.abs(f2) ** 2).ravel(), minlength=R_MAX + 1)
    den = np.sqrt(d1 * d2)
    return (num / np.where(den > 0, den, np.nan))[1:R_MAX + 1]


def curves(pairs) -> np.ndarray:
    """(n_pairs, R_MAX) FRC curves for [(path_a, path_b), ...]."""
    return np.stack([frc(spectrum(a), spectrum(b)) for a, b in pairs])


def summarize(comparison, dataset, arm, scale, c: np.ndarray) -> tuple[list, dict]:
    from lanes.p1.transfer_summary import mean_ci
    n = len(c)
    curve = [{"comparison": comparison, "dataset": dataset, "arm": arm, "removed_scale": scale, "r": r,
              "frc": round(float(v), 5), "n": n} for r, v in zip(range(1, R_MAX + 1), c.mean(axis=0))]
    band = {"comparison": comparison, "dataset": dataset, "arm": arm, "removed_scale": scale, "n": n}
    for name, sl in (("latent_1_8", slice(0, LATENT)), ("decoder_9_128", slice(LATENT, R_MAX))):
        for s, v in zip(("", "_lo", "_hi"), mean_ci(c[:, sl].mean(axis=1))):
            band[f"{name}{s}"] = round(v, 5)
    return curve, band


# ---------------------------------------------------------------- inputs

def phasef_pairs(dataset: str, crystal_dir: Path) -> dict:
    """{arm: [(output png, reference png, real image)]} for one Phase F job, via its transfer.csv."""
    job = Path("outputs/p1") / PHASEF[dataset]
    if dataset == "crystal":
        from lanes.p2.transfer import EXTS
        real = {p.name: p for p in sorted(crystal_dir.iterdir()) if p.suffix.lower() in EXTS}
    else:
        from lanes.p1.make_dog64 import MANIFEST, SRC, pick
        with MANIFEST.open(newline="") as f:
            rows = sorted((int(r["class_id"]), int(r["seed"])) for r in csv.DictReader(f))
        real = {f"dog{n:02d}_class{c:04d}_seed{s}.png": SRC / f"class{c:04d}_seed{s}.png"
                for n, (c, s) in enumerate((rows[i] for i in pick(rows)), start=1)}
    missing = [n for n, p in real.items() if not p.is_file()]
    if len(real) != 64 or missing:
        raise SystemExit(f"ERROR: {dataset}: {len(real)} real images found locally, {len(missing)} missing; expected 64")
    out = {}
    for arm in ("reference", *(f"remove_s{j}" for j in range(10))):
        rows = list(csv.DictReader(open(job / arm / "transfer.csv", newline="")))
        names = [r["image"] for r in rows]
        if sorted(names) != sorted(real):
            raise SystemExit(f"ERROR: {job}/{arm}/transfer.csv names images that do not match the local {dataset} set")
        out[arm] = [(job / arm / f"{r['file']}.png", job / "reference" / f"{r['file']}.png", real[r["image"]])
                    for r in rows]
    return out


def imagenet_pairs() -> dict:
    """{scale: [(arm png, baseline png)]} for the seed-1 single-scale arms, paired by (class_id, seed)."""
    from lanes.p1.summarize import job_dir
    out = {}
    for j, scales in IMAGENET.items():
        job = job_dir(j)
        runs = list(csv.DictReader(open(job / "tables" / "runs.csv", newline="")))
        base = job / next(r["run_dir"] for r in runs if r["role"] == "baseline")
        for r in runs:
            if r["role"] == "baseline":
                continue
            s = json.loads(r["params"])["scales"]
            if len(s) != 1 or s[0] not in scales:
                raise SystemExit(f"ERROR: {j} {r['arm']}: unexpected scales {s}")
            d = job / r["run_dir"]
            names = sorted(p.name for p in d.glob("class*_seed*.png"))
            if names != sorted(p.name for p in base.glob("class*_seed*.png")) or len(names) != 224:
                raise SystemExit(f"ERROR: {d}: PNGs do not match its baseline's 224")
            out[s[0]] = [(d / n, base / n) for n in names]
    if sorted(out) != list(range(10)):
        raise SystemExit(f"ERROR: ImageNet arms found for scales {sorted(out)}, expected 0-9")
    return out


def sanity():
    base = Path("outputs/p1/25916920/var/baseline")
    files = sorted(base.glob("class*_seed*.png"))
    same = curves([(p, p) for p in files[:16]])
    other = curves([(files[i], files[(i + 32) % len(files)]) for i in range(len(files))])
    print(f"sanity: image vs itself, 16 images: FRC min {np.nanmin(same):.6f}, max {np.nanmax(same):.6f} (expect 1)")
    print(f"sanity: unrelated pairs (image i vs i + 32: other class and seed), {len(other)} pairs: mean FRC "
          f"r 1-8 {other[:, :LATENT].mean():+.4f}, r 9-128 {other[:, LATENT:].mean():+.4f}, "
          f"max |mean curve| {np.abs(other.mean(axis=0)).max():.4f} (expect about 0)")
    return same, other


# ---------------------------------------------------------------- figure

def plot(curve_rows, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    SURFACE, INK, INK2, GRIDC = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
    RAMP = ("#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab",
            "#104281")   # sequential blue, light = scale 0 ... dark = scale 9
    panels = (("phasef_vs_reference", "crystal", "A  Crystals (Phase F): arm vs reference"),
              ("phasef_vs_reference", "dog", "B  Dogs (Phase F): arm vs reference"),
              ("imagenet_vs_baseline", "imagenet", "C  ImageNet free generation: arm vs baseline"))
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharey=True)
    fig.patch.set_facecolor(SURFACE)
    r = np.arange(1, R_MAX + 1)
    for ax, (comp, ds, title) in zip(axes, panels):
        ax.set_facecolor(SURFACE)
        for k, n in enumerate(GRID):
            if n / 2 >= 1:
                ax.axvline(n / 2, color=RAMP[k], linestyle="--", linewidth=0.9, zorder=1)
                ax.text(n / 2, 1.075, str(k), color=INK2, fontsize=7, ha="center", va="bottom")
        ax.axvline(LATENT + 0.5, color=INK2, linewidth=0.8, zorder=1)
        for k in range(10):
            y = [c["frc"] for c in curve_rows if c["comparison"] == comp and c["dataset"] == ds
                 and c["removed_scale"] == k]
            ax.plot(r, y, color=RAMP[k], linewidth=1.6, zorder=2)
        ax.set_xscale("log")
        ax.set_xlim(1, R_MAX)
        ax.set_ylim(-0.1, 1.07)
        ax.set_xticks([1, 2, 4, 8, 16, 32, 64, 128], ["1", "2", "4", "8", "16", "32", "64", "128"])
        ax.minorticks_off()
        ax.yaxis.grid(True, color=GRIDC, linewidth=0.8)
        ax.axhline(0, color=INK2, linewidth=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(INK2)
        ax.tick_params(colors=INK2, labelcolor=INK2)
        ax.set_xlabel("spatial frequency (cycles / image)", color=INK)
        ax.set_title(title, color=INK, loc="left", fontsize=11, pad=18)
        ax.text(LATENT + 0.7, -0.07, "decoder band →", color=INK2, fontsize=8, ha="left")
    axes[0].set_ylabel("FRC (mean over images)", color=INK)
    handles = [Line2D([], [], color=RAMP[k], linewidth=2, label=f"scale {k} removed") for k in range(10)]
    fig.legend(handles=handles, loc="lower center", ncol=10, frameon=False, fontsize=8.5, labelcolor=INK,
               bbox_to_anchor=(0.5, -0.01), handlelength=1.6, columnspacing=1.0)
    fig.text(0.01, 0.075, "Dashed: n_k / 2, the highest frequency scale k's token grid can carry (number above = "
             "k; scale 0 at 0.5 is off the axis). Solid: r = 8.5, end of the latent band.", color=INK2, fontsize=8)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=SURFACE)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--crystal-dir", type=Path, default=Path("../crystal_64_p1"))
    ap.add_argument("--sanity-only", action="store_true")
    args = ap.parse_args(argv)
    sanity()
    if args.sanity_only:
        return
    curve_rows, band_rows = [], []

    def add(comp, ds, arm, scale, pairs):
        c, b = summarize(comp, ds, arm, scale, curves(pairs))
        curve_rows.extend(c)
        band_rows.append(b)

    for ds in PHASEF:
        arms = phasef_pairs(ds, args.crystal_dir)
        for j in range(10):
            add("phasef_vs_reference", ds, f"remove_s{j}", j, [(o, ref) for o, ref, _ in arms[f"remove_s{j}"]])
        add("phasef_vs_real", ds, "reference", "", [(o, real) for o, _, real in arms["reference"]])
        for j in range(10):
            add("phasef_vs_real", ds, f"remove_s{j}", j, [(o, real) for o, _, real in arms[f"remove_s{j}"]])
        _FFT.clear()
    for k, pairs in sorted(imagenet_pairs().items()):
        add("imagenet_vs_baseline", "imagenet", f"remove_s{k}", k, pairs)

    from lanes.p1.summarize import _write
    _write(OUT / "frc.csv", CURVE_COLUMNS, curve_rows)
    _write(OUT / "frc_bands.csv", BAND_COLUMNS, band_rows)
    plot(curve_rows, OUT / "fig1e_frc.png")
    print(f"wrote {OUT / 'frc.csv'} ({len(curve_rows)} rows), {OUT / 'frc_bands.csv'} ({len(band_rows)} rows), "
          f"{OUT / 'fig1e_frc.png'}")


if __name__ == "__main__":
    main()
