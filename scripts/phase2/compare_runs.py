"""Compare runner outputs for the phase2_check job. Loads no model and imports no upstream code.

    compare_runs.py regression --out F  NAME PT EXPECTED_SHA_PREFIX [...]
        tensor sha256 of PT must start with the prefix                         (exit 1 if not)
    compare_runs.py repeat     --out F  NAME DIR_A DIR_B [...]
        two manifest runs (runner.generate --out-dir): every row's hash equal  (exit 1 if not)
    compare_runs.py upstream   --out F  NAME RUNNER_DIR REF_DIR [...]
        runner run vs scripts/phase2/upstream_ref.py output: every row equal   (exit 1 if not)
    compare_runs.py batch      --out F [--ref bs16] [--pixel-threshold 1.0] [--contact-sheet PNG] NAME ROOT [...]
        ROOT/<tag>/ runs at several batch sizes; per row vs the --ref tag: max|diff| of the raw tensor,
        mean/max abs difference of the saved uint8 PNG. Rows whose mean abs pixel diff exceeds
        --pixel-threshold (grey levels, 0-255) count as visibly different. Measured only (exit 0).
        --contact-sheet: for the first NAME, a few rows side by side: <tag> | ref | |diff| x8.

Hashes use runner.generate.tensor_sha256 (dtype + shape + bytes; same as scripts/phase1/compare.py).
"""

import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from runner.generate import tensor_sha256  # noqa: E402

CONTACT_ROWS = 4
DIFF_GAIN = 8


def groups(items: list[str], n: int, ap) -> list[list[str]]:
    if not items or len(items) % n:
        ap.error(f"positional arguments must come in groups of {n}")
    return [items[i:i + n] for i in range(0, len(items), n)]


def run_rows(d: Path) -> dict[str, str]:
    """{stem: sha256} of a runner manifest run."""
    return {r["file"]: r["tensor_sha256"] for r in json.loads((d / "run.json").read_text())["rows"]}


def tensor_diff(a: Path, b: Path) -> float:
    return float((torch.load(a).double() - torch.load(b).double()).abs().max())


def pixel_diff(a: Path, b: Path) -> np.ndarray:
    return np.abs(np.asarray(Image.open(a).convert("RGB"), dtype=np.int16)
                  - np.asarray(Image.open(b).convert("RGB"), dtype=np.int16))


def cmd_regression(groups_, _args) -> tuple[dict, bool]:
    res, ok = {}, True
    for name, pt, prefix in groups_:
        try:
            sha = tensor_sha256(torch.load(pt))
        except OSError as e:
            res[name] = {"pt": pt, "error": str(e), "expected_prefix": prefix, "match": False}
            ok = False
            continue
        match = sha.startswith(prefix.lower())
        res[name] = {"pt": pt, "sha256": sha, "expected_prefix": prefix, "match": match}
        ok &= match
        print(f"{name}: {sha[:12]} expected {prefix}  {'MATCH' if match else 'MISMATCH'}")
    return res, ok


def paired(name: str, a: dict, b: dict, a_dir: Path, b_dir: Path) -> dict:
    rows = list(a)
    missing = [r for r in rows if r not in b]
    differing = [r for r in rows if r in b and a[r] != b[r]]
    rec = {"rows": len(rows), "identical_rows": len(rows) - len(differing) - len(missing),
           "differing": differing, "missing": missing}
    if differing:
        rec["max_abs_diff"] = max(tensor_diff(a_dir / f"{r}.pt", b_dir / f"{r}.pt") for r in differing)
    print(f"{name}: {rec['identical_rows']}/{rec['rows']} rows identical"
          + (f", differing {differing} (max|diff| {rec['max_abs_diff']:.3e})" if differing else "")
          + (f", missing {missing}" if missing else ""))
    return rec


def cmd_repeat(groups_, _args) -> tuple[dict, bool]:
    res, ok = {}, True
    for name, a, b in groups_:
        a, b = Path(a), Path(b)
        try:
            res[name] = paired(name, run_rows(a), run_rows(b), a, b)
        except OSError as e:
            res[name] = {"error": str(e)}
        ok &= res[name].get("identical_rows", -1) == res[name].get("rows")
    return res, ok


def cmd_upstream(groups_, _args) -> tuple[dict, bool]:
    res, ok = {}, True
    for name, runner_dir, ref_dir in groups_:
        runner_dir, ref_dir = Path(runner_dir), Path(ref_dir)
        try:
            res[name] = paired(name, run_rows(runner_dir), json.loads((ref_dir / "hashes.json").read_text()),
                               runner_dir, ref_dir)
        except OSError as e:
            res[name] = {"error": str(e)}
        ok &= res[name].get("identical_rows", -1) == res[name].get("rows")
    return res, ok


def cmd_batch(groups_, args) -> tuple[dict, bool]:
    res, ok = {}, True
    for gi, (name, root) in enumerate(groups_):
        root = Path(root)
        tags = sorted((p.name for p in root.iterdir() if (p / "run.json").exists() and p.name != args.ref),
                      key=lambda t: (not t.startswith("bs"), int(t[2:]) if t[2:].isdigit() else 0, t))
        if not (root / args.ref / "run.json").exists():
            res[name] = {"error": f"missing {root / args.ref / 'run.json'}"}
            ok = False
            continue
        ref_rows = run_rows(root / args.ref)
        rec = {"ref": args.ref, "pixel_threshold": args.pixel_threshold, "tags": {}}
        for tag in tags:
            rows = run_rows(root / tag)
            per_row = {}
            for stem in ref_rows:
                if stem not in rows:
                    per_row[stem] = {"missing": True}
                    continue
                pd = pixel_diff(root / tag / f"{stem}.png", root / args.ref / f"{stem}.png")
                per_row[stem] = {
                    "identical": rows[stem] == ref_rows[stem],
                    "max_abs_diff": tensor_diff(root / tag / f"{stem}.pt", root / args.ref / f"{stem}.pt"),
                    "mean_pixel_diff": float(pd.mean()),
                    "max_pixel_diff": int(pd.max()),
                }
            present = [v for v in per_row.values() if not v.get("missing")]
            t = rec["tags"][tag] = {
                "rows": per_row,
                "identical_rows": sum(v["identical"] for v in present),
                "max_abs_diff": max((v["max_abs_diff"] for v in present), default=None),
                "visible_rows": [s for s, v in per_row.items()
                                 if not v.get("missing") and v["mean_pixel_diff"] > args.pixel_threshold],
                "missing": [s for s, v in per_row.items() if v.get("missing")],
            }
            ok &= not t["missing"]
            print(f"{name} {tag} vs {args.ref}: {t['identical_rows']}/{len(per_row)} identical, "
                  f"max|diff| {t['max_abs_diff']}, visibly different (mean |pixel diff| > {args.pixel_threshold}): "
                  f"{len(t['visible_rows'])}")
        res[name] = rec
        if args.contact_sheet and gi == 0 and "bs1" in rec["tags"]:
            res[name]["contact_sheet"] = contact_sheet(root, "bs1", args.ref, rec["tags"]["bs1"]["rows"],
                                                       args.contact_sheet)
    return res, ok


def contact_sheet(root: Path, tag: str, ref: str, per_row: dict, out: Path) -> dict:
    """Rows with the largest mean pixel difference (manifest order breaks ties): tag | ref | |diff| x gain."""
    stems = sorted((s for s, v in per_row.items() if not v.get("missing")),
                   key=lambda s: -per_row[s]["mean_pixel_diff"])[:CONTACT_ROWS]
    label_h, tiles = 14, []
    for s in stems:
        a = Image.open(root / tag / f"{s}.png").convert("RGB")
        b = Image.open(root / ref / f"{s}.png").convert("RGB")
        d = np.clip(pixel_diff(root / tag / f"{s}.png", root / ref / f"{s}.png") * DIFF_GAIN, 0, 255).astype(np.uint8)
        tiles.append((s, [a, b, Image.fromarray(d)]))
    w, h = tiles[0][1][0].size
    sheet = Image.new("RGB", (3 * w, len(tiles) * (h + label_h) + label_h), "white")
    draw = ImageDraw.Draw(sheet)
    for ci, col in enumerate((tag, ref, f"|diff| x{DIFF_GAIN}")):
        draw.text((ci * w + 4, 1), col, fill="black")
    for ri, (s, imgs) in enumerate(tiles):
        y = label_h + ri * (h + label_h)
        v = per_row[s]
        draw.text((4, y + 1), f"{s}  mean|pixel diff|={v['mean_pixel_diff']:.3f}  max={v['max_pixel_diff']}", fill="black")
        for ci, im in enumerate(imgs):
            sheet.paste(im, (ci * w, y + label_h))
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    print(f"contact sheet: {out}")
    return {"path": str(out), "rows": stems, "columns": [tag, ref, f"|diff| x{DIFF_GAIN}"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("regression", "repeat", "upstream", "batch"))
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--ref", default="bs16", help="batch: reference tag")
    ap.add_argument("--pixel-threshold", type=float, default=1.0, help="batch: grey levels (0-255)")
    ap.add_argument("--contact-sheet", type=Path, help="batch: PNG path")
    ap.add_argument("items", nargs="+")
    args = ap.parse_args()
    n = {"regression": 3, "repeat": 3, "upstream": 3, "batch": 2}[args.mode]
    fn = {"regression": cmd_regression, "repeat": cmd_repeat, "upstream": cmd_upstream, "batch": cmd_batch}[args.mode]
    res, ok = fn(groups(args.items, n, ap), args)
    args.out.write_text(json.dumps(res, indent=2))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
