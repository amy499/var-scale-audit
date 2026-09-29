"""Where in VAR's scale sequence do two runs (e.g. batch 1 vs 16) first sample different tokens?

    tokens.py config  --config configs/var_d20.yaml --autocast none [--tf32 false] --out DIR/config.yaml
        Copy of a config with a different precision (checkpoint paths made absolute).
    tokens.py analyze --name fp16 --a RUN_BS1 --b RUN_BS16 [--phase2-batch batch.json] --out F
        Both dirs are `runner.generate --manifest ... --capture-tokens` runs. Per row: first scale with
        any differing token, fraction of differing tokens per scale, mean |pixel diff| of the PNGs.
        Across rows: how many rows first diverge at each scale; Spearman correlation between the
        first-diverging scale and the mean pixel difference (own PNGs, and Phase 2 batch.json if given);
        for the first flipped token of up to FLIP_ROWS rows, the top-2 probabilities in both runs.
    tokens.py plot    --out PNG NAME=F [NAME=F ...]
        Divergence per scale for each analyze output.

Token files hold, per scale: idx (l,), top2_p / top2_i (l, 2), p_chosen (l,). Probabilities are the
model's post-CFG softmax before top-k/top-p filtering (see runner/var_model.py).
"""

import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402

REPO = Path(__file__).resolve().parents[2]

import numpy as np  # noqa: E402

FLIP_ROWS = 6


def cmd_config(args):
    import yaml
    cfg = yaml.safe_load(args.config.read_text())
    cfg["precision"]["autocast_dtype"] = None if args.autocast == "none" else args.autocast
    if args.tf32 is not None:
        cfg["precision"]["tf32"] = args.tf32 == "true"
    for c in cfg["checkpoints"].values():
        p = Path(c["path"])
        c["path"] = str(p if p.is_absolute() else REPO / p)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(f"wrote {args.out}: precision={cfg['precision']}")


def spearman(x: list[float], y: list[float]) -> float | None:
    def ranks(v):
        v = np.asarray(v, dtype=float)
        order = v.argsort(kind="stable")
        r = np.empty(len(v))
        r[order] = np.arange(len(v))
        for val in np.unique(v):   # average ranks for ties
            m = v == val
            r[m] = r[m].mean()
        return r
    if len(x) < 3:
        return None
    rx, ry = ranks(x), ranks(y)
    if rx.std() == 0 or ry.std() == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def mean_pixel_diff(a: Path, b: Path) -> float:
    from PIL import Image
    pa = np.asarray(Image.open(a).convert("RGB"), dtype=np.int16)
    pb = np.asarray(Image.open(b).convert("RGB"), dtype=np.int16)
    return float(np.abs(pa - pb).mean())


def flip_record(ta: dict, tb: dict, s: int, pn: int) -> dict:
    """The first flipped token at scale s (lowest position), with both runs' top-2 probabilities."""
    sa, sb = ta["scales"][s], tb["scales"][s]
    pos = int((sa["idx"] != sb["idx"]).nonzero()[0, 0])
    gap_a = (sa["top2_p"][:, 0] - sa["top2_p"][:, 1]).numpy()
    rec = {"scale": s, "pn": pn, "position": pos, "row_col": list(divmod(pos, pn))}
    for tag, sc in (("a", sa), ("b", sb)):
        rec[tag] = {"token": int(sc["idx"][pos]), "p_chosen": float(sc["p_chosen"][pos]),
                    "top2_p": [float(v) for v in sc["top2_p"][pos]], "top2_i": [int(v) for v in sc["top2_i"][pos]],
                    "chosen_is_top1": int(sc["idx"][pos]) == int(sc["top2_i"][pos, 0])}
    # How close to a tie is this position, relative to all positions at this scale (run a)?
    rec["gap_top1_top2_a"] = float(gap_a[pos])
    rec["gap_percentile_at_scale_a"] = float((gap_a <= gap_a[pos]).mean())
    rec["median_gap_at_scale_a"] = float(np.median(gap_a))
    return rec


def cmd_analyze(args):
    import torch
    run_a = json.loads((args.a / "run.json").read_text())
    run_b = json.loads((args.b / "run.json").read_text())
    hash_b = {r["file"]: r["tensor_sha256"] for r in run_b["rows"]}
    stems = [r["file"] for r in run_a["rows"]]
    phase2 = None
    if args.phase2_batch and args.phase2_batch.exists():
        rows = json.loads(args.phase2_batch.read_text()).get("var", {}).get("tags", {}).get("bs1", {}).get("rows")
        if rows:
            phase2 = {s: v["mean_pixel_diff"] for s, v in rows.items() if not v.get("missing")}

    per_row, patch_nums = {}, None
    flips = []
    for r in run_a["rows"]:
        stem = r["file"]
        ta = torch.load(args.a / f"{stem}.tokens.pt")
        tb = torch.load(args.b / f"{stem}.tokens.pt")
        patch_nums = ta["patch_nums"]
        fracs = [float((sa["idx"] != sb["idx"]).float().mean()) for sa, sb in zip(ta["scales"], tb["scales"])]
        first = next((s for s, f in enumerate(fracs) if f > 0), None)
        per_row[stem] = {
            "first_scale": first, "first_pn": None if first is None else patch_nums[first],
            "frac_differing": fracs, "tensor_identical": r["tensor_sha256"] == hash_b.get(stem),
            "mean_pixel_diff": mean_pixel_diff(args.a / f"{stem}.png", args.b / f"{stem}.png"),
            "phase2_mean_pixel_diff": None if phase2 is None else phase2.get(stem),
        }
        if first is not None and len(flips) < FLIP_ROWS:
            flips.append({"row": stem, **flip_record(ta, tb, first, patch_nums[first])})

    n_scales = len(patch_nums)
    first_counts = [sum(v["first_scale"] == s for v in per_row.values()) for s in range(n_scales)]
    never = sum(v["first_scale"] is None for v in per_row.values())
    # Rows that never diverge are ranked as scale n_scales (after the last one).
    x = [n_scales if v["first_scale"] is None else v["first_scale"] for v in per_row.values()]
    corr = {"own_png": spearman(x, [v["mean_pixel_diff"] for v in per_row.values()])}
    if phase2 is not None:
        keep = [(xi, v["phase2_mean_pixel_diff"]) for xi, v in zip(x, per_row.values())
                if v["phase2_mean_pixel_diff"] is not None]
        corr["phase2_batch_json"] = spearman([k[0] for k in keep], [k[1] for k in keep])
    res = {
        "name": args.name, "a": str(args.a), "b": str(args.b), "patch_nums": patch_nums,
        "precision_a": run_a["precision"], "precision_b": run_b["precision"],
        "batch_sizes": [run_a["batch_size"], run_b["batch_size"]],
        "rows": per_row,
        "per_scale": {
            "first_divergence_rows": first_counts, "never_diverge_rows": never,
            "rows_with_any_diff": [sum(v["frac_differing"][s] > 0 for v in per_row.values()) for s in range(n_scales)],
            "mean_frac_differing": [float(np.mean([v["frac_differing"][s] for v in per_row.values()]))
                                    for s in range(n_scales)],
        },
        "spearman_first_scale_vs_pixel_diff": corr,
        "spearman_note": f"rows that never diverge ranked as scale {n_scales}; negative = earlier divergence, larger pixel diff",
        "first_flips": flips,
    }
    args.out.write_text(json.dumps(res, indent=2))
    print(f"{args.name}: first divergence per scale {first_counts}, never {never}; spearman {corr}")


def cmd_plot(args):
    from PIL import Image, ImageDraw
    series = []
    for item in args.inputs:
        name, path = item.split("=", 1)
        try:
            series.append((name, json.loads(Path(path).read_text())))
        except (OSError, ValueError):
            print(f"skip {name}: cannot read {path}")
    if not series:
        raise SystemExit("nothing to plot")
    colors = [(31, 119, 180), (214, 39, 40), (44, 160, 44), (148, 103, 189)]
    pns = series[0][1]["patch_nums"]
    labels = [str(p) for p in pns] + ["none"]
    W, H, L, R, T, gap = 900, 780, 70, 20, 60, 60
    ph = (H - T - 2 * gap) // 2
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)

    def panel(y0, title, ymax, values_per_series, bars):
        x0, x1, y1 = L, W - R, y0 + ph
        d.text((x0, y0 - 18), title, fill="black")
        d.line([(x0, y0), (x0, y1), (x1, y1)], fill="black")
        n = len(values_per_series[0])
        step = (x1 - x0) / n
        for k in range(5):   # y grid + labels
            v = ymax * k / 4
            y = y1 - (y1 - y0) * k / 4
            d.line([(x0, y), (x1, y)], fill=(225, 225, 225))
            d.text((4, y - 6), f"{v:.3g}" if ymax <= 1 else f"{v:.0f}", fill="black")
        for i in range(n):
            d.text((x0 + step * (i + 0.5) - 8, y1 + 4), labels[i] if i < len(labels) else "", fill="black")
        for si, vals in enumerate(values_per_series):
            c = colors[si % len(colors)]
            pts = []
            for i, v in enumerate(vals):
                y = y1 - (y1 - y0) * (v / ymax if ymax else 0)
                if bars:
                    bw = step * 0.8 / len(values_per_series)
                    bx = x0 + step * i + step * 0.1 + bw * si
                    d.rectangle([bx, y, bx + bw - 1, y1], fill=c)
                else:
                    pts.append((x0 + step * (i + 0.5), y))
            if pts:
                d.line(pts, fill=c, width=2)
                for px, py in pts:
                    d.ellipse([px - 3, py - 3, px + 3, py + 3], fill=c)
        d.text((x0 + (x1 - x0) // 2 - 80, y1 + 20), "scale (patch side length)", fill="black")

    counts = [s["per_scale"]["first_divergence_rows"] + [s["per_scale"]["never_diverge_rows"]] for _, s in series]
    fracs = [s["per_scale"]["mean_frac_differing"] for _, s in series]
    n_rows = len(series[0][1]["rows"])
    panel(T, f"rows whose FIRST differing token is at this scale (of {n_rows}; 'none' = never differ)",
          max(1, max(max(c) for c in counts)), counts, bars=True)
    panel(T + ph + gap, "mean fraction of tokens that differ at this scale",
          max(1e-9, max(max(f) for f in fracs)) * 1.05, fracs, bars=False)
    for si, (name, s) in enumerate(series):   # legend, one row above the panels
        c = colors[si % len(colors)]
        lx, ly = L + 260 * si, 8
        d.rectangle([lx, ly, lx + 10, ly + 10], fill=c)
        d.text((lx + 16, ly - 1), f"{name} (batch {s['batch_sizes'][0]} vs {s['batch_sizes'][1]})", fill="black")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    img.save(args.out)
    print(f"wrote {args.out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)
    c = sub.add_parser("config")
    c.add_argument("--config", required=True, type=Path)
    c.add_argument("--autocast", required=True, choices=("none", "float16", "bfloat16"))
    c.add_argument("--tf32", choices=("true", "false"))
    c.add_argument("--out", required=True, type=Path)
    a = sub.add_parser("analyze")
    a.add_argument("--name", required=True)
    a.add_argument("--a", required=True, type=Path)
    a.add_argument("--b", required=True, type=Path)
    a.add_argument("--phase2-batch", type=Path)
    a.add_argument("--out", required=True, type=Path)
    p = sub.add_parser("plot")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("inputs", nargs="+", metavar="NAME=JSON")
    args = ap.parse_args()
    {"config": cmd_config, "analyze": cmd_analyze, "plot": cmd_plot}[args.mode](args)


if __name__ == "__main__":
    main()
