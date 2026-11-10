"""Phase D synthesis: a dated snapshot of the other lanes' results, read-only adapters, and Figure 5.

Run from the repo root.

    python -m lanes.p4.synthesis snapshot --out outputs/p4/handoff/2026-11-03
    python -m lanes.p4.synthesis figure5 outputs/p4/handoff/2026-11-03 --out-dir outputs/p4/summary/<date> \
        [--p4-summary var=VAR_SUMMARY.csv --p4-summary dit=DIT_SUMMARY.csv]
    python -m lanes.p4.synthesis scores  outputs/p4/handoff/2026-11-03 --out scores.csv

**snapshot** copies named files from named git refs (`git show <ref>:<path>`), never from a working
tree, into one dated folder, and writes `SOURCES.json`: the ref, the commit it resolved to, the path,
and each copy's sha256. A file the ref does not have is recorded as missing, not fatal. Every adapter
reads only the snapshot, so a number in the report traces to a commit. P1 and P2 are read from
`origin/main`. P3 is read from `origin/p3/dit-and-cross-lane` (`lanes/p3/results/`) until it merges.

**Adapters** turn each lane into one importance number per native stage, **higher = more important**:

    P1  cls_prob drop when that one stage is removed (results_sweep.csv drop_cls_prob, remove_sK).
        DiT has band removals only, so each P4 band's timesteps carry P1's value for that band
        (P1's DiT bands are steps 1-83 / 84-166 / 167-249, P4's are 83/83/84 steps from step 0).
    P2  cls_prob_base - cls_prob at severity 1.0 (p2_summary.csv), as P3's RANKING_SPEC.md asks.
        Duplicate rows (re-runs) are averaged. Each row's `p` must equal P4's p_place for its stage.
    P3  per-stage contribution of the cumulative curves (rel_dist: delta, inverted; identity_progress:
        delta), through lanes.p3.agreement.load_rank, as P3's own agreement note prescribes.
    P4  -(damage - control) on cls_prob at the band's largest budget, from lanes/p4/export.py summary:
        how much worse damaging the band is than spending the same budget anywhere. Band-level, so each
        band's stages carry its value. P4 is drawn in Figure 5 but never used by the shortlist.

**Figure 5** is one figure per model (`figure5_var.png`, `figure5_dit.png`), one panel per lane, each
normalized within itself (lanes/p4/figure5.py). Lanes are not averaged in the figure. Absent lanes are
named in the title.
"""

import argparse
import csv
import datetime
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

sys.dont_write_bytecode = True

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from lanes.p4 import bands, fixtures, plotting, progress  # noqa: E402

MAIN_REF = "origin/main"
P3_REF = "origin/p3/dit-and-cross-lane"     # P3's tables are not on main yet; switch to MAIN_REF once merged
P3_DIR = "lanes/p3/results"
# (lane, ref, path in the repo). The snapshot copies each to <lane>/<file name>.
SOURCES = (
    ("p1", MAIN_REF, "lanes/p1/results/results_sweep.csv"),
    ("p1", MAIN_REF, "lanes/p1/results/ranking.csv"),
    ("p2", MAIN_REF, "lanes/p2/results/phasec/p2_summary.csv"),
    ("p2", MAIN_REF, "lanes/p2/results/phasec/recovery_ranking.csv"),
    ("p2", MAIN_REF, "lanes/p2/results/phasec/p2_curves.csv"),
    *[("p3", P3_REF, f"{P3_DIR}/{model}_{kind}.csv") for model in progress.MODELS
      for kind in ("curves", "semantic")],
)
P2_SEVERITY = 1.0
P_TOLERANCE = 1e-3
LANE_LABELS = {"p1": "P1 necessity (removal)", "p2": "P2 corruption (sev 1.0)",
               "p3_energy": "P3 energy contribution", "p3_identity": "P3 identity contribution",
               "p4": "P4 damage - control (largest m)"}
SHORTLIST_LANES = ("p1", "p2", "p3_energy", "p3_identity")    # never p4: no lane selects its own test
P3_SPECS = {"p3_energy": ("curves", "rel_dist", "delta,invert"),
            "p3_identity": ("semantic", "identity_progress", "delta")}


class SynthesisError(ValueError):
    """A snapshot or lane file that cannot be read honestly, with the reason named."""


# ---------------------------------------------------------------- snapshot

def _git(*args) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True)


def snapshot(out_dir, sources=SOURCES) -> dict:
    """Copy every source into out_dir/<lane>/<name> and write SOURCES.json. Returns the record."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    record = {"taken": datetime.datetime.now().astimezone().isoformat(timespec="seconds"), "files": []}
    for lane, ref, path in sources:
        commit = _git("rev-parse", "--verify", f"{ref}^{{commit}}")
        entry = {"lane": lane, "ref": ref, "path": path,
                 "commit": commit.stdout.decode().strip() if commit.returncode == 0 else None}
        shown = _git("show", f"{ref}:{path}") if commit.returncode == 0 else None
        if shown is None or shown.returncode != 0:
            entry.update({"missing": True, "copy": None, "sha256": None,
                          "reason": "ref not found" if shown is None else "path not on ref"})
        else:
            dest = out_dir / lane / Path(path).name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(shown.stdout)
            entry.update({"missing": False, "copy": dest.relative_to(out_dir).as_posix(),
                          "sha256": hashlib.sha256(shown.stdout).hexdigest()})
        record["files"].append(entry)
    (out_dir / "SOURCES.json").write_text(json.dumps(record, indent=2))
    return record


def _snap_file(snap_dir: Path, lane: str, name: str) -> Path | None:
    path = Path(snap_dir) / lane / name
    return path if path.is_file() else None


def _read(path: Path) -> list[dict]:
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------- the frozen stage tables

def stage_table(model: str) -> list[dict]:
    """Frozen stages with p_place and band, in sampling order."""
    return bands.assign_bands(progress.map_stages(model, fixtures.frozen_stages(model)))


def band_stages(model: str) -> dict:
    out = {band: [] for band in bands.BANDS}
    for row in stage_table(model):
        out[row["band"]].append(row["stage"])
    return out


# ---------------------------------------------------------------- adapters

def p1_scores(snap_dir) -> dict:
    """{model: {stage: drop}}. VAR: single-scale removals. DiT: band removals spread over P4's bands."""
    path = _snap_file(snap_dir, "p1", "results_sweep.csv")
    if path is None:
        return {}
    out = {"var": {}, "dit": {}}
    dit_bands = {}
    for row in _read(path):
        arm, model = row["arm"], row["model"]
        if not row.get("drop_cls_prob"):
            continue
        single = re.fullmatch(r"remove_s(\d)", arm)
        if model == "var" and single:
            stage = int(single.group(1))
            if stage in out["var"]:
                raise SynthesisError(f"P1 has two rows for VAR {arm}")
            out["var"][stage] = float(row["drop_cls_prob"])
        elif model == "dit" and arm in ("remove_early", "remove_mid", "remove_late"):
            dit_bands[{"remove_early": "early", "remove_mid": "middle", "remove_late": "late"}[arm]] = \
                float(row["drop_cls_prob"])
    if dit_bands:
        for band, stages in band_stages("dit").items():
            if band in dit_bands:
                for stage in stages:
                    out["dit"][stage] = dit_bands[band]
    return {m: s for m, s in out.items() if s}


def p1_band_removal(snap_dir, model: str, arm: str) -> float | None:
    """P1's cls_prob after one band removal, for the report's cross-check against P4's damage arm."""
    path = _snap_file(snap_dir, "p1", "results_sweep.csv")
    if path is None:
        return None
    hits = [float(r["cls_prob"]) for r in _read(path) if r["model"] == model and r["arm"] == arm]
    return hits[0] if hits else None


def p2_scores(snap_dir, severity: float = P2_SEVERITY) -> dict:
    """{model: {stage: cls_prob_base - cls_prob}} at one severity; duplicates averaged; p checked."""
    path = _snap_file(snap_dir, "p2", "p2_summary.csv")
    if path is None:
        return {}
    places = {m: {r["stage"]: r["p_place"] for r in stage_table(m)} for m in progress.MODELS}
    acc = {}
    for row in _read(path):
        if float(row["severity"]) != severity or not row.get("cls_prob"):
            continue
        model, stage = row["model"], int(row["stage"])
        if stage not in places[model]:
            raise SynthesisError(f"P2 {model} stage {stage} is not a stage of P4's frozen {model} table")
        if abs(float(row["p"]) - places[model][stage]) > P_TOLERANCE:
            raise SynthesisError(f"P2 {model} stage {stage} has p = {float(row['p']):.4f}, but P4's "
                                 f"p_place is {places[model][stage]:.4f}; the lanes would not share an axis")
        acc.setdefault(model, {}).setdefault(stage, []).append(
            float(row["cls_prob_base"]) - float(row["cls_prob"]))
    return {m: {s: sum(v) / len(v) for s, v in by.items()} for m, by in acc.items()}


def p3_scores(snap_dir) -> dict:
    """{lane: {model: {stage: contribution}}} for P3's energy and identity curves, when present."""
    from lanes.p3.agreement import load_rank
    out = {}
    for lane, (kind, column, opts) in P3_SPECS.items():
        for model in progress.MODELS:
            path = _snap_file(snap_dir, "p3", f"{model}_{kind}.csv")
            if path is None:
                continue
            _label, stages, values = load_rank(f"{lane}={path}:{column}:{opts}")
            out.setdefault(lane, {})[model] = {int(s): float(v) for s, v in zip(stages, values)}
    return out


def p4_scores(summary_csv, model: str, metric: str = "cls_prob") -> dict:
    """{stage: -(damage - control)} at each band's largest budget, spread over the band's stages."""
    rows = [r for r in _read(summary_csv) if r["model"] == model and r["arm"] == "damage"
            and r.get(f"{metric}_vs_control")]
    if not rows:
        raise SynthesisError(f"{summary_csv} has no {model} damage rows with a {metric} contrast")
    out = {}
    stages_of = band_stages(model)
    for band in bands.BANDS:
        here = [r for r in rows if r["band"] == band]
        if not here:
            continue
        top = max(here, key=lambda r: int(r["m"]))
        sign = 1.0 if plotting.P3_METRIC_SPECS[metric].direction == "higher_is_better" else -1.0
        value = -sign * float(top[f"{metric}_vs_control"])
        for stage in stages_of[band]:
            out[stage] = value
    return out


def lane_scores(snap_dir, p4_summaries: dict | None = None) -> dict:
    """{model: {lane: {stage: score}}} for every lane present; higher = more important."""
    out = {m: {} for m in progress.MODELS}
    for model, scores in p1_scores(snap_dir).items():
        out[model]["p1"] = scores
    for model, scores in p2_scores(snap_dir).items():
        out[model]["p2"] = scores
    for lane, by_model in p3_scores(snap_dir).items():
        for model, scores in by_model.items():
            out[model][lane] = scores
    for model, path in (p4_summaries or {}).items():
        out[model]["p4"] = p4_scores(path, model)
    return out


def absent_lanes(scores: dict, model: str) -> list[str]:
    have = set(scores.get(model, {}))
    missing = [lane for lane in ("p1", "p2") if lane not in have]
    if not have & set(P3_SPECS):
        missing.append("p3")
    return missing


# ---------------------------------------------------------------- Figure 5

def lane_tidy(model: str, lane: str, scores: dict) -> list[dict]:
    """A lane's per-stage scores as a tidy table on P4's frozen stage table."""
    record = fixtures.run_record(model, fixtures.frozen_stages(model))
    metric = f"{lane}_importance"
    rows = [{"metric": metric, "value": v, "stage": s} for s, v in sorted(scores.items())]
    spec = plotting.MetricSpec(metric, "higher_is_better", "per_set", f"{LANE_LABELS[lane]} (higher = more important)")
    return plotting.tidy_from_run(record, rows, [spec], arm=lane, lane=lane, run_id=f"{model}/{lane}")


def figure5_tables(scores: dict, model: str) -> list[tuple[str, list[dict]]]:
    order = ("p1", "p2", "p3_energy", "p3_identity", "p4")
    return [(LANE_LABELS[lane], lane_tidy(model, lane, scores[model][lane]))
            for lane in order if lane in scores.get(model, {})]


def write_figure5(scores: dict, out_dir, backend: str = "auto") -> dict:
    from lanes.p4 import figure5
    out_dir = Path(out_dir)
    written = {}
    for model in progress.MODELS:
        tables = figure5_tables(scores, model)
        if not tables:
            continue
        missing = absent_lanes(scores, model)
        title = (f"Figure 5 ({model.upper()}): stage importance across lanes"
                 + (f"  -  absent: {', '.join(m.upper() for m in missing)}" if missing else ""))
        for label, rows in tables:
            plotting.write_tidy(rows, out_dir / f"figure5_{model}_{rows[0]['lane']}_tidy.csv")
        # wider than figure5's default panels, so the three band labels with their ranges do not collide
        written[model] = figure5.figure5(tables, out_dir / f"figure5_{model}.png", title=title,
                                         backend=backend, size=(520 * len(tables) + 120, 500))
    return written


def write_scores(scores: dict, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(("model", "lane", "stage", "p_place", "band", "score"))
        for model in progress.MODELS:
            table = {r["stage"]: r for r in stage_table(model)}
            for lane, by_stage in sorted(scores.get(model, {}).items()):
                for stage, value in sorted(by_stage.items(), key=lambda kv: table[kv[0]]["p_place"]):
                    w.writerow((model, lane, stage, f"{table[stage]['p_place']:.6g}", table[stage]["band"],
                                f"{value:.6g}"))
    return path


def _p4_summaries(items) -> dict:
    out = {}
    for item in items or []:
        model, sep, path = item.partition("=")
        if not sep or model not in progress.MODELS:
            raise SystemExit(f"--p4-summary {item!r}: expected var=PATH or dit=PATH")
        out[model] = Path(path)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="action", required=True)
    p = sub.add_parser("snapshot")
    p.add_argument("--out", type=Path,
                   default=REPO / "outputs" / "p4" / "handoff" / datetime.date.today().isoformat())
    for name in ("figure5", "scores"):
        p = sub.add_parser(name)
        p.add_argument("snapshot", type=Path)
        p.add_argument("--p4-summary", action="append", metavar="MODEL=SUMMARY.csv")
        p.add_argument("--out-dir" if name == "figure5" else "--out", type=Path, required=True)
        if name == "figure5":
            p.add_argument("--backend", default="auto", choices=("auto", *plotting.BACKENDS))
    args = ap.parse_args(argv)

    if args.action == "snapshot":
        record = snapshot(args.out)
        for e in record["files"]:
            print(f"   {'MISSING' if e['missing'] else 'ok     '}  {e['lane']}  {e['ref']}:{e['path']}"
                  + (f"  ({e['reason']})" if e["missing"] else f"  @ {e['commit'][:10]}"))
        print(f"wrote {args.out / 'SOURCES.json'}")
        return 0
    scores = lane_scores(args.snapshot, _p4_summaries(args.p4_summary))
    if args.action == "scores":
        print(f"wrote {write_scores(scores, args.out)}")
        return 0
    for model, path in write_figure5(scores, args.out_dir, backend=args.backend).items():
        missing = absent_lanes(scores, model)
        print(f"wrote {path}" + (f"   absent: {', '.join(missing)}" if missing else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
