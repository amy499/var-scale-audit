"""P1 results: one row per arm of one or more gateb.pbs jobs, with paired drops vs the job's baseline.

Run from the repo root. numpy + csv only; no GPU.

    python -m lanes.p1.summarize outputs/p1/25893211 outputs/p1/25893212 ... --out lanes/p1/results/x.csv
        (a bare job id is looked up as outputs/p1/<id>)
    python -m lanes.p1.summarize --compare 25893213:remove_mid,remove_late --compare ... --compare-out y.csv
        (both forms can be given in one call)

Per arm (--out), over the final-image rows of metrics.csv (stage empty):
    n                       images
    cls_top1 .. img_mad255  means (img_mad255 is distance from the baseline image, so empty for a baseline)
    drop_cls_top1, drop_cls_prob
                            paired drop, mean over images of baseline - arm (positive = worse than baseline),
                            each with a 95% CI (_lo, _hi): paired bootstrap over (class_id, seed) rows,
                            10000 resamples, numpy default_rng(0), percentile interval
    kid_vs_base             lanes.p2.analyze.kid(arm, baseline, paired=True) on inception_features.npz;
                            empty if either run has no features file

Per pair (--compare JOB:ARM_A,ARM_B, repeatable; --compare-out): two arms of the same job, over the same
(class_id, seed) rows. diff_cls_top1, diff_cls_prob = mean over images of ARM_A - ARM_B (positive = A
scores higher), with the same paired bootstrap 95% CI, and each arm's mean.

Ranking (--ranking-out, with --out): read back from the --out CSV just written, every removal arm per
model, ordered by drop_cls_top1 (largest first; ties by drop_cls_prob, then job and arm), with what it
removed (VAR scales, DiT step indices and timesteps; from runs.csv params), how many scales or steps, and
for VAR the tokens removed (sum of n_tokens over the removed scales, from the run's run.json stages table).

Arms come from JOB/tables/runs.csv (written by `schema collect` at the end of every job). Metrics are
read from RUN_DIR/metrics.csv, or, when the run directory was not copied, from JOB/tables/metrics.csv
(the same rows, merged by collect).
"""

import argparse
import csv
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True

import numpy as np  # noqa: E402

from lanes.p2.analyze import kid  # noqa: E402

MEANS = ("cls_top1", "cls_top5", "cls_prob", "img_mad255")
DROPS = ("cls_top1", "cls_prob")
N_BOOT, SEED = 10000, 0
COLUMNS = ("job", "model", "arm", "n", *MEANS,
           *(f"drop_{m}{s}" for m in DROPS for s in ("", "_lo", "_hi")), "kid_vs_base")
COMPARE_COLUMNS = ("job", "model", "arm_a", "arm_b", "n",
                   *(c for m in DROPS for c in (f"{m}_a", f"{m}_b", f"diff_{m}", f"diff_{m}_lo", f"diff_{m}_hi")))
RANKING_COLUMNS = ("model", "rank", "job", "arm", "removed", "n_removed", "unit", "tokens_removed", "cls_top1",
                   *(f"drop_{m}{s}" for m in DROPS for s in ("", "_lo", "_hi")), "kid_vs_base")


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def final_metrics(rows) -> dict:
    """{metric: {(class_id, seed): value}} over final-image rows (stage empty)."""
    out = {}
    for r in rows:
        if r["stage"] == "":
            out.setdefault(r["metric"], {})[(int(r["class_id"]), int(r["seed"]))] = float(r["value"])
    return out


def features(run_dir: Path):
    """{(class_id, seed): feature vector}, or None without inception_features.npz."""
    path = run_dir / "inception_features.npz"
    if not path.is_file():
        return None
    z = np.load(path)
    return {(int(c), int(s)): f for c, s, f in zip(z["class_id"], z["seed"], z["feat"])}


def paired_diff(x: dict, y: dict, keys: list) -> tuple:
    """Mean over keys of x - y, and its 95% CI: paired bootstrap over keys (N_BOOT resamples, seed SEED)."""
    d = np.array([x[k] - y[k] for k in keys])
    idx = np.random.default_rng(SEED).integers(0, len(keys), size=(N_BOOT, len(keys)))
    boot = d[idx].mean(axis=1)
    return float(d.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def job_dir(j) -> Path:
    job = Path(j) if Path(j).is_dir() else Path("outputs/p1") / str(j)
    if not job.is_dir():
        raise SystemExit(f"ERROR: job directory not found: {j}")
    return job


def load_job(job: Path) -> tuple:
    """(runs.csv rows, {run_id: final metrics}, {run_id: features or None})."""
    tables = job / "tables"
    if not (tables / "runs.csv").is_file():
        raise SystemExit(f"ERROR: {tables / 'runs.csv'} not found: run `python -m lanes.p2.schema collect {job}`")
    runs = _read_csv(tables / "runs.csv")
    merged = None
    metrics, feats = {}, {}
    for r in runs:
        run_dir = job / r["run_dir"]
        if (run_dir / "metrics.csv").is_file():
            metrics[r["run_id"]] = final_metrics(_read_csv(run_dir / "metrics.csv"))
        else:
            if merged is None:
                merged = _read_csv(tables / "metrics.csv")
            metrics[r["run_id"]] = final_metrics(m for m in merged if m["run_id"] == r["run_id"])
        feats[r["run_id"]] = features(run_dir)
    return runs, metrics, feats


def _same_rows(a: dict, b: dict, what: str) -> list:
    keys = sorted(a.get("cls_top1", {}))
    if sorted(b.get("cls_top1", {})) != keys:
        raise SystemExit(f"ERROR: {what}: the two runs have different images; not a paired comparison")
    return keys


def summarize_job(job: Path) -> list[dict]:
    runs, metrics, feats = load_job(job)
    out = []
    for r in runs:
        rid, base_id = r["run_id"], r["baseline_run_id"] or None
        m = metrics[rid]
        keys = sorted(m.get("cls_top1", {}))
        row = {"job": job.name, "model": r["model"], "arm": r["arm"], "n": len(keys)}
        for name in MEANS:
            vals = m.get(name, {})
            row[name] = round(float(np.mean([vals[k] for k in keys])), 4) if vals and keys else ""
        if base_id is not None:
            b = metrics[base_id]
            _same_rows(b, m, f"{rid} vs its baseline {base_id}")
            for name in DROPS:
                for s, v in zip(("", "_lo", "_hi"), paired_diff(b[name], m[name], keys)):
                    row[f"drop_{name}{s}"] = round(v, 4)
            fa, fb = feats[rid], feats[base_id]
            if fa is not None and fb is not None:
                row["kid_vs_base"] = round(kid(np.stack([fa[k] for k in keys]), np.stack([fb[k] for k in keys]),
                                               paired=True), 5) + 0.0   # + 0.0: no "-0.0"
        out.append(row)
    return out


def compare(spec: str) -> dict:
    """One --compare JOB:ARM_A,ARM_B: ARM_A - ARM_B per image, over the same (class_id, seed) rows."""
    j, sep, arms = spec.rpartition(":")
    a, _, b = arms.partition(",")
    if not sep or not j or not a or not b or "," in b:
        raise SystemExit(f"ERROR: --compare {spec!r}: expected JOB:ARM_A,ARM_B")
    job = job_dir(j)
    runs, metrics, _ = load_job(job)
    by_arm = {r["arm"]: r for r in runs}
    for arm in (a, b):
        if arm not in by_arm:
            raise SystemExit(f"ERROR: --compare {spec!r}: no arm {arm!r} in {job} (arms: {', '.join(by_arm)})")
    if by_arm[a]["model"] != by_arm[b]["model"]:
        raise SystemExit(f"ERROR: --compare {spec!r}: {a} and {b} are different models")
    ma, mb = metrics[by_arm[a]["run_id"]], metrics[by_arm[b]["run_id"]]
    keys = _same_rows(ma, mb, f"--compare {spec}")
    row = {"job": job.name, "model": by_arm[a]["model"], "arm_a": a, "arm_b": b, "n": len(keys)}
    for name in DROPS:
        row[f"{name}_a"] = round(float(np.mean([ma[name][k] for k in keys])), 4)
        row[f"{name}_b"] = round(float(np.mean([mb[name][k] for k in keys])), 4)
        for s, v in zip(("", "_lo", "_hi"), paired_diff(ma[name], mb[name], keys)):
            row[f"diff_{name}{s}"] = round(v, 4)
    return row


def _removal(job: Path, arm: str) -> dict:
    """What one arm removed: removed, n_removed, unit, tokens_removed (VAR only)."""
    runs = {r["arm"]: r for r in _read_csv(job / "tables" / "runs.csv")}
    if arm not in runs:
        raise SystemExit(f"ERROR: no arm {arm!r} in {job}/tables/runs.csv")
    r = runs[arm]
    p = json.loads(r["params"] or "{}")
    if "scales" in p:
        stages = json.loads((job / r["run_dir"] / "run.json").read_text())["stages"]
        tokens = {s["stage"]: s["n_tokens"] for s in stages}
        return {"removed": "{" + ",".join(map(str, p["scales"])) + "}", "n_removed": len(p["scales"]),
                "unit": "scales", "tokens_removed": sum(tokens[s] for s in p["scales"])}
    if "step_first" in p:
        return {"removed": f"steps {p['step_first']}-{p['step_last']} (t {p['timestep_first']}-{p['timestep_last']})",
                "n_removed": p["n_removed"], "unit": "steps", "tokens_removed": ""}
    raise SystemExit(f"ERROR: {job} arm {arm}: params do not say what was removed: {r['params']}")


def ranking(results_csv: Path) -> list[dict]:
    """Every removal arm of results_csv (an --out CSV), per model, by drop in top-1, largest first."""
    rows = [r for r in _read_csv(results_csv) if r["drop_cls_top1"] != ""]
    out = []
    for model in sorted({r["model"] for r in rows}, key=lambda m: (m != "var", m)):
        ranked = sorted((r for r in rows if r["model"] == model),
                        key=lambda r: (-float(r["drop_cls_top1"]), -float(r["drop_cls_prob"]), r["job"], r["arm"]))
        for i, r in enumerate(ranked, 1):
            out.append({"model": model, "rank": i, "job": r["job"], "arm": r["arm"],
                        **_removal(job_dir(r["job"]), r["arm"]), "cls_top1": r["cls_top1"],
                        **{c: r[c] for c in RANKING_COLUMNS[9:]}})
    return out


def _write(path: Path, columns, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns, restval="")
        w.writeheader()
        w.writerows(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("jobs", nargs="*", help="job directories (outputs/p1/<id>) or bare job ids")
    ap.add_argument("--out", type=Path, help="per-arm CSV to write (needs jobs)")
    ap.add_argument("--compare", action="append", default=[], metavar="JOB:ARM_A,ARM_B",
                    help="paired difference ARM_A - ARM_B within one job (repeatable; needs --compare-out)")
    ap.add_argument("--compare-out", type=Path, help="CSV for the --compare rows")
    ap.add_argument("--ranking-out", type=Path, help="CSV ranking the removal arms of --out (needs --out)")
    args = ap.parse_args(argv)
    if bool(args.jobs) != bool(args.out):
        ap.error("give job directories together with --out")
    if args.ranking_out and not args.out:
        ap.error("--ranking-out ranks the --out CSV: give jobs and --out too")
    if bool(args.compare) != bool(args.compare_out):
        ap.error("give --compare together with --compare-out")
    if not args.out and not args.compare:
        ap.error("nothing to do: give jobs with --out, and/or --compare with --compare-out")
    if args.out:
        rows = [row for j in args.jobs for row in summarize_job(job_dir(j))]
        _write(args.out, COLUMNS, rows)
        print(f"wrote {len(rows)} arms to {args.out}")
        if args.ranking_out:
            ranked = ranking(args.out)
            _write(args.ranking_out, RANKING_COLUMNS, ranked)
            print(f"wrote {len(ranked)} ranked removal arms to {args.ranking_out}")
    if args.compare:
        rows = [compare(spec) for spec in args.compare]
        _write(args.compare_out, COMPARE_COLUMNS, rows)
        print(f"wrote {len(rows)} comparisons to {args.compare_out}")


if __name__ == "__main__":
    main()
