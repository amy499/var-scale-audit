"""P1 Phase E: the removal results on seed set 1 (manifest v1.1) vs seed set 2 (lanes/p1/manifests/seed2.csv).
numpy + csv only; no GPU.

    python -m lanes.p1.compare_seeds [--single JOB] [--groups JOB] [--dit JOB]
                                     [--results-out lanes/p1/results/seed2_results.csv]
                                     [--compare-out lanes/p1/results/seed_compare.csv]

--single / --groups / --dit are the seed-2 jobs (outputs/p1/<id> or a bare id) of seed2_var_single.env,
seed2_var_groups.env and seed2_dit.env; give any that have finished (at least one). Seed-1 jobs are fixed:
single scales 25893211 (s0, s9) + 25916920 (s1-s8), groups 25893212, DiT 25893213.

Writes:
  seed2_results.csv  every run of the given seed-2 jobs, same columns as results_sweep.csv
                     (lanes.p1.summarize.summarize_job).
  seed_compare.csv   one row per removal arm present in both seed sets, matched on what was removed (params
                     in runs.csv, not the arm name): for top-1 and cls_prob, the drop vs the same job's baseline
                     with its 95% paired bootstrap CI on each seed set (summarize.paired_diff: 10000 resamples,
                     seed 0), and diff = seed 2 - seed 1 with a 95% CI from resampling the two independent image
                     sets separately (10000 resamples, seed 0).
Prints: Spearman rho between the two seed sets' rankings of the 10 single VAR scales by cls_prob drop (when
all 10 are on both sides), the top-3 scales of each, and each seed set's DiT order (largest cls_prob drop
first). Stops if the two seed sets share an image (class_id, seed), or a seed set lacks 32 images per class.
Existing results files are never written.
"""

import argparse
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True

import numpy as np  # noqa: E402

from lanes.p1.summarize import (COLUMNS, DROPS, N_BOOT, SEED, _write, job_dir, load_job,  # noqa: E402
                                paired_diff, summarize_job)

SEED1 = {"single": ("25893211", "25916920"), "groups": ("25893212",), "dit": ("25893213",)}
N_PER_CLASS = 32
PROTECTED = {"results_sweep.csv", "ranking.csv", "compare_dit.csv", "heatmap_by_class.csv"}
COMPARE_COLUMNS = ("model", "set", "removed", "arm_seed1", "job_seed1", "arm_seed2", "job_seed2", "n_seed1",
                   "n_seed2", *(f"{c}_{m}{s}" for m in DROPS
                                for c, s in (("drop_s1", ""), ("drop_s1", "_lo"), ("drop_s1", "_hi"),
                                             ("drop_s2", ""), ("drop_s2", "_lo"), ("drop_s2", "_hi"),
                                             ("diff", ""), ("diff", "_lo"), ("diff", "_hi"))))


def _removed(params: dict) -> tuple:
    """(sort key, label) for what an arm removed: VAR scales or DiT step range."""
    if "scales" in params:
        s = params["scales"]
        return (s[0], len(s)), (str(s[0]) if len(s) == 1 else "{" + ",".join(map(str, s)) + "}")
    if "step_first" in params:
        return (params["step_first"], 0), f"{params['step_first']}-{params['step_last']}"
    raise SystemExit(f"ERROR: params do not say what was removed: {params}")


def arm_drops(jobs, what: str) -> dict:
    """{removed label: arm info with per-image baseline-minus-arm drops} over the removal arms of `jobs`."""
    out = {}
    for j in jobs:
        job = job_dir(j)
        runs, metrics, _ = load_job(job)
        for r in runs:
            if not r["baseline_run_id"]:
                continue
            key, label = _removed(json.loads(r["params"] or "{}"))
            if label in out:
                raise SystemExit(f"ERROR: {what}: {label} removed by two arms ({out[label]['job']}, {job.name})")
            base, run = metrics[r["baseline_run_id"]], metrics[r["run_id"]]
            keys = sorted(run["cls_top1"])
            if sorted(base["cls_top1"]) != keys:
                raise SystemExit(f"ERROR: {job.name} {r['arm']}: images differ from its baseline")
            per_class = {c: sum(k[0] == c for k in keys) for c in {k[0] for k in keys}}
            if len(per_class) != 7 or set(per_class.values()) != {N_PER_CLASS}:
                raise SystemExit(f"ERROR: {job.name} {r['arm']}: expected 7 classes x {N_PER_CLASS}, got {per_class}")
            out[label] = {"key": key, "model": r["model"], "arm": r["arm"], "job": job.name, "keys": keys,
                          **{m: paired_diff(base[m], run[m], keys) for m in DROPS},
                          **{f"d_{m}": np.array([base[m][k] - run[m][k] for k in keys]) for m in DROPS}}
    return out


def diff_ci(d1: np.ndarray, d2: np.ndarray) -> tuple:
    """mean(d2) - mean(d1) and its 95% CI: the two independent image sets resampled separately."""
    rng = np.random.default_rng(SEED)
    b1 = d1[rng.integers(0, len(d1), size=(N_BOOT, len(d1)))].mean(axis=1)
    b2 = d2[rng.integers(0, len(d2), size=(N_BOOT, len(d2)))].mean(axis=1)
    boot = b2 - b1
    return float(d2.mean() - d1.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def spearman(x, y) -> float:
    """Spearman rho: Pearson correlation of the ranks (ties get their average rank)."""
    def ranks(v):
        v = np.asarray(v, dtype=float)
        order = v.argsort(kind="stable")
        r = np.empty(len(v))
        r[order] = np.arange(1, len(v) + 1)
        for u in np.unique(v):
            r[v == u] = r[v == u].mean()
        return r
    return float(np.corrcoef(ranks(x), ranks(y))[0, 1])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--single", help="seed-2 job of seed2_var_single.env")
    ap.add_argument("--groups", help="seed-2 job of seed2_var_groups.env")
    ap.add_argument("--dit", help="seed-2 job of seed2_dit.env")
    ap.add_argument("--results-out", type=Path, default=Path("lanes/p1/results/seed2_results.csv"))
    ap.add_argument("--compare-out", type=Path, default=Path("lanes/p1/results/seed_compare.csv"))
    args = ap.parse_args(argv)
    seed2 = {s: (getattr(args, s),) for s in SEED1 if getattr(args, s)}
    if not seed2:
        ap.error("give at least one seed-2 job (--single, --groups, --dit)")
    for p in (args.results_out, args.compare_out):
        if p.name in PROTECTED:
            raise SystemExit(f"ERROR: {p} is an existing results file other lanes read; choose another name")

    rows, compare, sets = [], [], {}
    for s, jobs2 in seed2.items():
        rows += summarize_job(job_dir(jobs2[0]))
        a1, a2 = arm_drops(SEED1[s], f"seed 1 {s}"), arm_drops(jobs2, f"seed 2 {s}")
        imgs1 = {k for a in a1.values() for k in a["keys"]}
        imgs2 = {k for a in a2.values() for k in a["keys"]}
        if imgs1 & imgs2:
            raise SystemExit(f"ERROR: {s}: the seed sets share {len(imgs1 & imgs2)} images; "
                             f"is {jobs2[0]} really a seed-2 job?")
        only = sorted(set(a1) ^ set(a2))
        if only:
            print(f"note: {s}: removed in one seed set only, not compared: {only}")
        sets[s] = (a1, a2)
        for label in sorted(set(a1) & set(a2), key=lambda lb: a1[lb]["key"]):
            x, y = a1[label], a2[label]
            row = {"model": x["model"], "set": s, "removed": label, "arm_seed1": x["arm"], "job_seed1": x["job"],
                   "arm_seed2": y["arm"], "job_seed2": y["job"], "n_seed1": len(x["keys"]), "n_seed2": len(y["keys"])}
            for m in DROPS:
                for c, vals in (("drop_s1", x[m]), ("drop_s2", y[m]), ("diff", diff_ci(x[f"d_{m}"], y[f"d_{m}"]))):
                    for suf, v in zip(("", "_lo", "_hi"), vals):
                        row[f"{c}_{m}{suf}"] = round(v, 4) + 0.0
            compare.append(row)

    _write(args.results_out, COLUMNS, rows)
    _write(args.compare_out, COMPARE_COLUMNS, compare)
    print(f"wrote {args.results_out} ({len(rows)} runs) and {args.compare_out} ({len(compare)} arms)")

    if "single" in sets:
        a1, a2 = sets["single"]
        scales = [str(i) for i in range(10)]
        if all(s in a1 and s in a2 for s in scales):
            p1 = [a1[s]["cls_prob"][0] for s in scales]
            p2 = [a2[s]["cls_prob"][0] for s in scales]
            print(f"single VAR scales, cls_prob drop: Spearman rho (seed 1 vs seed 2, n = 10) = {spearman(p1, p2):.3f}")
            for name, p in (("seed 1", p1), ("seed 2", p2)):
                top = sorted(range(10), key=lambda i: -p[i])[:3]
                print(f"  top-3 {name}: " + ", ".join(f"scale {i} ({p[i]:.3f})" for i in top))
        else:
            print("single VAR scales: not all of 0-9 on both sides; no Spearman rho")
    if "dit" in sets:
        names = {"1-83": "early", "84-166": "mid", "167-249": "late"}
        for name, arms in zip(("seed 1", "seed 2"), sets["dit"]):
            order = sorted(arms, key=lambda lb: -arms[lb]["cls_prob"][0])
            print(f"DiT order by cls_prob drop, {name}: " + " > ".join(
                f"{names.get(lb, lb)} ({arms[lb]['cls_prob'][0]:.3f})" for lb in order))


if __name__ == "__main__":
    main()
