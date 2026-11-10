"""The predeclared shortlist (plan KTD6/KTD7): which stage set the other lanes jointly point at.

Run from the repo root, after `python -m lanes.p4.synthesis snapshot`.

    python -m lanes.p4.shortlist outputs/p4/handoff/<date> --out-dir outputs/p4/summary/<date>

Writes `shortlist_var.json` and `shortlist_dit.json`. The rule was fixed before any P4 sweep result
was seen, and P4's own results never enter it:

1. Inputs are P1-P3 only (synthesis.SHORTLIST_LANES): one number per stage, higher = more important
   (lanes/p4/synthesis.py states each lane's direction rule).
2. A lane that is constant on a model carries no ranking; it is excluded and recorded as excluded.
3. Each remaining lane is min-max normalized to 0-1 over its own stages. A stage's value is the mean
   over the lanes that have it.
4. VAR: every contiguous window at each frozen band size (3 and 4 scales) is scored by the mean of
   its stages' values. DiT: the lanes resolve only bands (P1) or 8 timesteps (P2), so the candidates are
   the three frozen bands, each scored by the mean over lanes of that lane's mean inside the band.
5. The highest score wins. A tie goes to a predeclared band.
6. Verdict: `covered by <band>` when the winner is a frozen band's stage set -- the Phase C sweep already
   ran the four-arm test on exactly that set, deterministically, so it is the confirmation (KTD7).
   Otherwise `new candidate set`: confirmation runs are submitted with STAGES=<set> and repeated on
   d24 (VAR, exported against the pooled control) or with SEED=20261070 (DiT) (RUNBOOK section B5).

Agreement between the lanes is reported beside the verdict, never used to select: pairwise Spearman
(rho from lanes.p3.agreement.spearman, P3's own function; p from the t distribution with n - 2 df,
as scipy computes it, whether or not scipy is installed) on each pair's shared stages, or "not
computable" when a pair shares fewer than 3 stages. With n = 10, |rho| must exceed about 0.65 for
p < 0.05.
"""

import argparse
import itertools
import json
import math
import sys
from pathlib import Path

sys.dont_write_bytecode = True

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from lanes.p4 import arms, bands, progress, synthesis  # noqa: E402

TIE = 1e-12
MIN_SHARED = 3
RULE = ("KTD6: P1-P3 only; constant lanes excluded; each lane min-max normalized; stage value = mean over "
        "lanes; VAR windows of each frozen band size scored by mean, DiT picks among its three bands; "
        "ties to a predeclared band. KTD7: a winner equal to a frozen band is covered by the sweep.")


def _normalize(scores: dict) -> dict:
    lo, hi = min(scores.values()), max(scores.values())
    return {s: (v - lo) / (hi - lo) for s, v in scores.items()}


def usable_lanes(model_scores: dict) -> tuple[dict, dict]:
    """({lane: normalized scores}, {lane: why excluded}) over the shortlist lanes only."""
    used, excluded = {}, {}
    for lane in synthesis.SHORTLIST_LANES:
        scores = model_scores.get(lane)
        if not scores:
            continue
        if max(scores.values()) - min(scores.values()) <= TIE:
            excluded[lane] = "constant on this model: it ranks no stage above another"
            continue
        used[lane] = _normalize(scores)
    for lane in model_scores:
        if lane not in synthesis.SHORTLIST_LANES:
            excluded[lane] = "not a shortlist input (P4 never selects its own confirmation set)"
    return used, excluded


def _candidates_var(used: dict, ordered: list[int], band_sets: dict) -> list[dict]:
    stage_value = {}
    for stage in ordered:
        vals = [lane[stage] for lane in used.values() if stage in lane]
        if vals:
            stage_value[stage] = sum(vals) / len(vals)
    sizes = sorted({len(s) for s in band_sets.values()})
    out = []
    for size in sizes:
        for start in range(len(ordered) - size + 1):
            window = ordered[start:start + size]
            if any(s not in stage_value for s in window):
                continue
            out.append({"stages": window, "score": sum(stage_value[s] for s in window) / size})
    return out


def _candidates_dit(used: dict, band_sets: dict) -> list[dict]:
    out = []
    for band, stages in band_sets.items():
        per_lane = []
        for lane in used.values():
            inside = [lane[s] for s in stages if s in lane]
            if inside:
                per_lane.append(sum(inside) / len(inside))
        if per_lane:
            out.append({"stages": list(stages), "score": sum(per_lane) / len(per_lane)})
    return out


def _betainc(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta I_x(a, b) by Lentz's continued fraction (no scipy needed)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    if x > (a + 1) / (a + b + 2):                      # the fraction converges fast only below this
        return 1.0 - _betainc(b, a, 1.0 - x)
    front = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                     + a * math.log(x) + b * math.log1p(-x)) / a
    tiny = 1e-300
    f, c, d = 1.0, 1.0, 0.0
    for i in range(400):
        m = i // 2
        if i == 0:
            num = 1.0
        elif i % 2 == 0:
            num = m * (b - m) * x / ((a + 2 * m - 1) * (a + 2 * m))
        else:
            num = -(a + m) * (a + b + m) * x / ((a + 2 * m) * (a + 2 * m + 1))
        d = 1.0 + num * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + num / c
        c = c if abs(c) > tiny else tiny
        f *= c * d
        if abs(1.0 - c * d) < 1e-14:
            break
    return front * (f - 1.0)


def spearman_p(rho: float, n: int) -> float:
    """Two-sided p of a Spearman rho on n pairs: Student t with n - 2 df, as scipy.stats.spearmanr.

    P3's spearman() falls back to a normal approximation when scipy is missing (the NSCC var-dit
    env), which at n = 8 understates p by orders of magnitude, so P4 computes p itself from P3's rho.
    """
    if rho != rho or n < 3:
        return float("nan")
    if abs(rho) >= 1.0:
        return 0.0
    df = n - 2
    t2 = rho * rho * df / (1.0 - rho * rho)
    return _betainc(df / 2.0, 0.5, df / (df + t2))


def agreement(used_raw: dict) -> list[dict]:
    from lanes.p3.agreement import spearman
    rows = []
    for (a, sa), (b, sb) in itertools.combinations(sorted(used_raw.items()), 2):
        shared = sorted(set(sa) & set(sb))
        if len(shared) < MIN_SHARED:
            rows.append({"a": a, "b": b, "n": len(shared), "rho": None, "p": None,
                         "note": "not computable: fewer than 3 shared stages"})
            continue
        rho, _ = spearman([sa[s] for s in shared], [sb[s] for s in shared])
        p = spearman_p(rho, len(shared))
        resolved = p == p and p < 0.05
        rows.append({"a": a, "b": b, "n": len(shared), "rho": None if rho != rho else round(rho, 4),
                     "p": None if p != p else float(f"{p:.3g}"),
                     "note": "" if resolved else "not significant at this n: this cannot resolve it"})
    return rows


def shortlist(scores: dict, model: str) -> dict:
    """The rule applied to one model's lane scores (synthesis.lane_scores output)."""
    model_scores = scores.get(model, {})
    used, excluded = usable_lanes(model_scores)
    table = synthesis.stage_table(model)
    ordered = [r["stage"] for r in table]
    band_sets = synthesis.band_stages(model)
    record = {"model": model, "rule": RULE, "lanes_used": sorted(used), "lanes_excluded": excluded,
              "agreement": agreement({lane: model_scores[lane] for lane in used})}
    if not used:
        record.update({"verdict": "no lanes: nothing to shortlist", "window": None, "score": None,
                       "covering_band": None, "candidates": []})
        return record
    cands = (_candidates_var(used, ordered, band_sets) if model == "var"
             else _candidates_dit(used, band_sets))
    as_band = {tuple(sorted(s)): b for b, s in band_sets.items()}
    for c in cands:
        c["band"] = as_band.get(tuple(sorted(c["stages"])))
    best = max(c["score"] for c in cands)
    tied = [c for c in cands if best - c["score"] <= TIE]
    winner = next((c for c in tied if c["band"]), tied[0])
    ranked = sorted(cands, key=lambda c: (-c["score"], c["band"] is None, c["stages"]))
    record.update({"window": winner["stages"], "score": round(winner["score"], 6),
                   "covering_band": winner["band"],
                   "verdict": f"covered by {winner['band']}" if winner["band"] else "new candidate set",
                   "candidates": [{**c, "score": round(c["score"], 6)} for c in ranked[:6]]})
    if not winner["band"]:
        stages = ",".join(str(s) for s in winner["stages"])
        tag = arms.candidate_label(winner["stages"])
        first = f'qsub -P <project> -v "STAGES={stages},OUT_TAG={tag}" lanes/p4/jobs/sweep_{model}.pbs'
        if model == "var":
            # VAR's control is the pooled control (export --control-pool), which already averages over
            # every placement, so a second control seed adds nothing; the d24 repeat still does.
            submit = [first, f'qsub -P <project> -v "STAGES={stages},OUT_TAG={tag}_d24,'
                             f'CONFIG=configs/var_d24.yaml" lanes/p4/jobs/sweep_var.pbs']
            note = "export each root with --control-pool <the d20 or d24 pool root> (RUNBOOK section B5)"
        else:
            submit = [first, f'qsub -P <project> -v "STAGES={stages},OUT_TAG={tag}_seed2,'
                             f'SEED={arms.PHASE_E_SEED}" lanes/p4/jobs/sweep_{model}.pbs']
            note = ""
        record["confirm"] = {"stages": stages, "submit": submit, "note": note}
    return record


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("snapshot", type=Path, help="a folder written by lanes.p4.synthesis snapshot")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args(argv)
    scores = synthesis.lane_scores(args.snapshot)          # never with P4's summary
    sources = args.snapshot / "SOURCES.json"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for model in progress.MODELS:
        record = shortlist(scores, model)
        record["snapshot"] = str(args.snapshot)
        record["snapshot_sources"] = json.loads(sources.read_text()) if sources.is_file() else None
        path = args.out_dir / f"shortlist_{model}.json"
        path.write_text(json.dumps(record, indent=2))
        print(f"{model}: window {record['window'] if model == 'var' else record['covering_band']}  "
              f"score {record['score']}  lanes {record['lanes_used']}  -> {record['verdict']}")
        for row in record["agreement"]:
            print(f"     {row['a']} vs {row['b']}: n={row['n']} rho={row['rho']} p={row['p']} {row['note']}")
        if "confirm" in record:
            print("     confirmation runs:\n       " + "\n       ".join(record["confirm"]["submit"]))
            if record["confirm"]["note"]:
                print(f"     then {record['confirm']['note']}")
        print(f"     wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
