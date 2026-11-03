"""P4 self-check for Phase D: the snapshot, the lane adapters, Figure 5 and the shortlist.

Run from the repo root, in the var-dit env. CPU only, no GPU, no network.

    python lanes/p4/check_p4_synthesis.py

Ends in one line: ALL P4 SYNTHESIS CHECKS PASS, or P4 SYNTHESIS CHECKS FAILED.
"""

import hashlib
import json
import sys
import tempfile
from pathlib import Path

sys.dont_write_bytecode = True

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from lanes.p4 import arms, bands, export, fixtures, pilot, synthesis  # noqa: E402
from lanes.p4.check_p4_shared import Checks  # noqa: E402

VAR_STAGES = list(range(10))


def write_csv(path: Path, header: list[str], rows: list[list]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(",".join(header) + "\n" + "".join(",".join(str(x) for x in r) + "\n" for r in rows))
    return path


def fake_snapshot(root: Path, *, p1_var=None, p1_dit=None, p2_var=None, p2_dit=None, p3_var=None,
                  p2_p_offset=0.0) -> Path:
    """A snapshot folder in the shape `synthesis.snapshot` writes, from per-stage values."""
    root.mkdir(parents=True, exist_ok=True)
    p1_rows = []
    for stage, drop in (p1_var or {}).items():
        p1_rows.append(["1", "var", f"remove_s{stage}", "224", "0", "0", 0.8 - drop, "", "", "", "", drop, "", ""])
    for band, drop in (p1_dit or {}).items():
        arm = {"early": "remove_early", "middle": "remove_mid", "late": "remove_late"}[band]
        p1_rows.append(["2", "dit", arm, "224", "0", "0", 0.76 - drop, "", "", "", "", drop, "", ""])
    if p1_rows:
        p1_rows.append(["1", "var", "baseline", "224", "0", "0", "0.81", "", "", "", "", "", "", ""])
        write_csv(root / "p1" / "results_sweep.csv",
                  ["job", "model", "arm", "n", "cls_top1", "cls_top5", "cls_prob", "img_mad255", "drop_cls_top1",
                   "drop_cls_top1_lo", "drop_cls_top1_hi", "drop_cls_prob", "drop_cls_prob_lo",
                   "drop_cls_prob_hi"], p1_rows)
    p2_rows = []
    for model, values in (("var", p2_var), ("dit", p2_dit)):
        places = {r["stage"]: r["p_place"] for r in synthesis.stage_table(model)}
        for stage, drop in (values or {}).items():
            for sev in (0.5, 1.0):
                p2_rows.append([model, stage, places[stage] + p2_p_offset, sev, 0.8 - (drop if sev == 1.0 else 0),
                                0.8])
    if p2_rows:
        write_csv(root / "p2" / "p2_summary.csv", ["model", "stage", "p", "severity", "cls_prob", "cls_prob_base"],
                  p2_rows)
    if p3_var:
        cumulative, total = [], 0.0
        for stage in VAR_STAGES:
            total += p3_var[stage]
            cumulative.append([stage, total])
        write_csv(root / "p3" / "var_semantic.csv", ["stage", "identity_progress"], cumulative)
    return root


def check_snapshot(c: Checks):
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        record = synthesis.snapshot(tmp / "snap", sources=(
            ("p4", "HEAD", "lanes/p4/bands.py"),
            ("p4", "HEAD", "lanes/p4/no_such_file.csv"),
            ("p9", "refs/heads/no-such-branch-p4check", "lanes/p4/bands.py")))
        files = record["files"]
        c.that("a file on the ref is copied", not files[0]["missing"] and (tmp / "snap" / files[0]["copy"]).is_file())
        c.equal("its sha256 in SOURCES.json matches the copy", files[0]["sha256"],
                hashlib.sha256((tmp / "snap" / files[0]["copy"]).read_bytes()).hexdigest())
        c.that("the commit the ref resolved to is recorded", len(files[0]["commit"] or "") == 40)
        c.that("a path the ref lacks is recorded as missing, not fatal",
               files[1]["missing"] and files[1]["reason"] == "path not on ref")
        c.that("a ref that does not exist is recorded as missing, not fatal",
               files[2]["missing"] and files[2]["reason"] == "ref not found")
        on_disk = json.loads((tmp / "snap" / "SOURCES.json").read_text())
        c.equal("SOURCES.json holds every requested file", len(on_disk["files"]), 3)


def check_adapters(c: Checks):
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        p1 = {s: 0.1 * s for s in VAR_STAGES}
        snap = fake_snapshot(tmp / "s", p1_var=p1, p1_dit={"early": 0.05, "middle": 0.2, "late": 0.4},
                             p2_var={s: 0.3 - 0.02 * s for s in VAR_STAGES}, p2_dit={798: 0.1, 397: 0.5, 96: 0.7})
        scores = synthesis.lane_scores(snap)
        c.equal("P1: remove_s3 maps to VAR stage 3", scores["var"]["p1"][3], 0.30000000000000004)
        dit_bands = synthesis.band_stages("dit")
        c.that("P1: DiT remove_mid spans exactly P4's middle band",
               {s for s, v in scores["dit"]["p1"].items() if v == 0.2} == set(dit_bands["middle"]))
        c.equal("P1: DiT band values cover all 250 steps", len(scores["dit"]["p1"]), 250)
        c.equal("P2: one value per stage at severity 1.0, sev 0.5 ignored",
                round(scores["var"]["p2"][0], 9), 0.3)
        c.equal("P2: DiT timesteps keep their own ids", sorted(scores["dit"]["p2"]), [96, 397, 798])
        for model in ("var", "dit"):
            for lane, by_stage in scores[model].items():
                top = max(by_stage, key=by_stage.get)
                truth = {("var", "p1"): 9, ("var", "p2"): 0, ("dit", "p2"): 96}.get((model, lane))
                if truth is not None:
                    c.equal(f"direction: {model} {lane}'s most important stage has the largest value", top, truth)
        c.raises("P2: a p that disagrees with P4's p_place fails",
                 lambda: synthesis.p2_scores(fake_snapshot(tmp / "bad", p2_var={0: 0.1, 5: 0.2}, p2_p_offset=0.01)),
                 "would not share an axis", synthesis.SynthesisError)
        c.equal("absent lanes are named", synthesis.absent_lanes(scores, "var"), ["p3"])

        # P3: cumulative curve -> per-stage contribution, through P3's own load_rank
        bump = {s: (0.5 if s == 4 else 0.05) for s in VAR_STAGES}
        snap3 = fake_snapshot(tmp / "s3", p3_var=bump)
        p3 = synthesis.p3_scores(snap3)
        c.equal("P3: the stage that adds the most identity is the most important",
                max(p3["p3_identity"]["var"], key=p3["p3_identity"]["var"].get), 4)

        # P4: -(damage - control) at the largest budget, per band
        root = fixtures.write_sweep_root(tmp / "sweep", "var", lambda arm, band, m, metric, c_, s: (
            0.5 - (0.3 if (arm == "damage" and band == "middle") else 0.1 if arm == "damage" else 0.0)))
        summary = export.write_rows(export.summary_rows(export.load_root(root)), export.SUMMARY_COLUMNS,
                                    tmp / "sum.csv")
        p4 = synthesis.p4_scores(summary, "var")
        c.close("P4: the band whose damage hurts most scores highest", p4[4], 0.3, 1e-9)
        c.close("P4: a band whose damage hurts less scores lower", p4[0], 0.1, 1e-9)


def check_figure5(c: Checks):
    from lanes.p4 import plotting
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        snap = fake_snapshot(tmp / "s", p1_var={s: 0.1 * s for s in VAR_STAGES},
                             p2_var={s: 0.3 - 0.02 * s for s in VAR_STAGES},
                             p1_dit={"early": 0.05, "middle": 0.2, "late": 0.4}, p2_dit={798: 0.1, 397: 0.5})
        root = fixtures.write_sweep_root(tmp / "sweep", "var", lambda *a: 0.4 if a[0] == "damage" else 0.5)
        summary = export.write_rows(export.summary_rows(export.load_root(root)), export.SUMMARY_COLUMNS,
                                    tmp / "sum.csv")
        scores = synthesis.lane_scores(snap, {"var": summary})
        tables = synthesis.figure5_tables(scores, "var")
        c.equal("P1 + P2 + P4 snapshot: three VAR panels, in lane order",
                [label for label, _rows in tables],
                [synthesis.LANE_LABELS[k] for k in ("p1", "p2", "p4")])
        written = synthesis.write_figure5(scores, tmp / "fig", backend="pillow")
        c.that("Figure 5 renders for both models", sorted(written) == ["dit", "var"]
               and all(p.exists() for p in written.values()))
        c.that("each lane's tidy table is written beside the figure",
               (tmp / "fig" / "figure5_var_p1_tidy.csv").is_file())
        back = plotting.read_tidy(tmp / "fig" / "figure5_var_p2_tidy.csv")
        c.equal("the tidy table round-trips", len(back), 10)


def _scores(var=None, dit=None) -> dict:
    """synthesis.lane_scores-shaped input: {model: {lane: {stage: value}}}."""
    return {"var": var or {}, "dit": dit or {}}


def check_shortlist(c: Checks):
    from lanes.p4 import shortlist
    bump = lambda top: {s: (1.0 if s in top else 0.1 * (s % 3)) for s in VAR_STAGES}   # noqa: E731
    r = shortlist.shortlist(_scores(var={"p1": bump({0, 1, 2}), "p2": bump({0, 1, 2})}), "var")
    c.equal("a top window of {0,1,2} is covered by the early band",
            (r["window"], r["verdict"], r["covering_band"]), ([0, 1, 2], "covered by early", "early"))
    c.that("a covered set plans no confirmation runs", "confirm" not in r)

    r = shortlist.shortlist(_scores(var={"p1": bump({2, 3, 4}), "p2": bump({2, 3, 4})}), "var")
    c.equal("a top window of {2,3,4} is a new candidate set", (r["window"], r["verdict"]),
            ([2, 3, 4], "new candidate set"))
    c.that("a VAR confirmation is submitted on d20 and d24, and exported against the pooled control",
           len(r["confirm"]["submit"]) == 2 and all("STAGES=2,3,4" in s for s in r["confirm"]["submit"])
           and not any("SEED=" in s for s in r["confirm"]["submit"])
           and any("var_d24" in s for s in r["confirm"]["submit"])
           and "--control-pool" in r["confirm"]["note"])
    runs = pilot.plan_runs("var", (2, 3, 4), REPO / "configs" / "var_d20.yaml",
                           Path("manifest/frozen_v1_1.csv"), Path("/tmp/x"))
    c.equal("its confirmation plan is one baseline plus 9 arms", len(runs), 10)

    # a tie between a predeclared band and a non-band window goes to the band
    tie = {s: (1.0 if s in (3, 4, 5, 6, 7, 8) else 0.0) for s in VAR_STAGES}
    r = shortlist.shortlist(_scores(var={"p1": tie}), "var")
    c.equal("a tie between {3,4,5} (a band) and {4,5,6} goes to the band", r["window"], [3, 4, 5])

    # only one lane, a constant lane, and P4 never selecting
    r = shortlist.shortlist(_scores(var={"p2": bump({6, 7, 8}), "p1": {s: 0.2 for s in VAR_STAGES},
                                         "p4": bump({0, 1, 2})}), "var")
    c.equal("with one usable lane the rule still runs and records it", r["lanes_used"], ["p2"])
    c.that("a constant lane is excluded with a reason", "constant" in r["lanes_excluded"].get("p1", ""))
    c.that("P4 is excluded from selection", "p4" in r["lanes_excluded"] and r["window"] != [0, 1, 2])

    # agreement: computed on shared stages, or recorded as not computable
    r = shortlist.shortlist(_scores(var={"p1": bump({2, 3, 4}), "p2": {0: 0.1, 1: 0.5, 9: 0.2},
                                         "p3_identity": bump({2, 3, 4})}), "var")
    by_pair = {(a["a"], a["b"]): a for a in r["agreement"]}
    c.that("identical rankings agree with rho = 1", by_pair[("p1", "p3_identity")]["rho"] == 1.0)
    c.that("a pair sharing 3 stages is computed", by_pair[("p1", "p2")]["n"] == 3)
    r = shortlist.shortlist(_scores(var={"p1": bump({2, 3, 4}), "p2": {0: 0.1, 9: 0.2}}), "var")
    c.that("a pair sharing fewer than 3 stages is 'not computable'",
           r["agreement"][0]["rho"] is None and "not computable" in r["agreement"][0]["note"])

    # DiT picks among its three bands only, so its verdict is always a covering band
    dit_bands = synthesis.band_stages("dit")
    for top in bands.BANDS:
        p1 = {s: (0.9 if b == top else 0.1) for b, ss in dit_bands.items() for s in ss}
        r = shortlist.shortlist(_scores(dit={"p1": p1, "p2": {dit_bands[top][5]: 0.8, dit_bands["early"][9]: 0.0,
                                                               dit_bands["late"][9]: 0.0}}), "dit")
        c.equal(f"DiT: lanes pointing at {top} give 'covered by {top}'", r["verdict"], f"covered by {top}")
    c.equal("no lanes at all is reported, not crashed",
            shortlist.shortlist(_scores(), "var")["verdict"], "no lanes: nothing to shortlist")


def check_explicit_sets(c: Checks):
    for model in ("var", "dit"):
        stages = fixtures.frozen_stages(model)
        middle = synthesis.band_stages(model)["middle"]
        for m in arms.budget_grid(model, stages, "middle"):
            by_name = arms.arm_plan(model, stages, "middle", m)
            by_set = arms.arm_plan(model, stages, tuple(middle), m)
            c.equal(f"{model} m={m}: an explicit set equal to the middle band plans the same stages",
                    {a: by_set[a]["stages"] for a in arms.ARMS}, {a: by_name[a]["stages"] for a in arms.ARMS})
        c.equal(f"{model}: an explicit set has the same budget grid as its band",
                arms.budget_grid(model, stages, tuple(middle)), arms.budget_grid(model, stages, "middle"))
    stages = fixtures.frozen_stages("var")
    order = [s["stage"] for s in stages]
    for m in arms.budget_grid("var", stages, (2, 3, 4)):
        plan = arms.arm_plan("var", stages, (2, 3, 4), m)
        c.that(f"VAR {{2,3,4}} m={m}: damage inside, protect outside",
               set(plan["damage"]["stages"]) <= {2, 3, 4} and not set(plan["protect"]["stages"]) & {2, 3, 4})
        c.equal(f"VAR {{2,3,4}} m={m}: window lengths match across arms",
                {plan[a]["windows"] for a in arms.INTERVENED_ARMS}, {plan["damage"]["windows"]})
        c.equal(f"VAR {{2,3,4}} m={m}: the record names the set", plan["damage"]["band"], "s2+3+4")
    c.equal("VAR {2,3,4}: m is capped at 3", max(arms.budget_grid("var", stages, (2, 3, 4))), 3)
    c.raises("a budget above the set's size is refused",
             lambda: arms.arm_plan("var", stages, (2, 3, 4), 4), "exceeds", arms.ArmError)
    dit = fixtures.frozen_stages("dit")
    first = [s["stage"] for s in dit][0]
    plan = arms.arm_plan("dit", dit, (first, *[s["stage"] for s in dit][1:6]), 5)
    c.that("DiT: a set holding the first step never damages it", first not in plan["damage"]["stages"])
    c.raises("DiT: that set's eligible count excludes the first step",
             lambda: arms.arm_plan("dit", dit, (first, *[s["stage"] for s in dit][1:6]), 6), "unskippable",
             arms.ArmError)
    c.raises("a stage id the table does not have is refused",
             lambda: arms.arm_plan("var", stages, (2, 3, 42), 1), "not native", arms.ArmError)
    c.that("the stage order is untouched", order == list(range(10)))


def main(argv=None):
    print("== P4 synthesis self-check (CPU; snapshot, adapters, Figure 5, shortlist)")
    sections = [("snapshot from git refs", check_snapshot), ("lane adapters", check_adapters),
                ("Figure 5 from a snapshot", check_figure5), ("shortlist rule", check_shortlist),
                ("explicit candidate sets in the arm construction", check_explicit_sets)]
    ok = True
    for title, fn in sections:
        c = Checks()
        try:
            fn(c)
        except Exception as e:   # noqa: BLE001
            c.that(f"{title} ran to completion", False, f"{type(e).__name__}: {e}")
        c.report(title)
        ok = ok and c.ok
    print("ALL P4 SYNTHESIS CHECKS PASS" if ok else "P4 SYNTHESIS CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
