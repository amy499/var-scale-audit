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

from lanes.p4 import bands, export, fixtures, synthesis  # noqa: E402
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


def main(argv=None):
    print("== P4 synthesis self-check (CPU; snapshot, adapters, Figure 5, shortlist)")
    sections = [("snapshot from git refs", check_snapshot), ("lane adapters", check_adapters),
                ("Figure 5 from a snapshot", check_figure5)]
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
