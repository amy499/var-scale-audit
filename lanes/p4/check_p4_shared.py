"""P4 self-check for the shared deliverable: the progress mapping, the frozen bands and the template.

Run from the repo root. Standard library plus PyYAML. No torch, no GPU, no checkpoint, and
deliberately **no plotting library** -- the data layer must work without one.

    python lanes/p4/check_p4_shared.py
    python lanes/p4/check_p4_shared.py --runs DIR     # also check against real run.json dirs under DIR

Ends in one line: ALL P4 SHARED CHECKS PASS, or P4 SHARED CHECKS FAILED with the failing invariants
named. Exit code 0 only when every assertion holds. This is the only gate between a wrong band list
and GPU time spent on it, so it asserts the band lists against the frozen configs, not against
constants copied from the plan.
"""

import argparse
import sys
from pathlib import Path

sys.dont_write_bytecode = True

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from lanes.p4 import bands, fixtures, progress  # noqa: E402

TINY_PATCH_NUMS = [1, 2, 3, 4]    # scripts/phase2/check_seeding.py VAR_BUILD: 4 scales, 30 tokens
TINY_DIT_STEPS = 20               # scripts/phase2/check_seeding.py DIT tiny sampler


class Checks:
    """Collects one PASS/FAIL line per invariant; any failure decides the exit code."""

    def __init__(self):
        self.results = []

    def that(self, name: str, ok: bool, detail: str = ""):
        self.results.append((name, bool(ok), detail if not ok else ""))
        return ok

    def equal(self, name: str, got, want):
        return self.that(name, got == want, f"got {got!r}, want {want!r}")

    def close(self, name: str, got: float, want: float, tol: float = 5e-4):
        return self.that(name, abs(got - want) <= tol, f"got {got!r}, want {want!r} +/- {tol}")

    def raises(self, name: str, fn, needle: str, exc_type: type = Exception):
        """Assert fn() raises `exc_type` and that its message mentions `needle`.

        The type matters: without it an assertion passes when the call fails for an unrelated reason
        whose message happens to contain the needle, which is a check that cannot fail for the thing
        it was written to catch.
        """
        try:
            fn()
        except exc_type as e:
            return self.that(name, needle in str(e),
                             f"{type(e).__name__}({e}) does not mention {needle!r}")
        except Exception as e:   # noqa: BLE001 - the wrong failure is still a failure
            return self.that(name, False,
                             f"raised {type(e).__name__}({e}), expected {exc_type.__name__}")
        return self.that(name, False, f"no exception raised; expected {exc_type.__name__}")

    @property
    def ok(self) -> bool:
        return all(ok for _, ok, _ in self.results)

    def report(self, title: str):
        print(f"-- {title}")
        for name, ok, detail in self.results:
            print(f"   {'PASS' if ok else 'FAIL'}  {name}" + ("" if ok else f"   {detail}"))


# ---------------------------------------------------------------- U1: progress mapping

def check_progress(c: Checks):
    var = progress.map_stages("var", fixtures.frozen_stages("var"))
    c.equal("var d20 has 10 stages", len(var), 10)
    c.equal("var d20 native stages are si 0..9", [r["stage"] for r in var], list(range(10)))
    c.close("var d20 placement starts at 0", var[0]["p_place"], 0.0, 0)
    c.close("var d20 placement ends at 1", var[-1]["p_place"], 1.0, 0)
    c.close("var d20 functional starts at 1/680", var[0]["p_func"], 1 / 680, 1e-12)
    c.close("var d20 functional ends at 1", var[-1]["p_func"], 1.0, 0)
    # The Appendix table, which is what the report prints.
    appendix = [(0, 0.000, 0.001), (1, 0.111, 0.007), (2, 0.222, 0.021), (3, 0.333, 0.044), (4, 0.444, 0.081),
                (5, 0.556, 0.134), (6, 0.667, 0.228), (7, 0.778, 0.375), (8, 0.889, 0.624), (9, 1.000, 1.000)]
    for si, p_place, p_func in appendix:
        c.close(f"var d20 si {si} placement", var[si]["p_place"], p_place)
        c.close(f"var d20 si {si} functional", var[si]["p_func"], p_func)

    dit = progress.map_stages("dit", fixtures.frozen_stages("dit"))
    c.equal("dit 250-step has 250 stages", len(dit), 250)
    c.close("dit placement starts at 0", dit[0]["p_place"], 0.0, 0)
    c.close("dit placement ends at 1", dit[-1]["p_place"], 1.0, 0)
    c.that("dit functional starts above 0", 0.0 < dit[0]["p_func"] < 0.01,
           f"p_func[0] = {dit[0]['p_func']!r}")
    c.close("dit functional reaches 1 at the last step", dit[-1]["p_func"], 1.0, 1e-4)
    for j, tau, p_place, p_func in ((0, 999, 0.000, 0.006), (83, 666, 0.333, 0.105), (124, 502, 0.498, 0.276),
                                    (166, 333, 0.667, 0.564), (210, 156, 0.843, 0.878), (249, 0, 1.000, 1.000)):
        c.equal(f"dit step {j} is timestep {tau}", dit[j]["stage"], tau)
        c.close(f"dit step {j} placement", dit[j]["p_place"], p_place)
        c.close(f"dit step {j} functional", dit[j]["p_func"], p_func)

    # The respacing rule, against a transcription of upstream's accumulation loop, at step counts
    # either side of the real and tiny schedules. 250 and 20 agree under either arithmetic, so
    # checking only those two cannot catch a multiply-instead-of-accumulate shortcut.
    def upstream_timestep_map(count, total=1000):
        if count <= 1:
            return [0]
        frac, cur, taken = (total - 1) / (count - 1), 0.0, []
        for _ in range(count):
            taken.append(round(cur))
            cur += frac
        return sorted(set(taken))

    for count in (2, 10, 20, 21, 23, 31, 43, 100, 125, 249, 250, 251, 500, 1000):
        c.equal(f"timestep_map reproduces upstream accumulation at S={count}",
                fixtures.timestep_map(count), upstream_timestep_map(count))

    # No hardcoded 10-scale or 680-token constant, and a schedule that is not 250 steps, so the
    # mapping holds for VAR d16-d30 and for the tiny CPU models.
    tiny_var = progress.map_stages("var", fixtures.var_stages(TINY_PATCH_NUMS))
    c.equal("tiny var maps 4 scales", len(tiny_var), 4)
    c.equal("tiny var total tokens is 30", tiny_var[-1]["total_tokens"], 30)
    c.close("tiny var functional ends at 1", tiny_var[-1]["p_func"], 1.0, 0)
    c.close("tiny var si 1 functional is 5/30", tiny_var[1]["p_func"], 5 / 30, 1e-12)
    tiny_dit = progress.map_stages("dit", fixtures.dit_stages(TINY_DIT_STEPS))
    c.equal("tiny dit maps 20 steps", len(tiny_dit), 20)
    c.close("tiny dit placement ends at 1", tiny_dit[-1]["p_place"], 1.0, 0)
    for depth_patch_nums in ([1, 2, 3, 4, 5, 6, 8, 10, 13, 16], [1, 2, 3], [1]):
        mapped = progress.map_stages("var", fixtures.var_stages(depth_patch_nums))
        c.equal(f"var {len(depth_patch_nums)}-scale table maps every scale", len(mapped), len(depth_patch_nums))

    # Single stage: p = 1.0, no division by zero (the runner's own convention).
    one_var = progress.map_stages("var", fixtures.var_stages([1]))
    c.close("single-scale var placement is 1.0", one_var[0]["p_place"], 1.0, 0)
    one_dit = progress.map_stages("dit", fixtures.dit_stages(1))
    c.close("single-step dit placement is 1.0", one_dit[0]["p_place"], 1.0, 0)

    # Monotonic non-decreasing in sampling order, on every fixture.
    for label, model, stages in (("var d20", "var", fixtures.frozen_stages("var")),
                                 ("dit 250", "dit", fixtures.frozen_stages("dit")),
                                 ("var tiny", "var", fixtures.var_stages(TINY_PATCH_NUMS)),
                                 ("dit tiny", "dit", fixtures.dit_stages(TINY_DIT_STEPS))):
        mapped = progress.map_stages(model, stages)
        for axis in ("p_place", "p_func"):
            values = [r[axis] for r in mapped]
            c.that(f"{label} {axis} is monotonic non-decreasing and in [0, 1]",
                   all(0.0 <= v <= 1.0 for v in values) and all(b >= a for a, b in zip(values, values[1:])),
                   f"{values[:4]} ...")

    # A missing field is named, not a KeyError (and the error type is this module's).
    E = progress.ProgressError
    c.raises("missing var cum_tokens names the field",
             lambda: progress.map_stages("var", [{"stage": 0, "total_tokens": 30}]), "cum_tokens", E)
    c.raises("missing dit alpha_bar_in names the field",
             lambda: progress.map_stages("dit", [{"stage": 999, "p": 0.0}]), "alpha_bar_in", E)
    c.raises("missing stage names the field",
             lambda: progress.map_stages("var", [{"cum_tokens": 1}]), "'stage'", E)
    c.raises("empty stages table is rejected", lambda: progress.map_stages("var", []), "empty", E)
    c.raises("unknown model is rejected", lambda: progress.map_stages("vqgan", []), "vqgan", E)
    c.that("ProgressError is a ValueError", issubclass(progress.ProgressError, ValueError))

    # A VAR table that does not cover every scale is rejected rather than silently mis-mapped.
    partial = fixtures.frozen_stages("var")[:5]
    c.raises("partial var table is rejected",
             lambda: progress.map_stages("var", partial), "every scale", progress.ProgressError)


# ---------------------------------------------------------------- U2: frozen bands

def check_bands(c: Checks):
    c.equal("band names", bands.BANDS, ("early", "middle", "late"))
    c.equal("cuts are thirds of the progress value", bands.CUTS, (1 / 3, 2 / 3))

    var = bands.assign_bands(progress.map_stages("var", fixtures.frozen_stages("var")))
    by_band = {b: [r["stage"] for r in var if r["band"] == b] for b in bands.BANDS}
    c.equal("var d20 early is si 0-2", by_band["early"], [0, 1, 2])
    c.equal("var d20 middle is si 3-5", by_band["middle"], [3, 4, 5])
    c.equal("var d20 late is si 6-9", by_band["late"], [6, 7, 8, 9])
    c.equal("var d20 frozen list matches the derivation", bands.VAR_D20_BANDS,
            {b: tuple(by_band[b]) for b in bands.BANDS})

    dit = bands.assign_bands(progress.map_stages("dit", fixtures.frozen_stages("dit")))
    dit_by_band = {b: [r["stage"] for r in dit if r["band"] == b] for b in bands.BANDS}
    c.equal("dit 250-step band sizes are 83/83/84", [len(dit_by_band[b]) for b in bands.BANDS], [83, 83, 84])
    c.equal("dit early runs t 999..670", (dit_by_band["early"][0], dit_by_band["early"][-1]), (999, 670))
    c.equal("dit middle runs t 666..337", (dit_by_band["middle"][0], dit_by_band["middle"][-1]), (666, 337))
    c.equal("dit late runs t 333..0", (dit_by_band["late"][0], dit_by_band["late"][-1]), (333, 0))
    c.that("dit late ends at timestep 0", dit_by_band["late"][-1] == 0)
    for band in bands.BANDS:
        frozen = bands.DIT_250_BANDS[band]
        c.equal(f"dit frozen {band} step range", frozen["steps"],
                (next(r["step"] for r in dit if r["stage"] == dit_by_band[band][0]),
                 next(r["step"] for r in dit if r["stage"] == dit_by_band[band][-1])))
        c.equal(f"dit frozen {band} timestep endpoints", frozen["timesteps"],
                (dit_by_band[band][0], dit_by_band[band][-1]))
        c.equal(f"dit frozen {band} size", frozen["n"], len(dit_by_band[band]))

    # Total coverage: every stage lands in exactly one band, and the union is every stage.
    for label, mapped in (("var d20", var), ("dit 250", dit)):
        c.that(f"{label}: every stage has exactly one band",
               all(r.get("band") in bands.BANDS for r in mapped))
        covered = sorted(r["stage"] for b in bands.BANDS for r in mapped if r["band"] == b)
        c.equal(f"{label}: bands cover every stage exactly once", covered, sorted(r["stage"] for r in mapped))

    # The half-open rule: a stage exactly on a boundary goes to the upper band, and p = 1.0 is late.
    c.equal("p exactly at the first cut is middle", bands.band_of(1 / 3), "middle")
    c.equal("p exactly at the second cut is late", bands.band_of(2 / 3), "late")
    c.equal("p just below the first cut is early", bands.band_of(1 / 3 - 1e-12), "early")
    c.equal("p = 1.0 is late, not outside every band", bands.band_of(1.0), "late")
    c.equal("p = 0.0 is early", bands.band_of(0.0), "early")
    c.raises("p outside [0, 1] is rejected", lambda: bands.band_of(1.5), "1.5", bands.BandError)

    # A tiny model still produces three non-empty bands.
    tiny = bands.assign_bands(progress.map_stages("var", fixtures.var_stages(TINY_PATCH_NUMS)))
    tiny_by_band = {b: [r["stage"] for r in tiny if r["band"] == b] for b in bands.BANDS}
    c.that("tiny var has three non-empty bands", all(tiny_by_band[b] for b in bands.BANDS),
           f"{tiny_by_band}")
    tiny_dit = bands.assign_bands(progress.map_stages("dit", fixtures.dit_stages(TINY_DIT_STEPS)))
    c.that("tiny dit has three non-empty bands",
           all(any(r["band"] == b for r in tiny_dit) for b in bands.BANDS))

    # The numbers that rejected the alternatives, recomputed rather than quoted from
    # lanes/p4/comparison_logic.md, so the document cannot drift from the code.
    var_mapped = progress.map_stages("var", fixtures.frozen_stages("var"))
    token_bands = {b: [r["stage"] for r in bands.assign_bands(var_mapped, basis="p_func") if r["band"] == b]
                   for b in bands.BANDS}
    c.equal("rejected var token-mass split is 7/2/1 scales",
            [len(token_bands[b]) for b in bands.BANDS], [7, 2, 1])
    dit_mapped = progress.map_stages("dit", fixtures.frozen_stages("dit"))
    signal_bands = {b: [r["stage"] for r in bands.assign_bands(dit_mapped, basis="p_func") if r["band"] == b]
                    for b in bands.BANDS}
    c.equal("rejected dit signal split is 134/46/70 steps",
            [len(signal_bands[b]) for b in bands.BANDS], [134, 46, 70])
    # comparison_logic.md section 4 says every row of that table is recomputed here. The two rows
    # bands.py cannot cut -- noise removed and normalized timestep are neither the placement nor the
    # functional axis -- are cut directly from the stages table so the claim holds for the whole table.
    def thirds(values):
        return [sum(1 for v in values if v < bands.CUTS[0]),
                sum(1 for v in values if bands.CUTS[0] <= v < bands.CUTS[1]),
                sum(1 for v in values if v >= bands.CUTS[1])]

    noise_removed = [1.0 - r["sigma_in"] for r in dit_mapped]
    c.equal("rejected dit noise-removed split is 190/34/26 steps", thirds(noise_removed), [190, 34, 26])
    noise_bands = [[r["stage"] for r, v in zip(dit_mapped, noise_removed) if lo <= v < hi]
                   for lo, hi in ((0.0, bands.CUTS[0]), bands.CUTS, (bands.CUTS[1], 1.01))]
    c.equal("rejected dit noise-removed early band runs t 999..241",
            (noise_bands[0][0], noise_bands[0][-1]), (999, 241))
    c.equal("rejected dit noise-removed late band runs t 100..0",
            (noise_bands[2][0], noise_bands[2][-1]), (100, 0))

    normalized_timestep = [1.0 - r["stage"] / 999 for r in dit_mapped]
    c.equal("normalized timestep gives the same 83/83/84 split as the step index",
            thirds(normalized_timestep), [83, 83, 84])
    c.that("normalized timestep and the step index agree on every stage's band",
           all(bands.band_of(v) == bands.band_of(r["p_place"])
               for v, r in zip(normalized_timestep, dit_mapped)),
           "the two bases disagree somewhere, so they are not interchangeable after all")

    mid = next(r for r in dit_mapped if r["step"] == 124)
    c.close("dit index midpoint placement is 0.498", mid["p_place"], 0.498)
    c.close("dit index midpoint 1 - sigma_in is 0.039", 1 - mid["sigma_in"], 0.039)
    var_mid_tokens = sum(r["n_tokens"] for r in var_mapped if r["stage"] in bands.VAR_D20_BANDS["middle"])
    c.equal("var middle band holds 77 of 680 tokens", (var_mid_tokens, var_mapped[-1]["total_tokens"]), (77, 680))


# ---------------------------------------------------------------- U3: the shared plot template

def metric_rows(stages, names, rows=((207, 0), (207, 1)), start=1.0):
    """Synthetic per-image metric rows in the column shape of lanes/p2/SCHEMA.md section 4."""
    out = []
    for n, name in enumerate(names):
        for i, stage in enumerate(r["stage"] for r in stages):
            for j, (class_id, seed) in enumerate(rows):
                out.append({"class_id": class_id, "seed": seed, "metric": name,
                            "value": start + n * 10 + i * 0.5 + j * 0.01, "stage": stage})
    return out


def check_template(c: Checks):
    from lanes.p4 import plotting

    var_stages = fixtures.frozen_stages("var")
    var_run = fixtures.run_record("var", var_stages)
    specs = [plotting.MetricSpec("lpips", direction="lower_is_better"),
             plotting.MetricSpec("clip_sim", direction="higher_is_better")]
    tidy = plotting.tidy_from_run(var_run, metric_rows(var_stages, ["lpips", "clip_sim"]), specs,
                                 arm="baseline", lane="p4")

    c.equal("tidy row count is stages x metrics x images", len(tidy), 10 * 2 * 2)
    c.that("every tidy row carries both axes and a band",
           all(set(("p_place", "p_func", "band", "arm", "metric", "value")) <= set(r) for r in tidy))
    c.that("tidy columns are P2's plus band / p_func / p_place / direction / level",
           set(plotting.TIDY_COLUMNS) >= {"run_id", "lane", "model", "arm", "class_id", "seed", "stage",
                                          "metric", "value", "band", "p_place", "p_func",
                                          "direction", "level"})
    c.equal("band of si 4 is middle", {r["band"] for r in tidy if r["stage"] == 4}, {"middle"})

    # Metric names, labels and directions come from the input, never from a hardcoded list, so P3's
    # final metric definitions drop in without a template change (lanes/p4/PLOTTING.md section 3).
    spec = plotting.lane_plot_spec(tidy, title="two metrics")
    c.equal("one panel per input metric", [p["title"] for p in spec["panels"]], ["lpips", "clip_sim"])
    c.equal("the tidy table carries each metric's label",
            {r["metric"]: r["label"] for r in tidy}, {"lpips": "lpips", "clip_sim": "clip_sim"})
    labelled = plotting.tidy_from_run(
        var_run, metric_rows(var_stages, ["lpips"]),
        [plotting.MetricSpec("lpips", direction="lower_is_better", label="perceptual distance")],
        arm="baseline", lane="p4")
    c.that("a MetricSpec label reaches the axis rather than being dropped",
           "perceptual distance" in plotting.lane_plot_spec(labelled)["panels"][0]["y_label"],
           plotting.lane_plot_spec(labelled)["panels"][0]["y_label"])
    c.that("panel label carries the metric's own direction",
           "lower is better" in spec["panels"][0]["y_label"]
           and "higher is better" in spec["panels"][1]["y_label"],
           f"{[p['y_label'] for p in spec['panels']]}")
    invented = plotting.lane_plot_spec(
        plotting.tidy_from_run(var_run, metric_rows(var_stages, ["p3_semantic_drift_v2"]),
                              [plotting.MetricSpec("p3_semantic_drift_v2", level="per_image")],
                              arm="baseline", lane="p3"))
    c.equal("a metric name the template has never seen is used verbatim",
            [p["title"] for p in invented["panels"]], ["p3_semantic_drift_v2"])

    # Bands are drawn at their frozen extents, so an empty band does not shift the other two.
    partial = [r for r in tidy if r["band"] != "middle"]
    partial_spec = plotting.lane_plot_spec(partial)
    c.equal("an empty middle band keeps all three band extents",
            partial_spec["bands"], {"early": (0.0, bands.CUTS[0]), "middle": bands.CUTS,
                                    "late": (bands.CUTS[1], 1.0)})
    c.equal("the empty band is labelled as empty, not dropped",
            partial_spec["band_notes"]["middle"], "no stages")
    c.that("the two populated bands keep their extents",
           partial_spec["bands"]["early"] == spec["bands"]["early"]
           and partial_spec["bands"]["late"] == spec["bands"]["late"])

    # A mismatch between metric rows and the stages table is reported, not silently dropped.
    c.raises("a metric row naming a stage the run lacks is reported",
             lambda: plotting.tidy_from_run(var_run, [{"class_id": 207, "seed": 0, "metric": "lpips",
                                                      "value": 1.0, "stage": 42}],
                                            [plotting.MetricSpec("lpips")], arm="baseline"),
             "[42]")
    c.raises("a non-numeric metric value is reported",
             lambda: plotting.tidy_from_run(var_run, [{"class_id": 207, "seed": 0, "metric": "lpips",
                                                      "value": "n/a", "stage": 0}],
                                            [plotting.MetricSpec("lpips")], arm="baseline"),
             "not a number")
    c.raises("no matching metric row is reported",
             lambda: plotting.tidy_from_run(var_run, metric_rows(var_stages, ["lpips"]),
                                            [plotting.MetricSpec("fid")], arm="baseline"),
             "no metric row matched")
    c.raises("a per_set metric with several rows at one stage is reported",
             lambda: plotting.aggregate(plotting.tidy_from_run(
                 var_run, metric_rows(var_stages, ["fid"]),
                 [plotting.MetricSpec("fid", level="per_set")], arm="baseline")),
             "per_set")
    c.raises("an unknown direction is rejected",
             lambda: plotting.MetricSpec("x", direction="bigger"), "direction")
    c.raises("an unknown level is rejected", lambda: plotting.MetricSpec("x", level="per_run"), "level")

    # A final-image metric row (empty stage) still lands in a band, at the end of generation.
    final = plotting.tidy_from_run(var_run, [{"class_id": 207, "seed": 0, "metric": "fid",
                                             "value": 3.0, "stage": ""}],
                                  [plotting.MetricSpec("fid", level="per_set")], arm="baseline")
    c.equal("a final-image metric lands at the last stage", final[0]["stage"], 9)
    c.equal("a final-image metric lands in late", final[0]["band"], "late")

    # Both models go through the same entry point, and a 250-step axis keeps its ticks readable.
    dit_stages = fixtures.frozen_stages("dit")
    dit_tidy = plotting.tidy_from_run(fixtures.run_record("dit", dit_stages),
                                      metric_rows(dit_stages, ["lpips"], rows=((207, 0),)),
                                      [plotting.MetricSpec("lpips")], arm="baseline", lane="p4")
    c.equal("dit tidy covers all 250 stages", len({r["stage"] for r in dit_tidy}), 250)
    dit_spec = plotting.lane_plot_spec(dit_tidy)
    c.that("a 250-step axis is thinned to at most 12 ticks",
           len(dit_spec["panels"][0]["x_ticks"]) <= 12, f"{len(dit_spec['panels'][0]['x_ticks'])} ticks")
    c.that("dit x ticks are native timesteps", dit_spec["panels"][0]["x_ticks"][0][1] == "999")
    c.that("var x ticks are native scale indices", spec["panels"][0]["x_ticks"][0][1] == "0")

    # Two runs can share an arm label (P4's own pilot has a protect run per budget) and a VAR scale
    # index collides with a DiT step index. Averaging either into one point is a wrong figure, so it
    # must be refused rather than drawn.
    budget_a = plotting.tidy_from_run(var_run, metric_rows(var_stages, ["lpips"]),
                                      [plotting.MetricSpec("lpips")], arm="protect", lane="p4",
                                      run_id="p4/pilot/protect_m1")
    budget_b = plotting.tidy_from_run(var_run, metric_rows(var_stages, ["lpips"]),
                                      [plotting.MetricSpec("lpips")], arm="protect", lane="p4",
                                      run_id="p4/pilot/protect_m2")
    c.raises("two budgets sharing an arm label are refused, not averaged",
             lambda: plotting.aggregate(budget_a + budget_b), "same progress point", plotting.PlotError)
    var_rows = plotting.tidy_from_run(var_run, metric_rows(var_stages, ["lpips"]),
                                      [plotting.MetricSpec("lpips")], arm="baseline", lane="p4",
                                      run_id="var")
    dit_rows = plotting.tidy_from_run(fixtures.run_record("dit", fixtures.frozen_stages("dit")),
                                      metric_rows(fixtures.frozen_stages("dit"), ["lpips"],
                                                  rows=((207, 0),)),
                                      [plotting.MetricSpec("lpips")], arm="baseline", lane="p4",
                                      run_id="dit")
    c.raises("a VAR and a DiT run sharing an arm label are refused, not averaged",
             lambda: plotting.aggregate(var_rows + dit_rows), "same progress point", plotting.PlotError)

    # The render assertions below only mean something if a backend exists.
    c.that("at least one plotting backend is available to render with",
           bool(plotting.available_backends()),
           "neither matplotlib nor Pillow is importable, so every render assertion below is vacuous")

    # The round trip through CSV keeps the table usable.
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        path = plotting.write_tidy(tidy, Path(tmp) / "tidy.csv")
        back = plotting.read_tidy(path)
        c.equal("tidy table survives a CSV round trip", len(back), len(tidy))
        c.that("round-tripped rows still aggregate",
               plotting.aggregate(back).keys() == plotting.aggregate(tidy).keys())

        # The render layer: an explicitly named missing backend raises one actionable message.
        real = plotting.available_backends
        try:
            plotting.available_backends = lambda: []
            c.raises("render with no plotting library names matplotlib",
                     lambda: plotting.render(spec, Path(tmp) / "x.png", backend="matplotlib"), "matplotlib")
            c.raises("render with no plotting library names Pillow",
                     lambda: plotting.render(spec, Path(tmp) / "x.png", backend="pillow"), "Pillow")
            c.raises("backend='auto' with nothing installed says what to install",
                     lambda: plotting.render(spec, Path(tmp) / "x.png", backend="auto"), "matplotlib")
            try:
                plotting.render(spec, Path(tmp) / "x.png", backend="matplotlib")
            except plotting.PlotBackendMissing as e:
                c.that("the message is actionable, not an ImportError traceback",
                       "not installed" in str(e) and "pip install" in str(e))
            except Exception as e:   # noqa: BLE001
                c.that("the missing backend raises PlotBackendMissing", False, f"{type(e).__name__}: {e}")
            plotting.available_backends = lambda: ["pillow"]
            c.raises("asking for matplotlib when only Pillow is present names matplotlib",
                     lambda: plotting.render(spec, Path(tmp) / "x.png", backend="matplotlib"), "matplotlib")
        finally:
            plotting.available_backends = real
        c.raises("an unknown backend is rejected",
                 lambda: plotting.render(spec, Path(tmp) / "x.png", backend="gnuplot"), "gnuplot")

        # And it actually renders, through every backend this environment has.
        for backend in plotting.available_backends():
            out = plotting.lane_plot(tidy, Path(tmp) / f"var_{backend}.png", backend=backend,
                                     title="VAR d20, two metrics")
            c.that(f"var lane plot renders via {backend}", out.exists() and out.stat().st_size > 2000,
                   f"{out} is {out.stat().st_size if out.exists() else 'missing'} bytes")
            out = plotting.lane_plot(dit_tidy, Path(tmp) / f"dit_{backend}.png", backend=backend)
            c.that(f"dit lane plot renders via {backend} through the same entry point",
                   out.exists() and out.stat().st_size > 2000)
            out = plotting.lane_plot(partial, Path(tmp) / f"empty_band_{backend}.png", backend=backend)
            c.that(f"an empty band renders via {backend}", out.exists())


def check_figure5(c: Checks):
    from lanes.p4 import figure5, plotting

    def lane(metric, scale, shape, model="var", arm="baseline"):
        stages = fixtures.frozen_stages(model)
        rows = [{"class_id": 207, "seed": 0, "metric": metric,
                 "value": scale * shape(i / (len(stages) - 1)), "stage": st["stage"]}
                for i, st in enumerate(stages)]
        return plotting.tidy_from_run(fixtures.run_record(model, stages), rows,
                                      [plotting.MetricSpec(metric)], arm=arm, lane=metric[:2])

    rising, falling = (lambda t: t), (lambda t: 1 - t)
    p1 = lane("scale_importance", 1.0, rising)
    p2 = lane("corruption_recovery", 300.0, falling)
    p3 = lane("semantic_drift", 0.004, rising)
    p4 = lane("protect_gain", 12.0, falling, model="dit")

    four = figure5.figure5_spec([("P1 scales", p1), ("P2 corruption", p2),
                                 ("P3 semantics", p3), ("P4 protect", p4)])
    c.equal("four lanes give four panels", len(four["panels"]), 4)
    c.equal("each panel is labelled with its lane",
            [p["title"] for p in four["panels"]], ["P1 scales", "P2 corruption", "P3 semantics", "P4 protect"])
    c.that("each panel names its own metric",
           all(m in p["y_label"] for p, m in zip(four["panels"],
               ["scale_importance", "corruption_recovery", "semantic_drift", "protect_gain"])),
           f"{[p['y_label'] for p in four['panels']]}")
    c.equal("all four share one progress axis", four["bands"], dict(plotting.BAND_EXTENTS))
    c.that("panels keep their own actual range, stated on the label",
           [p["value_range"] for p in four["panels"]][1][1] == 300.0
           and [p["value_range"] for p in four["panels"]][0][1] == 1.0,
           f"{[p['value_range'] for p in four['panels']]}")

    # Disagreement is preserved: a rising lane stays rising, a falling lane stays falling.
    rising_pts = [v for _x, v in four["panels"][0]["series"][0]["points"]]
    falling_pts = [v for _x, v in four["panels"][1]["series"][0]["points"]]
    c.that("a rising lane stays rising after normalization",
           all(b >= a for a, b in zip(rising_pts, rising_pts[1:])), f"{rising_pts[:4]}")
    c.that("a falling lane stays falling after normalization",
           all(b <= a for a, b in zip(falling_pts, falling_pts[1:])), f"{falling_pts[:4]}")
    c.that("no lane is averaged or ranked into another",
           rising_pts != falling_pts and len({len(p["series"]) for p in four["panels"]}) == 1)

    # A lane with a very different value range does not rescale the others.
    alone = figure5.figure5_spec([("P1 scales", p1)])
    c.equal("a single lane renders as exactly one panel, with no empty padding", len(alone["panels"]), 1)
    c.equal("a lane's normalized points do not change when a 300x lane joins it",
            alone["panels"][0]["series"][0]["points"], four["panels"][0]["series"][0]["points"])
    c.equal("a lane's stated range does not change either",
            alone["panels"][0]["value_range"], four["panels"][0]["value_range"])

    # Mixed models: no single committed range is quoted, and the axis label stays honest.
    c.that("a mixed-model frame does not quote one model's committed range",
           all(note == "" for note in four["band_notes"].values()), f"{four['band_notes']}")
    one_model = figure5.figure5_spec([("P1 scales", p1), ("P3 semantics", p3)])
    c.that("a single-model frame does quote the committed range",
           one_model["band_notes"]["middle"].startswith("committed"), f"{one_model['band_notes']}")
    c.that("the native-stage tick label names the model when there is only one",
           "VAR scale si" in one_model["x_label"] and "native stage" in four["x_label"],
           f"{one_model['x_label']!r} / {four['x_label']!r}")

    # A lane may hand over several metrics; each must keep its own range and its own direction.
    mixed_stages = fixtures.frozen_stages("var")
    mixed = (plotting.tidy_from_run(fixtures.run_record("var", mixed_stages),
                                    metric_rows(mixed_stages, ["small_metric"], rows=((207, 0),),
                                                start=0.0),
                                    [plotting.MetricSpec("small_metric", direction="lower_is_better")],
                                    arm="a", lane="p9", run_id="p9/small")
             + plotting.tidy_from_run(fixtures.run_record("var", mixed_stages),
                                      metric_rows(mixed_stages, ["big_metric"], rows=((207, 0),),
                                                  start=300.0),
                                      [plotting.MetricSpec("big_metric", direction="higher_is_better")],
                                      arm="a", lane="p9", run_id="p9/big"))
    mixed_spec = figure5.figure5_spec([("P9 two metrics", mixed)])
    panel = mixed_spec["panels"][0]
    c.equal("a two-metric lane keeps a range per metric", sorted(panel["metric_ranges"]),
            ["big_metric", "small_metric"])
    c.that("the small metric is not squashed by the big one's range",
           max(v for s_ in panel["series"] if "small_metric" in s_["label"] for _x, v in s_["points"]) > 0.9,
           "the small metric was normalized against the big metric's range")
    c.equal("each metric keeps its own direction", panel["metric_directions"],
            {"small_metric": "lower_is_better", "big_metric": "higher_is_better"})
    c.that("the panel label states both directions",
           "lower is better" in panel["y_label"] and "higher is better" in panel["y_label"],
           panel["y_label"])

    c.raises("no lanes at all is reported", lambda: figure5.figure5_spec([]), "at least one",
             plotting.PlotError)
    c.raises("an empty lane table is reported",
             lambda: figure5.figure5_spec([("P1", [])]), "empty tidy table", plotting.PlotError)

    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        for backend in plotting.available_backends():
            out = figure5.figure5([("P1 scales", p1), ("P2 corruption", p2),
                                   ("P3 semantics", p3), ("P4 protect", p4)],
                                  Path(tmp) / f"f5_{backend}.png", backend=backend)
            c.that(f"four-lane frame renders via {backend}", out.exists() and out.stat().st_size > 3000)
            out = figure5.figure5([("P1 scales", p1)], Path(tmp) / f"f5_one_{backend}.png", backend=backend)
            c.that(f"one-lane frame renders via {backend} without collapsing", out.exists())


def check_data_layer_is_dependency_free(c: Checks):
    """Prove the data layer never imports a plotting library, by blocking both and doing the work."""
    import subprocess
    script = """
import sys, importlib.abc, importlib.machinery, tempfile
BLOCKED = ("matplotlib", "PIL")
class Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in BLOCKED:
            raise ImportError(f"blocked for this check: {name}")
        return None
sys.meta_path.insert(0, Blocker())
sys.path.insert(0, %r)
from lanes.p4 import fixtures, plotting
stages = fixtures.frozen_stages("var")
run = fixtures.run_record("var", stages)
rows = [{"class_id": 207, "seed": 0, "metric": "lpips", "value": 0.1 * i, "stage": s["stage"]}
        for i, s in enumerate(stages)]
tidy = plotting.tidy_from_run(run, rows, [plotting.MetricSpec("lpips")], arm="baseline", lane="p4")
spec = plotting.lane_plot_spec(tidy)
with tempfile.TemporaryDirectory() as tmp:
    plotting.write_tidy(tidy, tmp + "/t.csv")
    assert len(plotting.read_tidy(tmp + "/t.csv")) == len(tidy)
assert len(spec["panels"]) == 1 and len(spec["bands"]) == 3
from lanes.p4 import figure5
f5 = figure5.figure5_spec([("lane A", tidy), ("lane B", tidy)])
assert len(f5["panels"]) == 2
leaked = sorted(m for m in sys.modules if m.split(".")[0] in BLOCKED)
assert not leaked, f"data layer imported {leaked}"
print("DATA LAYER CLEAN")
""" % str(REPO)
    r = subprocess.run([sys.executable, "-c", script], cwd=REPO, capture_output=True, text=True,
                       env={"PYTHONDONTWRITEBYTECODE": "1", "PATH": "/usr/bin:/bin"})
    c.that("data layer builds a tidy table with matplotlib and Pillow blocked",
           r.returncode == 0 and "DATA LAYER CLEAN" in r.stdout,
           f"exit {r.returncode}: {(r.stderr or r.stdout).strip().splitlines()[-1:] }")


# ---------------------------------------------------------------- real run.json, when available

def check_real_runs(c: Checks, root: Path):
    """Assert the reference tables against whatever real run.json files are under `root`."""
    found = sorted(p for p in root.rglob("run.json"))
    c.that(f"found at least one run.json under {root}", bool(found), "none found")
    for run_json in found:
        run = progress.load_run(run_json)
        model = progress.model_of(run)
        mapped = progress.map_run(run)
        label = f"{run_json.parent.name} ({model}, {len(mapped)} stages)"
        c.equal(f"{label}: every recorded stage is mapped", len(mapped), len(run["stages"]))
        if run.get("skip_timesteps"):
            continue   # a skip run's table is checked against its own baseline, not rebuilt here
        c.equal(f"{label}: fixtures reproduce the runner's stages table bit-for-bit",
                run["stages"], fixtures.stages_for_settings(run, str(run_json))[1])


def check_skip_pairing(c: Checks, root: Path):
    """A --skip-timesteps run keeps the placement value of every timestep it still has."""
    pairs, baselines = [], {}
    for run_json in sorted(root.rglob("run.json")):
        run = progress.load_run(run_json)
        if progress.model_of(run) != "dit":
            continue
        if run.get("skip_timesteps"):
            pairs.append((run_json, run))
        else:
            baselines[run["sampler"]["num_sampling_steps"]] = progress.map_run(run)
    if not pairs:
        c.that("a --skip-timesteps run was available to check placement stability against", False,
               "no DiT skip run under --runs, so the kept-timestep invariant was not exercised")
        return
    for run_json, run in pairs:
        steps = run["sampler"]["num_sampling_steps"]
        base = baselines.get(steps)
        if base is None:
            c.that(f"{run_json.parent.name}: a baseline run exists to pair against", False,
                   f"no unskipped {steps}-step run under {root}")
            continue
        base_p = {r["stage"]: r["p_place"] for r in base}
        mapped = progress.map_run(run)
        skipped = set(run["skip_timesteps"])
        c.equal(f"{run_json.parent.name}: table is missing exactly the skipped timesteps",
                sorted(set(base_p) - {r['stage'] for r in mapped}), sorted(skipped))
        c.that(f"{run_json.parent.name}: kept timesteps keep their baseline placement value",
               all(r["p_place"] == base_p[r["stage"]] for r in mapped),
               "a kept step's p_place moved, so a skip run no longer lines up with its baseline")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, help="also check against every run.json under this directory")
    args = ap.parse_args(argv)

    print("== P4 shared deliverable self-check (CPU, no plotting library required)")
    sections = [("U1 progress mapping", check_progress), ("U2 frozen bands", check_bands),
                ("U3 shared plot template", check_template),
                ("U4 Figure 5 synthesis frame", check_figure5),
                ("U3/U4 data layer needs no plotting library", check_data_layer_is_dependency_free)]
    ok = True
    for title, fn in sections:
        c = Checks()
        try:
            fn(c)
        except Exception as e:   # noqa: BLE001 - a crash is a failure, reported like one
            c.that(f"{title} ran to completion", False, f"{type(e).__name__}: {e}")
        c.report(title)
        ok = ok and c.ok
    skipped = []
    if args.runs:
        c = Checks()
        try:
            check_real_runs(c, args.runs)
            check_skip_pairing(c, args.runs)
        except Exception as e:   # noqa: BLE001
            c.that("real-run checks ran to completion", False, f"{type(e).__name__}: {e}")
        c.report(f"real run.json under {args.runs}")
        ok = ok and c.ok
    else:
        skipped.append("real run.json comparison (no --runs)")
    # The pass line names what did not run, so a green line can never be read as more coverage than
    # the run actually had.
    note = f"  (NOT RUN: {'; '.join(skipped)})" if skipped else ""
    print(("ALL P4 SHARED CHECKS PASS" + note) if ok else "P4 SHARED CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
