"""P4 self-check for the shared deliverable: the progress mapping, the frozen bands and the template.

Run from the repo root. Standard library plus PyYAML. No torch, no GPU, no checkpoint, and
deliberately **no plotting library** -- the data layer must work without one (R11).

    python lanes/p4/check_p4_shared.py
    python lanes/p4/check_p4_shared.py --runs DIR     # also check against real run.json dirs under DIR

Ends in one line: ALL P4 SHARED CHECKS PASS, or P4 SHARED CHECKS FAILED with the failing invariants
named. Exit code 0 only when every assertion holds. This is the only gate between a wrong band list
and GPU time spent on it, so it asserts the band lists against the frozen configs, not against
constants copied from the plan.
"""

import argparse
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from lanes.p4 import fixtures, progress  # noqa: E402

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

    def raises(self, name: str, fn, needle: str):
        try:
            fn()
        except Exception as e:   # noqa: BLE001 - any exception type, the message is what is asserted
            return self.that(name, needle in str(e), f"{type(e).__name__}({e}) does not mention {needle!r}")
        return self.that(name, False, "no exception raised")

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

    # Generality (R4): no hardcoded 10-scale or 680-token constant, and a schedule that is not 250 steps.
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

    # Monotonic non-decreasing in sampling order, on every fixture (R5).
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
    c.raises("missing var cum_tokens names the field",
             lambda: progress.map_stages("var", [{"stage": 0, "total_tokens": 30}]), "cum_tokens")
    c.raises("missing dit alpha_bar_in names the field",
             lambda: progress.map_stages("dit", [{"stage": 999, "p": 0.0}]), "alpha_bar_in")
    c.raises("missing stage names the field", lambda: progress.map_stages("var", [{"cum_tokens": 1}]), "stage")
    c.raises("empty stages table is rejected", lambda: progress.map_stages("var", []), "empty")
    c.raises("unknown model is rejected", lambda: progress.map_stages("vqgan", []), "vqgan")
    c.that("ProgressError is a ValueError", issubclass(progress.ProgressError, ValueError))

    # A VAR table that does not cover every scale is rejected rather than silently mis-mapped.
    partial = fixtures.frozen_stages("var")[:5]
    c.raises("partial var table is rejected", lambda: progress.map_stages("var", partial), "every scale")


# ---------------------------------------------------------------- U2: frozen bands

def check_bands(c: Checks):
    from lanes.p4 import bands

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
    c.raises("p outside [0, 1] is rejected", lambda: bands.band_of(1.5), "1.5")

    # A tiny model still produces three non-empty bands.
    tiny = bands.assign_bands(progress.map_stages("var", fixtures.var_stages(TINY_PATCH_NUMS)))
    tiny_by_band = {b: [r["stage"] for r in tiny if r["band"] == b] for b in bands.BANDS}
    c.that("tiny var has three non-empty bands", all(tiny_by_band[b] for b in bands.BANDS),
           f"{tiny_by_band}")
    tiny_dit = bands.assign_bands(progress.map_stages("dit", fixtures.dit_stages(TINY_DIT_STEPS)))
    c.that("tiny dit has three non-empty bands",
           all(any(r["band"] == b for r in tiny_dit) for b in bands.BANDS))

    # The numbers that rejected the alternatives (R17), recomputed rather than quoted.
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
    mid = next(r for r in dit_mapped if r["step"] == 124)
    c.close("dit index midpoint placement is 0.498", mid["p_place"], 0.498)
    c.close("dit index midpoint 1 - sigma_in is 0.039", 1 - mid["sigma_in"], 0.039)
    var_mid_tokens = sum(r["n_tokens"] for r in var_mapped if r["stage"] in bands.VAR_D20_BANDS["middle"])
    c.equal("var middle band holds 77 of 680 tokens", (var_mid_tokens, var_mapped[-1]["total_tokens"]), (77, 680))


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
        c.that(f"{label}: maps without error", True)
        if run.get("skip_timesteps"):
            continue   # a skip run's table is checked against its own baseline, not rebuilt here
        if model == "var":
            reference = fixtures.var_stages(run["build"]["patch_nums"])
        else:
            reference = fixtures.dit_stages(run["sampler"]["num_sampling_steps"])
        c.equal(f"{label}: fixtures reproduce the runner's stages table bit-for-bit",
                run["stages"], reference)


def check_skip_pairing(c: Checks, root: Path):
    """A --skip-timesteps run keeps the placement value of every timestep it still has (R4)."""
    pairs = []
    for run_json in sorted(root.rglob("run.json")):
        run = progress.load_run(run_json)
        if progress.model_of(run) == "dit" and run.get("skip_timesteps"):
            pairs.append((run_json, run))
    if not pairs:
        return
    baselines = {}
    for run_json in sorted(root.rglob("run.json")):
        run = progress.load_run(run_json)
        if progress.model_of(run) == "dit" and not run.get("skip_timesteps"):
            baselines[run["sampler"]["num_sampling_steps"]] = progress.map_run(run)
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
    sections = [("U1 progress mapping", check_progress), ("U2 frozen bands", check_bands)]
    ok = True
    for title, fn in sections:
        c = Checks()
        try:
            fn(c)
        except Exception as e:   # noqa: BLE001 - a crash is a failure, reported like one
            c.that(f"{title} ran to completion", False, f"{type(e).__name__}: {e}")
        c.report(title)
        ok = ok and c.ok
    if args.runs:
        c = Checks()
        try:
            check_real_runs(c, args.runs)
            check_skip_pairing(c, args.runs)
        except Exception as e:   # noqa: BLE001
            c.that("real-run checks ran to completion", False, f"{type(e).__name__}: {e}")
        c.report(f"real run.json under {args.runs}")
        ok = ok and c.ok
    print("ALL P4 SHARED CHECKS PASS" if ok else "P4 SHARED CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
