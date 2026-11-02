"""P4: the frozen early / middle / late bands, cut on the placement axis.

Run from the repo root. Standard library plus PyYAML (through lanes.p4.fixtures).

    python -m lanes.p4.bands            # print the frozen band lists for both models
    python -m lanes.p4.bands --full     # ... including every DiT timestep

Boundary convention: half-open `[lo, hi)` on the placement axis, cut at exactly 1/3 and 2/3 of the
progress value, with the final band closed at 1.0. A stage whose progress lands exactly on a cut
belongs to the upper band, and the last stage (p = 1.0) is always late.

The cuts come from the schedule alone -- the number of scales or steps and nothing else. No P1-P3
finding feeds into them, which is what keeps P4's pilot independent of the other lanes. Because
other lanes' figures are drawn against these lists, a change here invalidates comparisons already
made: treat it as a group-level decision, not a lane edit (lanes/p4/comparison_logic.md).
"""

import argparse
import sys

sys.dont_write_bytecode = True

BANDS = ("early", "middle", "late")
CUTS = (1 / 3, 2 / 3)       # half-open [lo, hi) on the placement axis; the final band closes at 1.0
PLACEMENT = "p_place"       # bands are cut on this axis; "p_func" is a reporting convention
                            # carried beside it (lanes/p4/comparison_logic.md section 2)

# Frozen for the two schedules every lane reports on. Asserted against the frozen configs by
# lanes/p4/check_p4_shared.py, so these are a published record, never the source of the split.
VAR_D20_BANDS = {"early": (0, 1, 2), "middle": (3, 4, 5), "late": (6, 7, 8, 9)}

# DiT's 250 native timesteps are three contiguous runs in sampling order, so they are frozen by step
# range plus timestep endpoints; `python -m lanes.p4.bands --full` prints every timestep.
DIT_250_BANDS = {
    "early":  {"steps": (0, 82),    "timesteps": (999, 670), "n": 83},
    "middle": {"steps": (83, 165),  "timesteps": (666, 337), "n": 83},
    "late":   {"steps": (166, 249), "timesteps": (333, 0),   "n": 84},
}


class BandError(ValueError):
    """A progress value or basis that cannot be banded."""


def band_of(p: float) -> str:
    """The band a progress value falls in, under the half-open convention above."""
    if not isinstance(p, (int, float)) or isinstance(p, bool):
        raise BandError(f"progress must be a number, got {p!r}")
    if not 0.0 <= p <= 1.0:
        raise BandError(f"progress {p!r} is outside [0, 1]; every supported basis is bounded")
    if p < CUTS[0]:
        return "early"
    if p < CUTS[1]:
        return "middle"
    return "late"


def assign_bands(mapped: list[dict], basis: str = PLACEMENT) -> list[dict]:
    """Add a "band" to every mapped stage (lanes/p4/progress.py).

    basis defaults to the placement axis, which is what every published band list and every arm's
    stage set is cut on. Passing "p_func" bands on the functional axis instead; that is used only to
    recompute the numbers that rejected the alternatives, never to build an arm or a figure axis.
    """
    if basis not in (PLACEMENT, "p_func"):
        raise BandError(f"basis must be {PLACEMENT!r} or 'p_func', got {basis!r}")
    out = []
    for i, row in enumerate(mapped):
        if basis not in row:
            raise BandError(f"mapped stage row {i} has no {basis!r}; map it with lanes.p4.progress first")
        out.append({**row, "band": band_of(row[basis])})
    return out


def frozen_band_stages(model: str) -> dict:
    """{band: (native stage, ...)} for the frozen config of `model`, derived from its own schedule."""
    from lanes.p4 import fixtures, progress
    mapped = assign_bands(progress.map_stages(model, fixtures.frozen_stages(model)))
    return {band: tuple(r["stage"] for r in mapped if r["band"] == band) for band in BANDS}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--full", action="store_true", help="print every native stage, not just the endpoints")
    args = ap.parse_args(argv)

    print(f"cuts at {CUTS[0]:.6f} and {CUTS[1]:.6f} of the placement axis, half-open [lo, hi), "
          "final band closed at 1.0")
    for model in ("var", "dit"):
        print(f"\n{model}:")
        for band, stages in frozen_band_stages(model).items():
            native = "si" if model == "var" else "t"
            head = f"  {band:<7} {len(stages):>3} stages   {native} {stages[0]}..{stages[-1]}"
            print(head if not args.full else f"{head}\n    {list(stages)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
