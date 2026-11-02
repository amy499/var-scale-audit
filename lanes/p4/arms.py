"""P4: which native stages each arm intervenes on, at one shared reduction budget.

Run from the repo root. Standard library plus PyYAML (through lanes.p4.fixtures). No torch, no GPU.

    python -m lanes.p4.arms --model var --band middle --budget 3
    python -m lanes.p4.arms --model dit --band middle --budget 25

For a candidate band `B` of size `k` out of `n` native stages, at budget `m`, all three intervened
arms cut exactly `m` stages and differ only in **where** (lanes/p4/comparison_logic.md section 3):

    baseline   nothing                     m = 0
    protect    m stages outside B          spend the cut away from the candidate band
    damage     m stages inside B           spend the cut on it
    control    m stages across all n       is B special, or do any m stages behave alike?

**Equal budget is not automatically equal severity on DiT.** Consecutive skipped timesteps merge into
one long transition, so a packed selection and a spread one differ sharply at the same `m`. Damage is
the constrained arm -- it must fit inside `B` -- so its window structure is computed first, at an even
stride within `B`, and protect and control then place that *same multiset of window lengths* in their
own eligible stages. The three arms therefore match in window structure as well as in count, which is
what makes the contrast attributable to placement.

`m` cannot exceed `B`'s eligible-stage count: `k`, except for DiT's early band, where the unskippable
first step (`runner/dit_model.py` rejects skipping it) leaves `k - 1`. So the swept reduction is capped
at `k / n` -- 0.3 for VAR-d20, 0.332 for the 250-step DiT schedule.
"""

import argparse
import random
import sys

sys.dont_write_bytecode = True

from pathlib import Path  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from lanes.p4 import bands, fixtures, progress  # noqa: E402

BASELINE = "baseline"
ARMS = (BASELINE, "protect", "damage", "control")
INTERVENED_ARMS = ("protect", "damage", "control")

# Gate B runs every VAR arm at lambda = 0: the only severity with a tested precedent in the repo
# (scripts/phase3/hooks_lib.py restore_all) and the only one whose intent maps onto a removed DiT step.
VAR_GATE_B_LAMBDA = 0.0
CONTROL_SEED = 20261006        # fixed, recorded, and reported with every control arm
# The environment an arm's VAR hook reads (lanes/p4/hooks.py). Named here so the pilot driver can set
# them without importing the hook module, which would pull torch into a process that only shells out.
ENV_STAGES, ENV_LAMBDA, ENV_LOG = "P4_STAGES", "P4_LAMBDA", "P4_LOG"
ARM_ENV = (ENV_STAGES, ENV_LAMBDA, ENV_LOG)
REDRAW_ATTEMPTS = 64           # successive seeds tried before a control arm is called degenerate
FRACTIONS = (0.1, 0.2, 0.3)    # the swept reduction budgets, as a fraction of all native stages


class ArmError(ValueError):
    """An arm that cannot be built, with the numbers that make it impossible named."""


def _band_context(model: str, stages, band: str) -> tuple[list[int], set]:
    """(native stage ids in sampling order, the stages of `band`) from one mapping of the table."""
    mapped = bands.assign_bands(progress.map_stages(model, stages))
    return [row["stage"] for row in mapped], {row["stage"] for row in mapped if row["band"] == band}


def first_step_excluded(model: str, ordered: list[int]) -> tuple[int, ...]:
    """Stages no arm may ever intervene on. DiT's first step cannot be skipped; VAR has none."""
    return (ordered[0],) if model == "dit" else ()


def eligible(model: str, ordered: list[int], inside: set | None = None,
             outside: set | None = None) -> list[int]:
    """Stages an arm may cut, in sampling order, with the never-eligible ones removed."""
    blocked = set(first_step_excluded(model, ordered))
    out = []
    for stage in ordered:
        if stage in blocked:
            continue
        if inside is not None and stage not in inside:
            continue
        if outside is not None and stage in outside:
            continue
        out.append(stage)
    return out


def _stride_pick(items: list, m: int) -> list:
    """m of `items` at an even stride, endpoints included -- upstream's respacing rule (respace.py)."""
    if m > len(items):
        raise ArmError(f"cannot pick {m} of {len(items)}")
    if m <= 0:
        return []
    if m == 1:
        return [items[len(items) // 2]]       # a single cut goes to the middle of its eligible set
    stride = (len(items) - 1) / (m - 1)
    return [items[round(k * stride)] for k in range(m)]


def _runs(positions: list[int]) -> list[tuple[int, int]]:
    """Maximal runs of consecutive positions as (start, length). Consecutive = merging on DiT."""
    out = []
    for p in sorted(positions):
        if out and p == out[-1][0] + out[-1][1]:
            out[-1] = (out[-1][0], out[-1][1] + 1)
        else:
            out.append((p, 1))
    return out


def window_lengths(positions: list[int]) -> tuple[int, ...]:
    """The merged-window lengths a selection produces, longest first."""
    return tuple(sorted((length for _start, length in _runs(positions)), reverse=True))


def _random_gaps(rng: random.Random, spare: int, bins: int) -> list[int]:
    """Split `spare` free positions into `bins` ordered non-negative gaps, uniformly (stars and bars)."""
    if bins <= 1:
        return [spare]
    if spare == 0:
        return [0] * bins
    cuts = sorted(rng.sample(range(spare + bins - 1), bins - 1))
    gaps, prev = [], -1
    for c in cuts:
        gaps.append(c - prev - 1)
        prev = c
    gaps.append(spare + bins - 2 - prev)
    return gaps


def _place(runs: list[tuple[int, int]], lengths: tuple[int, ...],
           rng: random.Random | None = None) -> list[int]:
    """Place blocks of `lengths` inside `runs` of eligible positions.

    Without `rng` the blocks are spread evenly within each run, which is what protect and damage use.
    With `rng` the free positions are distributed at random, which is what makes the control arm a
    genuinely uniform draw across its eligible set rather than another evenly spread one -- at small
    `n` an even spread can land exactly where protect landed, which would leave the control arm
    testing nothing.

    Blocks never straddle two runs, because two positions either side of a gap are not consecutive
    steps and would not merge into one transition.
    """
    if lengths and not runs:
        raise ArmError(f"cannot place {len(lengths)} window(s) {tuple(lengths)}: this arm has no "
                       "eligible stages at all")
    remaining = [length for _start, length in runs]
    assigned: list[list[int]] = [[] for _ in runs]
    for w in sorted(lengths, reverse=True):
        best = max(range(len(runs)), key=lambda i: (remaining[i], -i))
        if remaining[best] < w:
            raise ArmError(f"cannot place a window of {w} consecutive stages: the largest run of "
                           f"eligible stages still free holds {remaining[best]}")
        assigned[best].append(w)
        remaining[best] -= w
    out = []
    for (start, length), widths in zip(runs, assigned):
        if not widths:
            continue
        widths = sorted(widths)
        q = len(widths)
        # One empty position between blocks, so two placed blocks never merge into a longer window
        # than the structure being reproduced.
        free = length - sum(widths) - (q - 1)
        if free < 0:
            raise ArmError(f"cannot keep {q} windows separate inside {length} eligible stages: "
                           f"they need {sum(widths) + q - 1} positions")
        gaps = (_random_gaps(rng, free, q + 1) if rng is not None else
                [round((i + 1) * free / (q + 1)) - round(i * free / (q + 1)) for i in range(q + 1)])
        cursor = start
        for i, w in enumerate(widths):
            cursor += gaps[i] + (1 if i else 0)
            out.extend(range(cursor, cursor + w))
            cursor += w
    return sorted(out)


def arm_plan(model: str, stages, band: str, m: int, *, seed: int = CONTROL_SEED,
             lam: float = VAR_GATE_B_LAMBDA) -> dict:
    """Every arm's stage set at one budget. Returns {arm: record}; see `arm_record` for the fields."""
    if band not in bands.BANDS:
        raise ArmError(f"band must be one of {bands.BANDS}, got {band!r}")
    if not isinstance(m, int) or isinstance(m, bool) or m < 0:
        raise ArmError(f"budget m must be a non-negative integer, got {m!r}")
    if model == "var" and lam != VAR_GATE_B_LAMBDA:
        raise ArmError(f"Gate B runs every VAR arm at lambda = {VAR_GATE_B_LAMBDA}; got {lam!r}. "
                       "Interior severities belong to the deferred severity sweep "
                       "(lanes/p4/comparison_logic.md section 3).")

    ordered, in_band = _band_context(model, stages, band)
    n, k = len(ordered), len(in_band)
    position = {stage: i for i, stage in enumerate(ordered)}

    damage_eligible = eligible(model, ordered, inside=in_band)
    if m > len(damage_eligible):
        blocked = first_step_excluded(model, ordered)
        extra = (f" ({k} stages in the band, minus the unskippable first step {blocked[0]})"
                 if blocked and blocked[0] in in_band else f" ({k} stages in the band)")
        raise ArmError(f"budget m = {m} exceeds the {band} band's eligible-stage count "
                       f"{len(damage_eligible)}{extra}; damage must fit inside the band, so "
                       f"m <= {len(damage_eligible)} and the intervened fraction is capped at "
                       f"{len(damage_eligible) / n:.3f}")

    # Damage is the constrained arm, so its window structure is computed first and reused.
    damage = _stride_pick(damage_eligible, m)
    lengths = window_lengths([position[s] for s in damage])

    protect_eligible = eligible(model, ordered, outside=in_band)
    control_eligible = eligible(model, ordered)
    protect = [ordered[p] for p in _place(_runs([position[s] for s in protect_eligible]), lengths)]

    # The control must not land on exactly the protect or damage set, or it tests nothing. Redraw from
    # successive seeds until it separates; record the seed actually used so the arm is reproducible.
    control, used_seed, degenerate = None, seed, False
    for attempt in range(REDRAW_ATTEMPTS):
        candidate_seed = seed + attempt
        candidate = [ordered[p] for p in _place(_runs([position[s] for s in control_eligible]),
                                                lengths, rng=random.Random(candidate_seed))]
        if sorted(candidate) not in (sorted(protect), sorted(damage)):
            control, used_seed = candidate, candidate_seed
            break
    if control is None:        # only reachable when the eligible set is too small to separate at all
        control, used_seed, degenerate = candidate, candidate_seed, True

    for name, selected in (("protect", protect), ("control", control)):
        got = window_lengths([position[st] for st in selected])
        if got != lengths:
            raise ArmError(f"{name} arm reproduces window lengths {got}, not damage's {lengths}; "
                           "equal budget would not mean equal window structure")

    sets = {"baseline": [], "damage": damage, "protect": protect, "control": control}
    out = {arm: arm_record(model, arm, band, sets[arm], ordered, k, lam, used_seed) for arm in ARMS}
    out["control"]["degenerate"] = degenerate
    return out


def arm_record(model: str, arm: str, band: str, stages_selected, ordered, k: int,
               lam: float, seed: int) -> dict:
    n = len(ordered)
    position = {stage: i for i, stage in enumerate(ordered)}
    selected = sorted(stages_selected, key=lambda s: position[s])
    positions = [position[s] for s in selected]
    lengths = window_lengths(positions)
    return {"model": model, "arm": arm, "band": band, "m": len(selected), "n": n, "k": k,
            "fraction": len(selected) / n, "stages": tuple(selected), "positions": tuple(positions),
            "windows": lengths, "n_windows": len(lengths),
            "severity_lambda": lam if model == "var" else None,
            "control_seed": seed if arm == "control" else None}


def arm_stages(model: str, stages, band: str, m: int, arm: str, **kw) -> dict:
    if arm not in ARMS:
        raise ArmError(f"arm must be one of {ARMS}, got {arm!r}")
    return arm_plan(model, stages, band, m, **kw)[arm]


def budget_grid(model: str, stages, band: str, fractions=FRACTIONS) -> list[int]:
    """The budgets a fraction sweep asks for, rounded to whole stages and capped at the band."""
    ordered, in_band = _band_context(model, stages, band)
    ceiling = len(eligible(model, ordered, inside=in_band))
    out = []
    for f in fractions:
        m = max(1, min(ceiling, round(f * len(ordered))))
        if m not in out:
            out.append(m)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=("var", "dit"))
    ap.add_argument("--band", default="middle", choices=bands.BANDS)
    ap.add_argument("--budget", type=int, help="m; default: the 0.1/0.2/0.3 sweep")
    ap.add_argument("--seed", type=int, default=CONTROL_SEED)
    ap.add_argument("--stages", action="store_true", help="print each arm's full stage list")
    args = ap.parse_args(argv)

    stages = fixtures.frozen_stages(args.model)
    budgets = [args.budget] if args.budget is not None else budget_grid(args.model, stages, args.band)
    for m in budgets:
        plan = arm_plan(args.model, stages, args.band, m, seed=args.seed)
        head = plan["damage"]
        print(f"\n{args.model} {args.band} band: m = {m} of n = {head['n']} "
              f"(fraction {head['fraction']:.3f}, band holds k = {head['k']})")
        for arm in ARMS:
            r = plan[arm]
            line = (f"  {arm:<9} {r['m']:>3} stages  {r['n_windows']:>3} windows  "
                    f"lengths {list(r['windows'])[:6]}")
            print(line if not args.stages else f"{line}\n    {list(r['stages'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
