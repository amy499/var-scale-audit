"""P4: turn a run's `stages` table into both progress axes, keyed by the native stage.

Run from the repo root. Standard library only: no torch, no numpy, no GPU, no upstream import.

    python -m lanes.p4.progress <run dir or run.json>        # print the mapping table

Two axes travel with every stage (lanes/p4/comparison_logic.md):

    p_place   the placement axis. Bands are cut on this one (lanes/p4/bands.py).
              VAR si/(SN-1); DiT the `p` the stages table already records.
    p_func    the functional axis, a reporting convention carried beside p_place so a reader can
              see how much was actually committed at a stage the placement axis calls "middle".
              VAR cum_tokens/total_tokens; DiT sqrt(alpha_bar_in), the signal committed.

Every field either axis needs is already in `run.json`, so this module needs no change to
`runner/` and no other lane's output. Field names are exactly docs/hook_interface.md section 3.
"""

import json
import math
import sys
from pathlib import Path

sys.dont_write_bytecode = True

MODELS = ("var", "dit")

# What each axis is, per model, for the report and for comparison_logic.md.
PLACEMENT_BASIS = {"var": "scale index, si/(SN-1)", "dit": "step index, j/(S-1)"}
FUNCTIONAL_BASIS = {"var": "cum_tokens / total_tokens", "dit": "sqrt(alpha_bar_in)"}

# Fields read per model, beyond "stage". Listed so a missing one is named rather than a KeyError.
REQUIRED_FIELDS = {"var": ("cum_tokens", "total_tokens"), "dit": ("p", "alpha_bar_in")}

# p_place derived from the table's own length must agree with the recorded `p` this closely.
# Both are the same rational number in float; the tolerance only guards against a partial table.
P_TOLERANCE = 1e-9


class ProgressError(ValueError):
    """A stages table this module cannot map, with the reason named."""


def load_run(path) -> dict:
    """Read a run.json, given either the file or the run directory that holds it."""
    path = Path(path)
    run_json = path / "run.json" if path.is_dir() else path
    try:
        run = json.loads(run_json.read_text())
    except OSError as e:
        raise ProgressError(f"cannot read {run_json}: {e}") from e
    except ValueError as e:
        raise ProgressError(f"{run_json} is not valid JSON: {e}") from e
    if not isinstance(run, dict):
        raise ProgressError(f"{run_json} does not hold a run record (got {type(run).__name__})")
    return run


def model_of(run: dict) -> str:
    model = run.get("model")
    if model not in MODELS:
        raise ProgressError(f"run record has model={model!r}; expected one of {MODELS}")
    return model


def stages_of(run: dict) -> list[dict]:
    stages = run.get("stages")
    if not isinstance(stages, list) or not stages:
        raise ProgressError("run record has no non-empty 'stages' table")
    return stages


def _number(row: dict, field: str, model: str, where: str):
    if field not in row:
        raise ProgressError(f"{model} stages table {where} is missing the field {field!r}; "
                            f"this mapping needs {('stage',) + REQUIRED_FIELDS[model]} "
                            "(docs/hook_interface.md section 3)")
    value = row[field]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProgressError(f"{model} stages table {where} has {field}={value!r}, which is not a number")
    return value


def _index_progress(i: int, n: int) -> float:
    """i of n stages in sampling order. A single-stage run is p = 1.0, as the runner defines it."""
    return i / (n - 1) if n > 1 else 1.0


def map_stages(model: str, stages: list[dict]) -> list[dict]:
    """One dict per stage in sampling order: the native stage, both axes, and the row's own fields.

    VAR: every count comes from the table, which always covers every scale, so d16-d30 and the tiny
    models all map with no hardcoded scale count or token total.
    DiT: p_place is read from the recorded `p`, never derived from the table's length -- a
    --skip-timesteps run has fewer rows, and deriving it would shift the denominator and stop the
    remaining stages lining up with their baseline.
    """
    if model not in MODELS:
        raise ProgressError(f"model must be one of {MODELS}, got {model!r}")
    if not stages:
        raise ProgressError("stages table is empty")
    n = len(stages)
    mapped = []
    for i, row in enumerate(stages):
        where = f"row {i}"
        if not isinstance(row, dict):
            raise ProgressError(f"{model} stages table {where} is {type(row).__name__}, not a dict")
        stage = _number(row, "stage", model, where)
        if model == "var":
            p_place = _index_progress(i, n)
            recorded = row.get("p")
            if isinstance(recorded, (int, float)) and not isinstance(recorded, bool):
                if abs(recorded - p_place) > P_TOLERANCE:
                    raise ProgressError(
                        f"var stages table {where} records p={recorded!r} but its position gives "
                        f"{p_place!r} over {n} rows; a VAR table must cover every scale in order")
            total = _number(row, "total_tokens", model, where)
            if total <= 0:
                raise ProgressError(f"var stages table {where} has total_tokens={total!r}; must be > 0")
            p_func = _number(row, "cum_tokens", model, where) / total
        else:
            p_place = _number(row, "p", model, where)
            alpha_bar = _number(row, "alpha_bar_in", model, where)
            if alpha_bar < 0:
                raise ProgressError(f"dit stages table {where} has alpha_bar_in={alpha_bar!r}; must be >= 0")
            p_func = math.sqrt(alpha_bar)
        mapped.append({**row, "stage": int(stage), "p_place": float(p_place), "p_func": float(p_func)})
    _check_axes(model, mapped)
    return mapped


def map_run(run: dict) -> list[dict]:
    """map_stages for a loaded run record."""
    return map_stages(model_of(run), stages_of(run))


def _check_axes(model: str, mapped: list[dict]):
    """Both axes are bounded in [0, 1] and non-decreasing in sampling order (R5)."""
    for axis in ("p_place", "p_func"):
        values = [row[axis] for row in mapped]
        for i, v in enumerate(values):
            if not 0.0 <= v <= 1.0:
                raise ProgressError(f"{model} {axis} is {v!r} at row {i}; every basis is bounded in [0, 1]")
        for i in range(1, len(values)):
            if values[i] < values[i - 1]:
                raise ProgressError(f"{model} {axis} decreases at row {i}: "
                                    f"{values[i - 1]!r} -> {values[i]!r}; sampling order must be monotonic")


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 1:
        print(__doc__.strip(), file=sys.stderr)
        return 2
    run = load_run(argv[0])
    model = model_of(run)
    mapped = map_run(run)
    print(f"model {model}   {len(mapped)} stages   "
          f"placement = {PLACEMENT_BASIS[model]}   functional = {FUNCTIONAL_BASIS[model]}")
    print(f"{'stage':>6}  {'p_place':>8}  {'p_func':>8}")
    for row in mapped:
        print(f"{row['stage']:>6}  {row['p_place']:>8.3f}  {row['p_func']:>8.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
