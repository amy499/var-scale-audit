"""P1 stage removal, VAR variant (b): drop the listed scales' contribution to the accumulator.

    P1_VAR_SCALES="3 4 5" python -m runner.generate --config configs/var_d20.yaml ... \
        --hook lanes/p1/hooks.py:remove_scales

After scale si is added, f_hat is set back to f_hat_before, so neither later scales (their input is
read from f_hat) nor the decoder see that scale. Its tokens are still sampled, so every row draws the
same random numbers as the baseline and the run stays paired (docs/hook_interface.md §7).
Same operation as scripts/examples/skip_scale_2.py, for a list of scales taken from the environment.

P1_VAR_SCALES: space-separated VAR scale indices (required). The runner checks they exist in the run.
"""

import dataclasses
import os

from runner.hooks import HookError, Modify


def _scales() -> list[int]:
    raw = os.environ.get("P1_VAR_SCALES", "").split()
    if not raw:
        raise ValueError("P1_VAR_SCALES is not set: give the VAR scales to remove, e.g. P1_VAR_SCALES=\"3 4 5\"")
    try:
        return sorted({int(s) for s in raw})
    except ValueError:
        raise ValueError(f"P1_VAR_SCALES must be space-separated ints, got {' '.join(raw)!r}") from None


def drop_scale(model, stage, p, when, state):
    if model != "var":
        raise HookError(f"lanes/p1/hooks.py:remove_scales is VAR only, got model {model!r}")
    return dataclasses.replace(state, f_hat=state.f_hat_before)


REMOVE = _scales()
remove_scales = Modify(drop_scale, when="after", stages=REMOVE,
                       name="p1_remove_scales_" + "_".join(map(str, REMOVE)))
