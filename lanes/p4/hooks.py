"""P4 hooks: the VAR severity-scaled degradation, loaded by runner.generate --hook.

    P4_STAGES=3,4,5 P4_LAMBDA=0 python -m runner.generate --config configs/var_d20.yaml \
        --manifest manifest/provisional_4x4.csv --batch-size 16 \
        --out-dir outputs/p4/<exp>/damage --hook lanes/p4/hooks.py:var_degrade

Environment:
  P4_STAGES   comma-separated VAR scale indices (si) this arm intervenes on. Required: the targeted
              hooks cannot be built without it, so none of them silently fires everywhere.
  P4_LAMBDA   severity, default 0.0. **Gate B runs every VAR arm at 0.0** (lanes/p4/arms.py).
  P4_LOG      optional absolute path; `trace` appends one JSON line per call there.

At `after(si)` the hook returns

    f_hat = lambda * f_hat + (1 - lambda) * f_hat_before

keeping fraction `lambda` of that scale's contribution to the running accumulator. Both endpoints are
**structural**, not a float coincidence: at lambda = 1 the incoming state is returned untouched, and at
lambda = 0 the accumulator is restored to `f_hat_before`, which is exactly the removal
`scripts/phase3/hooks_lib.py:restore_all` performs and `docs/hook_interface.md` section 10b documents.

Rejected: zeroing `h_BChw` at `before(si)`. Section 10a shows it leaves a `r * bias_k` residue, so it
means "this scale contributed the zero embedding", not "this scale contributed less".

The hook keeps shape, dtype and device, never touches a read-only field, never consumes a random
number, and holds no reference to a tensor it returned.
"""

import dataclasses
import json
import os
import sys

sys.dont_write_bytecode = True

from lanes.p4.arms import ENV_LAMBDA, ENV_LOG, ENV_STAGES  # noqa: E402
from runner.hooks import Modify, Observe  # noqa: E402

DEFAULT_LAMBDA = 0.0


class HookConfigError(RuntimeError):
    """A P4 hook that cannot be built from the environment, with the variable named."""


def _stages() -> list[int]:
    raw = os.environ.get(ENV_STAGES)
    if raw is None:
        raise HookConfigError(f"{ENV_STAGES} is not set; a P4 arm hook needs an explicit stage list "
                              "(build it with lanes/p4/arms.py)")
    try:
        stages = [int(part) for part in raw.replace(" ", "").split(",") if part]
    except ValueError as e:
        raise HookConfigError(f"{ENV_STAGES}={raw!r} is not a comma-separated list of integers") from e
    if not stages:
        raise HookConfigError(f"{ENV_STAGES} is empty; the hook would never fire")
    return stages


def _lambda() -> float:
    raw = os.environ.get(ENV_LAMBDA, str(DEFAULT_LAMBDA))
    try:
        lam = float(raw)
    except ValueError as e:
        raise HookConfigError(f"{ENV_LAMBDA}={raw!r} is not a number") from e
    if not 0.0 <= lam <= 1.0:
        raise HookConfigError(f"{ENV_LAMBDA}={lam} is outside [0, 1]; 0 removes the scale's "
                              "contribution, 1 is the baseline")
    return lam


def severity_blend(lam: float):
    """The Modify function for severity `lam`, as a closure so the severity is recorded in the name."""

    def var_severity_blend(model, stage, p, when, state):
        if model != "var":
            raise HookConfigError(f"this hook is VAR only; the runner called it for {model!r}. "
                                  "DiT reduces with --skip-timesteps, not a hook "
                                  "(lanes/p4/comparison_logic.md section 5).")
        if when != "after":
            raise HookConfigError(f"this hook belongs at after(si); the runner called it at {when!r}")
        if state.f_hat_before is None:
            raise HookConfigError("f_hat_before is None; it exists only at the 'after' hook point")
        if lam == 1.0:
            return state                                        # baseline, exactly
        if lam == 0.0:
            return dataclasses.replace(state, f_hat=state.f_hat_before.clone())   # restore_all
        return dataclasses.replace(state, f_hat=lam * state.f_hat + (1.0 - lam) * state.f_hat_before)

    return var_severity_blend


def trace(model, stage, p, when, state):
    """Optional observer: one JSON line per call to P4_LOG. Never writes a tensor.

    Debug only. It reopens the log and forces a device sync on every call, so `trace_all` on DiT
    fires 250x per side per batch -- never register it on a run whose numbers are a result.
    """
    path = os.environ.get(ENV_LOG)
    if not path:
        raise HookConfigError(f"{ENV_LOG} is not set; the trace observer needs an absolute output path")
    latent = state.f_hat if model == "var" else state.x
    record = {"model": model, "stage": stage, "p": p, "when": when, "n_rows": len(state.rows),
              "fields": state.fields, "latent_absmax": float(latent.abs().max())}
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def _var_degrade() -> Modify:
    lam = _lambda()
    return Modify(severity_blend(lam), when="after", stages=_stages(), name=f"p4_var_degrade_lambda{lam:g}")


_BUILD = {
    # name -> how to build it when runner.generate asks for it by name
    "var_degrade": _var_degrade,
    "trace_after": lambda: Observe(trace, when="after", stages=_stages(), name="p4_trace_after"),
    "trace_all": lambda: Observe(trace, when="after", name="p4_trace_all"),
}


def __getattr__(name):
    if name in _BUILD:
        return _BUILD[name]()
    raise AttributeError(name)
