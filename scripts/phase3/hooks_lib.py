"""Hooks used by scripts/phase3/check_hooks.py, loaded with runner.generate --hook scripts/phase3/hooks_lib.py:NAME.

Environment:
  PHASE3_LOG    observers append one JSON line per call here (hashes and shapes, never full tensors)
  PHASE3_STAGE  the stage (VAR si, DiT timestep_in) the targeted hooks are registered for; the
                targeted hooks cannot be loaded without it, so none of them silently fires everywhere
Every hook works for both models unless its name says otherwise.
"""

import dataclasses
import hashlib
import json
import os

import torch

from runner.hooks import Modify, Observe

TENSOR_FIELDS = ("h_BChw", "f_hat", "idx_Bl", "f_hat_before", "x", "x_before", "pred_xstart")


def _sha(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()[:16]


def _describe(t: torch.Tensor) -> dict:
    return {"shape": list(t.shape), "dtype": str(t.dtype).removeprefix("torch."), "sha": _sha(t),
            "sha_row0": _sha(t[:1]), "sha_rest": _sha(t[1:]), "absmax_row0": float(t[:1].abs().max())}


def record(model, stage, p, when, state):
    rec = {"model": model, "stage": stage, "p": p, "when": when, "fields": state.fields, "n_rows": len(state.rows),
           "tensors": {k: _describe(getattr(state, k)) for k in TENSOR_FIELDS
                       if getattr(state, k, None) is not None}}
    with open(os.environ["PHASE3_LOG"], "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")


def same(model, stage, p, when, state):
    return state


def _latent(model):
    return "f_hat" if model == "var" else "x"


def restore_row0(model, stage, p, when, state):
    """Destructive, row 0 only. VAR after(k): row 0 drops scale k (f_hat = f_hat_before). DiT after(j): x[0] = 0."""
    if model == "var":
        f = state.f_hat.clone()
        f[0] = state.f_hat_before[0]
        return dataclasses.replace(state, f_hat=f)
    x = state.x.clone()
    x[0] = 0
    return dataclasses.replace(state, x=x)


def zero_row0(model, stage, p, when, state):
    """before(k): zero row 0 of the latent (VAR f_hat, DiT x)."""
    name = _latent(model)
    t = getattr(state, name).clone()
    t[0] = 0
    return dataclasses.replace(state, **{name: t})


def restore_all(model, stage, p, when, state):
    """after(k), every row: VAR skip scale k (version b, f_hat = f_hat_before); DiT hold (x = x_before)."""
    if model == "var":
        return dataclasses.replace(state, f_hat=state.f_hat_before)
    return dataclasses.replace(state, x=state.x_before)


def boom(model, stage, p, when, state):
    raise RuntimeError("phase3 deliberate hook failure")


noop_before = Modify(same, when="before", name="noop_before")
noop_after = Modify(same, when="after", name="noop_after")
observe = [Observe(record, when="before", name="observe_before"), Observe(record, when="after", name="observe_after")]


def _stage() -> list[int]:
    v = os.environ.get("PHASE3_STAGE")
    if v is None:
        raise RuntimeError("PHASE3_STAGE is not set; targeted hooks need an explicit stage")
    return [int(v)]


_TARGETED = {   # name -> (kind, fn, when)
    "row0_after": (Modify, restore_row0, "after"),
    "zero_row0_before": (Modify, zero_row0, "before"),
    "restore_all_after": (Modify, restore_all, "after"),
    "raises_after": (Modify, boom, "after"),
    "observe_at_stage": (Observe, record, "after"),
}


def __getattr__(name):
    if name in _TARGETED:
        kind, fn, when = _TARGETED[name]
        return kind(fn, when=when, stages=_stage(), name=name)
    raise AttributeError(name)
