"""Hooks: functions the runner calls before and after every sampling stage (docs/hook_interface.md).

    def hook(model, stage, p, when, state): ...     # model "var"/"dit", when "before"/"after"
    model.sample(class_ids, seeds, hooks=[Observe(fn, when="after"), Modify(fn2, when="before", stages=[3])])

Observe: gets clones of the state, must return None. Modify: gets clones, must return a state of
the same class; its modifiable tensors must keep shape, dtype and device, and its read-only fields
must be unchanged. At one hook point, Modify hooks run in registration order, the runner writes the
result back, then Observe hooks see that final state. Hooks never receive a row's generator.
Exceptions raised by a hook propagate: the run fails.
"""

import dataclasses
from dataclasses import dataclass, field
from typing import Callable, ClassVar

import torch

WHENS = ("before", "after")


class HookError(RuntimeError):
    pass


@dataclass(frozen=True)
class Hook:
    """Register fn for one side of a stage. stages: native stage ids (VAR si, DiT timestep_in); None = all."""
    fn: Callable
    when: str = field(kw_only=True)
    stages: tuple[int, ...] | None = field(default=None, kw_only=True)
    name: str | None = field(default=None, kw_only=True)
    kind: ClassVar[str] = ""

    def __post_init__(self):
        if type(self) is Hook:
            raise TypeError("use Observe or Modify")
        if not callable(self.fn):
            raise HookError(f"hook fn is not callable: {self.fn!r}")
        if self.when not in WHENS:
            raise HookError(f"when must be one of {WHENS}, got {self.when!r}")
        if self.stages is not None:
            stages = tuple(self.stages)
            if not stages:
                raise HookError("stages is empty: the hook would never fire (use None for all stages)")
            if not all(isinstance(s, int) and not isinstance(s, bool) for s in stages):
                raise HookError(f"stages must be ints, got {stages}")
            object.__setattr__(self, "stages", tuple(sorted(set(stages))))
        if self.name is None:
            object.__setattr__(self, "name", getattr(self.fn, "__qualname__", None) or repr(self.fn))

    def fires_at(self, stage: int) -> bool:
        return self.stages is None or stage in self.stages

    def record(self) -> dict:
        return {"name": self.name, "kind": self.kind, "when": self.when,
                "stages": "all" if self.stages is None else list(self.stages)}


class Observe(Hook):
    kind = "observe"


class Modify(Hook):
    kind = "modify"


def check_hooks(hooks, stage_ids, model: str) -> tuple[Hook, ...]:
    """Validate before sampling: every entry is a Hook and every registered stage exists in this run."""
    hooks = tuple(hooks)
    valid = set(stage_ids)
    for h in hooks:
        if not isinstance(h, Hook):
            raise HookError(f"not an Observe/Modify hook: {h!r}")
        missing = [s for s in (h.stages or ()) if s not in valid]
        if missing:
            raise HookError(f"hook {h.name!r} ({h.when}) registered for {model} stages {missing}, "
                            f"which this run does not have; stages are {sorted(valid)}")
    return hooks


# ---------------------------------------------------------------- states

@dataclass
class _State:
    """Base: `rows` and `fields` are read-only; MODIFIABLE[when] names the tensors a Modify may replace."""
    rows: tuple                 # ((class_id, seed), ...) per batch row
    fields: dict                # per-stage fields (docs/hook_interface.md §3); identical for every row
    MODIFIABLE: ClassVar[dict] = {}

    def clone(self):
        return dataclasses.replace(self, fields=dict(self.fields), **{
            f.name: getattr(self, f.name).clone() for f in dataclasses.fields(self)
            if isinstance(getattr(self, f.name), torch.Tensor)})


@dataclass
class VARScaleState(_State):
    h_BChw: torch.Tensor = None       # this scale's embedding (B, Cvae, pn, pn)
    f_hat: torch.Tensor = None        # accumulator (B, Cvae, P, P); before: without this scale, after: with it
    idx_Bl: torch.Tensor = None       # this scale's sampled token ids (B, pn*pn), read-only
    f_hat_before: torch.Tensor = None  # after only: f_hat as passed to the upstream call, read-only
    MODIFIABLE: ClassVar[dict] = {"before": ("h_BChw", "f_hat"), "after": ("f_hat",)}


def _same_bits(a, b) -> bool:
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        if not (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)):
            return False
        if a.shape != b.shape or a.dtype != b.dtype or a.device != b.device:
            return False
        if a.is_floating_point():   # bitwise, so NaNs compare equal to themselves
            a, b = a.reshape(-1).view(torch.uint8), b.reshape(-1).view(torch.uint8)
        return torch.equal(a, b)
    return a == b


def _check_modified(hook: Hook, new, ref: _State, when: str):
    if type(new) is not type(ref):
        raise HookError(f"Modify hook {hook.name!r} must return a {type(ref).__name__}, got {type(new).__name__}")
    modifiable = ref.MODIFIABLE[when]
    for f in dataclasses.fields(ref):
        a, b = getattr(new, f.name), getattr(ref, f.name)
        if f.name in modifiable:
            if not isinstance(a, torch.Tensor) or (a.shape, a.dtype, a.device) != (b.shape, b.dtype, b.device):
                got = (tuple(a.shape), a.dtype, a.device) if isinstance(a, torch.Tensor) else type(a).__name__
                raise HookError(f"Modify hook {hook.name!r} ({when}): {f.name} must keep shape/dtype/device "
                                f"{(tuple(b.shape), b.dtype, b.device)}, got {got}")
        elif not _same_bits(a, b):
            raise HookError(f"Modify hook {hook.name!r} ({when}) changed read-only field {f.name!r}")


def run_point(hooks: tuple[Hook, ...], model: str, stage: int, p: float, when: str, state: _State) -> tuple[_State, bool]:
    """Run the hooks for one side of one stage. Returns (final state, whether any Modify hook ran).

    With no Modify hook, the returned state is `state` itself (the runner's own tensors).
    """
    here = [h for h in hooks if h.when == when and h.fires_at(stage)]
    modified = False
    for h in here:
        if isinstance(h, Modify):
            new = h.fn(model, stage, p, when, state.clone())
            _check_modified(h, new, state, when)
            state, modified = new, True
    for h in here:
        if isinstance(h, Observe):
            ret = h.fn(model, stage, p, when, state.clone())
            if ret is not None:
                raise HookError(f"Observe hook {h.name!r} returned {type(ret).__name__}; it must return None")
    return state, modified


def has_point(hooks: tuple[Hook, ...], when: str, stage: int) -> bool:
    return any(h.when == when and h.fires_at(stage) for h in hooks)


# ---------------------------------------------------------------- per-stage fields

def var_stage_table(patch_nums) -> list[dict]:
    """One dict per scale: stage (si), p, pn, n_tokens, cum_tokens, total_tokens."""
    sn = len(patch_nums)
    total = sum(pn * pn for pn in patch_nums)
    table, cum = [], 0
    for si, pn in enumerate(patch_nums):
        cum += pn * pn
        table.append({"stage": si, "p": si / (sn - 1) if sn > 1 else 1.0, "pn": pn, "n_tokens": pn * pn,
                      "cum_tokens": cum, "total_tokens": total})
    return table
