"""P2 hooks: noise corruption at chosen stages, and state traces for recovery curves (lanes/p2/README.md).

Load from the repo root with  --hook lanes/p2/hooks.py:NAME

Corruption (Modify). Gaussian noise is added to one tensor at the stages in P2_STAGE:

    NAME        model  when    tensor   meaning
    noise_h     VAR    before  h_BChw   corrupt scale k's own contribution, at its own resolution (pn x pn)
    noise_fhat  VAR    after   f_hat    corrupt the accumulator after scale k was added (16 x 16)
    noise_x     DiT    after   x        corrupt the latent leaving the step at timestep t

    P2_STAGE     required  native stage id(s), comma separated: VAR scale 0..9, DiT timestep_in (run.json "stages")
    P2_SEVERITY  required  noise strength s >= 0
    P2_SCALE     rel (default): noise std = s x that image's std of the tensor, the same rule for both models
                 abs: noise std = s
    P2_NOISE_SEED  default 0; a different value gives a different, independent noise draw

The noise for an image depends only on (class_id, seed, stage, tensor, P2_NOISE_SEED): not on the batch
it is in, not on the device, and not on the severity, so a severity sweep scales one fixed noise
pattern. It never touches the images' own generators, so runs stay paired with the baseline.

Traces (Observe, after every traced stage; VAR: f_hat, DiT: x and pred_xstart):

    save_states     save the state per batch to P2_TRACE_DIR (use on the baseline)
    compare_states  compare the state with the one a baseline run saved in P2_TRACE_REF and append, per
                    image, state_l2:<tensor> = |run - base|, state_rel_l2:<tensor> = |run - base| / |base|
                    and state_cos:<tensor> to P2_METRICS (a metrics.csv in the format of lanes/p2/schema.py)

    P2_TRACE_EVERY  default 1; trace every n-th stage (DiT: use e.g. 10). The last stage is always traced.

Both runs must use the same manifest and batch size (as every paired comparison does).
"""

import csv
import dataclasses
import hashlib
import os
from pathlib import Path

import torch

from runner.hooks import Modify, Observe

TRACED = {"var": ("f_hat",), "dit": ("x", "pred_xstart")}
METRIC_COLUMNS = ("class_id", "seed", "metric", "value", "stage")   # same as lanes/p2/schema.py


def _env(name: str, cast=str, default=None):
    v = os.environ.get(name)
    if v is None or v == "":
        if default is None:
            raise RuntimeError(f"{name} is not set; lanes/p2/hooks.py needs it for this hook")
        return default
    return cast(v)


# ---------------------------------------------------------------- corruption

def noise_like(t: torch.Tensor, rows, stage: int, tag: str, noise_seed: int = 0) -> torch.Tensor:
    """Standard normal noise shaped like t, one independent stream per image.

    Drawn on the CPU from a generator of our own, seeded from (tag, class_id, seed, stage, noise_seed),
    so an image's noise is the same in any batch and on any device.
    """
    out = []
    for class_id, seed in rows:
        key = f"p2|{tag}|{class_id}|{seed}|{stage}|{noise_seed}".encode()
        g = torch.Generator().manual_seed(int.from_bytes(hashlib.sha256(key).digest()[:8], "big") >> 1)
        out.append(torch.randn(tuple(t.shape[1:]), generator=g, dtype=torch.float32))
    return torch.stack(out).to(device=t.device, dtype=t.dtype)


def _add_noise(field: str, severity: float, scale: str, noise_seed: int):
    def corrupt(model, stage, p, when, state):
        t = getattr(state, field, None)
        if t is None:
            raise RuntimeError(f"p2 noise on {field!r}: a {model} state at {when!r} has no such tensor "
                               "(noise_h / noise_fhat are VAR hooks, noise_x is the DiT hook)")
        amp = severity
        if scale == "rel":   # per image, so it does not depend on the rest of the batch
            amp = severity * t.float().flatten(1).std(dim=1).to(t.dtype).view(-1, *[1] * (t.dim() - 1))
        return dataclasses.replace(state, **{field: t + amp * noise_like(t, state.rows, stage, field, noise_seed)})
    return corrupt


_CORRUPTIONS = {"noise_h": ("h_BChw", "before"), "noise_fhat": ("f_hat", "after"), "noise_x": ("x", "after")}


def _corruption(name: str) -> Modify:
    field, when = _CORRUPTIONS[name]
    stages = sorted({int(s) for s in _env("P2_STAGE").split(",")})
    severity = _env("P2_SEVERITY", float)
    scale = _env("P2_SCALE", default="rel")
    noise_seed = _env("P2_NOISE_SEED", int, default=0)
    if severity < 0 or scale not in ("rel", "abs"):
        raise RuntimeError(f"P2_SEVERITY must be >= 0 and P2_SCALE rel or abs; got {severity}, {scale!r}")
    label = f"p2_{name}_s{'+'.join(map(str, stages))}_sev{severity:g}_{scale}" + (f"_n{noise_seed}" if noise_seed else "")
    return Modify(_add_noise(field, severity, scale, noise_seed), when=when, stages=stages, name=label)


# ---------------------------------------------------------------- traces

def _is_traced(model: str, p: float, state) -> bool:
    index = state.fields["step"] if model == "dit" else state.fields["stage"]
    return p == 1.0 or index % _env("P2_TRACE_EVERY", int, default=1) == 0


def _trace_file(root: Path, model: str, stage: int, rows) -> Path:
    class_id, seed = rows[0]   # one file per batch, named by its first image
    return root / f"{model}_stage{stage:04d}_class{class_id:04d}_seed{seed}.pt"


def _save_states(model, stage, p, when, state):
    if not _is_traced(model, p, state):
        return
    path = _trace_file(Path(_env("P2_TRACE_DIR")), model, stage, state.rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"rows": [list(r) for r in state.rows],
                **{k: getattr(state, k).detach().float().cpu() for k in TRACED[model]}}, path)


def _compare_states(model, stage, p, when, state):
    if not _is_traced(model, p, state):
        return
    path = _trace_file(Path(_env("P2_TRACE_REF")), model, stage, state.rows)
    if not path.is_file():
        raise RuntimeError(f"no baseline trace {path}: run the baseline with save_states, the same manifest, "
                           "batch size and P2_TRACE_EVERY")
    ref = torch.load(path, map_location="cpu")
    if ref["rows"] != [list(r) for r in state.rows]:
        raise RuntimeError(f"{path}: baseline batch holds {ref['rows'][:2]}..., this batch {list(state.rows[:2])}...")
    out = Path(_env("P2_METRICS"))
    out.parent.mkdir(parents=True, exist_ok=True)
    new = not out.exists()
    with out.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(METRIC_COLUMNS)
        for k in TRACED[model]:
            a, b = getattr(state, k).detach().float().cpu().flatten(1), ref[k].flatten(1)
            l2 = (a - b).norm(dim=1)
            rel = l2 / b.norm(dim=1).clamp_min(1e-12)
            cos = torch.nn.functional.cosine_similarity(a, b, dim=1)
            for (class_id, seed), d, r, c in zip(state.rows, l2.tolist(), rel.tolist(), cos.tolist()):
                w.writerow([class_id, seed, f"state_l2:{k}", repr(d), stage])
                w.writerow([class_id, seed, f"state_rel_l2:{k}", repr(r), stage])
                w.writerow([class_id, seed, f"state_cos:{k}", repr(c), stage])


save_states = Observe(_save_states, when="after", name="p2_save_states")
compare_states = Observe(_compare_states, when="after", name="p2_compare_states")


def __getattr__(name):
    # Built on request, so the trace hooks load without P2_STAGE / P2_SEVERITY being set.
    if name in _CORRUPTIONS:
        return _corruption(name)
    raise AttributeError(name)
