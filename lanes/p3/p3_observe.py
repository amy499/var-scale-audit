"""P3 observe hook: save the model's intermediate state at every stage.

This is P3's data-collection step. It changes nothing about the generation — it is an
Observe hook, so it receives copies and cannot alter the run. The images produced are
bit-identical to the clean baseline, which means THIS CAN RIDE ALONG WITH P1's BASELINE
RUN instead of costing a separate GPU job.

What it saves:
    VAR   f_hat at every scale             the accumulator after scale si is added.
                                           10 scales x 224 images x 32KB ~= 70 MB.
    DiT   pred_xstart at chosen steps      "what the model currently thinks the finished
                                           image is". 250 steps is ~900 MB, so a subset
                                           is saved (see P3_DIT_STEPS below).

These are LATENTS, not images. Decoding needs the model's VAE and happens in a second
pass (`p3_decode.py`), because an observe hook receives only the string "var"/"dit",
not the model object.

RUN IT
------
    python -m runner.generate --config configs/var_d20.yaml \
        --manifest manifest/frozen_v1_1.csv --batch-size 16 \
        --out-dir outputs/p3/var_stages \
        --hook lanes/p3/p3_observe.py:var_stages

    python -m runner.generate --config configs/dit_xl2_256.yaml \
        --manifest manifest/frozen_v1_1.csv --batch-size 16 \
        --out-dir outputs/p3/dit_stages \
        --hook lanes/p3/p3_observe.py:dit_stages

ENVIRONMENT
-----------
    P3_OUT        where to write (default: outputs/p3/stages)
    P3_DIT_STEPS  how many DiT steps to keep (default 25). They are spaced denser near
                  the end of sampling, where the perceptible change happens: uniform
                  spacing in step index wastes most samples on near-identical noise.

WHEN TO RUN IT
--------------
**Validate this hook on your own 16-image job BEFORE asking anyone to attach it to a
shared run.** The hook cannot corrupt a run — `runner/hooks.py:run_point` hands Observe
hooks a `state.clone()` and discards the return value — but it CAN kill one: hook
functions are called bare, with no try/except around them, so an unhandled exception
propagates and ends the host job. `_save` therefore never raises (see its docstring), but
a path or permission problem is still worth finding on a 16-image job of your own rather
than inside someone else's baseline run.

VERIFY BEFORE THE FULL SWEEP
----------------------------
Run on 16 images first and check:
  1. The final images are byte-identical to the clean baseline's
     (`tensor_sha256` in run.json). If not, this is not behaving as an observe hook.
  2. `n_saved` in the manifest .json equals stages x images.
  3. Decode a handful and LOOK at them. VAR should go coarse->fine; DiT noisy->clean.
     If they do not, the extraction is wrong and every downstream number is meaningless.
  4. `_failures.log` does not exist in P3_OUT.

Checks 1 and 2 are automated by `verify_extraction.py`; check 3 is `decode_stages.py`.

Owner: P3.
"""

import json
import os
from pathlib import Path

import numpy as np

from runner.hooks import Observe

OUT = Path(os.environ.get("P3_OUT", "outputs/p3/stages"))
N_DIT_STEPS = int(os.environ.get("P3_DIT_STEPS", "25"))

_keep_cache = {}            # {total_steps: frozenset of kept step indices}


_failures = []


def _save(subdir: str, stage: int, rows, arr, extra=None):
    """Write one stage's batch to its own .npz. One file per (stage, batch) keeps a
    failed job from losing everything and makes the output resumable.

    NEVER RAISES. This hook is designed to ride along inside another lane's GPU job, and
    `runner/hooks.py:run_point` calls hook functions bare — an exception here propagates
    and kills the HOST run. Losing P3's states is cheap; killing P1's baseline run is not.
    So every failure is swallowed, recorded, and reported at the end of the run.

    Nothing is lost silently: `verify_extraction.py` check 2 counts rows per stage and
    fails if any stage is short, and `_failures.log` says why.
    """
    try:
        d = OUT / subdir
        d.mkdir(parents=True, exist_ok=True)
        class_ids = np.array([r[0] for r in rows], dtype=np.int32)
        seeds = np.array([r[1] for r in rows], dtype=np.int64)
        # seeds make the filename unique, so concurrent batches never collide
        tag = f"{stage:04d}_{seeds[0]}"
        np.savez_compressed(d / f"stage{tag}.npz", stage=stage, class_id=class_ids,
                            seed=seeds, latent=arr, **(extra or {}))
    except Exception as e:                      # noqa: BLE001 - deliberate: see docstring
        _failures.append(f"{subdir} stage {stage}: {type(e).__name__}: {e}")
        if len(_failures) <= 3:                 # noisy enough to notice, not to flood a log
            print(f"[p3_observe] WARNING: could not save {subdir} stage {stage}: {e}",
                  flush=True)
        try:
            OUT.mkdir(parents=True, exist_ok=True)
            (OUT / "_failures.log").write_text("\n".join(_failures) + "\n")
        except Exception:                       # noqa: BLE001 - last resort, stay silent
            pass


def _dit_step_set(total=250, n=N_DIT_STEPS):
    """Step INDICES to keep, spaced denser near the end of sampling.

    DDPM's schedule is non-linear: most of the perceptible change happens in the last
    stretch (low timestep). Uniform spacing in step index would spend most of the
    budget on near-identical noisy latents and miss the interesting part.
    """
    i = np.arange(n)
    frac = 1.0 - (1.0 - i / (n - 1)) ** 2          # 0 .. 1, dense at the top
    return sorted(set(np.round(frac * (total - 1)).astype(int).tolist()))


def _keep_for(total):
    """Kept step indices for a run of `total` steps, computed once per total."""
    if total not in _keep_cache:
        _keep_cache[total] = frozenset(_dit_step_set(total, N_DIT_STEPS))
    return _keep_cache[total]


def var_stage(model, stage, p, when, state):
    """VAR: save the accumulator after this scale is added. Fires at all 10 scales."""
    f = state.f_hat.detach().float().cpu().numpy()
    _save("var", stage, state.rows, f,
          extra={"p": np.float32(p),
                 "n_tokens": np.int32(state.fields.get("n_tokens", -1)),
                 "cum_tokens": np.int32(state.fields.get("cum_tokens", -1))})


def dit_stage(model, stage, p, when, state):
    """DiT: save pred_xstart (the current guess at the finished image) at kept steps.

    `stage` is the TIMESTEP (999..0), not the step index, so the index comes from
    `state.fields["step"]`, which the runner computes per step in `dit_model.plan()`.

    DO NOT COUNT CALLS TO GET THE INDEX. A module-level counter is not reset between
    batches, so on a 224-image run (14 batches x 250 steps) it reaches 250..3499 after the
    first batch and matches nothing in the kept set — only the first 16 images are saved,
    silently. That bug shipped and cost a DiT run; P1 caught it. `fields["step"]` restarts
    at 0 for every batch because it is a property of the schedule, not of the hook.

    The total step count is derived from p = step / (total - 1) rather than assumed, so a
    run with a different schedule still saves the right spread instead of saving nothing.
    """
    j = int(state.fields["step"])
    if j > 0:
        if p <= 0:
            return                              # inconsistent fields; skip rather than guess
        total = int(round(j / p)) + 1
        if j not in _keep_for(total):
            return
    # j == 0 is always kept: it is the first step, and `total` cannot be derived from p = 0

    x0 = state.pred_xstart.detach().float().cpu().numpy()
    _save("dit", stage, state.rows, x0,
          extra={"p": np.float32(p), "step_index": np.int32(j),
                 "alpha_bar_in": np.float32(state.fields.get("alpha_bar_in", float("nan")))})


var_stages = Observe(var_stage, when="after", name="p3_var_stages")
dit_stages = Observe(dit_stage, when="after", name="p3_dit_stages")


if __name__ == "__main__":
    # Not run by the runner; this is a self-check of the step selection.
    keep = _dit_step_set()
    print(json.dumps({
        "n_dit_steps_kept": len(keep),
        "step_indices": keep,
        "note": "denser near the end of sampling, where the image actually resolves",
    }, indent=2))
