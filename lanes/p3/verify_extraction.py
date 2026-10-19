"""Checks 1 and 2 of P3 Step 2: did the observe hook behave?

Runs anywhere — no GPU, no torch, no checkpoints. Reads JSON and .npz headers only.

    CHECK 1  the hook changed nothing
             Every image's tensor_sha256 must equal the clean baseline's for the same
             (class_id, seed). An Observe hook receives copies and cannot write back, so
             any difference means it is not behaving as one and every number derived from
             the run is suspect.

    CHECK 2  the right number of states came out
             One .npz per (stage, batch), and rows summed over files = stages x images.
             Also loads each latent to confirm it is finite and the right shape: a silently
             truncated or NaN-filled save looks fine until it reaches a figure.

CHECK 3 (decode five and look at them) is in `decode_stages.py` and needs the GPU.
**Do not skip it.** These two checks prove the plumbing; only your eyes prove the states
mean what you think they mean.

RUN IT
------
    python -m lanes.p3.verify_extraction \
        --stages   outputs/p3/pilot/stages \
        --run      outputs/p3/pilot/var_d20 \
        --baseline outputs/baseline/25665469/var_d20 \
        --model    var

Exit code 0 = all checks passed, 1 = something failed (so a job script notices).

Owner: P3.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True

VAR_SCALES = 10         # VAR-d20: patch_nums [1,2,3,4,5,6,8,10,13,16]

# Latent sizes are DERIVED from run.json, never assumed. The tiny random-weight models
# used for CPU testing (README "macOS laptop") keep the real stage counts but shrink the
# images: DiT-S/2 at image_size 64 gives an 8x8 latent, not the real model's 32x32.
# Hard-coding 32 here would fail every local run for the wrong reason.
REAL_LATENT_HW = {"var": 16, "dit": 32}


def expected_latent_hw(run, model):
    """The latent side length this run should produce, from its own run.json.

    VAR: f_hat is (B, Cvae, P, P) with P = patch_nums[-1], which run.json's stage table
    records as `pn`. DiT: not recorded, so None — shape CONSISTENCY is still checked.
    """
    stages = run.get("stages") or []
    if model == "var":
        pns = [s.get("pn") for s in stages if isinstance(s.get("pn"), int)]
        if pns:
            return max(pns)
    return None


def load_run(d):
    p = Path(d) / "run.json"
    if not p.is_file():
        raise SystemExit(f"ERROR: {p} not found — is {d} a runner.generate output directory?")
    return json.loads(p.read_text())


# ==========================================================================
# CHECK 1 — the hook changed nothing
# ==========================================================================

def check_unchanged(run_dir, baseline_dir):
    run, base = load_run(run_dir), load_run(baseline_dir)
    r = {(x["class_id"], x["seed"]): x for x in run["rows"]}
    b = {(x["class_id"], x["seed"]): x for x in base["rows"]}

    shared = sorted(set(r) & set(b))
    only_run, only_base = sorted(set(r) - set(b)), sorted(set(b) - set(r))

    differing = [k for k in shared if r[k]["tensor_sha256"] != b[k]["tensor_sha256"]]
    gen_diff = [k for k in shared
                if r[k].get("generator_state_sha256") != b[k].get("generator_state_sha256")]

    print("CHECK 1 — hook changed nothing")
    print(f"  images in run       : {len(r)}")
    print(f"  images in baseline  : {len(b)}")
    print(f"  compared            : {len(shared)}")

    ok = True
    if not shared:
        print("  FAIL  no (class_id, seed) pairs in common — different manifests?")
        ok = False
    if only_run:
        print(f"  note  {len(only_run)} image(s) only in the run, e.g. {only_run[:3]}")
    if only_base:
        print(f"  note  {len(only_base)} image(s) only in the baseline "
              f"(expected if the run is a 16-image subset), e.g. {only_base[:3]}")

    if differing:
        print(f"  FAIL  {len(differing)} of {len(shared)} images differ from the baseline")
        for k in differing[:5]:
            print(f"          class {k[0]} seed {k[1]}: "
                  f"{r[k]['tensor_sha256'][:12]} vs {b[k]['tensor_sha256'][:12]}")
        print("        -> the hook is NOT behaving as an Observe hook. Do not use this run.")
        print("        -> check it is built with Observe(...), not Modify(...), and that")
        print("           nothing writes back into state tensors.")
        ok = False
    elif shared:
        print(f"  PASS  all {len(shared)} images byte-identical to the baseline")

    if gen_diff:
        print(f"  FAIL  {len(gen_diff)} image(s) consumed different randomness")
        print("        -> the two runs did not draw the same random numbers; the comparison")
        print("           is invalid even if the images happen to match.")
        ok = False

    # A run whose settings differ is not a fair comparison even if hashes match.
    for field in ("precision", "batch_size", "manifest"):
        rv, bv = run.get(field), base.get(field)
        if rv != bv and field != "manifest":
            print(f"  FAIL  {field} differs: run={rv!r} baseline={bv!r}")
            ok = False
    return ok


# ==========================================================================
# CHECK 2 — the right states came out, and they are usable
# ==========================================================================

def check_states(stages_dir, model, n_images, expect_stages=None, want_hw=None):
    d = Path(stages_dir) / model
    if not d.is_dir():
        raise SystemExit(f"ERROR: {d} not found — did the hook write anywhere? (P3_OUT)")

    files = sorted(d.glob("stage*.npz"))
    print(f"\nCHECK 2 — states written  ({d})")
    print(f"  .npz files          : {len(files)}")
    if not files:
        print("  FAIL  no .npz files at all — the hook never fired, or P3_OUT pointed elsewhere.")
        return False

    ok = True
    per_stage = defaultdict(int)
    shapes, bad = set(), []

    for f in files:
        try:
            z = np.load(f)
            stage = int(z["stage"])
            lat = z["latent"]
            per_stage[stage] += lat.shape[0]
            shapes.add(lat.shape[1:])
            if not np.isfinite(lat).all():
                bad.append((f.name, "contains NaN or Inf"))
            elif want_hw is not None and lat.shape[-2:] != (want_hw, want_hw):
                bad.append((f.name, f"latent is {lat.shape[-2:]}, expected {(want_hw, want_hw)}"))
            elif len(z["class_id"]) != lat.shape[0]:
                bad.append((f.name, f"{len(z['class_id'])} rows but latent has {lat.shape[0]}"))
        except Exception as e:                      # a truncated save raises here
            bad.append((f.name, f"will not load: {e}"))

    stages = sorted(per_stage)
    total = sum(per_stage.values())
    print(f"  distinct stages     : {len(stages)}  {stages if len(stages) <= 12 else ''}")
    print(f"  latent shapes       : {sorted(shapes)}")
    print(f"  total rows saved    : {total}   (expect {len(stages)} stages x {n_images} images "
          f"= {len(stages) * n_images})")

    if bad:
        print(f"  FAIL  {len(bad)} unusable file(s):")
        for name, why in bad[:5]:
            print(f"          {name}: {why}")
        ok = False

    if len(shapes) > 1:
        print("  FAIL  latent shape is not constant across stages — the saves are inconsistent.")
        ok = False
    elif shapes:
        hw = next(iter(shapes))[-1]
        real = REAL_LATENT_HW.get(model)
        if real is not None and hw != real:
            print(f"  note  latent is {hw}x{hw}, not the real model's {real}x{real} — this looks")
            print("        like a tiny/CPU test model. Fine for checking the hook's mechanics;")
            print("        the images are noise, so check 3 means nothing on this run.")

    uneven = {s: n for s, n in per_stage.items() if n != n_images}
    if uneven:
        print(f"  FAIL  {len(uneven)} stage(s) do not have {n_images} rows, e.g. "
              f"{dict(list(uneven.items())[:4])}")
        print("        -> a batch failed to save, or two runs wrote into the same folder.")
        ok = False

    if expect_stages is not None and len(stages) != expect_stages:
        print(f"  FAIL  {len(stages)} stages, expected {expect_stages}")
        ok = False
    elif model == "var" and len(stages) != VAR_SCALES:
        print(f"  FAIL  {len(stages)} VAR scales, expected {VAR_SCALES}")
        ok = False

    if ok:
        print(f"  PASS  {len(stages)} stages x {n_images} images, all finite and consistent")
    return ok


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stages", required=True, type=Path, help="P3_OUT from the hook run")
    ap.add_argument("--run", required=True, type=Path, help="the run that had the hook attached")
    ap.add_argument("--baseline", required=True, type=Path, help="the clean baseline run")
    ap.add_argument("--model", required=True, choices=("var", "dit"))
    ap.add_argument("--expect-stages", type=int, default=None,
                    help="override the expected stage count (VAR defaults to 10)")
    ap.add_argument("--latent-hw", type=int, default=None,
                    help="assert the latent side length; derived from run.json when omitted")
    args = ap.parse_args(argv)

    run = load_run(args.run)
    n_images = len(run["rows"])
    want_hw = args.latent_hw if args.latent_hw else expected_latent_hw(run, args.model)
    one = check_unchanged(args.run, args.baseline)
    two = check_states(args.stages, args.model, n_images, args.expect_stages, want_hw)

    print()
    if one and two:
        print("CHECKS 1 AND 2 PASSED.")
        print("Now run CHECK 3 — it is the one that catches a wrong extraction:")
        print(f"    python -m lanes.p3.decode_stages --stages {args.stages} "
              f"--model {args.model} --config configs/<your config>.yaml --n 5")
        return 0
    print("SOMETHING FAILED — fix it before the full sweep. Every downstream number")
    print("depends on these states being what you think they are.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
