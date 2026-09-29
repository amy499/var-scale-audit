"""Generate images with exactly one model per process.

One image:
    python -m runner.generate --config configs/var_d20.yaml --class-id 207 --seed 0
    -> <out>.png, <out>.json (and <out>.pt with --save-raw)

Every row of a manifest (see runner/manifest.py), in batches:
    python -m runner.generate --config configs/var_d20.yaml --manifest manifest/provisional_4x4.csv --batch-size 16
    -> <out-dir>/class<c>_seed<s>.png and .pt per row, plus <out-dir>/run.json

Each image's randomness comes only from its own seed (one generator per row), so the
output for a row does not depend on --batch-size or on the other rows in the manifest.
The manifest size must be a multiple of --batch-size unless --allow-partial-batch is given.

Hooks (runner/hooks.py, docs/hooks_quickstart.md): --hook path/to/file.py:NAME or --hook package.module:NAME,
repeatable, where NAME is an Observe/Modify hook or a list of them. They are recorded in the output JSON.
DiT only: --skip-timesteps T [T ...] skips the model evaluation at those original timesteps
(docs/hook_interface.md section 11b).
"""

import argparse
import hashlib
import importlib
import importlib.util
import json
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True  # before anything from third_party/ is imported

import torch  # noqa: E402
from PIL import Image  # noqa: E402

from runner import OUTPUTS_DIR  # noqa: E402
from runner.config import load_config  # noqa: E402
from runner.manifest import Row, load_manifest  # noqa: E402
from runner.runtime import apply_precision, effective_precision, environment_record  # noqa: E402


def load_model(cfg: dict):
    # Import only the wrapper for the configured repo.
    if cfg["model"] == "var":
        from runner.var_model import VARModel
        return VARModel(cfg)
    from runner.dit_model import DiTModel
    return DiTModel(cfg)


def tensor_sha256(t: torch.Tensor) -> str:
    """Hash of dtype, shape and raw bytes (same definition as scripts/phase1/compare.py)."""
    h = hashlib.sha256(f"{t.dtype}{tuple(t.shape)}".encode())
    h.update(t.contiguous().numpy().tobytes())
    return h.hexdigest()


def capture_kw(capture: list | None) -> dict:
    """Pass capture only when on, so the default call is exactly today's."""
    return {} if capture is None else {"capture": capture}


def hooks_kw(hooks: list) -> dict:
    """Pass hooks only when some are registered, so the default call is exactly today's."""
    return {"hooks": hooks} if hooks else {}


def load_hooks(specs: list[str]) -> list:
    """Each spec is FILE.py:NAME or MODULE:NAME; NAME is a Hook or a list/tuple of Hooks."""
    from runner.hooks import Hook
    hooks = []
    for spec in specs:
        target, sep, attr = spec.rpartition(":")
        if not sep or not target or not attr:
            raise ValueError(f"--hook {spec!r}: expected FILE.py:NAME or MODULE:NAME")
        if target.endswith(".py"):
            path = Path(target).resolve()
            mod_spec = importlib.util.spec_from_file_location(f"_hooks_{path.stem}_{len(hooks)}", path)
            if mod_spec is None:
                raise ValueError(f"--hook {spec!r}: cannot load {path}")
            mod = importlib.util.module_from_spec(mod_spec)
            mod_spec.loader.exec_module(mod)
        else:
            mod = importlib.import_module(target)
        obj = getattr(mod, attr)
        items = list(obj) if isinstance(obj, (list, tuple)) else [obj]
        for h in items:
            if not isinstance(h, Hook):
                raise ValueError(f"--hook {spec!r}: {h!r} is not an Observe/Modify hook")
        hooks.extend(items)
    return hooks


def state_sha256(state: torch.Tensor) -> str:
    return hashlib.sha256(state.cpu().numpy().tobytes()).hexdigest()


def row_tokens(capture: list[dict], i: int, cfg: dict) -> dict:
    """Row i of a batch capture: per scale, that image's tensors (idx, top2_p, top2_i, p_chosen)."""
    return {"patch_nums": list(cfg["build"]["patch_nums"]),
            "scales": [{k: v[i].clone() for k, v in scale.items()} for scale in capture]}


def settings_record(cfg: dict, device: torch.device) -> dict:
    return {
        "model": cfg["model"],
        "config_path": cfg["_path"],
        "device": str(device),
        "sampler": cfg["sampler"],
        "build": cfg["build"],
        "checkpoints": {k: v["path"] for k, v in cfg["checkpoints"].items()},
        "precision": effective_precision(cfg["precision"], device),
        "env": environment_record(device),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--device", help="override config device (e.g. cpu for a smoke test)")
    ap.add_argument("--capture-tokens", action="store_true",
                    help="VAR only: also save each image's sampled token ids per scale as <stem>.tokens.pt "
                         "(read-only; outputs are unchanged)")
    ap.add_argument("--hook", action="append", default=[], metavar="FILE.py:NAME|MODULE:NAME",
                    help="register Observe/Modify hooks (repeatable; see runner/hooks.py)")
    ap.add_argument("--skip-timesteps", type=int, nargs="+", default=[], metavar="T",
                    help="DiT only: skip the model evaluation at these original timesteps; the previous step is "
                         "merged into the next kept one and the skipped step's noise is still drawn "
                         "(docs/hook_interface.md section 11b)")
    ap.add_argument("--no-burn-skipped", action="store_true", help=argparse.SUPPRESS)   # test only: unpairs the run
    one = ap.add_argument_group("one image")
    one.add_argument("--class-id", type=int)
    one.add_argument("--seed", type=int)
    one.add_argument("--out", type=Path, help="output path without extension (default: outputs/<config>/class<c>_seed<s>)")
    one.add_argument("--save-raw", action="store_true", help="also save the raw output tensor as <out>.pt")
    many = ap.add_argument_group("manifest")
    many.add_argument("--manifest", type=Path, help="generate every row of this manifest")
    many.add_argument("--batch-size", type=int, default=1, help="rows per model call (does not change the images)")
    many.add_argument("--out-dir", type=Path, help="default: outputs/<config>/<manifest name>")
    many.add_argument("--allow-partial-batch", action="store_true",
                      help="allow a manifest size that is not a multiple of --batch-size (smaller last batch)")
    args = ap.parse_args(argv)

    if args.manifest:
        if args.class_id is not None or args.seed is not None or args.out or args.save_raw:
            ap.error("--manifest cannot be combined with --class-id/--seed/--out/--save-raw")
        if args.batch_size < 1:
            ap.error("--batch-size must be >= 1")
        rows = load_manifest(args.manifest)
        if len(rows) % args.batch_size and not args.allow_partial_batch:
            ap.error(f"manifest has {len(rows)} rows, not a multiple of --batch-size {args.batch_size}; "
                     "pass --allow-partial-batch to run a smaller last batch")
    else:
        if args.class_id is None or args.seed is None:
            ap.error("give --class-id and --seed, or --manifest")
        if args.out_dir or args.batch_size != 1 or args.allow_partial_batch:
            ap.error("--out-dir/--batch-size/--allow-partial-batch only apply with --manifest")
        rows = [Row(args.class_id, args.seed)]

    cfg = load_config(args.config)
    if args.device:
        cfg["device"] = args.device
    num_classes = cfg["build"]["num_classes"]
    bad = [r for r in rows if not 0 <= r.class_id < num_classes]
    if bad:
        ap.error(f"class_id must be in [0, {num_classes}); got {bad[:5]}")

    if args.capture_tokens and cfg["model"] != "var":
        ap.error("--capture-tokens is VAR only")
    if (args.skip_timesteps or args.no_burn_skipped) and cfg["model"] != "dit":
        ap.error("--skip-timesteps is DiT only")
    if args.no_burn_skipped and not args.skip_timesteps:
        ap.error("--no-burn-skipped needs --skip-timesteps")
    # Passed only when given, so the default call is exactly today's.
    skip_kw = {"skip_timesteps": args.skip_timesteps} if args.skip_timesteps else {}
    try:
        hooks = load_hooks(args.hook)
    except (ValueError, ImportError, AttributeError, OSError) as e:
        ap.error(str(e))
    hook_records = [h.record() for h in hooks]
    sample_kw = {**hooks_kw(hooks), **skip_kw, **({"_burn_skipped": False} if args.no_burn_skipped else {})}

    device = torch.device(cfg["device"])
    apply_precision(cfg["precision"])
    t0 = time.time()
    model = load_model(cfg)
    t_load = time.time() - t0
    try:
        stages = model.stage_table(**skip_kw)
    except ValueError as e:
        ap.error(str(e))
    run_settings = {"hooks": hook_records, "skip_timesteps": args.skip_timesteps,
                    **({"burn_skipped": False} if args.no_burn_skipped else {})}

    if not args.manifest:
        capture = [] if args.capture_tokens else None
        gen_states = []
        imgs, raw, attention = model.sample([args.class_id], [args.seed], **capture_kw(capture), **sample_kw,
                                            gen_states=gen_states)
        t_sample = time.time() - t0 - t_load
        out = args.out or OUTPUTS_DIR / Path(cfg["_path"]).stem / rows[0].stem
        out.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(imgs[0]).save(out.with_suffix(".png"))
        if args.save_raw:
            torch.save(raw, out.with_suffix(".pt"))
        if capture is not None:
            torch.save(row_tokens(capture, 0, cfg), out.with_suffix(".tokens.pt"))
        record = {
            **settings_record(cfg, device),
            "class_id": args.class_id,
            "seed": args.seed,
            "attention": attention,
            "capture_tokens": args.capture_tokens,
            **run_settings,
            "generator_state_sha256": state_sha256(gen_states[0]),
            "seconds": {"load": round(t_load, 2), "sample": round(t_sample, 2)},
            "stages": stages,
        }
        out.with_suffix(".json").write_text(json.dumps(record, indent=2))
        print(f"wrote {out.with_suffix('.png')}  (attention: {attention['sdpa_kernel']['effective']})")
        return

    out_dir = args.out_dir or OUTPUTS_DIR / Path(cfg["_path"]).stem / args.manifest.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    results, t_sample = [], 0.0
    for start in range(0, len(rows), args.batch_size):
        batch = rows[start:start + args.batch_size]
        t1 = time.time()
        capture = [] if args.capture_tokens else None
        gen_states = []
        imgs, raw, attention = model.sample([r.class_id for r in batch], [r.seed for r in batch],
                                            **capture_kw(capture), **sample_kw, gen_states=gen_states)
        t_sample += time.time() - t1
        for i, r in enumerate(batch):
            one_raw = raw[i:i + 1].clone()  # 1xCxHxW, own storage (a view would save the whole batch)
            Image.fromarray(imgs[i]).save(out_dir / f"{r.stem}.png")
            torch.save(one_raw, out_dir / f"{r.stem}.pt")
            if capture is not None:
                torch.save(row_tokens(capture, i, cfg), out_dir / f"{r.stem}.tokens.pt")
            results.append({"class_id": r.class_id, "seed": r.seed, "file": r.stem,
                            "tensor_sha256": tensor_sha256(one_raw),
                            "generator_state_sha256": state_sha256(gen_states[i])})
        print(f"[{start + len(batch)}/{len(rows)}] rows done", flush=True)

    record = {
        **settings_record(cfg, device),
        "manifest": str(args.manifest.resolve()),
        "batch_size": args.batch_size,
        "allow_partial_batch": args.allow_partial_batch,
        "capture_tokens": args.capture_tokens,
        **run_settings,
        "attention": attention,
        "seconds": {"load": round(t_load, 2), "sample": round(t_sample, 2)},
        "stages": stages,
        "rows": results,
    }
    (out_dir / "run.json").write_text(json.dumps(record, indent=2))
    print(f"wrote {len(rows)} images to {out_dir}  (attention: {attention['sdpa_kernel']['effective']})")


if __name__ == "__main__":
    main()
