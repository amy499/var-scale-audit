"""Plain upstream sampling, one image per call (B=1), seeded the way upstream seeds.

    python scripts/phase2/upstream_ref.py --config configs/var_d20.yaml --manifest manifest/provisional_4x4.csv --out-dir DIR

For every manifest row: VAR autoregressive_infer_cfg(B=1, g_seed=seed); DiT torch.manual_seed(seed) +
p_sample_loop (the sample.py / sample_ddp.py per-batch body). Runs under the same runtime settings as
the runner (config precision flags, autocast dtype, SDPA kernel), so it isolates the runner's sampling
code. The runner wrapper is used only to build and load the models.

Writes DIR/<class..._seed...>.pt (1xCxHxW raw output) and DIR/hashes.json {stem: sha256}.
"""

import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import torch  # noqa: E402

from runner.config import load_config  # noqa: E402
from runner.generate import load_model, tensor_sha256  # noqa: E402
from runner.manifest import load_manifest  # noqa: E402
from runner.runtime import apply_precision, autocast_ctx, sdpa_ctx  # noqa: E402


def sample_var(m, cfg, class_id: int, seed: int) -> torch.Tensor:
    s = cfg["sampler"]
    torch.manual_seed(seed)
    with torch.inference_mode(), autocast_ctx(cfg["precision"], m.device), \
            sdpa_ctx(cfg["attention"]["sdpa_backend"], m.device):
        return m.var.autoregressive_infer_cfg(
            B=1, label_B=torch.tensor([class_id], device=m.device), g_seed=seed,
            cfg=s["cfg"], top_k=s["top_k"], top_p=s["top_p"], more_smooth=s["more_smooth"])


def sample_dit(m, cfg, class_id: int, seed: int) -> torch.Tensor:
    cfg_scale = cfg["sampler"]["cfg_scale"]
    dev = m.device
    torch.manual_seed(seed)
    with torch.no_grad(), autocast_ctx(cfg["precision"], dev), sdpa_ctx(cfg["attention"]["sdpa_backend"], dev):
        z = torch.randn(1, m.model.in_channels, m.latent_size, m.latent_size, device=dev)
        y = torch.tensor([class_id], device=dev)
        if cfg_scale > 1.0:
            z = torch.cat([z, z], 0)
            y = torch.cat([y, torch.tensor([m.num_classes], device=dev)], 0)
            fn, kw = m.model.forward_with_cfg, dict(y=y, cfg_scale=cfg_scale)
        else:
            fn, kw = m.model.forward, dict(y=y)
        samples = m.diffusion.p_sample_loop(fn, z.shape, z, clip_denoised=False, model_kwargs=kw,
                                            progress=False, device=dev)
        if cfg_scale > 1.0:
            samples, _ = samples.chunk(2, dim=0)
        return m.vae.decode(samples / 0.18215).sample


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--device", help="override config device")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.device:
        cfg["device"] = args.device
    apply_precision(cfg["precision"])
    m = load_model(cfg)
    sample = sample_var if cfg["model"] == "var" else sample_dit
    args.out_dir.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for r in load_manifest(args.manifest):
        raw = sample(m, cfg, r.class_id, r.seed).cpu().clone()
        torch.save(raw, args.out_dir / f"{r.stem}.pt")
        hashes[r.stem] = tensor_sha256(raw)
        print(f"{r.stem} {hashes[r.stem][:12]}", flush=True)
    (args.out_dir / "hashes.json").write_text(json.dumps(hashes, indent=2))


if __name__ == "__main__":
    main()
