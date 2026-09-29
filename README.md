# var-scale-audit

Shared generation runner for [VAR](https://github.com/FoundationVision/VAR) and [DiT](https://github.com/facebookresearch/DiT).

## Layout

| Path | Purpose |
|---|---|
| `third_party/VAR`, `third_party/DiT` | Upstream repos as git submodules, pinned. **Never modified.** |
| `runner/` | Shared generation runner (Python package) |
| `lanes/p1` … `lanes/p4` | Per-lane work |
| `configs/` | Run configurations |
| `manifest/` | Run manifests |
| `scripts/` | Utility scripts |
| `checkpoints/` | Model weights (gitignored) |
| `outputs/` | Generated samples (gitignored) |

## Setup

```bash
git clone --recurse-submodules https://github.com/amy499/var-scale-audit.git
# or, in an existing clone:
git submodule update --init

conda env create -f environment.yml
conda activate var-dit
python scripts/check_third_party.py   # confirms submodules are pinned and clean
```

On NSCC ASPIRE 2A, follow [docs/NSCC.md](docs/NSCC.md) instead of the commands below.

## Checkpoints (run on the login node)

```bash
python scripts/download_checkpoints.py            # all configs/*.yaml (~5.9 GB)
python scripts/download_checkpoints.py configs/var_d20.yaml
```

Standard library + PyYAML only (no torch/GPU). URLs and sha256 come from each config's `checkpoints:`
section; Hugging Face files are pinned to a repo revision. Downloads resume; `HF_ENDPOINT` selects a mirror.

| File | Size | Source |
|---|---|---|
| `checkpoints/var/var_d20.pth` | 2.4 GB | `FoundationVision/var` @ `6d0ee65` |
| `checkpoints/var/vae_ch160v4096z32.pth` | 436 MB | `FoundationVision/var` @ `6d0ee65` |
| `checkpoints/dit/DiT-XL-2-256x256.pt` | 2.7 GB | `dl.fbaipublicfiles.com` (no published hash; recorded in `.sha256`) |
| `checkpoints/sd-vae-ft-ema/` | 335 MB | `stabilityai/sd-vae-ft-ema` @ `f04b2c4` |

## Generating

```bash
python -m runner.generate --config configs/var_d20.yaml     --class-id 207 --seed 0
python -m runner.generate --config configs/dit_xl2_256.yaml --class-id 207 --seed 0
```

One process loads exactly one model (`runner/upstream.py` refuses to mix repos). Output:
`outputs/<config>/class<c>_seed<s>.png` plus a `.json` with the settings, attention backend, precision,
versions and submodule SHAs. All sampler settings come from the config; missing keys are an error.

## Pinned upstream commits

| Submodule | Commit | Date |
|---|---|---|
| `third_party/VAR` | `78b95394fc5896192e3a003e4b295f8ea743c48f` | 2025-11-10 |
| `third_party/DiT` | `ed81ce2229091fd4ecc9a223645f95cf379d582b` | 2023-06-02 |

## Environment

`environment.yml` is the single source of truth for dependencies. It resolves these conflicts between the two upstream specs:

| Package | VAR wants | DiT wants | Resolved to | Why |
|---|---|---|---|---|
| torch | `~=2.1.0` | `>=1.13` | `2.1.2` | Intersection of the two ranges. |
| CUDA | (unspecified) | `pytorch-cuda=11.7` | `11.8` | No torch 2.1 build exists for CUDA 11.7 (2.1 ships for 11.8 / 12.1). |
| python | (unspecified) | `>=3.8` | `3.10` | torch 2.1 supports only Python 3.8–3.11. |
| numpy | unpinned | (unspecified) | `>=1.24,<2` | torch 2.1 is compiled against NumPy 1.x. |
| timm / diffusers / transformers / huggingface_hub / accelerate | unpinned | unpinned | pinned | Current releases require newer torch or have broken APIs used here (e.g. `huggingface_hub` removed `cached_download`). |

Known caveats:
- **Module-name collision.** Both repos have a top-level `models` module (and VAR also has `utils`), so they cannot share one `sys.path` in a single process. The runner must isolate them (separate processes or per-repo `sys.modules` handling).
- `DiT/diffusion/timestep_sampler.py` uses `np.int`, which NumPy ≥1.24 removed. It is only reached through the `loss-second-moment` training sampler, not during sampling.
- VAR can optionally use `flash_attn` / `xformers`. They are left out; VAR falls back to `torch` SDPA, so attention kernels (and bit-level numerics) differ from runs that have them installed.
- `transformers` (VAR) and `accelerate` (DiT) are listed upstream but never imported.
- Running upstream code writes `__pycache__/` into `third_party/DiT`, which makes the submodule dirty. Run with `PYTHONDONTWRITEBYTECODE=1` or `python -B`.
