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

A manifest (CSV `class_id,seed`, one row per image; read only by `runner/manifest.py`) generates many images:

```bash
python -m runner.generate --config configs/var_d20.yaml --manifest manifest/provisional_4x4.csv --batch-size 16
```

Output: `outputs/<config>/<manifest>/class<c>_seed<s>.{png,pt}` plus `run.json` (settings and each row's
tensor sha256). Every row gets its own generator seeded with its own seed, so its random draws do not
depend on batch size or the other rows. Results are bit-identical across repeat runs at the same batch
size; across batch sizes they can differ by float rounding (~1e-5 on CPU; see
`scripts/phase2/check_seeding.py`). Use one batch size for a whole experiment.

Hooks (read or change the model's state at every sampling stage): see `docs/hooks_quickstart.md`.

## Precision and reference hashes

From Phase 3, VAR runs in **strict fp32**: `configs/var_d20.yaml` has autocast off and TF32 off.
DiT is unchanged (fp32 with TF32). The previous VAR settings (fp16 autocast with TF32, as in VAR's
`demo_sample.ipynb`) are kept in `configs/var_d20_fp16.yaml`. It is used only by the historical
Phase 1-2b jobs (below) and by the fp16 side of the timing comparison in `jobs/phase3_check.pbs`;
experiments use `configs/var_d20.yaml`.

Reference tensor hashes (sha256 prefix of the raw output, class 207 seed 0, batch 1, on the GPU):

| Model | Precision | Hash | Status |
|---|---|---|---|
| VAR-d20 | fp16 autocast + TF32 (`var_d20_fp16.yaml`) | `48db2d9e166a` | Historical: Phase 1-2b |
| VAR-d20 | strict fp32 (`var_d20.yaml`) | not yet recorded | Recorded by the first `jobs/phase3_check.pbs` run, which also checks runner = plain upstream at batch 1 in fp32 (all 16 manifest rows); pin it here and as `EXPECT_VAR` in that job |
| DiT-XL/2 | fp32 + TF32 (`dit_xl2_256.yaml`) | `2a3a0d6fbce9` | Current |

**Historical jobs.** `jobs/phase1_check.pbs`, `jobs/phase2_check.pbs` and `jobs/phase2b_tokens.pbs`
were run with the fp16 VAR settings and expect the fp16 hash. Their `VAR_CONFIG` now defaults to
`configs/var_d20_fp16.yaml`, which has exactly the settings they originally ran with, so re-running
them reproduces their fp16 results (only the config path recorded in their output JSON differs). Their
checks are otherwise unchanged: the Phase 2 and 2b regression steps still expect `48db2d9e166a`, and
`scripts/phase1/ref_var_notebook.py` (Phase 1, and Phase 2's divergence step) still asserts the fp16
precision. Don't point them at `configs/var_d20.yaml`: those checks would fail, and Phase 2b's "fp16"
arm would actually run in fp32.

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
