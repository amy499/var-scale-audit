"""Reference DiT sample following third_party/DiT/run_DiT.ipynb cell by cell.

Independent of runner/: imports only upstream DiT. Settings come from the same
config the runner uses. Deviations from the notebook are marked DEVIATION.

Writes two variants for the given class/seed:
  <out>_asis.{pt,png}     notebook code with PyTorch's default runtime flags
                          (matmul TF32 off, cudnn non-deterministic allowed, SDPA kernel auto)
  <out>_matched.{pt,png}  notebook code + the config's precision flags and SDPA kernel

    python scripts/phase1/ref_dit_notebook.py --config configs/dit_xl2_256.yaml --class-id 207 --seed 0 --out DIR/ref_dit
"""

import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import contextlib  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
from pathlib import Path  # noqa: E402

import yaml  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "third_party" / "DiT"))   # DEVIATION: notebook clones DiT and chdirs into it
os.environ["HF_HUB_OFFLINE"] = "1"

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True, type=Path)
ap.add_argument("--class-id", required=True, type=int)
ap.add_argument("--seed", required=True, type=int)
ap.add_argument("--out", required=True, type=Path)
args = ap.parse_args()
C = yaml.safe_load(args.config.read_text())
assert C["model"] == "dit" and C["build"]["num_classes"] == 1000
assert C["precision"]["autocast_dtype"] is None, "notebook samples in fp32"
ck = lambda p: str(p if Path(p).is_absolute() else REPO / p)  # noqa: E731

# ---- cell 1: DiT imports
import torch  # noqa: E402
from torchvision.utils import save_image  # noqa: E402
from diffusion import create_diffusion  # noqa: E402
from diffusers.models import AutoencoderKL  # noqa: E402
from download import find_model  # noqa: E402
from models import DiT_models  # noqa: E402  (notebook: `from models import DiT_XL_2`, = DiT_models['DiT-XL/2'])
torch.set_grad_enabled(False)
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cpu":
    print("GPU not found. Using CPU instead.")

# ---- cell 2: load model
image_size = C["build"]["image_size"]            # DEVIATION: from config (notebook: 256)
vae_model = ck(C["checkpoints"]["vae"]["path"])  # DEVIATION: local copy of stabilityai/sd-vae-ft-ema (notebook: hub id)
latent_size = int(image_size) // 8
# Load model:
model = DiT_models[C["build"]["model"]](input_size=latent_size).to(device)
state_dict = find_model(ck(C["checkpoints"]["dit"]["path"]))  # DEVIATION: local file (notebook: name -> downloads to ./pretrained_models)
model.load_state_dict(state_dict)
model.eval()  # important!
vae = AutoencoderKL.from_pretrained(vae_model).to(device)


# ---- cell 3: sample
def cell3(variant: str):
    if variant == "asis":   # PyTorch defaults (the notebook sets none of these)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.deterministic = False
        kernel = contextlib.nullcontext()
    else:                   # DEVIATION ("matched"): the runner's config flags
        P, backend = C["precision"], C["attention"]["sdpa_backend"]
        torch.backends.cuda.matmul.allow_tf32 = P["tf32"]
        torch.backends.cudnn.allow_tf32 = P["tf32"]
        torch.backends.cudnn.deterministic = P["cudnn_deterministic"]
        kernel = (torch.backends.cuda.sdp_kernel(
            enable_math=backend == "math", enable_flash=backend == "flash",
            enable_mem_efficient=backend == "mem_efficient")
            if backend != "auto" and device == "cuda" else contextlib.nullcontext())

    # Set user inputs:
    seed = args.seed                                            # DEVIATION: from CLI (notebook: 0)
    torch.manual_seed(seed)
    num_sampling_steps = C["sampler"]["num_sampling_steps"]    # DEVIATION: from config (notebook: 250)
    cfg_scale = C["sampler"]["cfg_scale"]                      # DEVIATION: from config (notebook: 4)
    class_labels = (args.class_id,)                            # DEVIATION: from CLI (notebook: 8 labels)
    samples_per_row = 4

    # Create diffusion object:
    diffusion = create_diffusion(str(num_sampling_steps))

    with kernel:
        # Create sampling noise:
        n = len(class_labels)
        z = torch.randn(n, 4, latent_size, latent_size, device=device)
        y = torch.tensor(class_labels, device=device)

        # Setup classifier-free guidance:
        z = torch.cat([z, z], 0)
        y_null = torch.tensor([1000] * n, device=device)
        y = torch.cat([y, y_null], 0)
        model_kwargs = dict(y=y, cfg_scale=cfg_scale)

        # Sample images:
        samples = diffusion.p_sample_loop(
            model.forward_with_cfg, z.shape, z, clip_denoised=False,
            model_kwargs=model_kwargs, progress=False, device=device   # DEVIATION: progress bar off
        )
        samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
        samples = vae.decode(samples / 0.18215).sample
    torch.save(samples.cpu().clone(), f"{args.out}_{variant}.pt")   # added: raw tensor for comparison

    # Save and display images:
    save_image(samples, f"{args.out}_{variant}.png", nrow=int(samples_per_row),
               normalize=True, value_range=(-1, 1))


args.out.parent.mkdir(parents=True, exist_ok=True)
for variant in ("asis", "matched"):
    cell3(variant)
    print(f"wrote {args.out}_{variant}.png")
Path(f"{args.out}.json").write_text(json.dumps({"notebook": "third_party/DiT/run_DiT.ipynb",
                                                "config_sdpa_backend": C["attention"]["sdpa_backend"]}, indent=2))
