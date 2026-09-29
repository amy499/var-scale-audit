"""Reference VAR sample following third_party/VAR/demo_sample.ipynb cell by cell.

Independent of runner/: imports only upstream VAR. Settings come from the same
config the runner uses. Deviations from the notebook are marked DEVIATION.

Writes two variants for the given class/seed:
  <out>_matched.{pt,png}  notebook code + the config's SDPA kernel restriction
                          (the only runtime setting where runner and notebook differ)
  <out>_asis.{pt,png}     notebook code, PyTorch picks the SDPA kernel itself

    python scripts/phase1/ref_var_notebook.py --config configs/var_d20.yaml --class-id 207 --seed 0 --out DIR/ref_var
"""

import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import contextlib  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402

import yaml  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "third_party" / "VAR"))

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True, type=Path)
ap.add_argument("--class-id", required=True, type=int)
ap.add_argument("--seed", required=True, type=int)
ap.add_argument("--out", required=True, type=Path)
args = ap.parse_args()
C = yaml.safe_load(args.config.read_text())
assert C["model"] == "var"
# The notebook hard-codes these; "matched" is only meaningful if the config agrees.
_nb = {"V": 4096, "Cvae": 32, "ch": 160, "share_quant_resi": 4, "num_classes": 1000, "shared_aln": False}
assert {k: C["build"][k] for k in _nb} == _nb, f"config [build] differs from notebook: {C['build']}"
assert C["precision"] == {"autocast_dtype": "float16", "tf32": True, "cudnn_deterministic": True}, \
    f"config [precision] differs from notebook (fp16 autocast, tf32, cudnn deterministic): {C['precision']}"
ck =lambda p: str(p if Path(p).is_absolute() else REPO / p)  # noqa: E731

################## 1. Download checkpoints and build models
import os  # noqa: E402,F401
import os.path as osp  # noqa: E402,F401
import torch, torchvision  # noqa: E402,E401
import random  # noqa: E402
import numpy as np  # noqa: E402
import PIL.Image as PImage  # noqa: E402
setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)     # disable default parameter init for faster speed
setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)  # disable default parameter init for faster speed
from models import VQVAE, build_vae_var  # noqa: E402,F401

MODEL_DEPTH = C["build"]["depth"]    # DEVIATION: from config (notebook: user-set constant)
assert MODEL_DEPTH in {16, 20, 24, 30} or C["device"] == "cpu"   # cpu: allow tiny smoke-test models

# DEVIATION: no wget into cwd; checkpoints already downloaded to the config paths
vae_ckpt, var_ckpt = ck(C["checkpoints"]["vae"]["path"]), ck(C["checkpoints"]["var"]["path"])

# build vae, var
patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
assert tuple(C["build"]["patch_nums"]) == patch_nums
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if 'vae' not in globals() or 'var' not in globals():
    vae, var = build_vae_var(
        V=4096, Cvae=32, ch=160, share_quant_resi=4,    # hard-coded VQVAE hyperparameters
        device=device, patch_nums=patch_nums,
        num_classes=1000, depth=MODEL_DEPTH, shared_aln=False,
    )

# load checkpoints
vae.load_state_dict(torch.load(vae_ckpt, map_location='cpu'), strict=True)
var.load_state_dict(torch.load(var_ckpt, map_location='cpu'), strict=True)
vae.eval(), var.eval()
for p in vae.parameters(): p.requires_grad_(False)  # noqa: E701
for p in var.parameters(): p.requires_grad_(False)  # noqa: E701
print(f'prepare finished.')  # noqa: F541

from models import basic_var  # noqa: E402
fused = {"flash_attn_installed": basic_var.flash_attn_func is not None,
         "xformers_installed": basic_var.memory_efficient_attention is not None}


############################# 2. Sample with classifier-free guidance
def cell2(variant: str):
    S = C["sampler"]
    # set args
    seed = args.seed                  # DEVIATION: from CLI (notebook: 0)
    torch.manual_seed(seed)
    cfg = S["cfg"]                    # DEVIATION: from config (notebook: 4)
    class_labels = (args.class_id,)   # DEVIATION: from CLI (notebook: 8 labels)
    more_smooth = S["more_smooth"]

    # seed
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # run faster
    tf32 = True
    torch.backends.cudnn.allow_tf32 = bool(tf32)
    torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
    torch.set_float32_matmul_precision('high' if tf32 else 'highest')

    # DEVIATION ("matched" only): restrict SDPA to the config's kernel, as the runner does
    backend = C["attention"]["sdpa_backend"]
    if variant == "matched" and backend != "auto" and device == "cuda":
        kernel = torch.backends.cuda.sdp_kernel(
            enable_math=backend == "math", enable_flash=backend == "flash",
            enable_mem_efficient=backend == "mem_efficient")
    else:
        kernel = contextlib.nullcontext()

    # sample
    B = len(class_labels)
    label_B: torch.LongTensor = torch.tensor(class_labels, device=device)
    with torch.inference_mode():
        with torch.autocast('cuda', enabled=True, dtype=torch.float16, cache_enabled=True):    # using bfloat16 can be faster
            with kernel:
                recon_B3HW = var.autoregressive_infer_cfg(B=B, label_B=label_B, cfg=cfg, top_k=S["top_k"], top_p=S["top_p"], g_seed=seed, more_smooth=more_smooth)  # DEVIATION: top_k/top_p from config (notebook: 900/0.95)
    torch.save(recon_B3HW.cpu().clone(), f"{args.out}_{variant}.pt")   # added: raw tensor for comparison

    chw = torchvision.utils.make_grid(recon_B3HW, nrow=8, padding=0, pad_value=1.0)
    # DEVIATION: mul instead of mul_. For B=1 make_grid returns the inference tensor itself, and the
    # notebook's in-place mul_ raises outside inference_mode (it only works for the notebook's 8 labels).
    chw = chw.permute(1, 2, 0).mul(255).cpu().numpy()
    chw = PImage.fromarray(chw.astype(np.uint8))
    chw.save(f"{args.out}_{variant}.png")   # DEVIATION: save instead of chw.show()


args.out.parent.mkdir(parents=True, exist_ok=True)
for variant in ("asis", "matched"):
    cell2(variant)
    print(f"wrote {args.out}_{variant}.png")
Path(f"{args.out}.json").write_text(json.dumps({"notebook": "third_party/VAR/demo_sample.ipynb", **fused,
                                                "config_sdpa_backend": C["attention"]["sdpa_backend"]}, indent=2))
