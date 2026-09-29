"""Precision, attention-kernel and provenance helpers shared by both models."""

import contextlib
import platform
import subprocess

import torch

from runner import DIT_ROOT, VAR_ROOT

# torch 2.1 API: torch.backends.cuda.sdp_kernel(enable_flash, enable_math, enable_mem_efficient)
_SDPA_FLAGS = {
    "math": dict(enable_math=True, enable_flash=False, enable_mem_efficient=False),
    "flash": dict(enable_math=False, enable_flash=True, enable_mem_efficient=False),
    "mem_efficient": dict(enable_math=False, enable_flash=False, enable_mem_efficient=True),
}


def apply_precision(precision: dict):
    torch.backends.cuda.matmul.allow_tf32 = precision["tf32"]
    torch.backends.cudnn.allow_tf32 = precision["tf32"]
    torch.backends.cudnn.deterministic = precision["cudnn_deterministic"]


def autocast_ctx(precision: dict, device: torch.device):
    dtype = precision["autocast_dtype"]
    if dtype is None or device.type != "cuda":
        return contextlib.nullcontext()
    return torch.autocast("cuda", dtype=getattr(torch, dtype), enabled=True, cache_enabled=True)


def effective_precision(precision: dict, device: torch.device) -> dict:
    return {
        "autocast_dtype": precision["autocast_dtype"] if device.type == "cuda" else None,
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
    }


@contextlib.contextmanager
def sdpa_ctx(backend: str, device: torch.device):
    """Restrict torch SDPA to one CUDA kernel.

    With a single kernel enabled, SDPA raises instead of silently falling back,
    so a successful run proves that kernel was used.
    """
    if backend == "auto" or device.type != "cuda":
        yield
        return
    with torch.backends.cuda.sdp_kernel(**_SDPA_FLAGS[backend]):
        yield


def sdpa_record(backend: str, device: torch.device) -> dict:
    """Call inside sdpa_ctx: records the kernel flags actually in force."""
    if device.type != "cuda":
        return {"requested": backend, "effective": "cpu (CUDA kernel flags do not apply)"}
    flags = {
        "flash": torch.backends.cuda.flash_sdp_enabled(),
        "mem_efficient": torch.backends.cuda.mem_efficient_sdp_enabled(),
        "math": torch.backends.cuda.math_sdp_enabled(),
    }
    enabled = [k for k, v in flags.items() if v]
    effective = enabled[0] if len(enabled) == 1 else f"auto (PyTorch chooses among {enabled})"
    return {"requested": backend, "effective": effective, "cuda_sdp_flags": flags}


def row_generators(seeds: list[int], device: torch.device) -> list[torch.Generator]:
    """One generator per image, seeded with that image's own seed.

    Each row's random draws then depend only on its seed, not on batch size or the other rows.
    """
    return [torch.Generator(device=device).manual_seed(s) for s in seeds]


def _git_sha(path) -> str | None:
    try:
        return subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def environment_record(device: torch.device) -> dict:
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "third_party": {"VAR": _git_sha(VAR_ROOT), "DiT": _git_sha(DIT_ROOT)},
    }
