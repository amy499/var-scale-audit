"""Config loading and validation.

Only depends on PyYAML so scripts/download_checkpoints.py can use it on a login node.
Nothing here supplies defaults: every sampler/runtime setting must be in the YAML.
"""

from pathlib import Path

import yaml

from runner import REPO_ROOT

REQUIRED_TOP = ("model", "device", "checkpoints", "build", "sampler", "attention", "precision")
REQUIRED_CHECKPOINTS = {"var": {"vae", "var"}, "dit": {"dit", "vae"}}
REQUIRED_SAMPLER = {"var": {"cfg", "top_k", "top_p", "more_smooth"}, "dit": {"cfg_scale", "num_sampling_steps"}}
REQUIRED_PRECISION = {"autocast_dtype", "tf32", "cudnn_deterministic"}
SDPA_BACKENDS = ("math", "flash", "mem_efficient", "auto")
AUTOCAST_DTYPES = (None, "float16", "bfloat16")


class ConfigError(ValueError):
    pass


def resolve_path(p) -> Path:
    p = Path(p)
    return p if p.is_absolute() else REPO_ROOT / p


def _require_exact(section: dict, required: set, where: str):
    missing, extra = required - section.keys(), section.keys() - required
    if missing or extra:
        raise ConfigError(f"{where}: missing keys {sorted(missing)}, unknown keys {sorted(extra)}")


def load_config(path) -> dict:
    path = Path(path).resolve()
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    where = str(path)

    missing = [k for k in REQUIRED_TOP if k not in cfg]
    if missing:
        raise ConfigError(f"{where}: missing top-level keys {missing}")
    model = cfg["model"]
    if model not in REQUIRED_SAMPLER:
        raise ConfigError(f"{where}: model must be one of {sorted(REQUIRED_SAMPLER)}, got {model!r}")

    _require_exact(cfg["checkpoints"], REQUIRED_CHECKPOINTS[model], f"{where} [checkpoints]")
    _require_exact(cfg["sampler"], REQUIRED_SAMPLER[model], f"{where} [sampler]")
    _require_exact(cfg["precision"], REQUIRED_PRECISION, f"{where} [precision]")
    if cfg["attention"].get("sdpa_backend") not in SDPA_BACKENDS:
        raise ConfigError(f"{where} [attention]: sdpa_backend must be one of {SDPA_BACKENDS}")
    if cfg["precision"]["autocast_dtype"] not in AUTOCAST_DTYPES:
        raise ConfigError(f"{where} [precision]: autocast_dtype must be one of {AUTOCAST_DTYPES}")

    cfg["_path"] = str(path)
    return cfg


def iter_downloads(cfg: dict):
    """Yield (dest_path, url, sha256_or_None) for every file under [checkpoints]."""
    for name, entry in cfg["checkpoints"].items():
        if "files" in entry:  # directory of files (e.g. a diffusers model folder)
            root = resolve_path(entry["path"])
            for fname, f in entry["files"].items():
                yield root / fname, f["url"], f.get("sha256")
        else:
            yield resolve_path(entry["path"]), entry["url"], entry.get("sha256")
