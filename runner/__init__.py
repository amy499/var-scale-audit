"""Shared generation runner for VAR and DiT."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
THIRD_PARTY = REPO_ROOT / "third_party"
VAR_ROOT = THIRD_PARTY / "VAR"
DIT_ROOT = THIRD_PARTY / "DiT"
CHECKPOINTS_DIR = REPO_ROOT / "checkpoints"
OUTPUTS_DIR = REPO_ROOT / "outputs"
