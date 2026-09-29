"""Put exactly one upstream repo (VAR or DiT) on sys.path per process.

Both repos use a top-level `models` module (and VAR also `utils`/`dist`, DiT
`diffusion`/`download`), so importing both in one interpreter silently picks
whichever comes first on sys.path. activate() refuses to let that happen.
"""

import os
import sys
from pathlib import Path

from runner import DIT_ROOT, VAR_ROOT

ROOTS = {"var": VAR_ROOT, "dit": DIT_ROOT}
# Top-level importable names each repo provides.
TOP_LEVEL = {
    "var": ("models", "utils", "dist", "trainer"),
    "dit": ("models", "diffusion", "download"),
}

_active = None


def activate(repo: str) -> Path:
    """Make `repo` importable. Call once per process, before any upstream import."""
    global _active
    if _active is not None:
        if _active == repo:
            return ROOTS[repo]
        raise RuntimeError(f"upstream {_active!r} already active in this process; cannot also load {repo!r}")

    # Never write __pycache__ into third_party/ (also inherited by child processes).
    sys.dont_write_bytecode = True
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

    clashes = sorted({n for names in TOP_LEVEL.values() for n in names if n in sys.modules})
    if clashes:
        raise RuntimeError(f"modules {clashes} already imported before activate({repo!r}); refusing to mix repos")

    root = ROOTS[repo]
    sys.path.insert(0, str(root))
    _active = repo
    return root


def check_origin(module, repo: str):
    """Assert an imported upstream module really came from `repo`'s checkout."""
    origin = Path(module.__file__).resolve()
    if not origin.is_relative_to(ROOTS[repo].resolve()):
        raise RuntimeError(f"{module.__name__} imported from {origin}, expected under {ROOTS[repo]}")
