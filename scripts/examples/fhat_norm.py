# Observe hook: log the VAR accumulator's norm per image at every scale (docs/hooks_quickstart.md).
# Run from the repo root:  --hook scripts/examples/fhat_norm.py:fhat_norm
import json
import os
from pathlib import Path

from runner.hooks import Observe

LOG = Path(os.environ.get("FHAT_NORM_LOG", "outputs/examples/fhat_norms.jsonl"))


def log_fhat_norm(model, stage, p, when, state):
    norms = state.f_hat.float().flatten(1).norm(dim=1).tolist()   # one value per row
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a") as f:
        f.write(json.dumps({"stage": stage, "p": p, "rows": state.rows, "norm": norms}) + "\n")


fhat_norm = Observe(log_fhat_norm, when="after", name="fhat_norm")
