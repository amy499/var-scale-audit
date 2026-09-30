# Modify hook: remove VAR scale 2's contribution for every image ("skip-scale", version b).
# Run from the repo root:  --hook scripts/examples/skip_scale_2.py:skip_scale_2
import dataclasses

from runner.hooks import Modify


def drop_scale(model, stage, p, when, state):
    return dataclasses.replace(state, f_hat=state.f_hat_before)


skip_scale_2 = Modify(drop_scale, when="after", stages=[2], name="skip_scale_2")
