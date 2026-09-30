# Illustration of the hook API only; the corruption/severity design is P2's choice.
# Adds Gaussian noise to the VAR accumulator after scale 3. Copy into your lane (e.g. lanes/p2/hooks.py)
# and run from the repo root:  --hook lanes/p2/hooks.py:noise_scale_3
import dataclasses

import torch

from runner.hooks import Modify

SEVERITY = 0.5   # illustrative only; P2's severity schedule is P2's decision


def add_noise(model, stage, p, when, state):
    f = state.f_hat
    # Hook randomness comes from our OWN generators (never the images' generators), seeded per image
    # and stage, so each image's noise does not depend on which batch it is in.
    noise = torch.stack([
        torch.randn(f.shape[1:], generator=torch.Generator(device=f.device).manual_seed(10_000 * seed + stage),
                    device=f.device, dtype=f.dtype)
        for (_class_id, seed) in state.rows])
    scale = f.flatten(1).std(dim=1).view(-1, 1, 1, 1)   # per image, so it doesn't depend on the batch
    return dataclasses.replace(state, f_hat=f + SEVERITY * scale * noise)


noise_scale_3 = Modify(add_noise, when="after", stages=[3], name="noise_scale3_sev0.5")
