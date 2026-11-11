# Gate B pilot results (middle band, frozen 224-image manifest)

Generated from `metrics.csv`, `pilot.json` and `run.json` of VAR job 25969141 and DiT job 25969145. Per-image values, n = 224, batch 16, λ = 0 for VAR. `incep_cos` = Inception cosine distance to the paired baseline image (lower = closer to baseline). `cls_prob` = classifier probability of the image's own class (higher = better). Both are P3's metrics; neither is pixel-space.

## VAR-d20

Baseline `cls_prob` = 0.8122.

### Means, and the stages each arm intervened on

| m (scales) | arm | stages | incep_cos | cls_prob |
|---|---|---|---|---|
| 1 | protect | [8] | 0.088 | 0.756 |
| 1 | damage | [4] | 0.212 | 0.566 |
| 1 | control | [7] | 0.129 | 0.673 |
| 2 | protect | [1, 8] | 0.250 | 0.723 |
| 2 | damage | [3, 5] | 0.431 | 0.165 |
| 2 | control | [3, 6] | 0.450 | 0.120 |
| 3 | protect | [6, 7, 8] | 0.375 | 0.175 |
| 3 | damage | [3, 4, 5] | 0.546 | 0.001 |
| 3 | control | [7, 8, 9] | 0.441 | 0.064 |

### Paired difference against the control (mean, 95% CI over 224 images)

| m | metric | protect − control | damage − control |
|---|---|---|---|
| 1 | incep_cos | -0.041 [-0.048, -0.034] | +0.082 [+0.069, +0.095] |
| 1 | cls_prob | +0.083 [+0.054, +0.113] | -0.107 [-0.155, -0.060] |
| 2 | incep_cos | -0.200 [-0.219, -0.180] | -0.018 [-0.029, -0.008] |
| 2 | cls_prob | +0.603 [+0.555, +0.652] | +0.045 [+0.017, +0.072] |
| 3 | incep_cos | -0.067 [-0.078, -0.056] | +0.104 [+0.094, +0.115] |
| 3 | cls_prob | +0.111 [+0.078, +0.145] | -0.063 [-0.084, -0.041] |

## DiT-XL/2

Baseline `cls_prob` = 0.7587.

### Means, and the stages each arm intervened on

DiT stage ranges are first–last skipped timestep. The steps are strided inside each arm's eligible set, not contiguous: protect and control skip across the whole schedule, damage inside t 337–666. Exact lists are in each run's `pilot.json`.

| m (timesteps) | arm | stages | incep_cos | cls_prob |
|---|---|---|---|---|
| 25 | protect | 25 steps, t 16-975 | 0.190 | 0.754 |
| 25 | damage | 25 steps, t 337-666 | 0.221 | 0.756 |
| 25 | control | 25 steps, t 4-987 | 0.190 | 0.775 |
| 50 | protect | 50 steps, t 8-983 | 0.217 | 0.728 |
| 50 | damage | 50 steps, t 337-666 | 0.251 | 0.694 |
| 50 | control | 50 steps, t 4-991 | 0.227 | 0.699 |
| 75 | protect | 75 steps, t 36-967 | 0.237 | 0.743 |
| 75 | damage | 75 steps, t 337-666 | 0.261 | 0.678 |
| 75 | control | 75 steps, t 8-915 | 0.261 | 0.701 |

### Paired difference against the control (mean, 95% CI over 224 images)

| m | metric | protect − control | damage − control |
|---|---|---|---|
| 25 | incep_cos | -0.001 [-0.015, +0.014] | +0.031 [+0.017, +0.044] |
| 25 | cls_prob | -0.022 [-0.064, +0.020] | -0.020 [-0.062, +0.023] |
| 50 | incep_cos | -0.010 [-0.025, +0.005] | +0.024 [+0.008, +0.040] |
| 50 | cls_prob | +0.029 [-0.021, +0.078] | -0.005 [-0.061, +0.051] |
| 75 | incep_cos | -0.024 [-0.039, -0.009] | +0.001 [-0.015, +0.017] |
| 75 | cls_prob | +0.042 [-0.011, +0.095] | -0.023 [-0.076, +0.031] |
