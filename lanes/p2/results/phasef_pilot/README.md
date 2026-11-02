# P2 Phase F pilot: DiT corruption on real crystal images (job `26089558`)

Optional transfer check (`lanes/p2/transfer.py`, `jobs/phasef.pbs`): 16 real microscopy images, a real
image's VAE latent noised to a starting timestep and denoised by DiT-XL/2 (cfg 1, class 0 = tench),
once clean and once with `noise_x` at timestep 197 (severity 1, `sigma`).

Data: OpenCrystalData, "In-line Image Characterization using EasyViewer" (CC-BY-SA-4.0;
https://www.kaggle.com/datasets/opencrystaldata/easyviewer-based-image-characterization; Barhate et al.,
Digital Chemical Engineering, 2024). 16 files evenly spaced over the sorted file list (12 Wollastonite,
4 Glutamic Acid; 8 at 100x, 8 at 400x), saved as `ocd01_x100.png` ... `ocd16_x400.png`.

| Run | PSNR to the real image (dB) | Texture ratio (hf_ratio) |
|---|---|---|
| VAE reconstruction only | 29.3 | 0.88 |
| start t96 | 23.9 | 0.94 |
| start t297 | 19.6 | 0.89 |
| start t297 + noise at 197 | 15.2 | 1.27 |
| start t497 | 15.9 | 1.81 |
| start t497 + noise at 197 | 13.3 | 1.73 |

Corrupted vs clean from the same start: `img_mse` 0.018 (t297), 0.021 (t497); ImageNet, same setting: 0.028.

- **The codec works:** reconstructions keep the crystal edges (24.5-34.1 dB).
- **The corruption is applied and its damage persists:** speckle and broken crystal edges, layout kept
  (`grid.jpg`, columns 4-6), as at timestep 197 on ImageNet.
- **From t297 the class prior takes over:** at t497 most images become a person holding a fish (class 0,
  tench). Only starts at or below about t197 measure the crystals; there is no null class in the runner.

`grid.jpg`: 6 of the 16 images; columns: input, VAE reconstruction, start t96, t297, t297 + noise,
|difference| x 3, t497, t497 + noise.
