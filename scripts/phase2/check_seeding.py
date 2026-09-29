"""CPU check that each image depends only on its own seed, using tiny random-weight models.

    python scripts/phase2/check_seeding.py [--manifest manifest/provisional_4x4.csv] [--work DIR]

Builds tiny VAR and DiT models with random weights (same code paths as the real ones, small
sizes), then for every config runs the manifest through runner.generate (one process per run).

Asserted (identical tensor hash for every row; exit code 0 only if both pass):
  repeat     same manifest, batch 4, run twice
  upstream   runner at batch 1 vs the plain upstream call at B=1 with the same seed
             (scripts/phase2/upstream_ref.py: VAR g_seed=seed; DiT torch.manual_seed + p_sample_loop)
Measured only (count of identical rows and max|diff| of the raw tensors; never fails):
  batch      batch 1 vs 4, and 3 vs 4 (3 leaves a partial last batch). Each row's random draws
             do not depend on batch size, but batched kernels can round differently.
Needs no checkpoints, GPU or network.
"""

import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import tempfile  # noqa: E402
from pathlib import Path  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

BATCH_SIZES = (4, 1, 3)
BASE = {
    "device": "cpu",
    "attention": {"sdpa_backend": "math"},
    "precision": {"autocast_dtype": None, "tf32": True, "cudnn_deterministic": True},
}
VAR_BUILD = {"depth": 2, "patch_nums": [1, 2, 3, 4], "V": 4096, "Cvae": 32, "ch": 32,
             "share_quant_resi": 4, "num_classes": 1000, "shared_aln": False}
VAR_SAMPLER = {"cfg": 1.5, "top_k": 900, "top_p": 0.96, "more_smooth": False}   # as configs/var_d20.yaml
DIT_BUILD = {"model": "DiT-S/2", "image_size": 64, "num_classes": 1000}


def make_tiny(repo: str, work: Path):
    """Write random-weight checkpoints and configs for `repo` into `work` (own process: one repo per process)."""
    import torch
    import yaml

    from runner import upstream
    torch.manual_seed(1234)
    upstream.activate(repo)
    configs = {}
    if repo == "var":
        from models import build_vae_var
        b = {**VAR_BUILD, "patch_nums": tuple(VAR_BUILD["patch_nums"])}
        vae, var = build_vae_var(device="cpu", flash_if_available=False, fused_if_available=False, **b)
        # build_vae_var disables default init; the VQVAE's parameters are uninitialised memory.
        with torch.no_grad():
            for name, p in vae.named_parameters():
                if p.ndim > 1:
                    p.normal_(0, p[0].numel() ** -0.5)
                else:
                    p.normal_(1.0 if name.endswith("weight") else 0.0, 0.1)
        torch.save(vae.state_dict(), work / "var_tiny_vae.pth")
        torch.save(var.state_dict(), work / "var_tiny.pth")
        ckpts = {"vae": {"path": str(work / "var_tiny_vae.pth")}, "var": {"path": str(work / "var_tiny.pth")}}
        for name, smooth in (("var_tiny", False), ("var_tiny_smooth", True)):
            configs[name] = {"model": "var", **BASE, "checkpoints": ckpts, "build": VAR_BUILD,
                             "sampler": {**VAR_SAMPLER, "more_smooth": smooth}}
    else:
        from diffusers.models import AutoencoderKL
        from models import DiT_models
        dit = DiT_models[DIT_BUILD["model"]](input_size=DIT_BUILD["image_size"] // 8,
                                             num_classes=DIT_BUILD["num_classes"])
        with torch.no_grad():   # DiT zero-inits adaLN and the final layer; perturb so the net does something
            for p in dit.parameters():
                p.add_(torch.randn_like(p) * 0.02)
        torch.save(dit.state_dict(), work / "dit_tiny.pt")
        vae = AutoencoderKL(down_block_types=("DownEncoderBlock2D",) * 4, up_block_types=("UpDecoderBlock2D",) * 4,
                            block_out_channels=(32,) * 4, layers_per_block=1, latent_channels=4,
                            norm_num_groups=32, sample_size=DIT_BUILD["image_size"])
        vae.save_pretrained(work / "dit_tiny_vae")
        ckpts = {"dit": {"path": str(work / "dit_tiny.pt")}, "vae": {"path": str(work / "dit_tiny_vae")}}
        for name, cfg_scale in (("dit_tiny", 1.5), ("dit_tiny_nocfg", 1.0)):
            configs[name] = {"model": "dit", **BASE, "checkpoints": ckpts, "build": DIT_BUILD,
                             "sampler": {"cfg_scale": cfg_scale, "num_sampling_steps": 20}}
    for name, c in configs.items():
        (work / f"{name}.yaml").write_text(yaml.safe_dump(c, sort_keys=False))


# Must hold exactly (sha256 of the raw tensor, every row); these decide the exit code.
ASSERTED = {
    "repeat (bs4 run A == run B)": ("bs4_a", "bs4_b"),
    "upstream (runner bs1 == upstream B=1)": ("bs1", "upstream_b1"),
}
# Not expected to be bit-identical (batched kernels round differently); reported only.
MEASURED = {
    "batch (bs1 vs bs4)": ("bs1", "bs4_a"),
    "batch (bs3 vs bs4)": ("bs3", "bs4_a"),
}


def max_abs_diff(out: Path, a: str, b: str, stems: list[str]) -> float:
    """Largest elementwise difference between two runs' raw tensors over the given rows.

    A different random draw changes an image by O(1); float rounding from batched kernels by ~1e-5.
    """
    import torch
    return max(float((torch.load(out / a / f"{s}.pt").double() - torch.load(out / b / f"{s}.pt").double()).abs().max())
               for s in stems)


def run(*args):
    subprocess.run([sys.executable, *map(str, args)], cwd=REPO, check=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, default=REPO / "manifest" / "provisional_4x4.csv")
    ap.add_argument("--work", type=Path, help="scratch dir (default: a new temp dir)")
    ap.add_argument("--make-tiny", nargs=2, metavar=("REPO", "DIR"), help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.make_tiny:
        return make_tiny(args.make_tiny[0], Path(args.make_tiny[1]))

    work = args.work or Path(tempfile.mkdtemp(prefix="check_seeding_"))
    work.mkdir(parents=True, exist_ok=True)
    manifest = args.manifest.resolve()
    me = Path(__file__).resolve()
    for repo in ("var", "dit"):
        run(me, "--make-tiny", repo, work)

    report, ok = {}, True
    for name in ("var_tiny", "var_tiny_smooth", "dit_tiny", "dit_tiny_nocfg"):
        config = work / f"{name}.yaml"
        runs = {}
        for tag, bs in [("bs4_a", 4), ("bs4_b", 4), ("bs1", 1), ("bs3", 3)]:
            out = work / "out" / name / tag
            run("-m", "runner.generate", "--config", config, "--manifest", manifest, "--batch-size", bs, "--out-dir", out)
            rows = json.loads((out / "run.json").read_text())["rows"]
            runs[tag] = {r["file"]: r["tensor_sha256"] for r in rows}
        ref = work / "out" / name / "upstream_b1"
        run(REPO / "scripts" / "phase2" / "upstream_ref.py", "--config", config, "--manifest", manifest, "--out-dir", ref)
        runs["upstream_b1"] = json.loads((ref / "hashes.json").read_text())

        rows = list(runs["bs1"])
        rep = report[name] = {"n_rows": len(rows), "distinct_hashes": len(set(runs["bs1"].values())),
                              "asserted": {}, "measured": {}}
        for label, (a, b) in ASSERTED.items():
            diff = [r for r in rows if runs[a][r] != runs[b].get(r)]
            rep["asserted"][label] = {"identical_rows": len(rows) - len(diff), "differing": diff}
            ok &= not diff
        for label, (a, b) in MEASURED.items():
            rep["measured"][label] = {
                "identical_rows": sum(runs[a][r] == runs[b][r] for r in rows),
                "max_abs_diff": max_abs_diff(work / "out" / name, a, b, rows),
            }
        rep["hashes_bs1"] = runs["bs1"]

    (work / "report.json").write_text(json.dumps(report, indent=2))
    print()
    for name, rep in report.items():
        n = rep["n_rows"]
        print(f"{name}: {n} rows, {rep['distinct_hashes']} distinct images")
        for label, c in rep["asserted"].items():
            status = "PASS" if not c["differing"] else f"FAIL {c['differing']}"
            print(f"    assert   {label:38s} {c['identical_rows']:2d}/{n} identical  {status}")
        for label, c in rep["measured"].items():
            print(f"    measure  {label:38s} {c['identical_rows']:2d}/{n} identical  max|diff| {c['max_abs_diff']:.1e}")
    print(f"\nreport: {work / 'report.json'}")
    print("ASSERTED CHECKS PASS" if ok else "ASSERTED CHECKS FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
