"""Frozen ImageNet manifest v1.1 (DRAFT, pending P3 sign-off): one seed per image.

    python scripts/manifest/make_manifest_v1_1.py
    -> manifest/frozen_v1_1.csv  (header class_id,seed; LF line endings; 224 rows)

Same classes and master seed as P3's v1.0 (make_manifest_v1.0_P3.py), but v1.0 gives one seed per
batch of 16 while the runner seeds every image from its own row (runner/generate.py). v1.1 therefore
derives 224 per-image seeds from the master seed, taken in order: class 151 gets seeds 0-31, class 153
gets 32-63, and so on. SeedSequence output is prefix-consistent, so the first 14 seeds equal v1.0's 14
batch seeds. Re-running reproduces the file byte for byte. See docs/manifest_frozen.md.

Standard library + numpy only.
"""

import hashlib
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from runner.manifest import load_manifest  # noqa: E402  (standard library only)

MANIFEST_VERSION = "1.1"
MASTER_SEED = 61320017
IMAGES_PER_CLASS = 32
BATCH_SIZE = 16
# ILSVRC-2012 / ImageNet-1k class indices (0-indexed), in manifest order.
CLASSES = [151, 153, 162, 207, 235, 254, 258]
OUT = REPO_ROOT / "manifest" / "frozen_v1_1.csv"


def derive_seeds(master: int, n: int) -> list[int]:
    ss = np.random.SeedSequence(master)
    return [int(s) for s in ss.generate_state(n, dtype=np.uint32) % (2**31 - 1)]


def build() -> list[tuple[int, int]]:
    n = len(CLASSES) * IMAGES_PER_CLASS
    seeds = derive_seeds(MASTER_SEED, n)
    return [(c, seeds[k * IMAGES_PER_CLASS + i]) for k, c in enumerate(CLASSES) for i in range(IMAGES_PER_CLASS)]


def main():
    rows = build()
    seeds = [s for _, s in rows]
    assert len(rows) == 224, len(rows)
    assert len(set(seeds)) == len(seeds), "per-image seeds are not unique"
    assert len(rows) % BATCH_SIZE == 0, f"{len(rows)} rows is not a multiple of batch size {BATCH_SIZE}"

    text = "class_id,seed\n" + "".join(f"{c},{s}\n" for c, s in rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_bytes(text.encode("ascii"))   # bytes: LF line endings on every platform

    loaded = load_manifest(OUT)
    assert [(r.class_id, r.seed) for r in loaded] == rows, "load_manifest disagrees with what was written"
    assert len(loaded) % BATCH_SIZE == 0

    print(f"manifest v{MANIFEST_VERSION}: {OUT.relative_to(REPO_ROOT).as_posix()}")
    print(f"{len(CLASSES)} classes x {IMAGES_PER_CLASS} = {len(rows)} rows, master seed {MASTER_SEED}, "
          f"{len(set(seeds))} unique seeds, seed sum {sum(seeds)}")
    print(f"sha256 {hashlib.sha256(OUT.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
