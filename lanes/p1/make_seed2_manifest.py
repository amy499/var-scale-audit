"""P1 Phase E manifest: a second, independent seed set for the same 7 classes. Standard library + numpy.

    python -m lanes.p1.make_seed2_manifest
    -> lanes/p1/manifests/seed2.csv  (header class_id,seed; LF line endings; 224 rows)

Same classes and per-image layout as manifest v1.1 (scripts/manifest/make_manifest_v1_1.py): 7 classes x 32
images, one seed per image, class 151 takes derived seeds 0-31, class 153 takes 32-63, and so on. Only the
declared master seed differs: 61320017 + 1 = 61320018 (v1.1 uses 61320017). Seeds come from v1.1's own
derive_seeds (SeedSequence(master).generate_state, mod 2**31 - 1), imported, not copied. SeedSequence entropy
61320018 gives a stream unrelated to 61320017's (not a shifted prefix of it); the script still checks that no
seed is shared with manifest/frozen_v1_1.csv and stops if one is. Re-running reproduces the file byte for byte.
"""

import hashlib
import importlib.util
import sys
from pathlib import Path

sys.dont_write_bytecode = True

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from runner.manifest import load_manifest  # noqa: E402

_spec = importlib.util.spec_from_file_location("make_manifest_v1_1",
                                               REPO_ROOT / "scripts" / "manifest" / "make_manifest_v1_1.py")
V11 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(V11)

MASTER_SEED = V11.MASTER_SEED + 1   # 61320018
assert MASTER_SEED == 61320018
CLASSES, IMAGES_PER_CLASS, BATCH_SIZE = V11.CLASSES, V11.IMAGES_PER_CLASS, V11.BATCH_SIZE
OUT = REPO_ROOT / "lanes" / "p1" / "manifests" / "seed2.csv"
SEED1 = REPO_ROOT / "manifest" / "frozen_v1_1.csv"


def build() -> list[tuple[int, int]]:
    seeds = V11.derive_seeds(MASTER_SEED, len(CLASSES) * IMAGES_PER_CLASS)
    return [(c, seeds[k * IMAGES_PER_CLASS + i]) for k, c in enumerate(CLASSES) for i in range(IMAGES_PER_CLASS)]


def main():
    rows = build()
    seeds = [s for _, s in rows]
    assert len(rows) == 224, len(rows)
    assert len(set(seeds)) == len(seeds), "per-image seeds are not unique"
    assert len(rows) % BATCH_SIZE == 0

    seed1 = {r.seed for r in load_manifest(SEED1)}
    shared = sorted(set(seeds) & seed1)
    if shared:
        raise SystemExit(f"ERROR: {len(shared)} seed(s) also in {SEED1.name}: {shared[:10]}")

    text = "class_id,seed\n" + "".join(f"{c},{s}\n" for c, s in rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_bytes(text.encode("ascii"))   # bytes: LF line endings on every platform
    loaded = load_manifest(OUT)
    assert [(r.class_id, r.seed) for r in loaded] == rows, "load_manifest disagrees with what was written"

    print(f"seed set 2: {OUT.relative_to(REPO_ROOT).as_posix()}")
    print(f"{len(CLASSES)} classes x {IMAGES_PER_CLASS} = {len(rows)} rows, master seed {MASTER_SEED}, "
          f"{len(set(seeds))} unique seeds, seed sum {sum(seeds)}")
    print(f"overlap with {SEED1.relative_to(REPO_ROOT).as_posix()} ({len(seed1)} seeds): none")
    print(f"sha256 {hashlib.sha256(OUT.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
