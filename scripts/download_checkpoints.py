"""Download the checkpoints a set of configs needs into checkpoints/.

Meant for the cluster login node: standard library + PyYAML only (no torch, no GPU).

    python scripts/download_checkpoints.py                                  # default: VAR-d20 + DiT
    python scripts/download_checkpoints.py --configs configs/var_d24.yaml   # named configs
    python scripts/download_checkpoints.py configs/var_d24.yaml             # same (positional form)
    python scripts/download_checkpoints.py --all-var-depths                 # VAR d16, d20, d24, d30
    python scripts/download_checkpoints.py --dry-run [...]                  # list files only

Files shared by several configs (e.g. the VAR VQVAE) are fetched once.

- Resumes interrupted downloads (<file>.part + HTTP Range).
- Verifies sha256 when the config pins one; otherwise records it in <file>.sha256.
- HF_ENDPOINT (e.g. a mirror) replaces https://huggingface.co; https_proxy is honoured.
"""

import argparse
import hashlib
import os
import sys
import urllib.request
from pathlib import Path

sys.dont_write_bytecode = True
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from runner.config import iter_downloads, load_config  # noqa: E402

CHUNK = 8 << 20
DEFAULT_CONFIGS = ("configs/var_d20.yaml", "configs/dit_xl2_256.yaml")   # the primary VAR depth and DiT
VAR_DEPTH_CONFIGS = tuple(f"configs/var_d{d}.yaml" for d in (16, 20, 24, 30))


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(CHUNK):
            h.update(block)
    return h.hexdigest()


def apply_endpoint(url: str) -> str:
    endpoint = os.environ.get("HF_ENDPOINT")
    if endpoint and url.startswith("https://huggingface.co/"):
        return endpoint.rstrip("/") + url[len("https://huggingface.co"):]
    return url


def fetch(url: str, dest: Path):
    part = dest.with_name(dest.name + ".part")
    have = part.stat().st_size if part.exists() else 0
    req = urllib.request.Request(url, headers={"User-Agent": "var-scale-audit"})
    if have:
        req.add_header("Range", f"bytes={have}-")
    with urllib.request.urlopen(req, timeout=60) as resp:
        if have and resp.status != 206:  # server ignored Range: start over
            have = 0
        total = have + int(resp.headers.get("Content-Length", 0))
        print(f"  {'resuming at ' + str(have >> 20) + ' MiB, ' if have else ''}total {total >> 20} MiB")
        with open(part, "ab" if have else "wb") as f:
            done, last_pct = have, -10
            while block := resp.read(CHUNK):
                f.write(block)
                done += len(block)
                pct = 100 * done // total if total else 0
                if pct >= last_pct + 10:
                    print(f"  {pct:3d}%", flush=True)
                    last_pct = pct
    part.replace(dest)


def ensure(dest: Path, url: str, expected: str | None) -> bool:
    rel = dest.relative_to(REPO_ROOT) if dest.is_relative_to(REPO_ROOT) else dest
    if dest.exists():
        print(f"[skip] {rel} exists")
    else:
        print(f"[get ] {rel}\n  <- {url}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        fetch(apply_endpoint(url), dest)

    digest = sha256_of(dest)
    if expected is None:
        dest.with_name(dest.name + ".sha256").write_text(f"{digest}  {dest.name}\n")
        print(f"  sha256 {digest} (not pinned in config; recorded)")
        return True
    if digest != expected:
        print(f"  [FAIL] sha256 {digest} != expected {expected}. Delete the file and re-run.")
        return False
    print("  sha256 OK")
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("positional", nargs="*", type=Path, metavar="CONFIG", help="config files (same as --configs)")
    ap.add_argument("--configs", nargs="+", type=Path, default=[], help="config files to download for")
    ap.add_argument("--all-var-depths", action="store_true", help="add " + ", ".join(VAR_DEPTH_CONFIGS))
    ap.add_argument("--dry-run", action="store_true", help="list the files that would be checked or downloaded, then stop")
    args = ap.parse_args()

    configs = [*args.positional, *args.configs]
    if args.all_var_depths:
        configs += [REPO_ROOT / c for c in VAR_DEPTH_CONFIGS]
    if not configs:
        configs = [REPO_ROOT / c for c in DEFAULT_CONFIGS]
    print("configs: " + ", ".join(Path(c).as_posix() for c in configs))
    seen, ok = set(), True
    for cfg_path in configs:
        for dest, url, sha in iter_downloads(load_config(cfg_path)):
            if dest not in seen:
                seen.add(dest)
                if args.dry_run:
                    rel = dest.relative_to(REPO_ROOT) if dest.is_relative_to(REPO_ROOT) else dest
                    print(f"  {rel.as_posix()}{'  (exists)' if dest.exists() else ''}  <- {url}")
                    continue
                ok &= ensure(dest, url, sha)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
