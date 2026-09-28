"""Verify third_party submodules are at their pinned commits and unmodified.

Usage: python scripts/check_third_party.py   (exit code 1 on any problem)
"""

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SUBMODULES = ["third_party/VAR", "third_party/DiT"]


def git(*args, cwd=REPO_ROOT):
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout


def main():
    ok = True
    for path in SUBMODULES:
        # Leading char: ' ' = at pinned commit, '+' = different commit, '-' = not initialised
        status = git("submodule", "status", "--", path).strip("\n")
        flag, sha = status[0], status[1:].split()[0]
        if flag == "-":
            print(f"[FAIL] {path}: not initialised (run: git submodule update --init)")
            ok = False
            continue
        if flag == "+":
            print(f"[FAIL] {path}: checked out {sha[:12]}, not the pinned commit")
            ok = False
        dirty = git("status", "--porcelain", cwd=REPO_ROOT / path)
        if dirty:
            print(f"[FAIL] {path}: has local modifications:\n{dirty}")
            ok = False
        if flag == " " and not dirty:
            print(f"[ OK ] {path} @ {sha[:12]}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
