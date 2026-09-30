"""Batch-1 regression for the phase3_check job: plain run vs expected hash, hooked run vs plain run.

    python scripts/phase3/regression.py --out regression.json  NAME PLAIN_PT HOOKED_PT EXPECTED [...]

EXPECTED is a sha256 prefix the plain run's tensor must start with, or `record` to record the plain
run's hash without checking it (VAR after the switch to strict fp32, until a reference is pinned).
The hooked run (no-op Modify + Observe on both sides) must always hash exactly like the plain run;
HOOKED_PT `-` means there is no hooked run (only the plain hash is recorded or checked).
Exit 1 if any check fails. Hashes use runner.generate.tensor_sha256.
"""

import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch  # noqa: E402

from runner.generate import tensor_sha256  # noqa: E402


def sha(pt: str):
    try:
        return tensor_sha256(torch.load(pt)), None
    except OSError as e:
        return None, str(e)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("items", nargs="+", help="NAME PLAIN_PT HOOKED_PT EXPECTED, repeated")
    args = ap.parse_args()
    if len(args.items) % 4:
        ap.error("items come in groups of 4: NAME PLAIN_PT HOOKED_PT EXPECTED")

    res, ok = {}, True
    for i in range(0, len(args.items), 4):
        name, plain_pt, hooked_pt, expected = args.items[i:i + 4]
        plain, err_p = sha(plain_pt)
        has_hooked = hooked_pt != "-"
        hooked, err_h = sha(hooked_pt) if has_hooked else (None, None)
        record_only = expected.lower() == "record"
        match = None if record_only or plain is None else plain.startswith(expected.lower())
        same = plain is not None and (plain == hooked if has_hooked else True)
        passed = same and (record_only or bool(match))
        res[name] = {"plain_sha256": plain, "hooked_sha256": hooked, "expected_prefix": None if record_only else expected,
                     "mode": "record" if record_only else "check", "matches_expected": match,
                     "hooked_equals_plain": same if has_hooked else None, "pass": passed,
                     **({"errors": [e for e in (err_p, err_h) if e]} if err_p or err_h else {})}
        ok &= passed
        print(f"{name}: plain {(plain or 'MISSING')[:12]} "
              + ("(recorded)" if record_only else f"expected {expected} {'MATCH' if match else 'MISMATCH'}")
              + (f"; hooked {'==' if same else '!='} plain" if has_hooked else "")
              + f"  {'PASS' if passed else 'FAIL'}")
    args.out.write_text(json.dumps(res, indent=2))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
