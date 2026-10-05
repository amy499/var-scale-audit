"""Shared logging / reproducibility schema on top of the runner's run.json (lanes/p2/SCHEMA.md).

Run from the repo root. Standard library only (no torch), so it also runs on a login node.

    python -m lanes.p2.schema record RUN_DIR --lane p2 --experiment NAME --arm NAME
                                     [--baseline BASELINE_DIR] [--param KEY=VALUE ...] [--vram-log FILE] [--note TEXT]
        -> RUN_DIR/experiment.json: what the run was for, which baseline it pairs with, the pairing
           check, code commit, job id and runtime/VRAM. run.json is never modified.

    python -m lanes.p2.schema check BASELINE_DIR RUN_DIR
        -> is RUN_DIR a valid paired comparison against BASELINE_DIR? Exit 1 if not.

    python -m lanes.p2.schema collect ROOT [--out DIR]
        -> runs.csv, images.csv, stages.csv, metrics.csv merged over every run under ROOT.

Lane code can also write per-image / per-stage numbers with append_metrics(run_dir, rows).
"""

import argparse
import csv
import datetime
import hashlib
import json
import os
import socket
import subprocess
import sys
from pathlib import Path

sys.dont_write_bytecode = True

SCHEMA_VERSION = 1
REPO = Path(__file__).resolve().parents[2]

# Settings a baseline and the run compared with it must share (docs/HANDOVER.md §3). Dotted paths into run.json.
MUST_MATCH = ("model", "depth", "device", "sampler", "build", "precision", "batch_size", "allow_partial_batch",
              "attention.sdpa_kernel.effective", "env.torch", "env.cuda", "env.third_party")
SHOULD_MATCH = ("env.gpu", "env.cudnn", "env.python")   # a difference is reported as a warning

METRIC_COLUMNS = ("class_id", "seed", "metric", "value", "stage")   # stage: native stage id, empty = final image
RUN_COLUMNS = ("run_id", "lane", "experiment", "arm", "role", "baseline_run_id", "pairing_ok", "model", "config",
               "depth", "manifest", "n_images", "batch_size", "hooks", "skip_timesteps", "params", "seconds_load",
               "seconds_sample", "seconds_per_image", "peak_vram_mb", "gpu", "torch", "code_commit", "code_dirty",
               "pbs_jobid", "created_utc", "run_dir")
IMAGE_COLUMNS = ("run_id", "class_id", "seed", "file", "tensor_sha256", "generator_state_sha256",
                 "identical_to_baseline")


class SchemaError(ValueError):
    pass


# ---------------------------------------------------------------- reading

def load_run(run_dir) -> dict:
    path = Path(run_dir) / "run.json"
    if not path.is_file():
        raise SchemaError(f"{path} not found: not a runner.generate --manifest output directory")
    return json.loads(path.read_text())


def load_experiment(run_dir) -> dict | None:
    path = Path(run_dir) / "experiment.json"
    return json.loads(path.read_text()) if path.is_file() else None


def file_sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _get(d: dict, dotted: str):
    for key in dotted.split("."):
        d = d.get(key) if isinstance(d, dict) else None
    return d


def _rows(run: dict) -> list[tuple]:
    return [(r["class_id"], r["seed"]) for r in run["rows"]]


# ---------------------------------------------------------------- pairing check

def check_pairing(baseline: dict, run: dict) -> dict:
    """Is `run` a valid paired comparison against `baseline` (two run.json dicts)?

    Errors (ok = False): a frozen setting differs, the images differ in number or order, the baseline
    itself was modified, or a row's generator state differs (the two runs did not consume the same
    random numbers). Warnings do not fail the check.
    """
    errors, warnings = [], []
    for key in MUST_MATCH:
        a, b = _get(baseline, key), _get(run, key)
        if a != b:
            errors.append(f"{key} differs: baseline {a!r}, run {b!r}")
    for key in SHOULD_MATCH:
        a, b = _get(baseline, key), _get(run, key)
        if a != b:
            warnings.append(f"{key} differs: baseline {a!r}, run {b!r}")
    for name, r in (("baseline", baseline), ("run", run)):
        if r.get("allow_partial_batch"):
            errors.append(f"{name} used --allow-partial-batch")
    modifying = [h["name"] for h in baseline.get("hooks", []) if h.get("kind") == "modify"]
    if modifying or baseline.get("skip_timesteps"):
        errors.append(f"baseline is not clean: modify hooks {modifying}, skip_timesteps {baseline.get('skip_timesteps')}")
    if Path(baseline["config_path"]).name != Path(run["config_path"]).name:
        warnings.append(f"config file names differ: {baseline['config_path']} vs {run['config_path']}")

    changed = identical = 0
    if _rows(baseline) != _rows(run):
        errors.append(f"manifest rows differ (baseline {len(baseline['rows'])} rows, run {len(run['rows'])} rows, "
                      "or a different order)")
    else:
        unpaired = [f"class {a['class_id']} seed {a['seed']}" for a, b in zip(baseline["rows"], run["rows"])
                    if a["generator_state_sha256"] != b["generator_state_sha256"]]
        if unpaired:
            errors.append(f"{len(unpaired)} rows are not paired (generator state differs), e.g. {unpaired[:3]}")
        identical = sum(a["tensor_sha256"] == b["tensor_sha256"] for a, b in zip(baseline["rows"], run["rows"]))
        changed = len(run["rows"]) - identical
    return {"ok": not errors, "errors": errors, "warnings": warnings, "n_rows": len(run["rows"]),
            "rows_changed": changed, "rows_identical": identical}


# ---------------------------------------------------------------- experiment.json

def _git(*args) -> str | None:
    try:
        return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True, check=True).stdout.rstrip("\n")
    except (OSError, subprocess.CalledProcessError):
        return None


def code_record() -> dict:
    """Commit of this repo (run.json only records the submodules).

    dirty: the working tree differs from that commit (edited or untracked files, ignored ones aside),
    so the commit alone does not reproduce the run; dirty_files names the first few.
    """
    status = _git("status", "--porcelain")
    files = None if status is None else [line[3:] for line in status.splitlines()]
    return {"commit": _git("rev-parse", "HEAD"), "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": None if files is None else bool(files), "dirty_files": files and files[:10]}


def peak_vram_mb(vram_log) -> float | None:
    """Max of an `nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -l 1` log (MiB)."""
    values = []
    for line in Path(vram_log).read_text().splitlines():
        try:
            values.append(float(line.split(",")[0]))
        except ValueError:
            continue
    return max(values) if values else None


def _run_id(lane: str, experiment: str, arm: str) -> str:
    for part in (lane, experiment, arm):
        if not part or "/" in part or "," in part or part != part.strip():
            raise SchemaError(f"lane/experiment/arm must be non-empty, without '/' or ',': got {part!r}")
    return f"{lane}/{experiment}/{arm}"


def record(run_dir, lane: str, experiment: str, arm: str, baseline_dir=None, params: dict | None = None,
           vram_log=None, note: str | None = None) -> dict:
    """Write RUN_DIR/experiment.json next to run.json and return it.

    Without baseline_dir the run is a baseline. With one, the run is an intervention: the pairing check
    is run and stored, and the baseline must already have its own experiment.json.
    """
    run_dir = Path(run_dir)
    run = load_run(run_dir)
    baseline = pairing = None
    if baseline_dir is not None:
        baseline_dir = Path(baseline_dir)
        base_exp = load_experiment(baseline_dir)
        if base_exp is None:
            raise SchemaError(f"{baseline_dir} has no experiment.json: record the baseline first")
        if base_exp["role"] != "baseline":
            raise SchemaError(f"{baseline_dir} is recorded as {base_exp['role']!r}, not a baseline")
        pairing = check_pairing(load_run(baseline_dir), run)
        baseline = {"run_id": base_exp["run_id"], "dir": os.path.relpath(baseline_dir, run_dir),
                    "run_json_sha256": file_sha256(baseline_dir / "run.json")}
    n = len(run["rows"])
    seconds = run.get("seconds", {})
    exp = {
        "schema_version": SCHEMA_VERSION,
        "run_id": _run_id(lane, experiment, arm),
        "lane": lane,
        "experiment": experiment,
        "arm": arm,
        "role": "baseline" if baseline is None else "intervention",
        "baseline": baseline,
        "params": params or {},
        "pairing": pairing,
        "code": code_record(),
        "job": {"pbs_jobid": os.environ.get("PBS_JOBID"), "host": socket.gethostname()},
        "resources": {
            "n_images": n,
            "seconds_load": seconds.get("load"),
            "seconds_sample": seconds.get("sample"),
            "seconds_per_image": round(seconds["sample"] / n, 4) if seconds.get("sample") is not None and n else None,
            "peak_vram_mb": peak_vram_mb(vram_log) if vram_log else None,
        },
        "run_json_sha256": file_sha256(run_dir / "run.json"),
        "created_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "note": note,
    }
    (run_dir / "experiment.json").write_text(json.dumps(exp, indent=2))
    return exp


# ---------------------------------------------------------------- metrics

def append_metrics(run_dir, rows) -> Path:
    """Append rows to RUN_DIR/metrics.csv (long format, one number per row).

    rows: dicts with class_id, seed, metric, value and optionally stage (the native stage id the number
    belongs to; leave out for a number about the final image).
    """
    path = Path(run_dir) / "metrics.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(METRIC_COLUMNS)
        for r in rows:
            unknown = set(r) - set(METRIC_COLUMNS)
            if unknown:
                raise SchemaError(f"unknown metric columns {sorted(unknown)}; allowed: {METRIC_COLUMNS}")
            stage = r.get("stage")
            w.writerow([int(r["class_id"]), int(r["seed"]), str(r["metric"]), repr(float(r["value"])),
                        "" if stage is None else int(stage)])
    return path


def replace_metrics(run_dir, names, rows) -> Path:
    """Like append_metrics, but first drop every existing row whose metric is in `names`, so re-running
    a metric script replaces its numbers instead of duplicating them. Other metrics are kept."""
    path = Path(run_dir) / "metrics.csv"
    if path.is_file():
        with path.open(newline="", encoding="utf-8") as f:
            kept = [row for row in csv.DictReader(f) if row["metric"] not in set(names)]
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=METRIC_COLUMNS)
            w.writeheader()
            w.writerows(kept)
    return append_metrics(run_dir, rows)


# ---------------------------------------------------------------- merged tables

def _write_csv(path: Path, columns, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        w.writerows(rows)


def collect(root, out=None) -> dict:
    """Merge every run under ROOT into four CSV tables in `out` (default ROOT/tables). Returns row counts.

    A run without experiment.json is still included; its run_id is its path relative to ROOT.
    """
    root = Path(root).resolve()
    out = Path(out) if out else root / "tables"
    runs, images, stages, metrics, seen = [], [], [], [], {}
    stage_columns = ["run_id"]
    for run_json in sorted(root.rglob("run.json")):
        d = run_json.parent
        run, exp = load_run(d), load_experiment(d)
        run_id = exp["run_id"] if exp else d.relative_to(root).as_posix()
        if run_id in seen:
            raise SchemaError(f"run_id {run_id!r} is used by both {seen[run_id]} and {d}")
        seen[run_id] = d
        if exp and exp["run_json_sha256"] != file_sha256(run_json):
            raise SchemaError(f"{d}: run.json changed after experiment.json was recorded; record it again")
        res = (exp or {}).get("resources", {})
        seconds = run.get("seconds", {})
        n = len(run["rows"])
        runs.append({
            "run_id": run_id, "lane": exp and exp["lane"], "experiment": exp and exp["experiment"],
            "arm": exp and exp["arm"], "role": exp and exp["role"],
            "baseline_run_id": exp and exp["baseline"] and exp["baseline"]["run_id"],
            "pairing_ok": exp and exp["pairing"] and exp["pairing"]["ok"],
            "model": run["model"], "config": Path(run["config_path"]).name, "depth": run.get("depth"),
            "manifest": Path(run["manifest"]).name, "n_images": n, "batch_size": run["batch_size"],
            "hooks": ";".join(h["name"] for h in run.get("hooks", [])),
            "skip_timesteps": ";".join(map(str, run.get("skip_timesteps", []))),
            "params": json.dumps(exp["params"], sort_keys=True) if exp else "",
            "seconds_load": seconds.get("load"), "seconds_sample": seconds.get("sample"),
            "seconds_per_image": round(seconds["sample"] / n, 4) if seconds.get("sample") is not None and n else None,
            "peak_vram_mb": res.get("peak_vram_mb"), "gpu": _get(run, "env.gpu"), "torch": _get(run, "env.torch"),
            "code_commit": exp and exp["code"]["commit"], "code_dirty": exp and exp["code"]["dirty"],
            "pbs_jobid": exp and exp["job"]["pbs_jobid"], "created_utc": exp and exp["created_utc"],
            "run_dir": d.relative_to(root).as_posix(),
        })
        base_hashes = None
        if exp and exp["baseline"]:
            base_dir = (d / exp["baseline"]["dir"]).resolve()
            if (base_dir / "run.json").is_file():
                base_hashes = {(r["class_id"], r["seed"]): r["tensor_sha256"] for r in load_run(base_dir)["rows"]}
        for r in run["rows"]:
            same = None if base_hashes is None else base_hashes.get((r["class_id"], r["seed"])) == r["tensor_sha256"]
            images.append({"run_id": run_id, **{k: r[k] for k in IMAGE_COLUMNS[1:6]}, "identical_to_baseline": same})
        for s in run.get("stages", []):
            stages.append({"run_id": run_id, **s})
            stage_columns += [k for k in s if k not in stage_columns]
        if (d / "metrics.csv").is_file():
            with (d / "metrics.csv").open(newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                if tuple(reader.fieldnames or ()) != METRIC_COLUMNS:
                    raise SchemaError(f"{d / 'metrics.csv'}: header must be {','.join(METRIC_COLUMNS)}")
                metrics += [{"run_id": run_id, **row} for row in reader]
    if not runs:
        raise SchemaError(f"no run.json found under {root}")
    out.mkdir(parents=True, exist_ok=True)
    _write_csv(out / "runs.csv", RUN_COLUMNS, runs)
    _write_csv(out / "images.csv", IMAGE_COLUMNS, images)
    _write_csv(out / "stages.csv", stage_columns, stages)
    _write_csv(out / "metrics.csv", ("run_id", *METRIC_COLUMNS), metrics)
    return {"out": str(out), "runs": len(runs), "images": len(images), "stages": len(stages), "metrics": len(metrics)}


# ---------------------------------------------------------------- command line

def _params(items: list[str]) -> dict:
    params = {}
    for item in items:
        key, sep, value = item.partition("=")
        if not sep or not key:
            raise SchemaError(f"--param {item!r}: expected KEY=VALUE")
        try:
            params[key] = json.loads(value)   # numbers, true/false, lists
        except json.JSONDecodeError:
            params[key] = value
    return params


def _print_pairing(p: dict):
    for w in p["warnings"]:
        print(f"  warning: {w}")
    for e in p["errors"]:
        print(f"  ERROR: {e}")
    print(f"pairing: {'OK' if p['ok'] else 'FAILED'}  ({p['rows_changed']}/{p['n_rows']} images changed, "
          f"{p['rows_identical']} identical to baseline)")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record", help="write RUN_DIR/experiment.json")
    r.add_argument("run_dir", type=Path)
    r.add_argument("--lane", required=True, help="p1 .. p4")
    r.add_argument("--experiment", required=True, help="groups a baseline with its interventions, e.g. var_noise_pilot_<JOBID>")
    r.add_argument("--arm", required=True, help="this run within the experiment, e.g. baseline or h_s3_sev0.5")
    r.add_argument("--baseline", type=Path, help="the baseline's run directory (omit when recording the baseline)")
    r.add_argument("--param", action="append", default=[], metavar="KEY=VALUE", help="intervention parameter (repeatable)")
    r.add_argument("--vram-log", type=Path, help="nvidia-smi memory.used log sampled during the run")
    r.add_argument("--note")
    c = sub.add_parser("check", help="pairing check of RUN_DIR against BASELINE_DIR")
    c.add_argument("baseline_dir", type=Path)
    c.add_argument("run_dir", type=Path)
    m = sub.add_parser("collect", help="merge every run under ROOT into CSV tables")
    m.add_argument("root", type=Path)
    m.add_argument("--out", type=Path, help="default: ROOT/tables")
    args = ap.parse_args(argv)
    try:
        if args.cmd == "record":
            exp = record(args.run_dir, args.lane, args.experiment, args.arm, args.baseline, _params(args.param),
                         args.vram_log, args.note)
            print(f"wrote {args.run_dir / 'experiment.json'}  ({exp['run_id']}, {exp['role']})")
            if exp["pairing"]:
                _print_pairing(exp["pairing"])
                sys.exit(0 if exp["pairing"]["ok"] else 1)
        elif args.cmd == "check":
            p = check_pairing(load_run(args.baseline_dir), load_run(args.run_dir))
            _print_pairing(p)
            sys.exit(0 if p["ok"] else 1)
        else:
            print(json.dumps(collect(args.root, args.out)))
    except SchemaError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
