"""Self-describing run directories built around the legacy ``cli.run`` writer.

Layout::

    <out>/config.resolved.json   every parameter explicit, including the seed
    <out>/manifest.json          status, command, times, code/dependency versions, seed
    <out>/summary.json           headline results with units
    <out>/schema.json            every file and column in outputs/, from the registry
    <out>/log.txt                messages that would otherwise go to the terminal
    <out>/outputs/run.json       legacy record (same as `simulate`)
    <out>/outputs/trajectory.npz legacy arrays (same as `simulate`)
    <out>/outputs/timeseries.csv sampled order parameters with step and time columns

The model is executed exactly as ``spp-physica-a simulate`` executes it.
"""
from __future__ import annotations

import csv
import hashlib
import json
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from . import registry as R
from . import configuration as C

PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def _dump(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def code_version() -> dict:
    from importlib import metadata
    info = {"package": "spp-physica-a", "version": None, "git_commit": None, "git_dirty": None}
    try:
        info["version"] = metadata.version("spp-physica-a")
    except metadata.PackageNotFoundError:
        pass
    if (PACKAGE_ROOT / ".git").exists():
        try:
            run = lambda *a: subprocess.run(["git", "-C", str(PACKAGE_ROOT), *a], capture_output=True,
                                            text=True, timeout=10)
            head = run("rev-parse", "HEAD")
            if head.returncode == 0:
                info["git_commit"] = head.stdout.strip()
                status = run("status", "--porcelain", "--untracked-files=no")
                info["git_dirty"] = bool(status.stdout.strip()) if status.returncode == 0 else None
        except (OSError, subprocess.SubprocessError):
            pass
    return info


def environment() -> dict:
    from importlib import metadata
    deps = {}
    for name in ("numpy", "scipy", "pandas", "matplotlib"):
        try:
            deps[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            deps[name] = None
    return {"python": platform.python_version(), "implementation": platform.python_implementation(),
            "platform": platform.platform(), "machine": platform.machine(), "dependencies": deps}


def prepare_out(out: Path, force: bool) -> None:
    out = Path(out)
    if out.exists() and not out.is_dir():
        raise C.ConfigError([C.error("--out", str(out), "exists and is not a directory",
                                     "Choose a new directory.")])
    if out.exists() and any(out.iterdir()):
        if not force:
            raise C.ConfigError([C.error("--out", str(out), "output directory exists and is not empty",
                                         "Choose a new directory or pass --force to replace it.")])
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)


def summary_document(record: dict, resolved: dict) -> dict:
    units = {o["name"]: o["unit"] for o in R.SUMMARY_OUTPUTS}
    dunits = {o["name"]: o["unit"] for o in R.DIAGNOSTIC_OUTPUTS}
    sim = resolved["simulation"]
    return {
        "kind": resolved["kind"],
        "seed": sim["seed"],
        "averaging_window": {"from_step_index": sim["sample_start"], "to_step_index": sim["steps"] - 1,
                             "every": sim["sample_every"],
                             "note": "phi..temporal_stability are means over this sampling window"},
        "values": {k: {"value": v, "unit": units.get(k)} for k, v in record["summary"].items()},
        "diagnostics": {k: {"value": v, "unit": dunits.get(k)} for k, v in record["diagnostics"].items()},
    }


def write_timeseries(outputs: Path, resolved: dict) -> int:
    import numpy as np
    sim = resolved["simulation"]
    steps = [s for s in range(sim["sample_start"], sim["steps"], sim["sample_every"])]
    with np.load(outputs / "trajectory.npz", allow_pickle=False) as data:
        cols = {k: data[f"timeseries_{k}"] for k in ("phi", "nematic", "milling", "clustering", "segregation")}
        com = data["timeseries_com"]
    with (outputs / "timeseries.csv").open("w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["step", "time", *cols, "com_x", "com_y"])
        for i, s in enumerate(steps):
            w.writerow([s + 1, repr((s + 1) * float(sim["dt"])), *(repr(float(cols[k][i])) for k in cols),
                        repr(float(com[i, 0])), repr(float(com[i, 1]))])
    return len(steps)


def schema_document(resolved: dict) -> dict:
    sim = resolved["simulation"]
    return {
        "run_directory": {
            "config.resolved.json": "Complete configuration; `spp-physica-a run` on it reproduces outputs/ and summary.json.",
            "manifest.json": "Status, command line, times, code and dependency versions, platform, seed, error.",
            "summary.json": "Headline values with units; averages over the sampling window.",
            "schema.json": "This file.",
            "log.txt": "Messages from the run.",
        },
        "outputs": {
            "timeseries.csv": {
                "format": "CSV with header", "rows": "one per sampled step",
                "columns": [{"name": "step", "unit": "updates",
                             "description": "One-based update count after which the sample was taken (loop index + 1)."},
                            {"name": "time", "unit": R.TIME, "description": f"step * dt (dt = {sim['dt']})."}]
                           + [{"name": s["name"], "unit": s["unit"], "description": s["description"]}
                              for s in R.SERIES_OUTPUTS]},
            "trajectory.npz": {"format": "NumPy npz (compressed, no pickle)",
                               "arrays": R.TRAJECTORY_ARRAYS},
            "run.json": {"format": "JSON", "description": "Legacy record written by `simulate`: kind, simulation, parameters, state_every, summary, diagnostics.",
                         "summary_fields": R.SUMMARY_OUTPUTS, "diagnostic_fields": R.DIAGNOSTIC_OUTPUTS},
        },
    }


def execute(cfg: dict, out: Path, *, force: bool = False, argv=None, log=None) -> dict:
    """Validate, resolve, and run one configuration into ``out``. Returns the manifest."""
    log = log or (lambda msg: print(msg, file=sys.stderr))
    warnings = C.check(cfg)
    resolved, chosen = C.resolve(cfg)
    out = Path(out)
    prepare_out(out, force)
    lines = []

    def say(msg):
        lines.append(msg)
        log(msg)

    for w in warnings:
        say(f"warning: {w['field']}={w['value']}: {w['reason']}")
    if chosen:
        say(f"note: simulation.seed was not set; chose seed {resolved['simulation']['seed']} "
            "and recorded it in config.resolved.json")
    _dump(out / "config.resolved.json", resolved)
    manifest = {
        "status": "running",
        "command": list(argv) if argv is not None else sys.argv,
        "cwd": str(Path.cwd()),
        "started": _now(), "finished": None, "duration_s": None,
        "seed": resolved["simulation"]["seed"], "seed_chosen_by_run": chosen,
        "config_sha256": hashlib.sha256(json.dumps(resolved, sort_keys=True).encode()).hexdigest(),
        "estimate": C.estimate(resolved),
        "code": code_version(), "environment": environment(), "error": None,
    }
    _dump(out / "manifest.json", manifest)
    t0 = time.perf_counter()
    try:
        from .cli import run as legacy_run
        say(f"running kind={resolved['kind']} n={manifest['estimate']['particles']} "
            f"steps={resolved['simulation']['steps']} seed={resolved['simulation']['seed']}")
        record = legacy_run(C.to_legacy(resolved), out / "outputs")
        write_timeseries(out / "outputs", resolved)
        _dump(out / "summary.json", summary_document(record, resolved))
        _dump(out / "schema.json", schema_document(resolved))
        manifest["status"] = "completed"
    except BaseException as exc:          # record interruptions too
        manifest["status"] = "failed"
        manifest["error"] = {"type": type(exc).__name__, "message": str(exc)}
        say(f"error: {type(exc).__name__}: {exc}")
        raise
    finally:
        manifest["finished"] = _now()
        manifest["duration_s"] = round(time.perf_counter() - t0, 4)
        if manifest["status"] == "completed":
            say(f"completed in {manifest['duration_s']} s")
        _dump(out / "manifest.json", manifest)
        (out / "log.txt").write_text("\n".join(lines) + "\n")
    return manifest


def sweep(cfg: dict, out: Path, *, force: bool = False, argv=None, log=None) -> dict:
    log = log or (lambda msg: print(msg, file=sys.stderr))
    C.check(cfg, allow_sweep=True)
    points = C.expand_sweep(cfg)
    out = Path(out)
    prepare_out(out, force)
    entries = []
    for i, point in enumerate(points):
        name = f"point_{i:04d}"
        log(f"[{i + 1}/{len(points)}] {point['parameters']}")
        entry = {"index": i, "parameters": point["parameters"], "directory": name}
        try:
            execute(point["config"], out / name, argv=argv, log=log)
            summary = json.loads((out / name / "summary.json").read_text())
            entry.update(status="completed",
                         summary={k: v["value"] for k, v in summary["values"].items()})
        except Exception as exc:  # noqa: BLE001 - one failed point must not hide the others
            entry.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        entries.append(entry)
    index = {"sweep": cfg["sweep"], "points": len(points),
             "completed": sum(e["status"] == "completed" for e in entries),
             "units": {o["name"]: o["unit"] for o in R.SUMMARY_OUTPUTS}, "runs": entries}
    _dump(out / "index.json", index)
    return index
