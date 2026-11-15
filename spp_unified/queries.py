"""Compact, read-only answers about finished run directories.

Every function reads files only; none reruns the model. Answers carry units and
state any approximation (downsampling, nearest-time lookup).
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

from . import registry as R
from .configuration import ConfigError, error

SERIES_UNITS = {s["name"]: s["unit"] for s in R.SERIES_OUTPUTS}


def _need(run: Path, rel: str) -> Path:
    path = Path(run) / rel
    if not path.exists():
        raise ConfigError([error("run", str(run), f"not a completed run directory: missing {rel}",
                                 "Pass a directory written by `spp-physica-a run`.")])
    return path


def _json(run: Path, rel: str):
    return json.loads(_need(run, rel).read_text())


def _table(run: Path) -> dict[str, list[float]]:
    with _need(run, "outputs/timeseries.csv").open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return {}
    return {k: [int(r[k]) if k == "step" else float(r[k]) for r in rows] for k in rows[0]}


def _check_series(name: str, table: dict):
    if name not in SERIES_UNITS:
        import difflib
        m = difflib.get_close_matches(name, list(SERIES_UNITS), n=1)
        raise ConfigError([error("name", name, "unknown series",
                                 (f"Did you mean '{m[0]}'? " if m else "") + f"Series: {', '.join(SERIES_UNITS)}.")])


def summary(run) -> dict:
    return _json(run, "summary.json")


def outputs(run) -> dict:
    run = Path(run)
    files = []
    for p in sorted((run / "outputs").glob("*")):
        files.append({"file": f"outputs/{p.name}", "bytes": p.stat().st_size})
    table = _table(run)
    import numpy as np
    with np.load(_need(run, "outputs/trajectory.npz"), allow_pickle=False) as data:
        arrays = {k: {"shape": list(data[k].shape), "dtype": str(data[k].dtype)} for k in data.files}
    return {"files": files,
            "series": [{"name": n, "unit": SERIES_UNITS[n], "shape": [len(table.get("step", []))],
                        "file": "outputs/timeseries.csv"} for n in SERIES_UNITS],
            "time_axis": {"column": "step", "unit": "updates",
                          "range": [table["step"][0], table["step"][-1]] if table else None},
            "arrays": arrays,
            "summary_fields": [{"name": o["name"], "unit": o["unit"]} for o in R.SUMMARY_OUTPUTS]}


def series(run, name: str, start=None, stop=None, points: int = 100) -> dict:
    table = _table(run)
    _check_series(name, table)
    if points < 2:
        raise ConfigError([error("--points", points, "must be at least 2", "Use --points 100.")])
    idx = [i for i, s in enumerate(table["step"])
           if (start is None or s >= start) and (stop is None or s <= stop)]
    n = len(idx)
    if n > points:
        sel = sorted({round(k * (n - 1) / (points - 1)) for k in range(points)})
        chosen = [idx[k] for k in sel]
        method = f"decimation: {len(chosen)} of {n} samples at evenly spaced indices, endpoints kept, no averaging"
    else:
        chosen, method = idx, "none: all samples returned"
    return {"name": name, "unit": SERIES_UNITS[name], "time_unit": "updates (step)",
            "samples_in_range": n, "returned": len(chosen), "downsampling": method,
            "step": [table["step"][i] for i in chosen], "value": [table[name][i] for i in chosen]}


def stats(run, name: str) -> dict:
    table = _table(run)
    _check_series(name, table)
    v, s = table[name], table["step"]
    imax, imin = max(range(len(v)), key=v.__getitem__), min(range(len(v)), key=v.__getitem__)
    mean = sum(v) / len(v)
    sd = math.sqrt(sum((x - mean) ** 2 for x in v) / (len(v) - 1)) if len(v) > 1 else 0.0
    return {"name": name, "unit": SERIES_UNITS[name], "samples": len(v),
            "min": v[imin], "step_of_min": s[imin], "max": v[imax], "step_of_max": s[imax],
            "mean": mean, "sd": sd, "final": v[-1], "final_step": s[-1],
            "note": "computed over the sampling window (outputs/timeseries.csv); first occurrence for ties"}


def _frame_summary(x, th, species) -> dict:
    import numpy as np
    z = np.exp(1j * th).mean()
    out = {"particles": int(len(x)), "polarization": float(abs(z)),
           "mean_heading_rad": float(np.angle(z)),
           "position_mean": [float(v) for v in x.mean(0)],
           "position_min": [float(v) for v in x.min(0)], "position_max": [float(v) for v in x.max(0)]}
    if species is not None and species.any():
        out["species_counts"] = {"A": int((species == 0).sum()), "B": int((species == 1).sum())}
        for lab, key in ((0, "A"), (1, "B")):
            sel = species == lab
            out[f"polarization_{key}"] = float(abs(np.exp(1j * th[sel]).mean())) if sel.any() else None
    return out


def at(run, step: float) -> dict:
    import numpy as np
    run = Path(run)
    table = _table(run)
    with np.load(_need(run, "outputs/trajectory.npz"), allow_pickle=False) as data:
        rec = data["recorded_steps"]
        k = int(np.argmin(np.abs(rec - step)))
        frame = _frame_summary(data["positions"][k], data["headings"][k], data["species"])
        rec_step = int(rec[k])
    out = {"requested_step": step, "frame": {"recorded_step": rec_step, **frame,
                                             "lookup": "nearest recorded frame (positions/headings)"}}
    if table:
        i = min(range(len(table["step"])), key=lambda j: abs(table["step"][j] - step))
        out["sample"] = {"step": table["step"][i], "lookup": "nearest sampled step (order parameters)",
                         **{n: {"value": table[n][i], "unit": SERIES_UNITS[n]} for n in SERIES_UNITS}}
    out["length_unit"] = R.LENGTH
    return out


def final(run) -> dict:
    import numpy as np
    run = Path(run)
    with np.load(_need(run, "outputs/trajectory.npz"), allow_pickle=False) as data:
        frame = _frame_summary(data["positions"][-1], data["headings"][-1], data["species"])
        last = int(data["recorded_steps"][-1])
    s = summary(run)
    return {"final_step": last, "state": {**frame, "note": "summarized; raw arrays in outputs/trajectory.npz"},
            "diagnostics": s["diagnostics"], "summary": s["values"], "length_unit": R.LENGTH}


def diagnostics(run) -> dict:
    s = summary(run)
    d = {k: v["value"] for k, v in s["diagnostics"].items()}
    return {"diagnostics": s["diagnostics"],
            "overlap_free": d.get("final_overlapping_pairs") == 0,
            "note": "Residual overlaps after the finite projection passes (docs/NUMERICS.md)."}


def descriptor(run) -> dict:
    import numpy as np
    with np.load(_need(run, "outputs/trajectory.npz"), allow_pickle=False) as data:
        d = data["descriptor"].astype(float)
    return {"length": int(d.size), "unit": R.DIMENSIONLESS,
            "description": "Learned-representation input descriptor of the last 10 recorded frames "
                           "(51 single, 107 two; layout in docs/NUMERICS.md)",
            "values": [float(v) for v in d]}


def _flatten(cfg: dict, prefix="") -> dict:
    out = {}
    for k, v in cfg.items():
        name = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, name + "."))
        else:
            out[name] = v
    return out


def compare(runs: list) -> dict:
    configs = [_flatten(_json(r, "config.resolved.json")) for r in runs]
    sums = [summary(r) for r in runs]
    keys = sorted(set().union(*configs))
    differ = [{"name": k, "values": [c.get(k) for c in configs]} for k in keys
              if len({json.dumps(c.get(k)) for c in configs}) > 1]
    metrics = []
    for name in sums[0]["values"]:
        vals = [s["values"].get(name, {}).get("value") for s in sums]
        entry = {"name": name, "unit": sums[0]["values"][name]["unit"], "values": vals}
        if all(isinstance(v, (int, float)) for v in vals):
            entry["difference_vs_first"] = [v - vals[0] for v in vals[1:]]
        metrics.append(entry)
    return {"runs": [str(r) for r in runs], "parameters_differing": differ,
            "summary": metrics, "note": "differences are run_i - run_0"}
