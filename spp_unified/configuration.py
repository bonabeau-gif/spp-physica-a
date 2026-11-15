"""Declarative run configurations: load, validate, resolve, estimate, template, schema.

A configuration is one JSON file in the format already used by ``configs/*.json``
and the legacy ``simulate`` command::

    {"schema": 1, "kind": "single",
     "parameters": {...}, "mixture": {...}, "simulation": {..., "seed": 42},
     "state_every": 8}

Full-line ``//`` comments are allowed (``template`` writes them). Everything is
driven by :mod:`spp_unified.registry`.
"""
from __future__ import annotations

import copy
import difflib
import itertools
import json
import math
import secrets
from pathlib import Path

from . import registry as R

GROUPS = ("parameters", "mixture", "simulation")
TOP_LEVEL = ("schema", "kind", "parameters", "mixture", "simulation", "state_every", "sweep")


class ConfigError(Exception):
    """Raised with a list of structured errors (field, value, reason, hint)."""

    def __init__(self, errors: list[dict]):
        super().__init__("; ".join(f"{e['field']}: {e['reason']}" for e in errors))
        self.errors = errors


def error(field, value, reason, hint=None) -> dict:
    return {"field": field, "value": value, "reason": reason, "hint": hint}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def strip_comments(text: str) -> str:
    return "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("//"))


def loads(text: str, source: str = "<config>") -> dict:
    try:
        data = json.loads(strip_comments(text))
    except json.JSONDecodeError as exc:
        raise ConfigError([error(source, None, f"not valid JSON: {exc}",
                                 "Use JSON; only full-line // comments are allowed.")])
    if not isinstance(data, dict):
        raise ConfigError([error(source, type(data).__name__, "configuration must be a JSON object",
                                 "Start from `spp-physica-a template`.")])
    return data


def load(path) -> tuple[dict, Path]:
    """Return (config, base_dir). ``-`` reads standard input; base_dir is then the cwd."""
    import sys
    if str(path) == "-":
        return loads(sys.stdin.read(), "<stdin>"), Path.cwd()
    path = Path(path)
    try:
        text = path.read_text()
    except OSError as exc:
        raise ConfigError([error("config", str(path), f"cannot read file: {exc.strerror}",
                                 "Pass an existing JSON configuration file or - for stdin.")])
    return loads(text, str(path)), path.resolve().parent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _group_keys(group: str) -> list[str]:
    return [p["name"].split(".", 1)[1] for p in R.PARAMETERS if p["name"].startswith(group + ".")]


def _suggest(key: str, candidates) -> str | None:
    match = difflib.get_close_matches(key, list(candidates), n=1, cutoff=0.5)
    return f"Did you mean '{match[0]}'?" if match else None


def _get(cfg: dict, dotted: str, default=None):
    node = cfg
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


def _set(cfg: dict, dotted: str, value):
    parts = dotted.split(".")
    node = cfg
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value


def _range_text(p: dict) -> str:
    if p.get("options") is not None:
        return "one of " + ", ".join(map(str, p["options"]))
    r = p.get("range")
    if not r:
        return "any finite value (no range stated in the source)" if p["type"] == "number" else "any"
    lo = "-inf" if r.get("min") is None else r["min"]
    hi = "inf" if r.get("max") is None else r["max"]
    left = "(" if r.get("min_exclusive") or r.get("min") is None else "["
    right = ")" if r.get("max") is None else "]"
    return f"{left}{lo}, {hi}{right}"


def _check_value(p: dict, value) -> dict | None:
    name, typ = p["name"], p["type"]
    if typ == "number":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return error(name, value, "must be a number", f"Accepted: {_range_text(p)}.")
        if not math.isfinite(value):
            return error(name, value, "must be finite", f"Accepted: {_range_text(p)}.")
    elif typ == "integer":
        if isinstance(value, bool) or not isinstance(value, int):
            return error(name, value, "must be an integer", f"Accepted: {_range_text(p)}.")
    elif typ == "boolean":
        if not isinstance(value, bool):
            return error(name, value, "must be true or false", None)
    elif typ == "enum":
        if value not in p["options"]:
            return error(name, value, "not an accepted option",
                         _suggest(str(value), map(str, p["options"])) or f"Accepted: {_range_text(p)}.")
        return None
    r = p.get("range")
    if r and typ in ("number", "integer"):
        lo, hi = r.get("min"), r.get("max")
        bad = (lo is not None and (value <= lo if r.get("min_exclusive") else value < lo)) or \
              (hi is not None and value > hi)
        if bad:
            return error(name, value, "out of range", f"Accepted: {_range_text(p)}.")
    return None


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def particle_count(rho: float, S: float, r0: float) -> int:
    n_max = max(2, int(round((math.sqrt(3.0) / 6.0) * (S / r0) ** 2)))
    return max(8, int(round(rho * n_max)))


def validate(cfg: dict, *, allow_sweep: bool = False, base_dir: Path | None = None) -> tuple[list, list]:
    """Return (errors, warnings). Never raises for content problems."""
    errors: list[dict] = []
    warnings: list[dict] = []
    index = R.parameter_index()

    for key in cfg:
        if key not in TOP_LEVEL:
            dotted = [n for n in index if n.split(".")[-1] == key]
            hint = (f"'{key}' belongs under '{dotted[0].rsplit('.', 1)[0]}'." if dotted and "." in dotted[0]
                    else _suggest(key, TOP_LEVEL) or f"Top-level keys: {', '.join(TOP_LEVEL)}.")
            errors.append(error(key, cfg[key], "unknown key", hint))
    if "schema" in cfg and cfg["schema"] != R.CONFIG_SCHEMA_VERSION:
        errors.append(error("schema", cfg["schema"], "unsupported configuration format version",
                            f"Use schema: {R.CONFIG_SCHEMA_VERSION}."))
    if "sweep" in cfg and not allow_sweep:
        errors.append(error("sweep", cfg["sweep"], "sweep block given to a single run",
                            "Use `spp-physica-a sweep <config> --out <dir>`, or remove the sweep block."))

    kind = cfg.get("kind", "single")
    if (e := _check_value(index["kind"], kind)):
        errors.append(e)
        kind = "single"

    for group in GROUPS:
        if group not in cfg:
            continue
        block = cfg[group]
        if not isinstance(block, dict):
            errors.append(error(group, block, "must be an object", f"Keys: {', '.join(_group_keys(group))}."))
            continue
        if group == "mixture" and kind != "two":
            errors.append(error("mixture", block, "mixture parameters apply only to kind 'two'",
                                "Set \"kind\": \"two\" or remove the mixture block."))
        valid = _group_keys(group)
        for key, value in block.items():
            name = f"{group}.{key}"
            if key not in valid:
                elsewhere = [g for g in GROUPS if key in _group_keys(g)]
                hint = (f"'{key}' belongs under '{elsewhere[0]}'." if elsewhere
                        else _suggest(key, valid) or f"Valid keys: {', '.join(valid)}.")
                errors.append(error(name, value, "unknown key", hint))
                continue
            if name == "simulation.seed" and value is None:
                continue
            if (e := _check_value(index[name], value)):
                errors.append(e)
    if "state_every" in cfg and (e := _check_value(index["state_every"], cfg["state_every"])):
        errors.append(e)

    bad_fields = {e["field"] for e in errors}

    def val(name):
        return _get(cfg, name, index[name]["default"])

    def ok(*names):
        return not any(n in bad_fields for n in names)

    if ok("simulation.steps", "simulation.sample_start"):
        if not val("simulation.sample_start") < val("simulation.steps"):
            errors.append(error("simulation.sample_start", val("simulation.sample_start"),
                                "violates constraint simulation.sample_start < simulation.steps",
                                f"Choose sample_start below steps={val('simulation.steps')}."))
    if ok("simulation.S", "simulation.r0"):
        S, r0 = val("simulation.S"), val("simulation.r0")
        if not S > 2 * r0:
            errors.append(error("simulation.S", S, "violates constraint simulation.S > 2 * simulation.r0",
                                f"Use S > {2 * r0}."))
        elif ok("parameters.rho") and val("parameters.rho") > 0:
            n = particle_count(val("parameters.rho"), S, r0)
            if not _placement_ok(n, S, r0):
                errors.append(error("parameters.rho", val("parameters.rho"),
                                    f"violates constraint particle_placement: {n} particles do not fit "
                                    f"on the initial grid (spacing >= 2.02*r0) in a box of side {S}",
                                    "Lower rho or enlarge simulation.S."))
        if ok("parameters.interaction_range_ratio") and S > 2 * r0:
            r1 = val("parameters.interaction_range_ratio") * r0
            if r1 > S / 2:
                warnings.append(error("parameters.interaction_range_ratio",
                                      val("parameters.interaction_range_ratio"),
                                      f"r1 = {r1:g} exceeds S/2 = {S / 2:g}; clustering normalization "
                                      "and minimum-image geometry assume r1 <= S/2",
                                      f"Use interaction_range_ratio <= {S / 2 / r0:g} for clean geometry."))
    if ok("parameters.adhesion") and abs(val("parameters.adhesion")) > 0.95:
        warnings.append(error("parameters.adhesion", val("parameters.adhesion"),
                              "|adhesion| > 0.95 is clipped to +/-0.95 by beta_from_adhesion",
                              "Values beyond +/-0.95 behave identically."))
    for p in R.PARAMETERS:
        er = p.get("explored_range")
        v = _get(cfg, p["name"])
        if er and isinstance(v, (int, float)) and not isinstance(v, bool) and p["name"] not in bad_fields \
                and not er[0] <= v <= er[1]:
            warnings.append(error(p["name"], v, f"outside the range explored in the paper {er}",
                                  "Valid, but results there were not studied."))

    if allow_sweep and "sweep" in cfg:
        errors.extend(_validate_sweep(cfg, index))
    return errors, warnings


def _placement_ok(n: int, S: float, r0: float) -> bool:
    min_sep = 2.02 * r0          # mirrors geometry.initialize_positions
    n_side = int(math.ceil(math.sqrt(n)))
    if S / n_side >= min_sep:
        return True
    n_side = max(1, int(math.floor(S / min_sep)))
    return n_side * n_side >= n


def _validate_sweep(cfg: dict, index: dict) -> list[dict]:
    errs = []
    block = cfg["sweep"]
    if not isinstance(block, dict) or not block:
        return [error("sweep", block, "must be a non-empty object of dotted-name: [values]",
                      'Example: {"parameters.noise": [0.1, 0.2], "simulation.seed": [1, 2]}')]
    for name, values in block.items():
        if name not in index or name in ("kind",):
            errs.append(error(f"sweep.{name}", values, "not a sweepable parameter",
                              _suggest(name, [n for n in index if n != "kind"])))
        elif not isinstance(values, list) or not values:
            errs.append(error(f"sweep.{name}", values, "must be a non-empty list of values", None))
    if errs:
        return errs
    for i, point in enumerate(expand_sweep(cfg)):
        e, _ = validate(point["config"])
        errs.extend({**x, "field": f"sweep[{i}].{x['field']}"} for x in e)
    return errs


def expand_sweep(cfg: dict) -> list[dict]:
    block = cfg["sweep"]
    names = list(block)
    base = {k: v for k, v in cfg.items() if k != "sweep"}
    points = []
    for combo in itertools.product(*(block[n] for n in names)):
        c = copy.deepcopy(base)
        for n, v in zip(names, combo):
            _set(c, n, v)
        points.append({"parameters": dict(zip(names, combo)), "config": c})
    return points


def check(cfg: dict, **kw) -> list[dict]:
    errors, warnings = validate(cfg, **kw)
    if errors:
        raise ConfigError(errors)
    return warnings


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------
def resolve(cfg: dict, *, seed_if_missing: bool = True) -> tuple[dict, bool]:
    """Return (fully explicit config, seed_was_chosen). ``cfg`` must already validate."""
    kind = cfg.get("kind", "single")
    out = {"schema": R.CONFIG_SCHEMA_VERSION, "kind": kind}
    for group in GROUPS:
        if group == "mixture" and kind != "two":
            continue
        block = {}
        for key in _group_keys(group):
            name = f"{group}.{key}"
            block[key] = _get(cfg, name, R.parameter_index()[name]["default"])
        out[group] = block
    out["state_every"] = cfg.get("state_every", R.parameter_index()["state_every"]["default"])
    chosen = False
    seed = out["simulation"]["seed"]
    if not isinstance(seed, int) or isinstance(seed, bool):
        if not seed_if_missing:
            raise ConfigError([error("simulation.seed", seed, "seed required", "Set simulation.seed.")])
        out["simulation"]["seed"] = secrets.randbelow(2**31)
        chosen = True
    return out, chosen


def to_legacy(resolved: dict) -> dict:
    """The resolved config in the exact shape the legacy ``cli.run`` consumes."""
    return {k: v for k, v in resolved.items() if k != "schema"}


# ---------------------------------------------------------------------------
# Cost estimate (calibrated on the dense path, Apple M-series, numpy 2.5)
# ---------------------------------------------------------------------------
PER_STEP_S = 2.0e-4
PER_STEP_PAIR_S = 4.2e-8
TWO_SPECIES_FACTOR = 1.2
FIXED_S = 0.05


def estimate(cfg: dict) -> dict:
    idx = R.parameter_index()

    def val(name):
        return _get(cfg, name, idx[name]["default"])
    S, r0, steps = val("simulation.S"), val("simulation.r0"), val("simulation.steps")
    n = particle_count(val("parameters.rho"), S, r0)
    kind = cfg.get("kind", "single")
    seconds = steps * (PER_STEP_S + PER_STEP_PAIR_S * n * n)
    if kind == "two":
        seconds *= TWO_SPECIES_FACTOR
    seconds += FIXED_S
    every = val("state_every")
    frames = len([i for i in range(steps) if i % every == 0 or i == steps - 1])
    start, sample_every = val("simulation.sample_start"), val("simulation.sample_every")
    samples = len(range(start, steps, sample_every))
    raw_npz = 8 * (frames * n * 3 + samples * (n + 7) + n + frames + 107)
    csv_bytes = 160 * samples + 100
    memory = 8 * n * n * 14 + raw_npz + 120e6
    note = "dense all-pairs calibration; use_cell_list changes scaling" if val("simulation.use_cell_list") else \
        "calibrated on the dense all-pairs path; scale by machine speed"
    return {"particles": n, "steps": steps, "recorded_frames": frames, "samples": samples,
            "runtime_s": round(seconds, 3),
            "runtime_note": note + "; excludes ~0.5 s interpreter start-up",
            "peak_memory_mb": round(memory / 1e6, 1),
            "output_bytes_upper_bound": int(raw_npz + csv_bytes + 8000)}


# ---------------------------------------------------------------------------
# Template and JSON Schema
# ---------------------------------------------------------------------------
TEMPLATE_SEED = 0


def default_config(kind: str) -> dict:
    cfg, _ = resolve({"kind": kind, "simulation": {"seed": TEMPLATE_SEED}})
    return cfg


def _comment(p: dict) -> str:
    unit = p.get("unit") or "none"
    parts = [p["description"], f"unit: {unit}", f"accepted: {_range_text(p)}"]
    if p.get("explored_range"):
        parts.append(f"paper explored: {p['explored_range']}")
    if p["name"] == "simulation.seed":
        parts.append("always set it explicitly")
    return " | ".join(parts)


def template_text(kind: str) -> str:
    cfg = default_config(kind)
    idx = R.parameter_index()
    lines = ["{", f"  // {R.SCHEMA_KEY['description']}", f'  "schema": {cfg["schema"]},',
             f"  // {_comment(idx['kind'])}", f'  "kind": {json.dumps(kind)},']
    for group in GROUPS:
        if group not in cfg:
            continue
        lines.append(f'  "{group}": {{')
        keys = list(cfg[group])
        for i, key in enumerate(keys):
            lines.append(f"    // {_comment(idx[group + '.' + key])}")
            comma = "," if i < len(keys) - 1 else ""
            lines.append(f'    "{key}": {json.dumps(cfg[group][key])}{comma}')
        lines.append("  },")
    lines.append(f"  // {_comment(idx['state_every'])}")
    lines.append(f'  "state_every": {cfg["state_every"]}')
    lines.append("}")
    return "\n".join(lines) + "\n"


def _json_type(p: dict) -> dict:
    t = {"number": {"type": "number"}, "integer": {"type": "integer"},
         "boolean": {"type": "boolean"}, "string": {"type": "string"}}.get(p["type"])
    if p["type"] == "enum":
        t = {"enum": p["options"]}
    t = dict(t)
    r = p.get("range")
    if r and p["type"] in ("number", "integer"):
        if r.get("min") is not None:
            t["exclusiveMinimum" if r.get("min_exclusive") else "minimum"] = r["min"]
        if r.get("max") is not None:
            t["maximum"] = r["max"]
    if p["name"] != "simulation.seed":
        t["default"] = p["default"]
    t["description"] = p["description"] + (f" Unit: {p['unit']}." if p.get("unit") else "")
    return t


def json_schema() -> dict:
    idx = R.parameter_index()
    props = {"schema": {"const": R.CONFIG_SCHEMA_VERSION, "description": R.SCHEMA_KEY["description"]},
             "kind": _json_type(idx["kind"]), "state_every": _json_type(idx["state_every"])}
    for group in GROUPS:
        props[group] = {"type": "object", "additionalProperties": False,
                        "properties": {k: _json_type(idx[f"{group}.{k}"]) for k in _group_keys(group)}}
    props["sweep"] = {"type": "object", "description": "Only for `sweep`: dotted parameter name -> list of values (Cartesian product).",
                      "propertyNames": {"enum": [n for n in idx if n != "kind"]},
                      "additionalProperties": {"type": "array", "minItems": 1}}
    return {"$schema": "https://json-schema.org/draft/2020-12/schema",
            "$id": "spp-physica-a/config.schema.json",
            "title": "spp-physica-a run configuration",
            "description": "Generated from spp_unified/registry.py by `spp-physica-a describe --schema`. "
                           "Cross-parameter constraints are checked by `validate`, not by this schema.",
            "type": "object", "additionalProperties": False, "properties": props,
            "x-constraints": R.CONSTRAINTS}
