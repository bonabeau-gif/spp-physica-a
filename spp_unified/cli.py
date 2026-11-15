"""Command-line entry point for reproducible single- and two-species runs."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from . import SingleParams, TwoSpeciesParams, SimulationConfig
from . import simulate_single, simulate_two_species
from .geometry import pairwise_displacements
from .learned import behavior_descriptor, species_behavior_descriptor


def run(config: dict, output: Path) -> dict:
    """Save parameters, observables, frames, and descriptors without pickle."""
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output directory is not empty: {output}")
    kind = config.get("kind", "single")
    if kind not in ("single", "two"):
        raise ValueError("kind must be 'single' or 'two'")
    cfg = SimulationConfig(**config.get("simulation", {}))
    base = SingleParams(**config.get("parameters", {}))
    interval = int(config.get("state_every", 8))
    if interval < 1:
        raise ValueError("state_every must be positive")
    if kind == "two":
        params = TwoSpeciesParams(base=base, **config.get("mixture", {}))
        result = simulate_two_species(params, cfg, record_states=True, state_every=interval)
        descriptor = species_behavior_descriptor(result.states, result.species, cfg)
    else:
        params = base
        result = simulate_single(params, cfg, record_states=True, state_every=interval)
        descriptor = behavior_descriptor(result.states, cfg)
    positions = np.stack([x for x, _ in result.states])
    headings = np.stack([th for _, th in result.states])
    _, distances = pairwise_displacements(positions[-1], cfg.S, cfg.boundary == "periodic")
    np.fill_diagonal(distances, np.inf)
    diagnostics = {
        "final_minimum_separation": float(distances.min()),
        "diameter": 2 * cfg.r0,
        "final_overlapping_pairs": int(np.triu(distances < 2 * cfg.r0 - 1e-8, 1).sum()),
    }
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / "trajectory.npz", positions=positions, headings=headings,
        species=result.species if result.species is not None else np.zeros(len(positions[0]), dtype=np.int8),
        descriptor=descriptor,
        recorded_steps=np.array([i + 1 for i in range(cfg.steps)
                                 if i % interval == 0 or i == cfg.steps - 1]),
        **{f"timeseries_{k}": v for k, v in result.timeseries.items()},
    )
    record = {"kind": kind, "simulation": asdict(cfg), "parameters": asdict(params),
              "state_every": interval, "summary": result.summary, "diagnostics": diagnostics}
    (output / "run.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    return record


EPILOG = "Start with `spp-physica-a describe`. Exit codes: 0 success, 1 runtime failure, 2 invalid usage or configuration."


class _Parser(argparse.ArgumentParser):
    """Argument errors become structured JSON on stdout when --json was requested."""

    json_mode = False

    def error(self, message):
        if _Parser.json_mode:
            _emit({"ok": False, "errors": [{"field": "arguments", "value": None, "reason": message,
                                             "hint": f"See `{self.prog} --help`."}]})
            raise SystemExit(2)
        super().error(message)


def _emit(obj) -> None:
    print(json.dumps(obj, indent=2, allow_nan=False))


def _fail(errors, as_json, code=2, warnings=None):
    import sys
    if as_json:
        doc = {"ok": False, "errors": errors}
        if warnings:
            doc["warnings"] = warnings
        _emit(doc)
    else:
        for e in errors:
            hint = f" ({e['hint']})" if e.get("hint") else ""
            print(f"error: {e['field']}={json.dumps(e['value'])}: {e['reason']}{hint}", file=sys.stderr)
    return code


def _add(sub, name, help_text, example, parents):
    return sub.add_parser(name, help=help_text, description=help_text, parents=parents,
                          epilog=f"Example: {example}", formatter_class=argparse.RawDescriptionHelpFormatter)


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--json", action="store_true",
                        help="Print exactly one JSON document on stdout (errors included); logs go to stderr.")
    parser = _Parser(prog="spp-physica-a", description=__doc__, epilog=EPILOG)
    parser.add_argument("--version", action="version", version="spp-physica-a 1.0.0")
    sub = parser.add_subparsers(dest="command", required=True, parser_class=_Parser)

    sim = _add(sub, "simulate", "Legacy: run a JSON configuration and save run.json + trajectory.npz",
               "spp-physica-a simulate --config configs/single_demo.json --output outputs/single", [])
    sim.add_argument("--config", type=Path, required=True)
    sim.add_argument("--output", type=Path, required=True)

    d = _add(sub, "describe", "Report models, parameters, outputs, and interventions from the registry",
             "spp-physica-a describe --json --param parameters.noise", [common])
    d.add_argument("--model", choices=["single", "two"], help="Only parameters this model uses.")
    d.add_argument("--param", help="Full entry for one dotted parameter name.")
    d.add_argument("--outputs", action="store_true", help="Only outputs.")
    d.add_argument("--schema", action="store_true", help="Print the configuration JSON Schema.")

    t = _add(sub, "template", "Print a complete, valid, commented configuration for a model",
             "spp-physica-a template --model two > my_run.json", [common])
    t.add_argument("--model", choices=["single", "two"], default="single")

    v = _add(sub, "validate", "Check a configuration without running it; report all problems and a cost estimate",
             "spp-physica-a validate examples/minimal_single.json --json", [common])
    v.add_argument("config", help="Configuration file, or - for stdin.")

    r = _add(sub, "run", "Execute one configuration and write a self-describing run directory",
             "spp-physica-a run examples/minimal_single.json --out runs/minimal", [common])
    r.add_argument("config")
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--force", action="store_true", help="Replace a non-empty output directory.")

    s = _add(sub, "sweep", "Run every point of the configuration's sweep block (Cartesian product), plus index.json",
             "spp-physica-a sweep examples/sweep_noise.json --out runs/noise_sweep", [common])
    s.add_argument("config")
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--force", action="store_true")

    q = _add(sub, "query", "Answer a targeted question about a finished run directory (reads files only)",
             "spp-physica-a query runs/minimal stats phi --json", [common])
    q.add_argument("run", type=Path, help="Run directory (or sweep directory for `summary`).")
    qs = q.add_subparsers(dest="what", required=True, parser_class=_Parser)
    _add(qs, "summary", "Contents of summary.json (sweep directory: compact table from index.json)",
         "spp-physica-a query runs/minimal summary", [common])
    _add(qs, "outputs", "Recorded files, series, arrays, units, and sizes",
         "spp-physica-a query runs/minimal outputs", [common])
    qse = _add(qs, "series", "One sampled time series, downsampled to at most --points values",
               "spp-physica-a query runs/minimal series phi --points 20", [common])
    qse.add_argument("name")
    qse.add_argument("--from", dest="start", type=float, help="First step (inclusive).")
    qse.add_argument("--to", dest="stop", type=float, help="Last step (inclusive).")
    qse.add_argument("--points", type=int, default=100)
    qst = _add(qs, "stats", "Min, max, mean, sd, final value, and step of the max for one series",
               "spp-physica-a query runs/minimal stats milling", [common])
    qst.add_argument("name")
    qat = _add(qs, "at", "State at or nearest to a step (or a time with --time)",
               "spp-physica-a query runs/minimal at 40", [common])
    qat.add_argument("value", type=float)
    qat.add_argument("--time", action="store_true", help="Interpret value as time (step = time / dt).")
    _add(qs, "final", "Final state, summarized, with diagnostics and summary",
         "spp-physica-a query runs/minimal final", [common])
    _add(qs, "diagnostics", "Residual-overlap diagnostics of the final frame",
         "spp-physica-a query runs/minimal diagnostics", [common])
    _add(qs, "descriptor", "The 51/107-component trajectory descriptor",
         "spp-physica-a query runs/minimal descriptor", [common])

    c = _add(sub, "compare", "Differences in configuration and summary results between runs",
             "spp-physica-a compare runs/a runs/b --json", [common])
    c.add_argument("runs", nargs="+", type=Path)
    return parser


def _describe(args) -> dict | str:
    from . import registry as R
    from . import configuration as C
    if args.schema:
        return C.json_schema()
    if args.param:
        idx = R.parameter_index()
        if args.param not in idx:
            import difflib
            m = difflib.get_close_matches(args.param, list(idx), n=1)
            raise C.ConfigError([C.error("--param", args.param, "unknown parameter",
                                         f"Did you mean '{m[0]}'?" if m else "See `describe`.")])
        p = dict(idx[args.param])
        p["constraint_details"] = {k: R.CONSTRAINTS[k] for k in p.get("constraints", [])}
        return p
    outputs = {"summary": R.SUMMARY_OUTPUTS, "diagnostics": R.DIAGNOSTIC_OUTPUTS,
               "series": R.SERIES_OUTPUTS, "trajectory_arrays": R.TRAJECTORY_ARRAYS}
    if args.outputs:
        return {"outputs": outputs}
    params = R.parameters_for(args.model) if args.model else R.PARAMETERS
    models = {args.model: R.MODELS[args.model]} if args.model else R.MODELS
    return {"package": "spp-physica-a", "registry_version": R.REGISTRY_VERSION,
            "config_schema_version": R.CONFIG_SCHEMA_VERSION,
            "models": models, "parameters": params, "constraints": R.CONSTRAINTS,
            "outputs": outputs, "interventions": R.INTERVENTIONS, "interventions_note": R.INTERVENTIONS_NOTE,
            "fixed_constants": R.FIXED_CONSTANTS,
            "commands": ["describe", "template", "validate", "run", "sweep", "query", "compare", "simulate"]}


def _describe_text(doc: dict) -> str:
    if "parameters" not in doc:
        return json.dumps(doc, indent=2)
    lines = ["Models:"]
    lines += [f"  {k}: {v['description'][:110]}..." for k, v in doc["models"].items()]
    lines.append("Parameters (name | type | default | unit | applies_to):")
    for p in doc["parameters"]:
        lines.append(f"  {p['name']} | {p['type']} | {p['default']} | {p.get('unit')} | {','.join(p['applies_to'])}")
    lines.append("Summary outputs: " + ", ".join(o["name"] for o in doc["outputs"]["summary"]))
    lines.append("Series: " + ", ".join(o["name"] for o in doc["outputs"]["series"]))
    lines.append("Interventions: none. " + doc["interventions_note"])
    lines.append("Details: describe --param <name> | describe --json | describe --schema")
    return "\n".join(lines)


def _query(args) -> dict:
    from . import queries as Q
    w = args.what
    if w == "summary":
        if (args.run / "index.json").exists() and not (args.run / "summary.json").exists():
            index = json.loads((args.run / "index.json").read_text())
            return {"sweep": index["sweep"], "points": index["points"], "completed": index["completed"],
                    "units": index["units"],
                    "rows": [{"directory": e["directory"], "status": e["status"], **e["parameters"],
                              **{k: e.get("summary", {}).get(k) for k in ("phi", "milling", "clustering", "nematic")}}
                             for e in index["runs"]],
                    "note": "phi, milling, clustering, nematic per point; full values per point via query <point dir> summary"}
        return Q.summary(args.run)
    if w == "outputs":
        return Q.outputs(args.run)
    if w == "series":
        return Q.series(args.run, args.name, args.start, args.stop, args.points)
    if w == "stats":
        return Q.stats(args.run, args.name)
    if w == "at":
        step = args.value
        if args.time:
            dt = json.loads((args.run / "config.resolved.json").read_text())["simulation"]["dt"]
            step = args.value / dt
        return Q.at(args.run, step)
    if w == "final":
        return Q.final(args.run)
    if w == "diagnostics":
        return Q.diagnostics(args.run)
    return Q.descriptor(args.run)


def main(argv=None):
    import sys
    raw = list(sys.argv[1:] if argv is None else argv)
    _Parser.json_mode = "--json" in raw
    try:
        args = build_parser().parse_args(raw)
    except SystemExit as exc:              # --help, --version, or a usage error
        return exc.code
    as_json = getattr(args, "json", False)

    if args.command == "simulate":           # legacy behaviour, unchanged
        parser = build_parser()
        try:
            record = run(json.loads(args.config.read_text()), args.output)
        except (ValueError, TypeError, OSError) as exc:
            parser.error(str(exc))
        print(json.dumps(record["summary"], indent=2))
        return 0

    from . import configuration as C
    try:
        if args.command == "describe":
            doc = _describe(args)
            print(json.dumps(doc, indent=2) if as_json or args.schema or args.param else _describe_text(doc))
        elif args.command == "template":
            if as_json:
                _emit(C.default_config(args.model))
            else:
                print(C.template_text(args.model), end="")
        elif args.command == "validate":
            cfg, _ = C.load(args.config)
            errors, warnings = C.validate(cfg, allow_sweep=True)
            if errors:
                return _fail(errors, as_json, 2, warnings)
            doc = {"ok": True, "config": args.config, "kind": cfg.get("kind", "single"),
                   "seed": (cfg.get("simulation") or {}).get("seed"), "warnings": warnings}
            if "sweep" in cfg:
                pts = C.expand_sweep(cfg)
                ests = [C.estimate(p["config"]) for p in pts]
                doc["sweep_points"] = len(pts)
                doc["estimate"] = {"runtime_s": round(sum(e["runtime_s"] for e in ests), 3),
                                   "peak_memory_mb": max(e["peak_memory_mb"] for e in ests),
                                   "output_bytes_upper_bound": sum(e["output_bytes_upper_bound"] for e in ests),
                                   "per_point": ests[0] if len(ests) == 1 else "varies; first: " + json.dumps(ests[0])}
            else:
                doc["estimate"] = C.estimate(cfg)
            if doc["seed"] is None:
                doc["warnings"].append(C.error("simulation.seed", None, "no seed set; run will choose and record one",
                                               "Set simulation.seed for reproducibility."))
            _emit(doc)
        elif args.command in ("run", "sweep"):
            from . import rundir
            cfg, _ = C.load(args.config)
            argv_full = ["spp-physica-a", *raw]
            if args.command == "run":
                m = rundir.execute(cfg, args.out, force=args.force, argv=argv_full)
                s = json.loads((args.out / "summary.json").read_text())
                _emit({"ok": True, "run_dir": str(args.out), "status": m["status"], "seed": m["seed"],
                       "duration_s": m["duration_s"], "summary": {k: v["value"] for k, v in s["values"].items()}})
            else:
                index = rundir.sweep(cfg, args.out, force=args.force, argv=argv_full)
                _emit({"ok": index["completed"] == index["points"], "sweep_dir": str(args.out),
                       "points": index["points"], "completed": index["completed"],
                       "index": str(args.out / "index.json")})
                if index["completed"] != index["points"]:
                    return 1
        elif args.command == "query":
            _emit(_query(args))
        elif args.command == "compare":
            from . import queries as Q
            _emit(Q.compare(args.runs))
    except C.ConfigError as exc:
        return _fail(exc.errors, as_json, 2)
    except Exception as exc:  # noqa: BLE001 - runtime failure -> exit 1 with a structured message
        return _fail([{"field": args.command, "value": None, "reason": f"{type(exc).__name__}: {exc}",
                       "hint": "See log.txt and manifest.json in the run directory, if one was created."}],
                     as_json, 1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

