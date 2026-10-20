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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", action="version", version="spp-physica-a 1.0.0")
    sub = parser.add_subparsers(dest="command", required=True)
    sim = sub.add_parser("simulate", help="Run a JSON configuration and save a trajectory")
    sim.add_argument("--config", type=Path, required=True)
    sim.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        record = run(json.loads(args.config.read_text()), args.output)
    except (ValueError, TypeError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps(record["summary"], indent=2))


if __name__ == "__main__":
    main()

