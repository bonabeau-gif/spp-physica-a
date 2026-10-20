from __future__ import annotations
import numpy as np
from .parameters import SingleParams, TwoSpeciesParams, SimulationConfig
from .model import simulate_single, simulate_two_species

SUMMARY_KEYS = (
    "phi", "nematic", "milling", "clustering", "segregation", "drift",
    "omega", "abs_omega", "rotation_r2", "chirality", "temporal_stability",
)


def _aggregate(rows: list[dict[str, float]]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key in rows[0]:
        vals = np.array([r[key] for r in rows], dtype=float)
        out[key] = float(np.mean(vals))
        out[f"{key}_sd"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    # Across-seed repeatability for behavior measures on comparable scales.
    spread = np.mean([
        out.get("phi_sd", 0.0),
        out.get("milling_sd", 0.0),
        min(out.get("clustering_sd", 0.0) / max(out.get("clustering", 1.0), 1.0), 1.0),
    ])
    out["seed_reproducibility"] = float(np.clip(1.0 - spread, 0.0, 1.0))
    return out


def evaluate_single(params: SingleParams, cfg: SimulationConfig, seeds: list[int] | np.ndarray) -> dict[str, float]:
    rows = [simulate_single(params, cfg.with_seed(int(s))).summary for s in seeds]
    return _aggregate(rows)


def evaluate_two_species(params: TwoSpeciesParams, cfg: SimulationConfig,
                         seeds: list[int] | np.ndarray) -> dict[str, float]:
    rows = [simulate_two_species(params, cfg.with_seed(int(s))).summary for s in seeds]
    return _aggregate(rows)
