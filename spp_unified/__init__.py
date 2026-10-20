"""Unified self-propelled-particle simulation and illumination package."""
from .parameters import SingleParams, TwoSpeciesParams, SimulationConfig
from .model import simulate_single, simulate_two_species
from .evaluation import evaluate_single, evaluate_two_species

__all__ = [
    "SingleParams", "TwoSpeciesParams", "SimulationConfig",
    "simulate_single", "simulate_two_species",
    "evaluate_single", "evaluate_two_species",
]
