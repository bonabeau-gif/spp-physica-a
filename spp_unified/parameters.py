from __future__ import annotations
from dataclasses import dataclass, replace
import numpy as np

# Full simulator parameter order, including the optional taxis cue.
PARAMETER_NAMES = (
    "rho", "v0", "inertia", "interaction_range_ratio", "force_strength",
    "adhesion", "alignment_range_ratio", "alignment_strength",
    "gradient_angle", "gradient_strength", "noise",
)

# Primary paper experiments set the external gradient to zero. Nine parameters
# remain free. These are broad but computationally tractable finite-system bounds.
ACTIVE_SINGLE_NAMES = (
    "rho", "v0", "inertia", "interaction_range_ratio", "force_strength",
    "adhesion", "alignment_range_ratio", "alignment_strength", "noise",
)
EXPERIMENT_BOUNDS = np.array([
    [0.05, 0.30],      # rho -> about 23-139 particles for S=4, r0=0.1
    [0.01, 0.12],
    [0.0, 5.0],
    [2.1, 18.0],       # r1 < S/2 for clean minimum-image geometry
    [0.0, 5.0],
    [-0.80, 0.80],     # avoids singular beta at |B|=1
    [2.1, 18.0],
    [0.0, 5.0],
    [0.0, 1.0],
], dtype=float)

TWO_SPECIES_EXTRA_NAMES = (
    "fraction_B", "alignment_symmetric", "alignment_antisymmetric",
    "force_symmetric", "force_antisymmetric",
)
TWO_SPECIES_EXTRA_BOUNDS = np.array([
    [0.20, 0.80],
    [-1.0, 1.5],
    [-1.5, 1.5],
    [-1.0, 1.0],
    [-1.5, 1.5],
], dtype=float)


@dataclass(frozen=True)
class SingleParams:
    rho: float = 0.15
    v0: float = 0.05
    inertia: float = 0.5
    interaction_range_ratio: float = 7.0
    force_strength: float = 1.0
    adhesion: float = 0.0
    alignment_range_ratio: float = 7.0
    alignment_strength: float = 2.0
    gradient_angle: float = 0.0
    gradient_strength: float = 0.0
    noise: float = 0.10

    def __post_init__(self):
        if not np.isfinite(self.as_array()).all():
            raise ValueError("Particle parameters must be finite")
        if self.rho <= 0 or self.v0 < 0:
            raise ValueError("rho must be positive and v0 must be nonnegative")
        if self.interaction_range_ratio <= 2 or self.alignment_range_ratio <= 0:
            raise ValueError("interaction_range_ratio must exceed 2; alignment range must be positive")
        if not 0 <= self.noise <= 1:
            raise ValueError("noise must lie in [0, 1]")

    def as_array(self) -> np.ndarray:
        return np.array([getattr(self, n) for n in PARAMETER_NAMES], dtype=float)

    @classmethod
    def from_active_vector(cls, x: np.ndarray, *, gradient_strength: float = 0.0,
                           gradient_angle: float = 0.0) -> "SingleParams":
        x = np.asarray(x, dtype=float)
        if x.shape != (len(ACTIVE_SINGLE_NAMES),):
            raise ValueError("Single-species vectors must contain exactly 9 parameters")
        vals = dict(zip(ACTIVE_SINGLE_NAMES, x))
        vals["gradient_strength"] = float(gradient_strength)
        vals["gradient_angle"] = float(gradient_angle)
        return cls(**vals)


@dataclass(frozen=True)
class TwoSpeciesParams:
    base: SingleParams = SingleParams()
    fraction_B: float = 0.5
    alignment_symmetric: float = 1.0
    alignment_antisymmetric: float = 0.0
    force_symmetric: float = 0.0
    force_antisymmetric: float = 0.0

    def __post_init__(self):
        values = [getattr(self, key) for key in TWO_SPECIES_EXTRA_NAMES]
        if not np.isfinite(values).all() or not 0 <= self.fraction_B <= 1:
            raise ValueError("Mixture parameters must be finite; fraction_B must lie in [0, 1]")

    def matrices(self) -> tuple[np.ndarray, np.ndarray]:
        """Return 2x2 alignment and additive radial-force coupling matrices.

        Rows are focal species and columns are neighbor species. Same-species
        alignment is one and same-species additive force is zero. Cross-species
        couplings are decomposed into symmetric and antisymmetric parts.
        """
        js = self.alignment_symmetric
        ja = self.alignment_antisymmetric
        fs = self.force_symmetric
        fa = self.force_antisymmetric
        J = np.array([[1.0, js + ja], [js - ja, 1.0]], dtype=float)
        C = np.array([[0.0, fs + fa], [fs - fa, 0.0]], dtype=float)
        return J, C

    @classmethod
    def from_active_vector(cls, x: np.ndarray, *, reciprocal: bool = False) -> "TwoSpeciesParams":
        x = np.asarray(x, dtype=float)
        if x.shape != (len(ACTIVE_SINGLE_NAMES) + len(TWO_SPECIES_EXTRA_NAMES),):
            raise ValueError("Two-species vectors must contain exactly 14 parameters")
        n = len(ACTIVE_SINGLE_NAMES)
        base = SingleParams.from_active_vector(x[:n])
        vals = dict(zip(TWO_SPECIES_EXTRA_NAMES, x[n:]))
        if reciprocal:
            vals["alignment_antisymmetric"] = 0.0
            vals["force_antisymmetric"] = 0.0
        return cls(base=base, **vals)


@dataclass(frozen=True)
class SimulationConfig:
    S: float = 4.0
    r0: float = 0.1
    steps: int = 240
    sample_start: int = 120
    sample_every: int = 2
    boundary: str = "periodic"  # periodic or reflecting
    dt: float = 1.0
    hard_core_strength: float = 50.0
    overlap_iterations: int = 3
    seed: int = 0
    use_cell_list: bool = False  # exact linked-cell accelerator (see spp_unified/neighbors.py)

    def __post_init__(self):
        if self.boundary not in ("periodic", "reflecting"):
            raise ValueError("boundary must be 'periodic' or 'reflecting'")
        for name in ("steps", "sample_start", "sample_every", "overlap_iterations", "seed"):
            value = getattr(self, name)
            if not isinstance(value, (int, np.integer)) or isinstance(value, bool):
                raise ValueError(f"{name} must be an integer")
        if self.steps < 1 or not 0 <= self.sample_start < self.steps or self.sample_every < 1:
            raise ValueError("Require steps > sample_start >= 0 and sample_every >= 1")
        if self.overlap_iterations < 0 or self.seed < 0:
            raise ValueError("overlap_iterations and seed must be nonnegative")
        if not np.isfinite([self.S, self.r0, self.dt, self.hard_core_strength]).all():
            raise ValueError("Simulation scales must be finite")
        if self.r0 <= 0 or self.S <= 2 * self.r0 or self.dt <= 0 or self.hard_core_strength <= 0:
            raise ValueError("Require S > 2*r0 > 0, dt > 0, and hard_core_strength > 0")

    def n_max(self) -> int:
        return max(2, int(round((np.sqrt(3.0) / 6.0) * (self.S / self.r0) ** 2)))

    def with_seed(self, seed: int) -> "SimulationConfig":
        return replace(self, seed=int(seed))
