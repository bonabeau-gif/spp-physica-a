"""Parameter spaces for the boundary-mapping subsystem.

Two frozen spaces are exposed, one per trained representation. Each space pins the
EXACT simulator config and bounds the corresponding autoencoder was trained on,
because Section 16 of the implementation plan requires the representation (and
therefore the descriptor standardization used by the sensitivity oracle) to be
frozen so sensitivity values are comparable across the whole study.

Divergences from the plan's nominal Table 1, logged deliberately:
  * two-species bounds use the code's EXPERIMENT_BOUNDS (rho in [0.05, 0.30], radius
    ratios to 18) stacked with TWO_SPECIES_EXTRA_BOUNDS, NOT the plan's rho->0.70 /
    box 3.2 values, because the two-species autoencoder was trained that way.
  * single-species bounds use EXPANDED_BOUNDS (rho in [0.05, 0.70], radius ratios to
    14), matching the single autoencoder's training corpus.
  * J_s and C_s bounds (plan says "not stated in the paper"): we adopt the code's
    alignment_symmetric in [-1.0, 1.5] and force_symmetric in [-1.0, 1.0].

Plan parameter names map to the code's names as:
    eps=inertia, r1_ratio=interaction_range_ratio, f0=force_strength, B=adhesion,
    ra_ratio=alignment_range_ratio, alpha=alignment_strength, eta=noise,
    f_B=fraction_B, J_s=alignment_symmetric, J_a=alignment_antisymmetric,
    C_s=force_symmetric, C_a=force_antisymmetric.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np

ROOT = Path(__import__("os").environ.get("SPP_PROJECT_ROOT", Path(__file__).resolve().parents[1]))
EXPANDED = ROOT / "results" / "expanded"

# ---- plan name <-> code name ----
PLAN_TO_CODE = {
    "rho": "rho", "v0": "v0", "eps": "inertia", "r1_ratio": "interaction_range_ratio",
    "f0": "force_strength", "B": "adhesion", "ra_ratio": "alignment_range_ratio",
    "alpha": "alignment_strength", "eta": "noise",
    "f_B": "fraction_B", "J_s": "alignment_symmetric", "J_a": "alignment_antisymmetric",
    "C_s": "force_symmetric", "C_a": "force_antisymmetric",
}
PARAM_ORDER_SINGLE = ["rho", "v0", "eps", "r1_ratio", "f0", "B", "ra_ratio", "alpha", "eta"]
PARAM_ORDER_TWO = PARAM_ORDER_SINGLE + ["f_B", "J_s", "J_a", "C_s", "C_a"]

# ---- bounds (physical), same row order as PARAM_ORDER_* ----
EXPANDED_BOUNDS = np.array([
    [0.05, 0.70], [0.01, 0.12], [0.0, 5.0], [2.1, 14.0], [0.0, 5.0],
    [-0.80, 0.80], [2.1, 14.0], [0.0, 5.0], [0.0, 1.0],
], float)
EXPERIMENT_BOUNDS = np.array([
    [0.05, 0.30], [0.01, 0.12], [0.0, 5.0], [2.1, 18.0], [0.0, 5.0],
    [-0.80, 0.80], [2.1, 18.0], [0.0, 5.0], [0.0, 1.0],
], float)
TWO_SPECIES_EXTRA_BOUNDS = np.array([
    [0.20, 0.80], [-1.0, 1.5], [-1.5, 1.5], [-1.0, 1.0], [-1.5, 1.5],
], float)
BOUNDS_TWO = np.vstack([EXPERIMENT_BOUNDS, TWO_SPECIES_EXTRA_BOUNDS])


@dataclass(frozen=True)
class Space:
    name: str                    # "single" | "two"
    param_order: tuple           # plan names, ordering of the unit vector
    bounds: np.ndarray           # (d, 2) physical bounds, same order
    # frozen simulator config (per trained representation)
    S: float
    r0: float
    base_steps: int
    sample_every: int
    state_every: int
    nbins: int
    max_frames: int
    descr_dim: int
    ae_path: Path
    descr_npz: Path              # holds corpus descriptors X (+ lo/hi for two)
    latent_npz: Path | None      # single: separate latent file with lo/hi

    @property
    def d(self) -> int:
        return len(self.param_order)


SINGLE = Space(
    name="single", param_order=tuple(PARAM_ORDER_SINGLE), bounds=EXPANDED_BOUNDS,
    S=3.2, r0=0.1, base_steps=160, sample_every=2, state_every=8, nbins=12,
    max_frames=10, descr_dim=51,
    ae_path=EXPANDED / "single_behavior_autoencoder.pt",
    descr_npz=EXPANDED / "single_behavior_descriptors.npz",
    latent_npz=EXPANDED / "single_behavior_latent.npz",
)
TWO = Space(
    name="two", param_order=tuple(PARAM_ORDER_TWO), bounds=BOUNDS_TWO,
    S=4.0, r0=0.1, base_steps=220, sample_every=2, state_every=10, nbins=8,
    max_frames=10, descr_dim=107,
    ae_path=EXPANDED / "multi_behavior_autoencoder.pt",
    descr_npz=EXPANDED / "multi_behavior_descriptors.npz",
    latent_npz=None,
)
SPACES = {"single": SINGLE, "two": TWO}


def get_space(name: str) -> Space:
    return SPACES[name]


# ---- unit <-> physical ----
def to_unit(x_phys: np.ndarray, space: Space) -> np.ndarray:
    lo, hi = space.bounds[:, 0], space.bounds[:, 1]
    return (np.asarray(x_phys, float) - lo) / (hi - lo)


def from_unit(u: np.ndarray, space: Space) -> np.ndarray:
    lo, hi = space.bounds[:, 0], space.bounds[:, 1]
    return lo + np.asarray(u, float) * (hi - lo)


def clip_unit(u: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(u, float), 0.0, 1.0)


def idx(space: Space, plan_name: str) -> int:
    return list(space.param_order).index(plan_name)
