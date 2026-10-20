from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .parameters import SingleParams, TwoSpeciesParams, SimulationConfig
from .geometry import (
    pairwise_displacements, initialize_positions, reflect_positions, resolve_overlaps,
)
from .metrics import polarization, nematic_order, milling, clustering, species_segregation, heading_rotation


def beta_from_adhesion(B: float) -> float:
    """Adhesion mapping beta = exp(tan(pi B / 2))."""
    B = float(np.clip(B, -0.95, 0.95))
    return float(np.exp(np.clip(np.tan(np.pi * B / 2.0), -8.0, 8.0)))


def manuscript_pair_force(r: np.ndarray, r0: float, r1: float, beta: float,
                          hard_core_strength: float) -> np.ndarray:
    """Piecewise radial force law from Section 2.1.

    For 2 r0 < r < r1:
        f(r) = ((r - 2 r0)/(r1 - 2 r0)) * (beta + 1) - 1.
    The force is zero outside r1 and a finite numerical replacement for -infinity
    inside the hard core. Positive values attract along the i-to-j direction;
    negative values repel.
    """
    f = np.zeros_like(r, dtype=float)
    denom = max(r1 - 2.0 * r0, 1e-12)
    shell = (r > 2.0 * r0) & (r < r1)
    u = (r[shell] - 2.0 * r0) / denom
    f[shell] = (beta + 1.0) * u - 1.0
    f[r <= 2.0 * r0] = -float(hard_core_strength)
    return f


@dataclass
class SimulationResult:
    summary: dict[str, float]
    timeseries: dict[str, np.ndarray]
    states: list[tuple[np.ndarray, np.ndarray]]
    species: np.ndarray | None = None


def _simulate(base: SingleParams, cfg: SimulationConfig, *, species: np.ndarray | None,
              alignment_matrix: np.ndarray | None, force_matrix: np.ndarray | None,
              record_states: bool, state_every: int,
              x_init=None, theta_init=None) -> SimulationResult:
    rng = np.random.default_rng(cfg.seed)
    periodic = cfg.boundary == "periodic"
    if x_init is not None:                          # seeded initial condition (Sec 5 protocol)
        x = np.asarray(x_init, float).copy(); n = len(x)
    else:
        n = max(8, int(round(base.rho * cfg.n_max())))
    if species is None:
        species = np.zeros(n, dtype=np.int8)
    elif len(species) != n:
        raise ValueError("species length must match particle count")

    if x_init is None:
        x = initialize_positions(n, cfg.S, cfg.r0, rng, periodic)
    x_unwrapped = x.copy()
    theta = (rng.uniform(-np.pi, np.pi, size=n) if theta_init is None
             else np.asarray(theta_init, float).copy())

    r1 = base.interaction_range_ratio * cfg.r0
    ra = base.alignment_range_ratio * cfg.r0
    beta = beta_from_adhesion(base.adhesion)
    G = np.array([np.cos(base.gradient_angle), np.sin(base.gradient_angle)])

    if alignment_matrix is None:
        alignment_matrix = np.ones((1, 1), dtype=float)
    if force_matrix is None:
        force_matrix = np.zeros((1, 1), dtype=float)

    ts_phi, ts_nem, ts_mill, ts_clust, ts_seg = [], [], [], [], []
    ts_com = []
    ts_theta = []
    states: list[tuple[np.ndarray, np.ndarray]] = []

    for step in range(cfg.steps):
        uhead = np.column_stack([np.cos(theta), np.sin(theta)])
        if cfg.use_cell_list:
            from spp_unified.neighbors import sparse_align_force
            align_dir, force = sparse_align_force(
                x, uhead, species, cfg.S, periodic, cfg.r0, r1, ra, beta,
                cfg.hard_core_strength, alignment_matrix, force_matrix)
        else:
            d, r = pairwise_displacements(x, cfg.S, periodic)
            np.fill_diagonal(r, np.inf)
            # Weighted alignment sum, including the focal particle itself with unit weight.
            weights = alignment_matrix[species[:, None], species[None, :]]
            amask = (r <= ra).astype(float) * weights
            avec = np.einsum("ij,jk->ik", amask, uhead, optimize=False) + uhead
            anorm = np.sqrt(np.sum(avec * avec, axis=1, keepdims=True))
            align_dir = np.divide(avec, anorm, out=np.zeros_like(avec), where=anorm > 1e-12)

            # Reciprocal force plus an additive species-dependent radial term.
            unit = np.divide(d, r[:, :, None], out=np.zeros_like(d), where=np.isfinite(r)[:, :, None] & (r[:, :, None] > 1e-12))
            fscalar = manuscript_pair_force(r, cfg.r0, r1, beta, cfg.hard_core_strength)
            if force_matrix.shape[0] > 1:
                cross = force_matrix[species[:, None], species[None, :]]
                shell = (r > 2.0 * cfg.r0) & (r < r1)
                envelope = np.zeros_like(r)
                envelope[shell] = 1.0 - (r[shell] - 2.0 * cfg.r0) / max(r1 - 2.0 * cfg.r0, 1e-12)
                fscalar = fscalar + cross * envelope
            force = (fscalar[:, :, None] * unit).sum(axis=1)

        drive = (base.inertia * uhead + base.alignment_strength * align_dir +
                 base.gradient_strength * G + base.force_strength * force)
        dn = np.sqrt(np.sum(drive * drive, axis=1))
        base_angle = theta.copy()
        nz = dn > 1e-12
        base_angle[nz] = np.arctan2(drive[nz, 1], drive[nz, 0])
        theta = base_angle + base.noise * rng.uniform(-np.pi, np.pi, size=n)
        theta = (theta + np.pi) % (2.0 * np.pi) - np.pi

        delta = cfg.dt * base.v0 * np.column_stack([np.cos(theta), np.sin(theta)])
        x_unwrapped += delta
        x += delta
        if periodic:
            x %= cfg.S
        else:
            x, theta = reflect_positions(x, theta, cfg.S)
            # For reflecting boundaries, unwrapped COM is simply the physical COM.
            x_unwrapped = x.copy()
        if cfg.use_cell_list:
            from spp_unified.neighbors import resolve_overlaps_cell
            x = resolve_overlaps_cell(x, cfg.S, cfg.r0, periodic, cfg.overlap_iterations)
        else:
            x = resolve_overlaps(x, cfg.S, cfg.r0, periodic, cfg.overlap_iterations)

        if step >= cfg.sample_start and ((step - cfg.sample_start) % cfg.sample_every == 0):
            ph = polarization(theta)
            ts_phi.append(ph)
            ts_nem.append(nematic_order(theta))
            ts_mill.append(milling(x, theta, cfg.S, periodic))
            ts_clust.append(clustering(x, r1, cfg.S, periodic))
            ts_com.append(x_unwrapped.mean(axis=0).copy())
            ts_theta.append(theta.copy())
            if alignment_matrix.shape[0] > 1:
                ts_seg.append(species_segregation(x, species, r1, cfg.S, periodic))
        if record_states and (step % max(1, state_every) == 0 or step == cfg.steps - 1):
            states.append((x.copy(), theta.copy()))

    phi = np.asarray(ts_phi)
    nem = np.asarray(ts_nem)
    mill = np.asarray(ts_mill)
    clust = np.asarray(ts_clust)
    com = np.asarray(ts_com)
    theta_series = np.asarray(ts_theta)
    seg = np.asarray(ts_seg) if ts_seg else np.zeros_like(phi)

    if len(com) >= 2:
        elapsed = (len(com) - 1) * cfg.sample_every * cfg.dt
        drift = float(np.sqrt(np.sum((com[-1] - com[0]) ** 2)) / max(base.v0 * elapsed, 1e-12))
        drift = float(np.clip(drift, 0.0, 1.0))
    else:
        drift = 0.0
    rot = heading_rotation(theta_series, phi) if len(phi) >= 3 else {"omega": 0.0, "rotation_r2": 0.0, "chirality": 0.0}
    omega_physical = rot["omega"] / max(cfg.sample_every * cfg.dt, 1e-12)
    chirality_score = float(np.clip(abs(omega_physical) / 0.05, 0.0, 1.0) * rot["rotation_r2"] * float(np.mean(phi)))

    cscale = max(float(np.mean(clust)), 1.0)
    temporal_instability = (
        float(np.std(phi)) + float(np.std(mill)) + min(float(np.std(clust)) / cscale, 1.0)
    ) / 3.0
    temporal_stability = float(np.clip(1.0 - temporal_instability, 0.0, 1.0))

    summary = {
        "n": float(n),
        "phi": float(np.mean(phi)),
        "nematic": float(np.mean(nem)),
        "milling": float(np.mean(mill)),
        "clustering": float(np.mean(clust)),
        "segregation": float(np.mean(seg)) if len(seg) else 0.0,
        "drift": drift,
        "omega": omega_physical,
        "abs_omega": abs(omega_physical),
        "rotation_r2": rot["rotation_r2"],
        "chirality": chirality_score,
        "temporal_stability": temporal_stability,
    }
    timeseries = {
        "phi": phi, "nematic": nem, "milling": mill, "clustering": clust,
        "segregation": seg, "com": com, "theta": theta_series,
    }
    if record_states and not states:
        states.append((x.copy(), theta.copy()))
    return SimulationResult(summary=summary, timeseries=timeseries, states=states,
                            species=species.copy() if alignment_matrix.shape[0] > 1 else None)


def simulate_single(params: SingleParams, cfg: SimulationConfig, *, record_states: bool = False,
                    state_every: int = 10) -> SimulationResult:
    return _simulate(params, cfg, species=None, alignment_matrix=None, force_matrix=None,
                     record_states=record_states, state_every=state_every)


def simulate_two_species(params: TwoSpeciesParams, cfg: SimulationConfig, *, record_states: bool = False,
                         state_every: int = 10, x_init=None, species_init=None) -> SimulationResult:
    if species_init is not None:                    # seeded initial condition
        species = np.asarray(species_init, dtype=np.int8)
    else:
        n = max(8, int(round(params.base.rho * cfg.n_max())))
        nB = int(round(params.fraction_B * n))
        species = np.zeros(n, dtype=np.int8)
        species[n - nB:] = 1
        # Randomize species labels independently of initial spatial placement.
        rng = np.random.default_rng(cfg.seed + 7919)
        rng.shuffle(species)
    J, C = params.matrices()
    return _simulate(params.base, cfg, species=species, alignment_matrix=J, force_matrix=C,
                     record_states=record_states, state_every=state_every,
                     x_init=x_init)
