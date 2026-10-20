from __future__ import annotations
import numpy as np
from .geometry import pairwise_displacements, relative_to_center


def polarization(theta: np.ndarray) -> float:
    return float(np.hypot(np.cos(theta).mean(), np.sin(theta).mean()))


def nematic_order(theta: np.ndarray) -> float:
    return float(np.hypot(np.cos(2.0 * theta).mean(), np.sin(2.0 * theta).mean()))


def milling(x: np.ndarray, theta: np.ndarray, S: float, periodic: bool) -> float:
    _, rel = relative_to_center(x, S, periodic)
    u = np.column_stack([np.cos(theta), np.sin(theta)])
    cross = rel[:, 0] * u[:, 1] - rel[:, 1] * u[:, 0]
    den = np.sqrt(np.sum(rel * rel, axis=1)).sum()
    return float(abs(cross.sum()) / den) if den > 1e-12 else 0.0


def pair_distance_cdf_square(radius: float, S: float) -> float:
    """Probability that two uniform points in a square are within radius.

    Exact for radius <= S. The expression is the CDF of square line-picking
    distances in the first branch. Experiments keep interaction radii <= S/2.
    """
    z = float(radius / S)
    if z <= 0:
        return 0.0
    if z <= 1.0:
        return float(np.clip(np.pi * z*z - (8.0/3.0) * z**3 + 0.5 * z**4, 0.0, 1.0))
    return 1.0


def clustering(x: np.ndarray, radius: float, S: float, periodic: bool) -> float:
    n = len(x)
    if n < 2 or radius <= 0:
        return 0.0
    _, r = pairwise_displacements(x, S, periodic)
    np.fill_diagonal(r, np.inf)
    observed = float((r < radius).sum(axis=1).mean())
    if periodic:
        p = min(np.pi * radius * radius / (S * S), 1.0)
    else:
        p = pair_distance_cdf_square(radius, S)
    expected = (n - 1) * p
    return observed / expected if expected > 1e-12 else 0.0


def species_segregation(x: np.ndarray, species: np.ndarray, radius: float,
                        S: float, periodic: bool) -> float:
    n = len(x)
    if n < 2:
        return 0.0
    _, r = pairwise_displacements(x, S, periodic)
    ii, jj = np.where(np.triu(r < radius, 1))
    if len(ii) == 0:
        return 0.0
    same = float(np.mean(species[ii] == species[jj]))
    pB = float(np.mean(species == 1))
    expected = pB*pB + (1-pB)*(1-pB)
    denom = max(1.0 - expected, 1e-12)
    return float(np.clip((same - expected) / denom, -1.0, 1.0))


def heading_rotation(theta_series: np.ndarray, phi_series: np.ndarray) -> dict[str, float]:
    """Estimate signed rotation rate and persistence of the global heading."""
    z = np.mean(np.exp(1j * theta_series), axis=1)
    ang = np.unwrap(np.angle(z))
    t = np.arange(len(ang), dtype=float)
    w = np.clip(phi_series, 0.0, 1.0) ** 2 + 1e-6
    tw = np.average(t, weights=w)
    aw = np.average(ang, weights=w)
    den = np.sum(w * (t - tw) ** 2)
    slope = float(np.sum(w * (t - tw) * (ang - aw)) / den) if den > 1e-12 else 0.0
    fit = aw + slope * (t - tw)
    sst = float(np.sum(w * (ang - aw) ** 2))
    sse = float(np.sum(w * (ang - fit) ** 2))
    r2 = 1.0 - sse / sst if sst > 1e-12 else 0.0
    r2 = float(np.clip(r2, 0.0, 1.0))
    mean_phi = float(np.mean(phi_series))
    # Dimensionless score for illumination, with physical rate retained separately.
    score = float(np.clip(abs(slope) / 0.10, 0.0, 1.0) * r2 * mean_phi)
    return {"omega": slope, "rotation_r2": r2, "chirality": score}
