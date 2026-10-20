from __future__ import annotations
import numpy as np


def pairwise_displacements(x: np.ndarray, S: float, periodic: bool) -> tuple[np.ndarray, np.ndarray]:
    d = x[None, :, :] - x[:, None, :]
    if periodic:
        d -= S * np.round(d / S)
    r = np.sqrt(np.sum(d * d, axis=2))
    return d, r


def periodic_center(x: np.ndarray, S: float) -> np.ndarray:
    """Circular center in each coordinate for a periodic square."""
    ang = 2.0 * np.pi * x / S
    z = np.exp(1j * ang).mean(axis=0)
    a = np.angle(z) % (2.0 * np.pi)
    return S * a / (2.0 * np.pi)


def relative_to_center(x: np.ndarray, S: float, periodic: bool) -> tuple[np.ndarray, np.ndarray]:
    center = periodic_center(x, S) if periodic else x.mean(axis=0)
    rel = x - center
    if periodic:
        rel -= S * np.round(rel / S)
    return center, rel


def _grid_initial_positions(n: int, S: float, min_sep: float, rng: np.random.Generator) -> np.ndarray:
    """Place particles on a randomized grid with spacing at least min_sep when feasible."""
    n_side = int(np.ceil(np.sqrt(n)))
    spacing = S / n_side
    if spacing < min_sep:
        # Hex-like rectangular fallback at the maximum feasible resolution.
        n_side = max(1, int(np.floor(S / min_sep)))
        if n_side * n_side < n:
            raise ValueError(f"Cannot place {n} non-overlapping particles in the selected box")
        spacing = S / n_side
    xs = (np.arange(n_side) + 0.5) * spacing
    pts = np.array(np.meshgrid(xs, xs)).reshape(2, -1).T
    rng.shuffle(pts)
    pts = pts[:n]
    jitter = min(0.15 * spacing, max(0.0, 0.45 * (spacing - min_sep)))
    if jitter > 0:
        pts += rng.uniform(-jitter, jitter, size=pts.shape)
    pts %= S
    # random global translation preserves separation under periodic boundaries
    pts = (pts + rng.uniform(0.0, S, size=2)) % S
    return pts


def initialize_positions(n: int, S: float, r0: float, rng: np.random.Generator,
                         periodic: bool) -> np.ndarray:
    # Grid initialization is robust at the moderate densities used here.
    x = _grid_initial_positions(n, S, 2.02 * r0, rng)
    if not periodic:
        x = np.clip(x, r0, S - r0)
    return x


def reflect_positions(x: np.ndarray, theta: np.ndarray, S: float) -> tuple[np.ndarray, np.ndarray]:
    for dim in (0, 1):
        # Repeat because a very large step could cross more than once.
        for _ in range(3):
            lo = x[:, dim] < 0.0
            hi = x[:, dim] > S
            if not (lo.any() or hi.any()):
                break
            x[lo, dim] = -x[lo, dim]
            x[hi, dim] = 2.0 * S - x[hi, dim]
            if dim == 0:
                theta[lo | hi] = np.pi - theta[lo | hi]
            else:
                theta[lo | hi] = -theta[lo | hi]
    return x, ((theta + np.pi) % (2.0 * np.pi)) - np.pi


def resolve_overlaps(x: np.ndarray, S: float, r0: float, periodic: bool,
                     iterations: int = 3) -> np.ndarray:
    """Project overlapping pairs apart while preserving each pair midpoint."""
    min_r = 2.0 * r0
    for _ in range(max(0, iterations)):
        d, r = pairwise_displacements(x, S, periodic)
        np.fill_diagonal(r, np.inf)
        ii, jj = np.where(np.triu(r < min_r, 1))
        if len(ii) == 0:
            break
        corr = np.zeros_like(x)
        rij = r[ii, jj]
        u = np.zeros((len(ii), 2), dtype=float)
        nz = rij > 1e-12
        u[nz] = d[ii[nz], jj[nz]] / rij[nz, None]
        u[~nz, 0] = 1.0
        delta = 0.5 * (min_r - rij + 1e-9)[:, None] * u
        np.add.at(corr, ii, -delta)
        np.add.at(corr, jj, delta)
        x += corr
        if periodic:
            x %= S
        else:
            x[:] = np.clip(x, r0, S - r0)
    return x
