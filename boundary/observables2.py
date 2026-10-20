"""Module A: species-resolved observables.

Each quantity is single-purpose and zero in one phase for a structural reason (the
composite chirality score C = min(|omega|/0.05,1) R^2 <phi> is deliberately NOT used
as a boundary observable). nan is propagated, never imputed: Omega_rel/Omega_tot are
nan below 0.1 mean polarization, alpha_WC nan when a minority count is below 5.
"""
from __future__ import annotations
import numpy as np
from spp_unified.geometry import pairwise_displacements
from spp_unified.learned import hexatic_order

# species labels: 0 -> A (s=+1), 1 -> B (s=-1)


def _species_heading(theta_row, mask):
    """Mean heading complex vector over a species subset for one frame."""
    if mask.sum() == 0:
        return 0.0 + 0.0j
    return np.mean(np.exp(1j * theta_row[mask]))


def _weighted_slope(y, w):
    """Weighted least-squares slope of y(t) vs t (unit step)."""
    t = np.arange(len(y), dtype=float)
    w = np.asarray(w, float) + 1e-12
    W = w.sum()
    tb = (w * t).sum() / W
    yb = (w * y).sum() / W
    num = (w * (t - tb) * (y - yb)).sum()
    den = (w * (t - tb) ** 2).sum()
    return num / den if den > 1e-12 else 0.0


def _phi_dem(x, s, S, n_shells=8):
    """Species structure factor peak over the lowest-k shells."""
    N = len(x)
    kbase = 2.0 * np.pi / S
    ks = []
    for a in range(0, 4):
        for b in range(0, 4):
            if a == 0 and b == 0:
                continue
            ks.append((a, b, a * a + b * b))
    ks.sort(key=lambda t: t[2])
    vals = []
    for a, b, _ in ks[:n_shells]:
        kv = kbase * np.array([a, b])
        amp = np.sum(s * np.exp(1j * (x @ kv)))
        vals.append((amp.real ** 2 + amp.imag ** 2) / N)
    return float(np.max(vals)) if vals else 0.0


def _alpha_wc(x, species, S, periodic, r0):
    """Warren-Cowley short-range order, symmetrized; nan if a minority count < 5."""
    _, r = pairwise_displacements(x, S, periodic)
    np.fill_diagonal(r, np.inf)
    rc = 4.0 * r0
    A = species == 0; B = species == 1
    fA = A.mean(); fB = B.mean()
    if A.sum() < 5 or B.sum() < 5:
        return np.nan
    neigh = r < rc
    p_unlike = []
    for grp, other, fother in ((A, B, fB), (B, A, fA)):
        vals = []
        for i in np.where(grp)[0]:
            k = neigh[i].sum()
            if k == 0:
                continue
            n_unlike = neigh[i][other].sum()
            vals.append(n_unlike / k)
        if not vals or fother < 1e-9:
            return np.nan
        p_unlike.append((1.0 - np.mean(vals) / fother))
    return float(fA * p_unlike[0] + fB * p_unlike[1])


def _chi_nn(x, species, S, periodic):
    """Mean over nearest-neighbor pairs of s_i s_j."""
    _, r = pairwise_displacements(x, S, periodic)
    np.fill_diagonal(r, np.inf)
    nn = np.argmin(r, axis=1)
    s = np.where(species == 0, 1.0, -1.0)
    return float(np.mean(s * s[nn]))


def compute_obs2(res, cfg, space) -> dict:
    species = np.asarray(res.species)
    A = species == 0; B = species == 1
    fA = float(A.mean()); fB = float(B.mean())
    periodic = cfg.boundary == "periodic"

    theta_series = np.asarray(res.timeseries["theta"])   # (T, n)
    PA = np.array([_species_heading(th, A) for th in theta_series])
    PB = np.array([_species_heading(th, B) for th in theta_series])
    tot = fA * PA + fB * PB
    p_plus = np.abs(tot)
    p_minus = np.abs(fA * PA - fB * PB)

    d_theta = np.unwrap(np.angle(PA) - np.angle(PB))
    R_lock = float(np.abs(np.mean(np.exp(1j * d_theta))))
    pol_inst = np.minimum(np.abs(PA), np.abs(PB))
    mean_pol = float(np.mean(np.minimum(np.abs(PA), np.abs(PB))))

    if mean_pol >= 0.1:
        Omega_rel = float(_weighted_slope(d_theta, pol_inst))
    else:
        Omega_rel = np.nan
    tot_ang = np.unwrap(np.angle(tot))
    if float(np.mean(np.abs(tot))) >= 0.1:
        Omega_tot = float(_weighted_slope(tot_ang, np.abs(tot)))
    else:
        Omega_tot = np.nan

    # spatial observables over the recorded frames (positions)
    frames = res.states[-space.max_frames:]
    awc, phidem, chinn, p6A, p6B = [], [], [], [], []
    for x, _th in frames:
        s = np.where(species == 0, 1.0, -1.0)
        awc.append(_alpha_wc(x, species, cfg.S, periodic, cfg.r0))
        phidem.append(_phi_dem(x, s, cfg.S))
        chinn.append(_chi_nn(x, species, cfg.S, periodic))
        if A.sum() >= 6:
            p6A.append(hexatic_order(x[A], cfg.S, periodic))
        if B.sum() >= 6:
            p6B.append(hexatic_order(x[B], cfg.S, periodic))

    return {
        "P_plus": float(np.mean(p_plus)),
        "P_minus": float(np.mean(p_minus)),
        "R_lock": R_lock,
        "Omega_rel": Omega_rel,
        "Omega_tot": Omega_tot,
        "alpha_WC": float(np.nanmean(awc)) if np.any(np.isfinite(awc)) else np.nan,
        "Phi_dem": float(np.mean(phidem)),
        "psi6_A": float(np.mean(p6A)) if p6A else np.nan,
        "psi6_B": float(np.mean(p6B)) if p6B else np.nan,
        "chi_nn": float(np.mean(chinn)),
        "D": float(res.summary.get("drift", np.nan)),
        "sigma_seg": float(res.summary.get("segregation", np.nan)),
    }
