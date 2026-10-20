"""Module G: boundary typing, controls, and reporting.

Boundary points are clustered into types by their normal loadings; each type is reported
with bootstrap intervals on the mean normal (quoted only where the interval excludes
zero, matching the sloppiness-table discipline) and subjected to two mandatory controls:
  * reciprocal control: re-evaluate with J_a = C_a = 0 held fixed;
  * single-species control: re-evaluate with f_B = 0 (a -> single species).
A type that fails either control is a single-species/reciprocal boundary in disguise.
"""
from __future__ import annotations
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist

from boundary import params as P
from boundary.params import clip_unit, idx
from boundary.adapters import run_many, seeds_for
from boundary.sensitivity import sensitivity

OBS_KEYS = ["P_plus", "P_minus", "R_lock", "Omega_rel", "Omega_tot",
            "alpha_WC", "Phi_dem", "chi_nn", "D", "sigma_seg"]


def dominant_observable(space: P.Space, u, normal, proto, cache=None, h=0.06, n_side=20):
    """Rank Module A observables by standardized mean difference across +/- normal."""
    n = np.asarray(normal, float); n = n / (np.linalg.norm(n) + 1e-12)
    out = {}
    for sgn, off in ((+1, 41000), (-1, 45000)):
        up = clip_unit(u + sgn * h * n)
        seeds = seeds_for(up, proto.hash(), n_side, offset=off)
        res = run_many(space, [(up, proto.n_steps, proto.boundary, s) for s in seeds],
                       cache=cache, want_obs2=True)
        out[sgn] = {k: np.array([r["obs2"][k] for r in res], float) for k in OBS_KEYS}
    best = None
    for k in OBS_KEYS:
        a, b = out[+1][k], out[-1][k]
        if not (np.isfinite(a).any() and np.isfinite(b).any()):
            continue
        ma, mb = np.nanmean(a), np.nanmean(b)
        sd = np.sqrt(np.nanvar(a) + np.nanvar(b)) + 1e-9
        eff = abs(ma - mb) / sd
        is_onset = (min(abs(ma), abs(mb)) < 0.05) and (max(abs(ma), abs(mb)) > 0.1)
        if best is None or eff > best["effect_size"]:
            best = {"dominant_obs": k, "effect_size": float(eff), "is_onset": bool(is_onset),
                    "val_plus": float(ma), "val_minus": float(mb)}
    return best or {"dominant_obs": None, "effect_size": 0.0, "is_onset": False}


def _obs_jump(space, u, normal, proto, cache, dom_obs, h=0.06, n_side=16, off=48000):
    """|mean dom_obs(+normal) - mean dom_obs(-normal)| at a point (the boundary's jump)."""
    n = np.asarray(normal, float); n = n / (np.linalg.norm(n) + 1e-12)
    vals = {}
    for sgn, o in ((+1, off), (-1, off + 3000)):
        up = clip_unit(np.asarray(u, float) + sgn * h * n)
        seeds = seeds_for(up, proto.hash(), n_side, offset=o)
        res = run_many(space, [(up, proto.n_steps, proto.boundary, s) for s in seeds],
                       cache=cache, want_obs2=True)
        vals[sgn] = np.nanmean([r["obs2"][dom_obs] for r in res])
    return abs(float(vals[+1] - vals[-1]))


def _control(space, u, normal, proto, cache, overrides, b_min, dom_obs=None, orig_jump=None):
    """Hold the coupling params fixed and test whether the boundary survives.

    A boundary survives a control if its dominant-observable jump across the normal
    persists (>= 40% of the unconstrained jump) with the coupling switched off; if the
    jump collapses, the boundary was coupling-driven and is a genuine mixture boundary.
    Falls back to the descriptor-b test when no dominant observable is available."""
    uc = np.asarray(u, float).copy()
    for name, unit_val in overrides.items():
        uc[idx(space, name)] = unit_val
    uc = clip_unit(uc)
    if dom_obs is not None and orig_jump is not None and orig_jump > 1e-6:
        jump_ctrl = _obs_jump(space, uc, normal, proto, cache, dom_obs)
        ratio = jump_ctrl / orig_jump
        return {"obs": dom_obs, "orig_jump": float(orig_jump), "control_jump": float(jump_ctrl),
                "jump_ratio": float(ratio), "survives": bool(ratio >= 0.4)}
    active = [j for j in range(space.d) if space.param_order[j] not in overrides]
    s = sensitivity(space, uc, proto, h=0.04, n_side=8, active=active, cache=cache)
    return {"b": float(s.b), "survives": bool(s.b >= b_min)}


def _phys_to_unit(space, name, phys_value):
    lo, hi = space.bounds[idx(space, name)]
    return float(np.clip((phys_value - lo) / (hi - lo), 0.0, 1.0))


def controls(space: P.Space, u, normal, proto, cache=None, b_min=3.0, dom_obs=None):
    res = {}
    if space.name == "two":
        orig_jump = _obs_jump(space, u, normal, proto, cache, dom_obs) if dom_obs else None
        # reciprocal control: hold J_a = C_a = 0
        rec = {"J_a": _phys_to_unit(space, "J_a", 0.0), "C_a": _phys_to_unit(space, "C_a", 0.0)}
        res["reciprocal"] = _control(space, u, normal, proto, cache, rec, b_min, dom_obs, orig_jump)
        # single-species control: f_B -> lowest available (species B minimized)
        res["single_species"] = _control(space, u, normal, proto, cache,
                                          {"f_B": _phys_to_unit(space, "f_B", 0.0)}, b_min, dom_obs, orig_jump)
    return res


def cluster_types(normals, min_per_type=5, dist_threshold=0.4):
    """Agglomerative clustering with cosine distance on |normal loadings|."""
    L = np.abs(np.asarray(normals))
    L = L / (np.linalg.norm(L, axis=1, keepdims=True) + 1e-12)
    if len(L) < 2:
        return np.zeros(len(L), dtype=int)
    Z = linkage(pdist(L, metric="cosine"), method="average")
    labels = fcluster(Z, t=dist_threshold, criterion="distance")
    # drop tiny clusters to a "-1" (unassigned)
    out = labels.copy()
    for c in np.unique(labels):
        if (labels == c).sum() < min_per_type:
            out[labels == c] = -1
    return out


def bootstrap_normal(normals, nb=400, seed=3):
    rng = np.random.default_rng(seed)
    L = np.asarray(normals)
    # sign-align each to the mean direction
    ref = L.mean(0); ref = ref / (np.linalg.norm(ref) + 1e-12)
    L = np.array([v if np.dot(v, ref) >= 0 else -v for v in L])
    boot = np.array([L[rng.integers(0, len(L), len(L))].mean(0) for _ in range(nb)])
    mean = L.mean(0)
    lo = np.percentile(boot, 5, 0); hi = np.percentile(boot, 95, 0)
    return mean, lo, hi
