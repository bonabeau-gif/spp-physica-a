"""Module F: boundary-manifold continuation (predictor-corrector).

Trace a (d-1)-dimensional boundary through the 14-D box. Two implicit definitions:
  * ridge: g(u)=0 at the local maximum of the descriptor rate along the normal
    (no order parameter needed); corrector is a 1-D maximization.
  * level: g(u)=O_dom(u)-O_c; corrector is a 1-D root find. Better conditioned when the
    dominant observable has a structural zero (onset boundaries), per the plan.

Local manifold dimension is estimated by PCA over accumulated points within a 3*ds ball
(number of components for 95% variance) — the quantitative "how thin is the boundary".
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

from boundary import params as P
from boundary.params import clip_unit
from boundary.adapters import run_many, seeds_for
from boundary.sensitivity import sensitivity, directional_sensitivity

_PHI = (np.sqrt(5.0) - 1.0) / 2.0


def project_face(u, t, eps=1e-6):
    """Project a tangent onto the feasible box face: zero any component that would push
    an already-saturated coordinate further out. Returns None if fully cornered."""
    t = np.asarray(t, float).copy()
    t[(u <= eps) & (t < 0)] = 0.0
    t[(u >= 1 - eps) & (t > 0)] = 0.0
    nrm = np.linalg.norm(t)
    return (t / nrm) if nrm > 1e-9 else None


def tangent_basis(normal, active=None):
    """Orthonormal complement of `normal`. If `active` (a list of coordinate indices)
    is given, the tangent space is restricted to that subspace, freezing all other
    parameters so the trace cannot drift onto a boundary in a frozen direction."""
    n = np.asarray(normal, float); n = n / (np.linalg.norm(n) + 1e-12)
    d = len(n)
    if active is None:
        Q, _ = np.linalg.qr(np.column_stack([n, np.eye(d)]))
        T = Q[:, 1:d]
    else:
        sub = np.zeros((d, len(active)))
        for k, j in enumerate(active):
            sub[j, k] = 1.0
        n_sub = n[active]
        Qs, _ = np.linalg.qr(np.column_stack([n_sub, np.eye(len(active))]))
        T = sub @ Qs[:, 1:len(active)]          # (d, |active|-1), lives in the active subspace
    for j in range(T.shape[1]):
        if T[np.argmax(np.abs(T[:, j])), j] < 0:
            T[:, j] *= -1
    return T


def _rate_along(space, u, n, proto, ds, cache, sigma):
    r = directional_sensitivity(space, u, proto, n, h=max(0.02, 0.5 * ds), n_side=8,
                                cache=cache, sigma=sigma)
    return r["rate"]


def corrector_ridge(space, u_pred, n, proto, ds, cache, sigma):
    """Maximize the directional rate along n over s in [-3ds, 3ds] (ridge crest)."""
    lo, hi = -3.0 * ds, 3.0 * ds
    x1 = hi - _PHI * (hi - lo); x2 = lo + _PHI * (hi - lo)
    f1 = _rate_along(space, clip_unit(u_pred + x1 * n), n, proto, ds, cache, sigma)
    f2 = _rate_along(space, clip_unit(u_pred + x2 * n), n, proto, ds, cache, sigma)
    for _ in range(4):
        if f1 < f2:
            lo = x1; x1 = x2; f1 = f2; x2 = lo + _PHI * (hi - lo)
            f2 = _rate_along(space, clip_unit(u_pred + x2 * n), n, proto, ds, cache, sigma)
        else:
            hi = x2; x2 = x1; f2 = f1; x1 = hi - _PHI * (hi - lo)
            f1 = _rate_along(space, clip_unit(u_pred + x1 * n), n, proto, ds, cache, sigma)
    s = 0.5 * (lo + hi)
    return clip_unit(u_pred + s * n), abs(s) < 3.0 * ds - 1e-6


@dataclass
class Trace:
    points: list = field(default_factory=list)     # list of u (d,)
    normals: list = field(default_factory=list)
    bvals: list = field(default_factory=list)
    manifold_dim: list = field(default_factory=list)
    terminated_by: str = ""
    total_sims: int = 0
    closed_loop: bool = False


def local_manifold_dim(points, ds, var=0.95):
    pts = np.asarray(points)
    if len(pts) < 3:
        return len(pts)
    c = pts[-1]
    ball = pts[np.linalg.norm(pts - c, axis=1) <= 3.0 * ds]
    if len(ball) < 3:
        return 1
    X = ball - ball.mean(0)
    sv = np.linalg.svd(X, compute_uv=False)
    ev = sv ** 2
    if ev.sum() <= 0:
        return 1
    frac = np.cumsum(ev) / ev.sum()
    return int(np.searchsorted(frac, var) + 1)


def _turning_curvature(points, moved):
    """Discrete curvature = turning angle between the last two polyline segments / step,
    kappa ~ ||dtangent||/ds. Used by the curvature-adaptive predictor (P2-E3)."""
    if len(points) < 3 or moved < 1e-9:
        return 0.0
    a = np.asarray(points[-1]) - np.asarray(points[-2])
    b = np.asarray(points[-2]) - np.asarray(points[-3])
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    cos = float(np.clip((a @ b) / (na * nb), -1.0, 1.0))
    return float(np.arccos(cos) / moved)


def continue_boundary(space: P.Space, u0, normal0, proto, b_min=3.0, ds0=0.03,
                      ds_min=0.005, ds_max=0.10, max_steps=40, cache=None,
                      refresh_every=5, n_side=8, active=None,
                      curvature_adaptive=False, eps_pred=0.004, refresh_angle_deg=5.0):
    u = clip_unit(np.asarray(u0, float))
    n = np.asarray(normal0, float)
    if active is not None:                     # restrict the normal to the active subspace
        mask = np.zeros(space.d); mask[active] = 1.0; n = n * mask
    n = n / (np.linalg.norm(n) + 1e-12)
    tr = Trace()
    ds = ds0
    prev_t = None
    below = 0
    turn_since_refresh = 0.0                    # accumulated turning angle (rad) since last refresh
    for step in range(max_steps):
        # floor (sigma) for cheap directional evaluations at this point
        seeds = seeds_for(u, proto.hash(), n_side)
        res = run_many(space, [(u, proto.n_steps, proto.boundary, s) for s in seeds], cache=cache)
        from boundary.adapters import standardize
        sigma = np.std([standardize(r["psi"], space) for r in res], axis=0)
        tr.total_sims += len(seeds)

        # choose tangent direction (projected onto the feasible box face)
        T = tangent_basis(n, active=active)
        if prev_t is None:
            # first step: try a few tangent axes (both signs), keep the best-converging
            best = None
            for j in range(min(T.shape[1], 3)):
                t = project_face(u, T[:, j])
                if t is None:
                    continue
                up = clip_unit(u + ds * t)
                unew, conv = corrector_ridge(space, up, n, proto, ds, cache, sigma)
                tr.total_sims += 6 * 2 * n_side
                rr = _rate_along(space, unew, n, proto, ds, cache, sigma)
                if best is None or rr > best[0]:
                    best = (rr, t, unew, conv)
            if best is None:
                tr.terminated_by = "cornered"; break
            _, t_dir, u_new, converged = best
        else:
            t_dir = T @ (T.T @ prev_t)                 # parallel transport
            t_dir = project_face(u, t_dir)
            if t_dir is None:
                tr.terminated_by = "cornered"; break
            u_pred = clip_unit(u + ds * t_dir)
            u_new, converged = corrector_ridge(space, u_pred, n, proto, ds, cache, sigma)
            tr.total_sims += 8 * 8

        # Cheap acceptance via the directional variant; refresh the full Jacobian only
        # every refresh_every accepted steps (plan Section 9), which is where the cost is.
        db = directional_sensitivity(space, u_new, proto, n, h=0.04, n_side=n_side,
                                     cache=cache, sigma=sigma)
        tr.total_sims += db["sims"]
        b_here = db["b"]; normal_here = n
        # refresh the full Jacobian every refresh_every steps, or (curvature-adaptive) once
        # the trace has turned more than refresh_angle_deg since the last refresh.
        do_refresh = (len(tr.points) % refresh_every == 0) or (
            curvature_adaptive and np.degrees(turn_since_refresh) >= refresh_angle_deg)
        if do_refresh:
            s = sensitivity(space, u_new, proto, h=0.04, n_side=n_side, active=active, cache=cache)
            tr.total_sims += s.sims
            b_here = s.b
            nh = s.normal
            if active is not None:
                mask = np.zeros(space.d); mask[active] = 1.0; nh = nh * mask
            normal_here = nh / (np.linalg.norm(nh) + 1e-12)
            turn_since_refresh = 0.0

        # directional b runs lower than the full multiscale b, so gate per-step at a
        # looser proxy and enforce the real b_min only at full-Jacobian refreshes.
        weak = (b_here < b_min) if do_refresh else (b_here < 0.6 * b_min)
        if (not converged) or weak:
            below += 1
            ds *= 0.5
            if ds < ds_min:
                tr.terminated_by = "ds_min"; break
            if below >= 3:
                tr.terminated_by = "dissolved"; break
            continue
        below = 0

        moved = float(np.linalg.norm(u_new - u))
        tr.points.append(u_new); tr.normals.append(normal_here.copy()); tr.bvals.append(b_here)
        tr.manifold_dim.append(local_manifold_dim(tr.points, ds))
        prev_t = t_dir
        n = normal_here
        u = u_new
        if curvature_adaptive:
            # ds_{k+1} = min(ds_max, sqrt(eps_pred / (kappa + eps))): small steps where the
            # surface bends, large steps where it is flat -- fixes the O(ds^2/r) curved drift.
            kappa = _turning_curvature(tr.points, moved)
            turn_since_refresh += kappa * moved
            ds = float(np.clip(np.sqrt(eps_pred / (kappa + 1e-6)), ds_min, ds_max))
        else:
            ds = min(1.5 * ds, ds_max)

        if moved < 0.3 * ds_min:
            tr.terminated_by = "left_box"; break
        # closed-loop detection
        if len(tr.points) > 5:
            d0 = np.linalg.norm(u - tr.points[0])
            if d0 < ds_min:
                tr.closed_loop = True; tr.terminated_by = "closed_loop"; break
    if not tr.terminated_by:
        tr.terminated_by = "max_steps"
    return tr
