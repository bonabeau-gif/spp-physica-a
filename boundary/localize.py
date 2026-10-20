"""Module E: 1-D localization along a segment.

Locate a boundary along u(t) = u0 + t (u1 - u0) by the descriptor rate
r(t) = ||psi_tilde(t+delta) - psi_tilde(t-delta)|| / (2 delta ||sigma_psi||),
refine the maximum by golden-section search, then run the full multi-scale oracle at
the refined t*. The rate (not the floor-normalized b) locates the boundary; b is used
only for acceptance, exactly as the plan prescribes.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

from boundary import params as P
from boundary.params import clip_unit
from boundary.adapters import run_many, standardize, seeds_for
from boundary.sensitivity import multiscale_accept

_PHI = (np.sqrt(5.0) - 1.0) / 2.0


def _psi_mean(space, u01, proto, n_side, cache, offset):
    seeds = seeds_for(u01, proto.hash(), n_side, offset)
    pts = [(u01, proto.n_steps, proto.boundary, s) for s in seeds]
    res = run_many(space, pts, cache=cache, want_obs2=False)
    return np.mean([standardize(r["psi"], space) for r in res], axis=0)


def rate_at(space, u0, u1, t, proto, delta=0.06, n_side=6, cache=None):
    """Descriptor rate along the segment at parameter t."""
    seg = np.asarray(u1, float) - np.asarray(u0, float)
    ua = clip_unit(np.asarray(u0, float) + (t - delta) * seg)
    ub = clip_unit(np.asarray(u0, float) + (t + delta) * seg)
    pa = _psi_mean(space, ua, proto, n_side, cache, offset=3000)
    pb = _psi_mean(space, ub, proto, n_side, cache, offset=7000)
    step = float(np.linalg.norm(ub - ua))
    return float(np.linalg.norm(pb - pa) / max(step, 1e-9))


@dataclass
class Localized:
    t_star: float
    u_star: np.ndarray
    rate_profile: list = field(default_factory=list)
    accept: dict = None


def localize_segment(space: P.Space, u0, u1, proto, n_coarse=11, delta=0.06,
                     n_side_coarse=6, n_side_refine=8, refine_iters=6,
                     b_min=3.0, cache=None, run_oracle=True) -> Localized:
    u0 = np.asarray(u0, float); u1 = np.asarray(u1, float)
    ts = np.linspace(0.0, 1.0, n_coarse)
    prof = [(float(t), rate_at(space, u0, u1, t, proto, delta, n_side_coarse, cache)) for t in ts]
    k = int(np.argmax([r for _, r in prof]))
    # golden-section on [a,c] bracketing the coarse max
    a = ts[max(0, k - 1)]; c = ts[min(len(ts) - 1, k + 1)]
    x1 = c - _PHI * (c - a); x2 = a + _PHI * (c - a)
    f1 = rate_at(space, u0, u1, x1, proto, delta, n_side_refine, cache)
    f2 = rate_at(space, u0, u1, x2, proto, delta, n_side_refine, cache)
    for _ in range(refine_iters):
        if f1 < f2:
            a = x1; x1 = x2; f1 = f2; x2 = a + _PHI * (c - a)
            f2 = rate_at(space, u0, u1, x2, proto, delta, n_side_refine, cache)
        else:
            c = x2; x2 = x1; f2 = f1; x1 = c - _PHI * (c - a)
            f1 = rate_at(space, u0, u1, x1, proto, delta, n_side_refine, cache)
    t_star = float((a + c) / 2.0)
    u_star = clip_unit(u0 + t_star * (u1 - u0))
    acc = multiscale_accept(space, u_star, proto, b_min=b_min, n_side=n_side_refine,
                            cache=cache) if run_oracle else None
    return Localized(t_star=t_star, u_star=u_star, rate_profile=prof, accept=acc)
