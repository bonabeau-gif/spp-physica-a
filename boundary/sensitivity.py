"""Module C: the sensitivity oracle (core primitive).

The boundary score b(u) = s1(J) / thr, where J is the Jacobian of the standardized
raw trajectory descriptor with respect to the (active) normalized parameters, and thr
is a Monte-Carlo estimate of the top singular value produced by seed noise alone. The
descriptor is standardized by the frozen autoencoder statistics (never per call), so
b is comparable across points. Multi-scale acceptance (plan Section 6) guards against
the known step-size dependence of resolved-direction counts.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

from boundary import params as P
from boundary.params import clip_unit
from boundary.adapters import run_many, standardize, seeds_for, load_standardizer


def mc_threshold(se: np.ndarray, q: float = 95.0, k: int = 2000, seed: int = 11) -> float:
    """95th percentile (plan default) of s1(E), E[a,j] ~ N(0, se[a,j]^2).

    se is strongly heteroscedastic in rows (depends only on the output), making it
    effectively rank-1, so the closed form (sqrt(m)+sqrt(d))*s is not used; see
    sloppiness_v4.mc_threshold, which this mirrors.
    """
    rng = np.random.default_rng(seed)
    m, n = se.shape
    vals = np.empty(k)
    for i in range(k):
        E = rng.standard_normal((m, n)) * se
        vals[i] = np.linalg.svd(E, compute_uv=False)[0]
    return float(np.percentile(vals, q))


@dataclass
class Sensitivity:
    s1: float
    thr: float
    b: float
    normal: np.ndarray          # v1 in full normalized parameter space (d,)
    n_resolved: int
    h: float
    sims: int
    sigma: np.ndarray = field(default=None, repr=False)   # per-output seed std of psi_tilde


def _eval_point(space, u01, proto, n, cache, offset=0):
    ph = proto.hash()
    seeds = seeds_for(u01, ph, n, offset)
    pts = [(u01, proto.n_steps, proto.boundary, s) for s in seeds]
    res = run_many(space, pts, cache=cache, want_obs2=False, desc="pt")
    psi = np.array([standardize(r["psi"], space) for r in res])   # (n, m)
    return psi, len(pts)


def sensitivity(space: P.Space, u01, proto, h=0.04, n_side=8, n_floor=16,
                active=None, cache=None, mc_k=2000, q=95.0) -> Sensitivity:
    u01 = clip_unit(np.asarray(u01, float))
    d = space.d
    active = list(range(d)) if active is None else list(active)
    ph = proto.hash()

    # Build ALL points for this oracle call and dispatch in ONE batch (floor + every
    # +/- perturbation), so the whole call runs at full parallelism instead of ~29
    # serial mini-batches.
    specs = [("floor", None, u01, n_floor, 0)]
    for j in active:
        e = np.zeros(d); e[j] = h
        specs.append(("p", j, clip_unit(u01 + e), n_side, 1000 + j))
        specs.append(("m", j, clip_unit(u01 - e), n_side, 5000 + j))
    tasks, spans = [], []
    for _tag, _j, pt, n, off in specs:
        seeds = seeds_for(pt, ph, n, off)
        start = len(tasks)
        tasks += [(pt, proto.n_steps, proto.boundary, s) for s in seeds]
        spans.append((start, len(tasks)))
    res = run_many(space, tasks, cache=cache, want_obs2=False, desc="oracle")
    sims = len(tasks)
    means = {}
    sigma = None
    for (tag, j, pt, n, off), (a, b) in zip(specs, spans):
        psi = np.array([standardize(res[k]["psi"], space) for k in range(a, b)])
        if tag == "floor":
            sigma = psi.std(axis=0); m = psi.shape[1]
        else:
            means[(tag, j)] = psi.mean(0)

    J = np.zeros((m, len(active)))
    for col, j in enumerate(active):
        e = np.zeros(d); e[j] = h
        step = float(np.linalg.norm(clip_unit(u01 + e) - clip_unit(u01 - e)))
        if step < 1e-9:
            continue
        J[:, col] = (means[("p", j)] - means[("m", j)]) / step

    se = np.sqrt(2.0) * sigma[:, None] / (np.sqrt(n_side) * 2.0 * h)     # (m, |active|)
    thr = mc_threshold(se, q=q, k=mc_k)
    U, sv, Vt = np.linalg.svd(J, full_matrices=False)
    s1 = float(sv[0]) if sv.size else 0.0
    normal = np.zeros(d)
    if sv.size:
        normal[active] = Vt[0]
    n_resolved = int((sv > thr).sum())
    b = s1 / thr if thr > 0 else 0.0
    return Sensitivity(s1=s1, thr=thr, b=b, normal=normal, n_resolved=n_resolved,
                       h=h, sims=sims, sigma=sigma)


def directional_sensitivity(space: P.Space, u01, proto, direction, h=0.04, n_side=8,
                            cache=None, sigma=None) -> dict:
    """Cheap corrector primitive. Returns rate and directional b along a unit vector.

    Cost 2*n_side sims (plus n_floor once if sigma not supplied)."""
    u01 = clip_unit(np.asarray(u01, float))
    dir_ = np.asarray(direction, float)
    dir_ = dir_ / (np.linalg.norm(dir_) + 1e-12)
    sims = 0
    if sigma is None:
        psi0, s0 = _eval_point(space, u01, proto, max(8, n_side), cache, offset=0)
        sigma = psi0.std(axis=0); sims += s0
    up = clip_unit(u01 + h * dir_); dn = clip_unit(u01 - h * dir_)
    step = float(np.linalg.norm(up - dn))
    psi_p, sp = _eval_point(space, up, proto, n_side, cache, offset=2000); sims += sp
    psi_m, sm = _eval_point(space, dn, proto, n_side, cache, offset=6000); sims += sm
    rate = float(np.linalg.norm(psi_p.mean(0) - psi_m.mean(0)) / max(step, 1e-9))
    se = np.sqrt(2.0) * sigma / (np.sqrt(n_side) * 2.0 * h)
    floor = float(np.linalg.norm(se))                     # expected |noise vector| scale
    return {"rate": rate, "b": rate / floor if floor > 0 else 0.0, "sims": sims, "sigma": sigma}


def multiscale_accept(space: P.Space, u01, proto, b_min=3.0, scales=(0.02, 0.04, 0.08),
                      n_side=8, n_floor=16, active=None, cache=None, cos_min=0.8) -> dict:
    """Accept a boundary point iff b >= b_min at >= 2 scales whose normals agree."""
    per = [sensitivity(space, u01, proto, h=h, n_side=n_side, n_floor=n_floor,
                       active=active, cache=cache) for h in scales]
    passing = [r for r in per if r.b >= b_min]
    ok = False
    best_cos = 0.0
    if len(passing) >= 2:
        for a in range(len(passing)):
            for bb in range(a + 1, len(passing)):
                c = abs(float(np.dot(passing[a].normal, passing[bb].normal)) /
                        (np.linalg.norm(passing[a].normal) * np.linalg.norm(passing[bb].normal) + 1e-12))
                best_cos = max(best_cos, c)
        ok = best_cos >= cos_min
    rep = max(per, key=lambda r: r.b)
    total_sims = sum(r.sims for r in per)
    return {"accepted": bool(ok), "best_cos": best_cos, "rep": rep, "per_scale": per,
            "b_max": max(r.b for r in per), "sims": total_sims}
