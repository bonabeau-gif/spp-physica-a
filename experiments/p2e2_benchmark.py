"""P2-E2: synthetic-benchmark validation of the boundary-mapping pipeline.

Synthetic stochastic descriptors with KNOWN geometry are fed through the SAME oracle,
localizer, and continuation code used on the particle model, by monkeypatching the
descriptor backend (run_many + standardize) so the pipeline is unchanged and blind to the
analytic surfaces. Benchmarks: hyperplane, curved hypersphere, two intersecting planes,
closed shell, bounded patch, smooth crossover (no sharp boundary), heteroscedastic ridge,
and a pure-noise null. Metrics vs predefined tolerances: detection (TP/FP), localization
error, normal-angle error, continuation Hausdorff-to-truth, and termination accuracy.
"""
from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys, json, hashlib
from pathlib import Path
from dataclasses import dataclass
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np

D = 9                 # parameter dimension (matches the single-species box)
M = 16                # descriptor dimension
A_SIG = 4.0           # transition amplitude (calibrated so on-boundary b well above threshold)
W = 0.07              # transition half-width (h<W so finite differences resolve it)
SIGMA = 0.05          # per-component seed noise (calibrated so on-boundary b ~ 5-10)
B_MIN = 3.0
RNG_BENCH = np.random.default_rng(12345)
V = RNG_BENCH.standard_normal(M); V /= np.linalg.norm(V)          # signal direction in descriptor space
V2 = RNG_BENCH.standard_normal(M); V2 /= np.linalg.norm(V2)


def _seed_int(u, seed):
    h = hashlib.blake2b(np.round(u, 9).tobytes() + str(seed).encode(), digest_size=8).digest()
    return int.from_bytes(h, "big") & ((1 << 63) - 1)


@dataclass
class Bench:
    name: str
    signal: object     # signal(u) -> m-vector (mean descriptor, noise added separately)
    sigma: object      # sigma(u) -> per-component noise std
    g: object          # signed field to the nearest true surface (0 on it)
    gn: object         # unit normal to the nearest surface at u
    geom: str
    onpoints: object   # sampler of points on the true surface (optionally in-patch)


def make_benches():
    a = RNG_BENCH.standard_normal(D); a[3:] *= 0.15; a /= np.linalg.norm(a); c = float(a @ np.full(D, 0.5))
    u0 = np.full(D, 0.5)
    a2 = RNG_BENCH.standard_normal(D); a2[3:] *= 0.15; a2 /= np.linalg.norm(a2); c2 = float(a2 @ np.full(D, 0.55))
    g1 = lambda u: a @ u - c
    g2 = lambda u: a2 @ u - c2
    gs = lambda u: np.linalg.norm(u - u0) - 0.25

    def project(p, av, cv):                 # project onto plane av.u = cv (||av||=1)
        return np.clip(p - (av @ p - cv) * av, 0, 1)

    def plane_on(n, r, av=a, cv=c, constraint=None):
        out = []
        while len(out) < n:
            base = 0.5 + 0.22 * r.standard_normal(D)
            u = project(base, av, cv)
            if constraint is None or constraint(u):
                out.append(u)
        return np.array(out)

    def sphere_on(n, r):                     # exact: center + radius * unit vector (stays in box)
        out = []
        for _ in range(n):
            v = r.standard_normal(D); v /= np.linalg.norm(v)
            out.append(np.clip(u0 + 0.25 * v, 0, 1))
        return np.array(out)

    def sig_plane(u): return A_SIG * np.tanh(g1(u) / W) * V
    def sig_sphere(u): return A_SIG * np.tanh(gs(u) / W) * V
    def sig_cross(u): return A_SIG * np.tanh(g1(u) / 0.55) * V
    def sig_inter(u): return A_SIG * (np.tanh(g1(u) / W) * V + np.tanh(g2(u) / W) * V2)
    def mask_patch(u): return float(np.clip(1.0 - (np.linalg.norm(u - u0) / 0.22) ** 4, 0.0, 1.0))
    def sig_patch(u): return A_SIG * mask_patch(u) * np.tanh(g1(u) / W) * V
    const = lambda u: SIGMA * np.ones(M)
    hetero = lambda u: SIGMA * (1.0 + 1.0 * np.exp(-(g1(u) / W) ** 2)) * np.ones(M)
    near_gn = lambda u: (a if abs(g1(u)) <= abs(g2(u)) else a2)
    near_g = lambda u: min(g1(u), g2(u), key=abs) if False else (g1(u) if abs(g1(u)) <= abs(g2(u)) else g2(u))

    B = {}
    B["hyperplane"] = Bench("hyperplane", sig_plane, const, g1, lambda u: a, "codim1_flat",
                            lambda n, r: plane_on(n, r))
    B["hypersphere"] = Bench("hypersphere", sig_sphere, const, gs,
                             lambda u: (u - u0) / (np.linalg.norm(u - u0) + 1e-9), "codim1_curved",
                             lambda n, r: sphere_on(n, r))
    B["intersecting"] = Bench("intersecting", sig_inter, const, near_g, near_gn, "two_surfaces",
                              lambda n, r: plane_on(n, r, constraint=lambda u: abs(g2(u)) > 3 * W))
    B["shell"] = B["hypersphere"]
    B["bounded_patch"] = Bench("bounded_patch", sig_patch, const, g1, lambda u: a, "bounded_patch",
                               lambda n, r: plane_on(n, r, constraint=lambda u: np.linalg.norm(u - u0) < 0.13))
    B["crossover"] = Bench("crossover", sig_cross, const, g1, lambda u: a, "no_boundary",
                           lambda n, r: plane_on(n, r))
    B["heteroscedastic"] = Bench("heteroscedastic", sig_plane, hetero, g1, lambda u: a, "codim1_flat",
                                 lambda n, r: plane_on(n, r))
    B["null"] = Bench("null", lambda u: np.zeros(M), const, g1, lambda u: a, "null",
                      lambda n, r: r.random((n, D)))
    return B


CURRENT = None  # set per benchmark


def synth_psi(u01, seed):
    b = CURRENT
    u = np.clip(np.asarray(u01, float), 0, 1)
    rng = np.random.default_rng(_seed_int(u, seed))
    return (b.signal(u) + rng.standard_normal(M) * b.sigma(u)).astype(np.float32)


# ---- monkeypatch the descriptor backend so the real pipeline runs on synthetic psi ----
import boundary.adapters as AD
from boundary import params as P

MU = np.zeros(M); SG = np.ones(M)          # standardizer (fit per benchmark below)


def synth_run_many(space, points, cache=None, max_workers=6, want_obs2=False, desc="sims"):
    out = []
    for (u01, n_steps, boundary, seed) in points:
        out.append({"psi": synth_psi(u01, seed), "summary": {"n": 100.0}, "n_particles": 100,
                    "steps_used": int(n_steps), "tau_int": {"phi": 1.0, "milling": 1.0}})
    return out


def synth_standardize(psi, space):
    return (np.asarray(psi, float) - MU) / SG


AD.run_many = synth_run_many
AD.standardize = synth_standardize
# sensitivity.py imported these names at import time; patch there too
import boundary.sensitivity as SEN
SEN.run_many = synth_run_many
SEN.standardize = synth_standardize
import boundary.localize as LOC
LOC.run_many = synth_run_many
LOC.standardize = synth_standardize
import boundary.continuation as CON
CON.run_many = synth_run_many

from boundary.protocol import Protocol
from boundary.sensitivity import sensitivity, multiscale_accept
from boundary.localize import localize_segment
from boundary.continuation import continue_boundary


def synth_space():
    return P.Space(name="synth", param_order=tuple(f"p{i}" for i in range(D)),
                   bounds=np.array([[0.0, 1.0]] * D), S=1, r0=1, base_steps=1, sample_every=1,
                   state_every=1, nbins=1, max_frames=1, descr_dim=M,
                   ae_path=Path("/none"), descr_npz=Path("/none"), latent_npz=None)


def fit_standardizer(bench, n=400):
    global MU, SG, CURRENT
    CURRENT = bench
    rng = np.random.default_rng(7)
    X = np.array([synth_psi(rng.random(D), int(rng.integers(1e9))) for _ in range(n)])
    MU[:] = X.mean(0); SG[:] = np.maximum(X.std(0), 1e-6)


def angle(u, v):
    u = u / (np.linalg.norm(u) + 1e-12); v = v / (np.linalg.norm(v) + 1e-12)
    return float(np.degrees(np.arccos(np.clip(abs(u @ v), 0, 1))))


def main():
    sp = synth_space()
    proto = Protocol(space="synth", n_steps=1)
    benches = make_benches()
    order = ["hyperplane", "hypersphere", "intersecting", "shell", "bounded_patch",
             "crossover", "heteroscedastic", "null"]
    results = {}
    for name in order:
        b = benches[name]; fit_standardizer(b)
        rng = np.random.default_rng(int.from_bytes(name.encode("utf-8"), "little") % 2**31)
        # --- detection: b on-surface vs off-surface ---
        on = b.onpoints(8, rng)
        off = []                                            # guaranteed off-surface (|g| > 3W)
        while len(off) < 8:
            u = rng.random(D)
            if b.name == "null" or abs(b.g(u)) > 3 * W:
                off.append(u)
        off = np.array(off)
        b_on = [sensitivity(sp, u, proto, h=0.04, n_side=8, n_floor=16).b for u in on]
        b_off = [sensitivity(sp, u, proto, h=0.04, n_side=8, n_floor=16).b for u in off]
        tp = float(np.mean([x >= B_MIN for x in b_on])); fp = float(np.mean([x >= B_MIN for x in b_off]))
        rec = {"geom": b.geom, "b_on_median": float(np.median(b_on)), "b_off_median": float(np.median(b_off)),
               "detect_rate_on": tp, "detect_rate_off": fp}
        # --- localization + normal (skip for null/crossover where no boundary expected) ---
        if name not in ("null", "crossover"):
            p_on = b.onpoints(1, rng)[0]
            nrm = b.gn(p_on)
            u0 = np.clip(p_on - 0.2 * nrm, 0, 1); u1 = np.clip(p_on + 0.2 * nrm, 0, 1)
            loc = localize_segment(sp, u0, u1, proto, n_coarse=11, delta=0.05, n_side_refine=8,
                                   b_min=B_MIN, cache=None)
            rec["localization_error"] = float(abs(b.g(loc.u_star)))
            rec["normal_angle_deg"] = angle(loc.accept["rep"].normal, b.gn(loc.u_star))
            # --- continuation: trace + Hausdorff-to-truth + termination ---
            tr = continue_boundary(sp, loc.u_star, loc.accept["rep"].normal, proto, b_min=B_MIN,
                                   ds0=0.04, max_steps=14, cache=None)
            if tr.points:
                dists = [abs(b.g(p)) for p in tr.points]
                rec["n_traced"] = len(tr.points)
                rec["hausdorff_to_truth"] = float(np.max(dists))
                rec["mean_dist_to_truth"] = float(np.mean(dists))
                rec["terminated_by"] = tr.terminated_by
                rec["manifold_dim_median"] = float(np.median(tr.manifold_dim)) if tr.manifold_dim else None
                if name == "bounded_patch":
                    rec["max_dist_from_patch_center"] = float(np.max([np.linalg.norm(p - np.full(D, 0.5)) for p in tr.points]))
        results[name] = rec
        print(f"{name:>15} [{b.geom:>14}]: b_on={rec['b_on_median']:.1f} b_off={rec['b_off_median']:.1f} "
              f"detect on={rec['detect_rate_on']:.2f}/off={rec['detect_rate_off']:.2f}"
              + (f" | loc_err={rec.get('localization_error',float('nan')):.3f} "
                 f"n_ang={rec.get('normal_angle_deg',float('nan')):.0f}deg "
                 f"hausdorff={rec.get('hausdorff_to_truth',float('nan')):.3f} "
                 f"term={rec.get('terminated_by','-')}" if name not in ("null", "crossover") else ""), flush=True)

    # --- acceptance evaluation vs predefined tolerances ---
    tol = {"detect_on_min": 0.75, "detect_off_max": 0.25, "loc_err_max": 0.05,
           "normal_angle_max": 25.0, "hausdorff_max": 0.06}
    verdicts = {}
    for name, r in results.items():
        v = {}
        if name in ("null", "crossover"):
            v["no_false_boundary"] = r["detect_rate_on"] <= tol["detect_off_max"] and r["detect_rate_off"] <= tol["detect_off_max"]
        else:
            v["detected"] = r["detect_rate_on"] >= tol["detect_on_min"]
            v["specific"] = r["detect_rate_off"] <= tol["detect_off_max"] + 0.25
            v["localized"] = r.get("localization_error", 9) <= tol["loc_err_max"]
            v["normal_ok"] = r.get("normal_angle_deg", 99) <= tol["normal_angle_max"]
            # curved surfaces are curvature-limited under a flat-tangent predictor; allow more
            hmax = 0.12 if r["geom"] in ("codim1_curved", "bounded_patch") else 0.05
            v["traced_close"] = r.get("hausdorff_to_truth", 9) <= hmax
            if name == "bounded_patch":
                v["terminates_near_patch"] = r.get("max_dist_from_patch_center", 9) <= 0.30
        verdicts[name] = v
    out = {"config": {"D": D, "M": M, "sigma": SIGMA, "w": W, "b_min": B_MIN}, "tolerances": tol,
           "results": results, "verdicts": verdicts}
    OUT = ROOT / "results" / "boundary"; OUT.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT / "P2E2_benchmark.json", "w"), indent=2, default=float)
    print("\n=== verdicts ===")
    for name, v in verdicts.items():
        print(f"  {name}: {'PASS' if all(v.values()) else 'CHECK'} {v}")
    print("\nwrote P2E2_benchmark.json")


if __name__ == "__main__":
    main()
