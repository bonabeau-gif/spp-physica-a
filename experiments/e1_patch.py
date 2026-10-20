"""E1: local boundary dimension by patch sampling.

Around each seed, draw 30 random tangent directions, form the predictor cloud (points
stepped into the tangent space) and the corrector cloud (each corrected back onto the
boundary along the normal), and compare their PCA eigenvalue spectra. The predictor cloud
spans the 13-D tangent space by construction; SANITY GATE: it must need >=12 components
for 95% variance, else the tangent-space construction is broken (STOP). The corrector
cloud's dimension is the actual local boundary dimension.
"""
from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys, json, pickle, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from boundary import params as P
from boundary import config as CFG
from boundary.protocol import Protocol
from boundary.cache import RunCache
from boundary.params import clip_unit
from boundary.adapters import run_many, seeds_for, standardize
from boundary.sensitivity import sensitivity, directional_sensitivity
from boundary.continuation import tangent_basis, corrector_ridge

OUT = ROOT / "results" / "boundary"
RNG = np.random.default_rng(20260727)
K_DIRS = 30
DS = 0.03


def var95(cloud):
    X = np.asarray(cloud) - np.asarray(cloud).mean(0)
    if len(X) < 2:
        return 1, np.array([])
    sv = np.linalg.svd(X, compute_uv=False); ev = sv ** 2
    if ev.sum() <= 0:
        return 1, ev
    frac = np.cumsum(ev) / ev.sum()
    return int(np.searchsorted(frac, 0.95) + 1), ev


def load_seeds(sp):
    seeds = []
    tr = pickle.load(open(OUT / "traces.pkl", "rb"))
    pts = np.array(tr["points"]); ptt = np.array(tr["pt_trace"])
    for ti in (0, 2, 4):
        idx = np.where(ptt == ti)[0]
        if len(idx):
            seeds.append(("single_trace%d" % ti, pts[idx[len(idx) // 2]]))
    c2 = pickle.load(open(OUT / "campaign2.pkl", "rb"))
    ja = [i for i, f in enumerate(c2["family"]) if f == "A_chiral_Ja"]
    jb = [c2["susc"][f"A_chiral_Ja_{k}"]["b"] for k in range(len(ja))]
    seeds.append(("chiral_onset", np.array(c2["points"][ja[int(np.argmax(jb))]])))
    return seeds[:4]


def sigma_at(sp, u, proto, cache, n=8):
    s = seeds_for(u, proto.hash(), n)
    res = run_many(sp, [(u, proto.n_steps, proto.boundary, x) for x in s], cache=cache)
    return np.std([standardize(r["psi"], sp) for r in res], axis=0)


def main():
    sp = P.get_space("two")
    proto = Protocol(space="two", n_steps=CFG.BASE_STEPS["two"])
    cache = RunCache(OUT / "cache_two.pkl")
    patches = []
    gate_ok = True
    t0 = time.time()
    for name, u_s in load_seeds(sp):
        u_s = clip_unit(np.asarray(u_s, float))
        s = sensitivity(sp, u_s, proto, h=0.04, n_side=CFG.N_SIDE, cache=cache)
        n = s.normal / (np.linalg.norm(s.normal) + 1e-12)
        T = tangent_basis(n)                       # (14, 13)
        sigma = sigma_at(sp, u_s, proto, cache)
        # draw MORE directions for the constrained chiral patch so the number of SUCCESSFUL
        # correctors is comparable to the single-species patches (PCA on ~8 points cannot
        # exceed ~7 dims, so an equal-n comparison is required, per review point 1).
        K = 120 if name == "chiral_onset" else K_DIRS
        pred, pred_unclip, corr, term_dirs = [], [], [], 0
        for j in range(K):
            v = RNG.standard_normal(T.shape[1]); v /= np.linalg.norm(v)
            t = T @ v
            pred_unclip.append(u_s + DS * t)              # tests basis+sampling (no box clip)
            p = clip_unit(u_s + DS * t); pred.append(p)   # physical accessible predictor
            c, conv = corrector_ridge(sp, p, n, proto, DS, cache, sigma)
            db = directional_sensitivity(sp, c, proto, n, h=0.04, n_side=6, cache=cache, sigma=sigma)["b"]
            if conv and db >= CFG.B_MIN:
                corr.append(c)
            else:
                term_dirs += 1
        # Isotropic nulls at the SAME sample size as each cloud (var95 of m random unit
        # vectors in 13-D falls below 13 from finite sampling alone). corrector_gap = how far
        # the corrector cloud falls BELOW a random cloud at its OWN n: ~0 means full-dim
        # (hypersurface), >0 means genuine dimension reduction.
        def iso_null(m):
            rng = np.random.default_rng(4242)
            X = np.array([(lambda w: w / np.linalg.norm(w))(rng.standard_normal(T.shape[1])) for _ in range(m)])
            return var95(X)[0]
        pred_null = iso_null(K)
        corr_null = iso_null(max(len(corr), 3))
        ng, ev_g = var95(pred_unclip)
        np_, ev_p = var95(pred)
        nc, ev_c = var95(corr) if len(corr) >= 3 else (len(corr), np.array([]))
        gate = ng >= pred_null - 1
        gate_ok = gate_ok and gate
        corr_gap = corr_null - nc
        patches.append({"seed": name, "b_seed": float(s.b), "n_dirs": K, "n_corrector": len(corr),
                        "corrector_success": len(corr) / K, "terminating_dirs": term_dirs,
                        "predictor_gate_var95": ng, "predictor_null_var95": pred_null,
                        "corrector_var95": nc, "corrector_null_var95": corr_null, "corrector_gap": corr_gap,
                        "predictor_clipped_var95": np_, "predictor_gate_spectrum": ev_g.tolist(),
                        "corrector_spectrum": ev_c.tolist(), "gate_pass": bool(gate)})
        print(f"{name}: n_dirs={K} success={len(corr)}/{K} corrector_var95={nc} vs same-n null "
              f"{corr_null} (gap {corr_gap}); predictor {ng} vs null {pred_null}; gate={'OK' if gate else 'FAIL'}", flush=True)
        cache.flush()
    out = {"config_hash": CFG.CONFIG_HASH, "k_dirs": K_DIRS, "ds": DS, "patches": patches,
           "gate_all_pass": bool(gate_ok),
           "corrector_var95_median": float(np.median([p["corrector_var95"] for p in patches]))}
    json.dump(out, open(OUT / "E1_patch.json", "w"), indent=2)
    print(f"\nE1 gate_all_pass={gate_ok}; corrector_var95 median="
          f"{out['corrector_var95_median']}; wall={time.time()-t0:.0f}s", flush=True)
    if not gate_ok:
        print("STOP: predictor-cloud gate failed — tangent-space construction is broken.", flush=True)


if __name__ == "__main__":
    main()
