"""E3: mixture-restricted descriptor psi_mix.

The general descriptor is single-species-dominated (Section 5.3), so a detector insensitive
to mixture structure finding few mixture boundaries would support a conclusion about the
detector, not the model. psi_mix is built from mixture-defined observables that are zero and
INSENSITIVE to parameters in the identical-species limit, standardized on the two-species
corpus with a variance floor. If the null control (200 identical-species points) keeps the
false-positive rate <= 5%, the detector is validated, and we ask whether psi_mix finds
control-confirmed boundaries beyond the chiral onset.

Note: P_minus and R_lock are NOT identical-species-zero (P_minus = |1-2 f_B||P|, R_lock->1),
so they are excluded; the mixture set used is Omega_rel, alpha_WC, Phi_dem, chi_nn.

Usage: e3_mixture.py [stage]  where stage in {null, full}
"""
from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys, json, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from scipy.stats import qmc
from boundary import params as P
from boundary import config as CFG
from boundary.protocol import Protocol
from boundary.cache import RunCache
from boundary.params import clip_unit
from boundary.adapters import run_many, seeds_for
from boundary.sensitivity import mc_threshold

OUT = ROOT / "results" / "boundary"
MIX_KEYS = ["Omega_rel", "alpha_WC", "Phi_dem", "chi_nn"]
STAGE = sys.argv[1] if len(sys.argv) > 1 else "null"
RNG = np.random.default_rng(20260727)


def mix_vec(obs2):
    return np.array([0.0 if not np.isfinite(obs2[k]) else obs2[k] for k in MIX_KEYS], float)


def eval_mix(sp, proto, u, cache, n, offset=0):
    seeds = seeds_for(u, proto.hash(), n, offset)
    res = run_many(sp, [(u, proto.n_steps, proto.boundary, s) for s in seeds],
                   cache=cache, want_obs2=True)
    return np.array([mix_vec(r["obs2"]) for r in res])          # (n, 4)


def identical_unit(sp, base_phys, fb):
    d = dict(base_phys); d["f_B"] = float(fb)
    d["J_s"] = 1.0; d["J_a"] = 0.0; d["C_s"] = 0.0; d["C_a"] = 0.0
    return clip_unit(P.to_unit(np.array([d[n] for n in sp.param_order]), sp))


def build_standardizer(sp, proto, cache):
    """mu, sigma of psi_mix over the two-species corpus, with a variance floor."""
    df = pd.read_csv(ROOT / "results/expanded/multi_behavior_corpus_latent.csv")
    NAMES = list(P.PARAM_ORDER_TWO)
    rows = df.sample(min(60, len(df)), random_state=0)
    vecs = []
    for _, r in rows.iterrows():
        phys = {n: float(r[P.PLAN_TO_CODE[n]]) for n in NAMES}
        u = clip_unit(P.to_unit(np.array([phys[n] for n in sp.param_order]), sp))
        vecs.append(eval_mix(sp, proto, u, cache, 3).mean(0))
    V = np.array(vecs)
    mu = V.mean(0); sig = V.std(0)
    floor = 0.25 * np.median(sig[sig > 0]) if np.any(sig > 0) else 1.0
    sig = np.maximum(sig, floor)
    return mu, sig, float(floor)


def mix_directional_b(sp, proto, u, direction, cache, mu, sig, sigma_floor, h=0.04, n_side=8):
    d = np.asarray(direction, float); d = d / (np.linalg.norm(d) + 1e-12)
    up = u + h * d; dn = u - h * d
    pu = ((eval_mix(sp, proto, up, cache, n_side, 2000) - mu) / sig)
    pd_ = ((eval_mix(sp, proto, dn, cache, n_side, 6000) - mu) / sig)
    rate = np.linalg.norm(pu.mean(0) - pd_.mean(0)) / (2 * h)
    return rate / (np.linalg.norm(sigma_floor) + 1e-12)


def mix_full_b(sp, proto, u, cache, mu, sig, h=0.04, n_side=8, n_floor=16):
    u = clip_unit(u); d = sp.d
    psi0 = (eval_mix(sp, proto, u, cache, n_floor, 0) - mu) / sig
    s0 = psi0.std(0)
    J = np.zeros((len(MIX_KEYS), d))
    for j in range(d):
        e = np.zeros(d); e[j] = h
        pu = (eval_mix(sp, proto, clip_unit(u + e), cache, n_side, 1000 + j) - mu) / sig
        pd_ = (eval_mix(sp, proto, clip_unit(u - e), cache, n_side, 5000 + j) - mu) / sig
        J[:, j] = (pu.mean(0) - pd_.mean(0)) / (2 * h)
    se = np.sqrt(2.0) * s0[:, None] / (np.sqrt(n_side) * 2 * h)
    thr = mc_threshold(se, q=CFG.MC_Q)
    U, sv, Vt = np.linalg.svd(J, full_matrices=False)
    normal = np.zeros(d); normal[:] = Vt[0]
    return (sv[0] / thr if thr > 0 else 0.0), normal, s0


def chiral_base_phys(sp):
    non = pd.read_csv(ROOT / "results" / "multi_qd_nonreciprocal.csv")
    row = non.loc[non.chirality.idxmax()]
    from spp_unified.parameters import SingleParams, ACTIVE_SINGLE_NAMES
    b = SingleParams.from_active_vector(np.array([row[n] for n in ACTIVE_SINGLE_NAMES], float))
    d = {n: getattr(b, P.PLAN_TO_CODE[n]) for n in P.PARAM_ORDER_SINGLE}
    d.update({"f_B": float(row.fraction_B), "J_s": float(row.alignment_symmetric),
              "J_a": float(row.alignment_antisymmetric), "C_s": float(row.force_symmetric),
              "C_a": float(row.force_antisymmetric)})
    return d


def mix_jump(sp, proto, u, normal, cache, h=0.06, n_side=16, off=41000):
    """Dominant mixture-observable jump across the normal (raw, for controls)."""
    n = np.asarray(normal, float); n = n / (np.linalg.norm(n) + 1e-12)
    a = eval_mix(sp, proto, clip_unit(u + h * n), cache, n_side, off).mean(0)
    b = eval_mix(sp, proto, clip_unit(u - h * n), cache, n_side, off + 3000).mean(0)
    return np.abs(a - b)                                   # per-component jump vector


def full_stage(sp, proto, cache, mu, sig):
    """Does psi_mix find control-confirmed boundaries beyond the chiral onset?"""
    import pickle
    from boundary.params import idx
    from boundary.catalog import _phys_to_unit
    cands = []
    c2 = pickle.load(open(OUT / "campaign2.pkl", "rb"))
    ja = [i for i, f in enumerate(c2["family"]) if f == "A_chiral_Ja"]
    cands.append(("chiral_onset", np.array(c2["points"][ja[0]])))
    c3 = pickle.load(open(OUT / "campaign3.pkl", "rb"))
    for fam in ("D_demix_Cs", "E_demix_Ca"):
        m = [i for i, f in enumerate(c3["family"]) if f == fam]
        if m:
            cands.append((fam, np.array(c3["points"][m[int(np.argmax([c3["susc"][i]["b"] for i in m]))]])))
    # archive elites (spread) + random coupling points (species-observable oracle is slow,
    # so keep the candidate set focused; the three named boundaries are the key test)
    for fn in ("multi_qd_nonreciprocal.csv", "multi_qd_reciprocal.csv"):
        d = pd.read_csv(ROOT / "results" / fn)
        d = d.iloc[np.linspace(0, len(d) - 1, 6).astype(int)]
        for k, (_, r) in enumerate(d.iterrows()):
            phys = {n: float(r.get(P.PLAN_TO_CODE[n], 0.0)) for n in sp.param_order}
            cands.append((f"{fn[:4]}_{k}", clip_unit(P.to_unit(np.array([phys[n] for n in sp.param_order]), sp))))
    for k in range(8):
        cands.append((f"rand_{k}", RNG.random(sp.d)))

    rows = []
    for name, u in cands:
        b, nrm, _ = mix_full_b(sp, proto, clip_unit(u), cache, mu, sig)
        rows.append({"name": name, "u": clip_unit(u), "b": float(b), "normal": nrm})
        cache.flush()
    rows.sort(key=lambda r: -r["b"])
    print("top psi_mix b:", [(r["name"], round(r["b"], 1)) for r in rows[:8]], flush=True)

    # controls on the top candidates with b >= b_min
    confirmed = []
    for r in [x for x in rows if x["b"] >= CFG.B_MIN][:10]:
        u = r["u"]; nrm = r["normal"]
        orig = mix_jump(sp, proto, u, nrm, cache)
        dom = int(np.argmax(orig)); oj = orig[dom]
        uc = u.copy(); uc[idx(sp, "J_a")] = _phys_to_unit(sp, "J_a", 0.0); uc[idx(sp, "C_a")] = _phys_to_unit(sp, "C_a", 0.0)
        rj = mix_jump(sp, proto, clip_unit(uc), nrm, cache)[dom]
        uf = u.copy(); uf[idx(sp, "f_B")] = _phys_to_unit(sp, "f_B", 0.0)
        fj = mix_jump(sp, proto, clip_unit(uf), nrm, cache)[dom]
        recip_survives = (rj / oj >= 0.4) if oj > 1e-6 else True
        single_survives = (fj / oj >= 0.4) if oj > 1e-6 else True
        is_mix = (not recip_survives) or (not single_survives)
        confirmed.append({"name": r["name"], "b": r["b"], "dom_mix": MIX_KEYS[dom],
                          "recip_survives": bool(recip_survives), "single_survives": bool(single_survives),
                          "is_mixture": bool(is_mix)})
        print(f"  {r['name']}: b={r['b']:.1f} dom={MIX_KEYS[dom]} recip_surv={recip_survives} "
              f"single_surv={single_survives} MIXTURE={is_mix}", flush=True)
        cache.flush()
    n_mix = sum(c["is_mixture"] for c in confirmed)
    n_beyond = sum(c["is_mixture"] and c["name"] != "chiral_onset" for c in confirmed)
    out = {"config_hash": CFG.CONFIG_HASH, "top_b": [(r["name"], r["b"]) for r in rows[:10]],
           "confirmed": confirmed, "n_mixture_confirmed": n_mix, "n_beyond_chiral": n_beyond}
    json.dump(out, open(OUT / "E3_full.json", "w"), indent=2)
    print(f"\nE3 FULL: {n_mix} control-confirmed mixture boundaries, {n_beyond} beyond the chiral onset", flush=True)


def discovery_null(sp, proto, cache, mu, sig, n=24):
    """Null for the DISCOVERY claim (not the identical-species null): what fraction of RANDOM
    two-species points pass the oracle (b>=b_min) AND fail a control (appear as a mixture
    boundary) purely by chance? This is the correct baseline for the five discoveries."""
    from boundary.params import idx
    from boundary.catalog import _phys_to_unit
    n_pass, n_mix = 0, 0
    for i in range(n):
        u = clip_unit(RNG.random(sp.d))
        b, nrm, _ = mix_full_b(sp, proto, u, cache, mu, sig)
        if b >= CFG.B_MIN:
            n_pass += 1
            orig = mix_jump(sp, proto, u, nrm, cache); dom = int(np.argmax(orig)); oj = orig[dom]
            uc = u.copy(); uc[idx(sp, "J_a")] = _phys_to_unit(sp, "J_a", 0.0); uc[idx(sp, "C_a")] = _phys_to_unit(sp, "C_a", 0.0)
            rj = mix_jump(sp, proto, clip_unit(uc), nrm, cache)[dom]
            uf = u.copy(); uf[idx(sp, "f_B")] = _phys_to_unit(sp, "f_B", 0.0)
            fj = mix_jump(sp, proto, clip_unit(uf), nrm, cache)[dom]
            is_mix = (oj > 1e-6) and ((rj / oj < 0.4) or (fj / oj < 0.4))
            n_mix += int(is_mix)
        cache.flush()
        if (i + 1) % 8 == 0:
            print(f"  dnull {i+1}/{n}: pass={n_pass} mix={n_mix}", flush=True)
    out = {"config_hash": CFG.CONFIG_HASH, "n_random": n, "n_pass_oracle": n_pass,
           "n_confirmed_mixture_by_chance": n_mix, "chance_mixture_rate": n_mix / n}
    json.dump(out, open(OUT / "E3_dnull.json", "w"), indent=2)
    print(f"\nE3 DISCOVERY NULL: of {n} random points, {n_pass} pass oracle, "
          f"{n_mix} appear as mixture by chance (rate {n_mix/n:.3f})", flush=True)


def main():
    sp = P.get_space("two")
    proto = Protocol(space="two", n_steps=CFG.BASE_STEPS["two"])
    cache = RunCache(OUT / "cache_two.pkl")
    t0 = time.time()
    mu, sig, floor = build_standardizer(sp, proto, cache)
    cache.flush()
    if STAGE == "full":
        full_stage(sp, proto, cache, mu, sig)
        return
    if STAGE == "dnull":
        discovery_null(sp, proto, cache, mu, sig)
        return

    # verify psi_mix ~ 0 on identical species
    base = chiral_base_phys(sp)
    idm = eval_mix(sp, proto, identical_unit(sp, base, 0.5), cache, 8, 0)
    id_std = ((idm - mu) / sig).mean(0)
    print(f"psi_mix identical-species standardized mean: {np.round(id_std,2)} (floor={floor:.3f})", flush=True)

    # NULL CONTROL: 200 identical-species points, directional mixture-oracle b
    bounds = sp.bounds
    Uq = qmc.Sobol(d=sp.d, scramble=True, seed=1).random(200)
    fp = 0; bvals = []
    axis = np.zeros(sp.d); axis[sp.param_order.index("f_B")] = 1.0
    for i in range(200):
        u = Uq[i].copy()
        phys = P.from_unit(u, sp)
        d = dict(zip(sp.param_order, phys))
        u_id = identical_unit(sp, d, d["f_B"])
        # probe sensitivity along a random parameter direction
        dirv = RNG.standard_normal(sp.d); dirv[sp.param_order.index("J_a")] = 0; dirv[sp.param_order.index("C_a")] = 0
        b = mix_directional_b(sp, proto, u_id, dirv, cache, mu, sig, sig)
        bvals.append(b); fp += int(b >= CFG.B_MIN)
        if (i + 1) % 50 == 0:
            print(f"  null {i+1}/200 fp={fp}", flush=True); cache.flush()
    fp_rate = fp / 200
    out = {"config_hash": CFG.CONFIG_HASH, "mix_keys": MIX_KEYS, "variance_floor": floor,
           "identical_std_mean": id_std.tolist(), "null_fp_rate": fp_rate,
           "null_b_median": float(np.median(bvals)), "null_b_p95": float(np.percentile(bvals, 95)),
           "null_pass": bool(fp_rate <= 0.05)}
    json.dump(out, open(OUT / "E3_null.json", "w"), indent=2)
    print(f"\nE3 NULL: fp_rate={fp_rate:.3f} (target<=0.05) b_med={out['null_b_median']:.2f} "
          f"b_p95={out['null_b_p95']:.2f} pass={out['null_pass']}; wall={time.time()-t0:.0f}s", flush=True)
    if not out["null_pass"]:
        print("STOP: psi_mix null control exceeds 5% -> standardization amplifies noise; add variance floor.", flush=True)


if __name__ == "__main__":
    main()
