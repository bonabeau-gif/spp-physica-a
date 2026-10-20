"""Targeted mixture-boundary campaign on the coupling planes (plan Section 14 priority).

Onset-from-zero coupling boundaries are located by their dominant observable (|omega|
susceptibility, as T4 established the descriptor rate is too noisy in the bistable chiral
regime). Three families are seeded:
  A: reciprocal-to-chiral onset in J_a (at C_a=0)      -> should fail the reciprocal control
  B: antisymmetric-force onset in C_a (at J_a=0)       -> should fail the reciprocal control
  C: composition edge in f_B (at strong non-reciprocity) -> should fail the single-species control
Each family is localized at several along-boundary offsets to give >=5 points; the oracle
normal is taken at each point for typing. Pooled points are clustered and controlled.

Usage: bd_campaign2.py [n_along]
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
import pandas as pd
from spp_unified.parameters import SingleParams, ACTIVE_SINGLE_NAMES
from boundary import params as P
from boundary.protocol import Protocol
from boundary.cache import RunCache
from boundary.adapters import run_many, seeds_for
from boundary.sensitivity import sensitivity
from boundary.catalog import dominant_observable, controls, cluster_types, bootstrap_normal

OUT = ROOT / "results" / "boundary"
B_MIN = 3.0
N_ALONG = int(sys.argv[1]) if len(sys.argv) > 1 else 6


def chiral_base():
    non = pd.read_csv(ROOT / "results" / "multi_qd_nonreciprocal.csv")
    row = non.loc[non.chirality.idxmax()]
    base = SingleParams.from_active_vector(np.array([row[n] for n in ACTIVE_SINGLE_NAMES], float))
    d = {P.PLAN_TO_CODE[n]: getattr(base, P.PLAN_TO_CODE[n]) for n in P.PARAM_ORDER_SINGLE}
    d.update({"fraction_B": float(row.fraction_B), "alignment_symmetric": float(row.alignment_symmetric),
              "alignment_antisymmetric": 0.0, "force_symmetric": float(row.force_symmetric),
              "force_antisymmetric": 0.0})
    return d


def to_unit(sp, phys_dict):
    phys = np.array([phys_dict[P.PLAN_TO_CODE[n]] for n in sp.param_order])
    return P.to_unit(phys, sp)


def susceptibility_peak(sp, proto, base_dict, scan_plan, grid, cache, k=20, obs="abs_omega"):
    e, chi = [], []
    for val in grid:
        d = dict(base_dict); d[P.PLAN_TO_CODE[scan_plan]] = float(val)
        u = to_unit(sp, d)
        seeds = seeds_for(u, proto.hash(), k, offset=90000)
        res = run_many(sp, [(u, proto.n_steps, proto.boundary, s) for s in seeds], cache=cache)
        o = np.array([r["summary"][obs] for r in res]); n = res[0]["n_particles"]
        e.append(float(val)); chi.append(float(n * o.var()))
    e = np.array(e); chi = np.array(chi)
    # Gaussian-smoothed parabolic peak, excluding scan edges
    ksig = 0.12 * (grid[-1] - grid[0])
    W = np.exp(-0.5 * ((e[:, None] - e[None, :]) / ksig) ** 2)
    sm = (W * chi[None, :]).sum(1) / W.sum(1)
    interior = np.arange(1, len(e) - 1)
    km = interior[np.argmax(sm[interior])]
    lo, hi = max(0, km - 2), min(len(e), km + 3)
    c = np.polyfit(e[lo:hi], sm[lo:hi], 2)
    star = float(np.clip(-c[1] / (2 * c[0]), e[lo], e[hi - 1])) if c[0] < 0 else float(e[km])
    return star, list(zip(e.tolist(), chi.tolist()))


FAMILIES = {
    "A_chiral_Ja": {"scan": "J_a", "grid": np.linspace(0.6, 1.5, 12),
                    "along": [("rho", d) for d in (0.0,)] + [("alpha", d) for d in (-0.6, 0.6)] +
                             [("f_B", d) for d in (-0.08, 0.08)] + [("v0", d) for d in (0.01,)]},
    "B_force_Ca": {"scan": "C_a", "grid": np.linspace(0.5, 1.5, 12), "fix": {"J_a": 0.0},
                   "along": [("rho", 0.0), ("alpha", -0.6), ("alpha", 0.6), ("f_B", -0.08),
                             ("f_B", 0.08), ("v0", 0.01)]},
    "C_comp_fB": {"scan": "f_B", "grid": np.linspace(0.22, 0.78, 12), "fix": {"J_a": 1.4},
                  "along": [("alpha", -0.6), ("alpha", 0.6), ("J_a", -0.15), ("J_a", 0.1),
                            ("rho", 0.0), ("v0", 0.01)]},
}


def main():
    sp = P.get_space("two")
    proto = Protocol(space="two", n_steps=240)   # observable config (matches T4)
    cache = RunCache(OUT / "cache_two.pkl")
    base0 = chiral_base()

    all_pts, all_norm, all_fam, susc_all = [], [], [], {}
    t0 = time.time()
    for fam, spec in FAMILIES.items():
        along = spec["along"][:N_ALONG]
        for ai, (aparam, adelta) in enumerate(along):
            d = dict(base0)
            for kf, vf in spec.get("fix", {}).items():
                d[P.PLAN_TO_CODE[kf]] = vf
            d[P.PLAN_TO_CODE[aparam]] = d[P.PLAN_TO_CODE[aparam]] + adelta
            star, cut = susceptibility_peak(sp, proto, d, spec["scan"], spec["grid"], cache)
            d[P.PLAN_TO_CODE[spec["scan"]]] = star
            u = P.clip_unit(to_unit(sp, d))
            s = sensitivity(sp, u, proto, h=0.04, n_side=8, cache=cache)
            all_pts.append(u); all_norm.append(s.normal); all_fam.append(fam)
            susc_all[f"{fam}_{ai}"] = {"scan": spec["scan"], "star": star, "cut": cut, "b": float(s.b)}
            print(f"  {fam}[{ai}] {spec['scan']}*={star:.3f} b={s.b:.2f}", flush=True)
        cache.flush()
    print(f"localized {len(all_pts)} points; wall={time.time()-t0:.0f}s", flush=True)

    labels = cluster_types(all_norm, min_per_type=4, dist_threshold=0.45)
    catalog = {"types": [], "n_points": len(all_pts), "families": list(FAMILIES)}
    for c in sorted(set(labels)):
        if c == -1:
            continue
        members = [i for i in range(len(all_pts)) if labels[i] == c]
        norms = [all_norm[i] for i in members]
        mean, lo, hi = bootstrap_normal(norms)
        rep = members[0]
        dom = dominant_observable(sp, np.asarray(all_pts[rep]), np.asarray(all_norm[rep]), proto, cache=cache, n_side=16)
        ctrl = controls(sp, np.asarray(all_pts[rep]), np.asarray(all_norm[rep]), proto, cache=cache, b_min=B_MIN)
        cache.flush()
        fams = sorted(set(all_fam[i] for i in members))
        sig = [{"param": sp.param_order[j], "mean": float(mean[j]), "ci": [float(lo[j]), float(hi[j])]}
               for j in range(sp.d) if lo[j] * hi[j] > 0]
        sig.sort(key=lambda r: -abs(r["mean"]))
        recip = ctrl.get("reciprocal", {}); single = ctrl.get("single_species", {})
        is_mixture = (not recip.get("survives", True)) or (not single.get("survives", True))
        catalog["types"].append({
            "type_id": int(c), "n_points": len(members), "families": fams,
            "dominant_obs": dom["dominant_obs"], "effect_size": dom["effect_size"],
            "significant_loadings": sig[:5], "reciprocal_control": recip,
            "single_species_control": single, "is_mixture_boundary": bool(is_mixture)})
        print(f"  type {c}: {len(members)}pts fams={fams} dom={dom['dominant_obs']} "
              f"recip_surv={recip.get('survives')} single_surv={single.get('survives')} "
              f"MIXTURE={is_mixture}", flush=True)

    n_mix = sum(t["is_mixture_boundary"] for t in catalog["types"])
    catalog["n_mixture_types"] = n_mix
    catalog["m6_pass"] = bool(n_mix >= 3)
    json.dump(catalog, open(OUT / "catalog2.json", "w"), indent=2)
    pickle.dump({"points": [u.tolist() for u in all_pts], "normals": [n.tolist() for n in all_norm],
                 "family": all_fam, "labels": labels.tolist(), "susc": susc_all},
                open(OUT / "campaign2.pkl", "wb"))
    print(f"\nCAMPAIGN2: {len(catalog['types'])} types, {n_mix} mixture; "
          f"M6 {'PASS' if catalog['m6_pass'] else 'PARTIAL'}", flush=True)


if __name__ == "__main__":
    main()
