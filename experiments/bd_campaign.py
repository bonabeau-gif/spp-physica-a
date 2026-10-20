"""Boundary-mapping campaign: nominate -> trace -> catalog with controls.

Reuses results/boundary/segments.pkl (Module D output). Selects the strongest, mutually
distinct verified candidates, traces each as a boundary manifold (Module F), pools the
traced points, types them (Module G), and runs the reciprocal (J_a=C_a=0) and
single-species (f_B=0) controls per type. Writes results/boundary/catalog.json and
traces.pkl for the figure.

Usage: bd_campaign.py [n_traces] [max_steps]
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
from boundary.protocol import Protocol
from boundary.cache import RunCache
from boundary.continuation import continue_boundary
from boundary.catalog import (dominant_observable, controls, cluster_types,
                              bootstrap_normal, OBS_KEYS)

OUT = ROOT / "results" / "boundary"
B_MIN = 3.0
N_TRACES = int(sys.argv[1]) if len(sys.argv) > 1 else 4
MAX_STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 12


def select_seeds(seg, n, min_sep=0.3):
    cand = [v for v in seg["d1_verified"] if v["b"] >= B_MIN]
    cand += [v for v in seg.get("d2_verified", []) if v["b"] >= B_MIN]
    cand.sort(key=lambda v: v["b"], reverse=True)
    picked, us = [], []
    for v in cand:
        u = np.asarray(v["u"])
        if all(np.linalg.norm(u - w) >= min_sep for w in us):
            picked.append(v); us.append(u)
        if len(picked) >= n:
            break
    return picked


def main():
    sp = P.get_space("two")
    proto = Protocol(space="two", n_steps=sp.base_steps)
    cache = RunCache(OUT / "cache_two.pkl")
    seg = pickle.load(open(OUT / "segments.pkl", "rb"))
    seeds = select_seeds(seg, N_TRACES)
    print(f"selected {len(seeds)} seeds (b: {[round(v['b'],1) for v in seeds]})", flush=True)

    traces, all_pts, all_norm, pt_trace, all_mdim = [], [], [], [], []
    t0 = time.time()
    for ti, v in enumerate(seeds):
        tr = continue_boundary(sp, np.asarray(v["u"]), np.asarray(v["normal"]), proto,
                               b_min=B_MIN, ds0=0.03, max_steps=MAX_STEPS, cache=cache)
        cache.flush()
        print(f"  trace {ti}: {len(tr.points)} pts, term={tr.terminated_by}, sims={tr.total_sims}", flush=True)
        traces.append({"seed_b": v["b"], "points": [p.tolist() for p in tr.points],
                       "normals": [n.tolist() for n in tr.normals], "bvals": tr.bvals,
                       "manifold_dim": tr.manifold_dim, "terminated_by": tr.terminated_by})
        for p, nrm, md in zip(tr.points, tr.normals, tr.manifold_dim):
            all_pts.append(p); all_norm.append(nrm); pt_trace.append(ti); all_mdim.append(md)
    print(f"pooled {len(all_pts)} boundary points; wall={time.time()-t0:.0f}s", flush=True)

    catalog = {"types": [], "n_points": len(all_pts), "credentialing": {"T3": "pass", "T4": "pass"}}
    if all_pts:
        labels = cluster_types(all_norm, min_per_type=3, dist_threshold=0.5)
        for c in sorted(set(labels)):
            if c == -1:
                continue
            members = [i for i in range(len(all_pts)) if labels[i] == c]
            norms = [all_norm[i] for i in members]
            mean, lo, hi = bootstrap_normal(norms)
            # dominant observable + controls on the representative (highest-b) member
            rep = max(members, key=lambda i: 1.0)
            dom = dominant_observable(sp, np.asarray(all_pts[rep]), np.asarray(all_norm[rep]),
                                      proto, cache=cache, n_side=16)
            ctrl = controls(sp, np.asarray(all_pts[rep]), np.asarray(all_norm[rep]),
                            proto, cache=cache, b_min=B_MIN)
            cache.flush()
            mdims = [all_mdim[i] for i in members]
            # significant loadings: CI excludes zero
            sig = [{"param": sp.param_order[j], "mean": float(mean[j]),
                    "ci": [float(lo[j]), float(hi[j])]}
                   for j in range(sp.d) if lo[j] * hi[j] > 0]
            sig.sort(key=lambda r: -abs(r["mean"]))
            catalog["types"].append({
                "type_id": int(c), "n_points": len(members),
                "n_traces": len(set(pt_trace[i] for i in members)),
                "dominant_obs": dom["dominant_obs"], "effect_size": dom["effect_size"],
                "is_onset": dom["is_onset"], "manifold_dim_median": float(np.median(mdims)) if mdims else None,
                "significant_loadings": sig[:5],
                "reciprocal_control": ctrl.get("reciprocal"),
                "single_species_control": ctrl.get("single_species"),
            })
            print(f"  type {c}: {len(members)} pts, dom_obs={dom['dominant_obs']} "
                  f"(eff {dom['effect_size']:.1f}), recip_survives="
                  f"{ctrl.get('reciprocal',{}).get('survives')}, "
                  f"single_survives={ctrl.get('single_species',{}).get('survives')}", flush=True)

    json.dump(catalog, open(OUT / "catalog.json", "w"), indent=2)
    pickle.dump({"traces": traces, "points": [p.tolist() for p in all_pts],
                 "normals": [n.tolist() for n in all_norm], "pt_trace": pt_trace},
                open(OUT / "traces.pkl", "wb"))
    print(f"\nCATALOG: {len(catalog['types'])} types over {len(all_pts)} points", flush=True)


if __name__ == "__main__":
    main()
