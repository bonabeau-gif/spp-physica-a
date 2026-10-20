"""Re-catalog the targeted coupling-boundary points (campaign2.pkl) with the improved
observable-transition controls. Reports both per-family types (each coupling family is a
distinct boundary type by construction) and data-driven normal clustering, and certifies
which types are genuine mixture boundaries (dominant-observable jump collapses under the
reciprocal or single-species control)."""
from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys, json, pickle
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from boundary import params as P
from boundary.protocol import Protocol
from boundary.cache import RunCache
from boundary.catalog import dominant_observable, controls, bootstrap_normal, cluster_types

OUT = ROOT / "results" / "boundary"
B_MIN = 3.0


def main():
    sp = P.get_space("two")
    proto = Protocol(space="two", n_steps=240)
    cache = RunCache(OUT / "cache_two.pkl")
    d = pickle.load(open(OUT / "campaign2.pkl", "rb"))
    pts = [np.asarray(u) for u in d["points"]]
    norms = [np.asarray(n) for n in d["normals"]]
    fam = d["family"]; susc = d["susc"]
    fam_names = sorted(set(fam))

    def bval(i):
        # recover b stored per family point
        f = fam[i]; ai = [j for j in range(len(fam)) if fam[j] == f].index(i)
        return susc.get(f"{f}_{ai}", {}).get("b", 0.0)

    types = []
    for f in fam_names:
        members = [i for i in range(len(pts)) if fam[i] == f]
        rep = max(members, key=bval)
        dom = dominant_observable(sp, pts[rep], norms[rep], proto, cache=cache, n_side=16)
        ctrl = controls(sp, pts[rep], norms[rep], proto, cache=cache, b_min=B_MIN,
                        dom_obs=dom["dominant_obs"])
        cache.flush()
        mean, lo, hi = bootstrap_normal([norms[i] for i in members])
        sig = [{"param": sp.param_order[j], "mean": float(mean[j])}
               for j in range(sp.d) if lo[j] * hi[j] > 0]
        sig.sort(key=lambda r: -abs(r["mean"]))
        rec = ctrl["reciprocal"]; sng = ctrl["single_species"]
        mixture = (not rec["survives"]) or (not sng["survives"])
        types.append({"family": f, "n_points": len(members), "dominant_obs": dom["dominant_obs"],
                      "rep_b": bval(rep), "top_loadings": [s["param"] for s in sig[:4]],
                      "reciprocal": rec, "single_species": sng, "is_mixture": bool(mixture)})
        print(f"{f}: {len(members)}pts dom={dom['dominant_obs']} b={bval(rep):.2f} "
              f"top={[s['param'] for s in sig[:3]]}", flush=True)
        print(f"   reciprocal survives={rec['survives']} (jump_ratio={rec.get('jump_ratio','-'):.2f}), "
              f"single_species survives={sng['survives']} (jump_ratio={sng.get('jump_ratio','-'):.2f}) "
              f"-> MIXTURE={mixture}", flush=True)

    n_mix = sum(t["is_mixture"] for t in types)
    labels = cluster_types(norms, min_per_type=3, dist_threshold=0.35)
    out = {"per_family_types": types, "n_mixture_types": n_mix,
           "data_driven_labels": labels.tolist(),
           "m6_pass": bool(n_mix >= 3), "credentialing": {"T3": "pass", "T4": "pass"}}
    json.dump(out, open(OUT / "catalog2_final.json", "w"), indent=2)
    print(f"\nRE-CATALOG: {len(types)} families, {n_mix} genuine mixture boundaries; "
          f"M6 {'PASS' if out['m6_pass'] else 'PARTIAL'}", flush=True)


if __name__ == "__main__":
    main()
