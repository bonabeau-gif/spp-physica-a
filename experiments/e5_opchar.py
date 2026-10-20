"""E5: operating characteristics + catalog table.

(1) False-positive rate of the GENERAL descriptor oracle in a deliberately flat region.
(2) Prescreen hit rate (re-analysis of M3 + E3). (3) Budget accounting. (4) Catalog table
assembled from E4 (controls), E1 (local dimension), and the campaign normal loadings.
"""
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
from boundary import config as CFG
from boundary.protocol import Protocol
from boundary.cache import RunCache
from boundary.params import clip_unit
from boundary.sensitivity import directional_sensitivity
from boundary.catalog import bootstrap_normal

OUT = ROOT / "results" / "boundary"
RNG = np.random.default_rng(20260727)


def false_positive_rate(sp, proto, cache, n=200):
    """Flat region: high eta, weak alignment, low rho; other params random."""
    lo = sp.bounds[:, 0]; hi = sp.bounds[:, 1]
    fp = 0; bvals = []
    for i in range(n):
        u = RNG.random(sp.d)
        u[sp.param_order.index("eta")] = RNG.uniform(0.65, 1.0)
        u[sp.param_order.index("alpha")] = RNG.uniform(0.0, 0.25)
        u[sp.param_order.index("rho")] = RNG.uniform(0.0, 0.15)
        u = clip_unit(u)
        dirv = RNG.standard_normal(sp.d)
        b = directional_sensitivity(sp, u, proto, dirv, h=0.04, n_side=8, cache=cache)["b"]
        bvals.append(b); fp += int(b >= CFG.B_MIN)
        if (i + 1) % 50 == 0:
            print(f"  fp {i+1}/{n}: {fp}", flush=True); cache.flush()
    return fp / n, bvals


def catalog_table():
    """One row per boundary type from E4 controls + E1 dimension + campaign loadings."""
    e4 = json.load(open(OUT / "E4_controls.json"))
    e1 = json.load(open(OUT / "E1_patch.json"))
    dim = {"chiral_Ja": None, "adhesion": None}
    for p in e1["patches"]:
        if p["seed"].startswith("single_trace"):
            dim["adhesion"] = p["corrector_var95"]
        if p["seed"] == "chiral_onset":
            dim["chiral_Ja"] = p["corrector_var95"]
    rows = []
    for t in e4["types"]:
        nm = t["type"]
        rows.append({
            "type": nm, "dominant_obs": t["dominant_obs"],
            "local_dimension": dim.get(nm, "n/a"),
            "C1_reciprocal_survives": t["C1_reciprocal"]["survives"],
            "C2_single_found": t["C2_single_subspace"]["found"],
            "C3_swap_invariant": t["C3_species_swap"]["invariant"],
            "label": t["label"],
        })
    return rows


def main():
    sp = P.get_space("two")
    proto = Protocol(space="two", n_steps=CFG.BASE_STEPS["two"])
    cache = RunCache(OUT / "cache_two.pkl")

    fp_rate, bvals = false_positive_rate(sp, proto, cache)
    cache.flush()
    print(f"general-oracle false-positive rate (flat region) = {fp_rate:.3f} "
          f"(b_med={np.median(bvals):.2f}, b_p95={np.percentile(bvals,95):.2f})", flush=True)

    # hit-rate re-analysis
    hit = {}
    try:
        m3 = json.load(open(OUT / "M3_nominate.json")); hit["general_D2"] = m3.get("hit_rate_d2")
    except Exception:
        pass
    try:
        e3 = json.load(open(OUT / "E3_full.json"))
        hit["psi_mix_confirmed"] = e3["n_mixture_confirmed"]; hit["psi_mix_beyond_chiral"] = e3["n_beyond_chiral"]
    except Exception:
        pass

    budget = {"cache_pkl_bytes": (OUT / "cache_two.pkl").stat().st_size,
              "note": "run length fixed 220-360 steps; adaptive escalation capped at 440 for E2/E6"}

    out = {"config_hash": CFG.CONFIG_HASH,
           "false_positive_rate_flat": fp_rate, "fp_target": 0.05, "fp_pass": bool(fp_rate <= 0.05),
           "fp_b_median": float(np.median(bvals)), "fp_b_p95": float(np.percentile(bvals, 95)),
           "hit_rates": hit, "budget": budget, "catalog_table": catalog_table()}
    json.dump(out, open(OUT / "E5_opchar.json", "w"), indent=2)
    print(f"\nE5: fp_flat={fp_rate:.3f} pass={out['fp_pass']}; "
          f"catalog rows={len(out['catalog_table'])}", flush=True)
    for r in out["catalog_table"]:
        print(f"  {r['type']}: dom={r['dominant_obs']} dim={r['local_dimension']} "
              f"C1surv={r['C1_reciprocal_survives']} C2found={r['C2_single_found']} "
              f"C3inv={r['C3_swap_invariant']} => {r['label']}", flush=True)


if __name__ == "__main__":
    main()
