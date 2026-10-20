"""Sanity + partial T5/T7 check for Module A species-resolved observables."""
from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys, json
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

OUT = ROOT / "results" / "boundary"


def chiral_base():
    non = pd.read_csv(ROOT / "results" / "multi_qd_nonreciprocal.csv")
    row = non.loc[non.chirality.idxmax()]
    base = SingleParams.from_active_vector(np.array([row[n] for n in ACTIVE_SINGLE_NAMES], float))
    return base, float(row.fraction_B), float(row.alignment_symmetric), float(row.force_symmetric)


def unit_for(sp, base, frac_b, j_sym, ja, cs, ca, fb=None):
    code = {**{P.PLAN_TO_CODE[n]: getattr(base, P.PLAN_TO_CODE[n]) for n in P.PARAM_ORDER_SINGLE},
            "fraction_B": frac_b if fb is None else fb, "alignment_symmetric": j_sym,
            "alignment_antisymmetric": ja, "force_symmetric": cs, "force_antisymmetric": ca}
    phys = np.array([code[P.PLAN_TO_CODE[n]] for n in sp.param_order])
    return P.to_unit(phys, sp)


def obs_mean(sp, proto, u, cache, k=10):
    seeds = seeds_for(u, proto.hash(), k, offset=77000)
    res = run_many(sp, [(u, proto.n_steps, proto.boundary, s) for s in seeds], cache=cache, want_obs2=True)
    keys = res[0]["obs2"].keys()
    return {kk: float(np.nanmean([r["obs2"][kk] for r in res])) for kk in keys}


def main():
    sp = P.get_space("two")
    proto = Protocol(space="two", n_steps=sp.base_steps)
    cache = RunCache(OUT / "cache_two.pkl")
    base, frac_b, j_sym, f_sym = chiral_base()

    cases = {
        "chiral(J_a=1.4)": unit_for(sp, base, frac_b, j_sym, 1.4, f_sym, 0.0),
        "reciprocal(J_a=C_a=0)": unit_for(sp, base, frac_b, j_sym, 0.0, f_sym, 0.0),
        "identical(J_s=1,rest=0)": unit_for(sp, base, frac_b, 1.0, 0.0, 0.0, 0.0, fb=0.5),
    }
    out = {}
    for name, u in cases.items():
        m = obs_mean(sp, proto, u, cache)
        out[name] = m
        print(f"\n{name}:")
        for kk in ("P_plus", "P_minus", "R_lock", "Omega_rel", "Omega_tot", "alpha_WC", "Phi_dem", "chi_nn", "D"):
            print(f"   {kk:10s} = {m[kk]:+.4f}")
    cache.flush()
    json.dump(out, open(OUT / "obs2_check.json", "w"), indent=2)
    print("\nOBS2 CHECK done")


if __name__ == "__main__":
    main()
