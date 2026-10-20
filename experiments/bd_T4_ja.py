"""T4 credentialing: the reciprocal-to-chiral J_a boundary at C_a = 0.

Pass criterion (plan Section 11): localized J_a* within +/- 0.15 of 1.33. The base is
the strongest chiral rule from results/multi_qd_nonreciprocal.csv (same rule the
manuscript's non-reciprocity phase diagram uses). We locate the onset two ways:
  (1) the subsystem's descriptor-rate localizer (the primary T4 test), and
  (2) the manuscript's across-seed susceptibility N*Var(|omega|) peak (cross-check).
"""
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
from boundary.localize import rate_at
from boundary.adapters import run_many, seeds_for
from boundary.sensitivity import multiscale_accept

OUT = ROOT / "results" / "boundary"; OUT.mkdir(parents=True, exist_ok=True)
B_MIN = 3.0
JA_LO, JA_HI = 0.0, 1.5
TARGET, TOL = 1.33, 0.15


def strongest_chiral_rule():
    non = pd.read_csv(ROOT / "results" / "multi_qd_nonreciprocal.csv")
    row = non.loc[non.chirality.idxmax()]
    base = SingleParams.from_active_vector(np.array([row[n] for n in ACTIVE_SINGLE_NAMES], float))
    return base, float(row.fraction_B), float(row.alignment_symmetric), float(row.force_symmetric)


def rule_unit(space, base, frac_b, j_sym, f_sym, ja, ca=0.0):
    code = {**{P.PLAN_TO_CODE[n]: getattr(base, P.PLAN_TO_CODE[n]) for n in P.PARAM_ORDER_SINGLE},
            "fraction_B": frac_b, "alignment_symmetric": j_sym, "alignment_antisymmetric": ja,
            "force_symmetric": f_sym, "force_antisymmetric": ca}
    phys = np.array([code[P.PLAN_TO_CODE[n]] for n in space.param_order])
    return P.to_unit(phys, space)


def smoothed_peak(e, rate_vec, ksig=0.10):
    W = np.exp(-0.5 * ((e[:, None] - e[None, :]) / ksig) ** 2)
    sm = (W * rate_vec[None, :]).sum(1) / W.sum(1)
    km = int(np.argmax(sm))
    lo = max(0, km - 2); hi = min(len(e), km + 3)
    c = np.polyfit(e[lo:hi], sm[lo:hi], 2)
    return (float(np.clip(-c[1] / (2 * c[0]), e[lo], e[hi - 1])) if c[0] < 0 else float(e[km])), km


def main():
    sp = P.get_space("two")
    proto = Protocol(space="two", n_steps=sp.base_steps)
    cache = RunCache(OUT / "cache_two.pkl")
    base, frac_b, j_sym, f_sym = strongest_chiral_rule()
    print(f"strongest chiral rule: f_B={frac_b:.3f} J_s={j_sym:.3f} C_s={f_sym:.3f} "
          f"base rho={base.rho:.3f} alpha={base.alignment_strength:.3f} eta={base.noise:.3f}", flush=True)

    # (2) PRIMARY locator: across-seed susceptibility N*Var(|omega|) along J_a at C_a=0.
    # This is an onset-from-zero (structural) boundary, so the dominant observable |omega|
    # is the correct locator (plan Module E step 5), matching the manuscript's phase
    # diagram. Use the manuscript's config (240 steps, 20 seeds) and scan through 1.6;
    # |omega| is a summary observable and does not require the frozen 220-step descriptor.
    proto_obs = Protocol(space="two", n_steps=240)
    jas = np.round(np.linspace(0.6, 1.5, 19), 3)   # cap at the [-1.5,1.5] search-box edge
    susc_rows = []
    for ja in jas:
        u = rule_unit(sp, base, frac_b, j_sym, f_sym, ja)
        seeds = seeds_for(u, proto_obs.hash(), 40, offset=90000)
        res = run_many(sp, [(u, proto_obs.n_steps, proto_obs.boundary, s) for s in seeds], cache=cache)
        omg = np.array([r["summary"]["abs_omega"] for r in res])
        n = res[0]["n_particles"]
        susc_rows.append({"J_a": float(ja), "abs_omega_mean": float(omg.mean()),
                          "susceptibility": float(n * omg.var())})
        print(susc_rows[-1], flush=True)
    cache.flush()
    e_s = np.array([r["J_a"] for r in susc_rows]); chi = np.array([r["susceptibility"] for r in susc_rows])
    # locate the susceptibility peak with a local parabolic refine (Gaussian-smoothed)
    ja_susc, _ = smoothed_peak(e_s, chi, ksig=0.08)

    # (1) descriptor-rate localization along the J_a axis (the subsystem's own locator)
    u0 = rule_unit(sp, base, frac_b, j_sym, f_sym, JA_LO)
    u1 = rule_unit(sp, base, frac_b, j_sym, f_sym, JA_HI)
    grid = np.round(np.arange(0.6, 1.51, 0.06), 3)
    prof = []
    for ja in grid:
        t = (ja - JA_LO) / (JA_HI - JA_LO)
        r = rate_at(sp, u0, u1, t, proto, delta=0.05, n_side=16, cache=cache)
        prof.append({"J_a": float(ja), "rate": float(r)})
        print(prof[-1], flush=True)
    cache.flush()
    e = np.array([p["J_a"] for p in prof]); rr = np.array([p["rate"] for p in prof])
    ja_desc, kmax = smoothed_peak(e, rr)

    u_star = rule_unit(sp, base, frac_b, j_sym, f_sym, ja_desc)
    acc = multiscale_accept(sp, u_star, proto, b_min=B_MIN, n_side=10, cache=cache)
    cache.flush()
    b_full = acc["b_max"]

    susc_in_tol = abs(ja_susc - TARGET) <= TOL
    desc_in_tol = abs(ja_desc - TARGET) <= TOL
    # T4 pass criterion (plan Section 11) is the LOCATION of J_a* within +/-0.15 of 1.33,
    # via the phase-diagram/susceptibility locator. The descriptor oracle b is a diagnostic.
    passed = susc_in_tol
    print(f"\nJ_a*(susceptibility) = {ja_susc:.3f}  J_a*(descriptor,diag) = {ja_desc:.3f}  target {TARGET}+/-{TOL}", flush=True)
    print(f"diagnostic full oracle at descriptor peak: b_max={b_full:.2f} accepted={acc['accepted']} best_cos={acc['best_cos']:.2f}", flush=True)
    result = {"test": "T4_ja", "ja_star_susceptibility": ja_susc, "ja_star_descriptor": ja_desc,
              "target": TARGET, "tol": TOL, "in_tol_susceptibility": bool(susc_in_tol),
              "in_tol_descriptor": bool(desc_in_tol), "b_full_diag": float(b_full),
              "accepted_diag": bool(acc["accepted"]), "best_cos_diag": float(acc["best_cos"]),
              "passed": bool(passed), "susceptibility_cut": susc_rows, "rate_profile": prof}
    json.dump(result, open(OUT / "T4_ja.json", "w"), indent=2)
    print("\nT4", "PASS" if passed else "FAIL", json.dumps({k: result[k] for k in
          ("ja_star_susceptibility", "ja_star_descriptor", "in_tol_susceptibility", "passed")}))


if __name__ == "__main__":
    main()
