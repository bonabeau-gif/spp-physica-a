"""M3: run Module D nomination (D1 archive mining + D2 GP-gradient prescreen) and
verify the top candidates with the real oracle. Gate: prescreen hit rate > 0.2.

Usage: bd_nominate.py [n_verify_d2] [n_verify_d1]
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
from boundary.nominate import load_corpus, d1_archive_pairs, fit_pc_gps, d2_prescreen
from boundary.sensitivity import sensitivity

OUT = ROOT / "results" / "boundary"; OUT.mkdir(parents=True, exist_ok=True)
B_MIN = 3.0
N_VERIFY_D2 = int(sys.argv[1]) if len(sys.argv) > 1 else 24
N_VERIFY_D1 = int(sys.argv[2]) if len(sys.argv) > 2 else 8


def main():
    sp = P.get_space("two")
    proto = Protocol(space="two", n_steps=sp.base_steps)
    cache = RunCache(OUT / "cache_two.pkl")

    U, psi_t = load_corpus(sp)
    print(f"corpus: {len(U)} rules, descriptor dim {psi_t.shape[1]}", flush=True)

    # D1: no new simulation
    d1 = d1_archive_pairs(U, psi_t, eps_pair=0.25, K=200, dedup=0.15)
    print(f"D1: {len(d1)} candidate segments (top ratio {d1[0]['score']:.1f})", flush=True)

    # D2: GP-gradient prescreen, no new simulation per candidate
    model = fit_pc_gps(U, psi_t, n_pc=10)
    d2, diag = d2_prescreen(model, sp.d, n_sample=100000, top=1000, n_clusters=60, seed=0)
    print(f"D2: {len(d2)} candidate points; g_top={diag['g_top']:.2f} "
          f"g_median_sample={diag['g_median_sample']:.2f}", flush=True)

    # verify with the real oracle (single-scale screen at h=0.04)
    def verify(u01):
        s = sensitivity(sp, u01, proto, h=0.04, n_side=8, n_floor=16, cache=cache)
        return {"b": float(s.b), "n_resolved": int(s.n_resolved), "sims": int(s.sims),
                "normal": s.normal.tolist()}

    t0 = time.time()
    d2_hits = 0; d2_ver = []
    for k, c in enumerate(d2[:N_VERIFY_D2]):
        v = verify(c["u"]); d2_ver.append({**v, "u": c["u"].tolist()})
        d2_hits += int(v["b"] >= B_MIN)
        print(f"  D2[{k+1}/{N_VERIFY_D2}] b={v['b']:.2f} resolved={v['n_resolved']}", flush=True)
        cache.flush()
    d1_hits = 0; d1_ver = []
    for k, seg in enumerate(d1[:N_VERIFY_D1]):
        mid = 0.5 * (np.asarray(seg["u_start"]) + np.asarray(seg["u_end"]))
        v = verify(mid); d1_ver.append({**v, "u": mid.tolist(), "score": seg["score"]})
        d1_hits += int(v["b"] >= B_MIN)
        print(f"  D1[{k+1}/{N_VERIFY_D1}] b={v['b']:.2f} ratio={seg['score']:.1f}", flush=True)
        cache.flush()

    hit_rate_d2 = d2_hits / max(1, len(d2_ver))
    hit_rate_d1 = d1_hits / max(1, len(d1_ver))
    print(f"\nD2 hit rate = {hit_rate_d2:.2f} ({d2_hits}/{len(d2_ver)})  "
          f"D1 hit rate = {hit_rate_d1:.2f} ({d1_hits}/{len(d1_ver)})  "
          f"wall {time.time()-t0:.0f}s", flush=True)

    # persist candidate segments/points for downstream localization + continuation
    with open(OUT / "segments.pkl", "wb") as fh:
        pickle.dump({"d1": d1, "d2": d2, "d2_verified": d2_ver, "d1_verified": d1_ver}, fh)
    result = {"n_corpus": len(U), "n_d1": len(d1), "n_d2": len(d2),
              "hit_rate_d2": hit_rate_d2, "hit_rate_d1": hit_rate_d1,
              "g_top": diag["g_top"], "g_median_sample": diag["g_median_sample"],
              "b_min": B_MIN, "passed": bool(hit_rate_d2 > 0.2)}
    json.dump(result, open(OUT / "M3_nominate.json", "w"), indent=2)
    print("\nM3", "PASS" if result["passed"] else "FAIL",
          json.dumps({k: result[k] for k in ("hit_rate_d2", "hit_rate_d1", "n_d1", "passed")}))


if __name__ == "__main__":
    main()
