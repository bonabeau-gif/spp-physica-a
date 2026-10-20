"""Diagnostic: phi susceptibility and descriptor rate for the C1 Vicsek eta cut over a
wide grid at a fixed moderate run length, to locate the true peaks and check bracketing."""
from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from boundary import params as P
from boundary.protocol import Protocol
from boundary.cache import RunCache
from boundary.adapters import run_many, seeds_for, standardize

OUT = ROOT / "results" / "boundary"


def main():
    sp = P.get_space("single")
    proto = Protocol(space="single", n_steps=320)
    cache = RunCache(OUT / "cache_single.pkl")
    base = dict(rho=0.15, v0=0.05, eps=0.3, r1_ratio=5.0, f0=0.0, B=0.0, ra_ratio=8.0, alpha=3.5, eta=0.0)
    ax = sp.param_order.index("eta"); e = np.zeros(sp.d); e[ax] = 1.0
    etas = np.round(np.linspace(0.20, 0.90, 15), 3)
    print("eta   phi_mean  N*Var(phi)   desc_rate")
    for eta in etas:
        d = dict(base); d["eta"] = eta
        u = P.clip_unit(P.to_unit(np.array([d[n] for n in sp.param_order]), sp))
        seeds = seeds_for(u, proto.hash(), 24, offset=110000)
        res = run_many(sp, [(u, proto.n_steps, proto.boundary, s) for s in seeds], cache=cache)
        phi = np.array([r["summary"]["phi"] for r in res]); N = res[0]["n_particles"]
        up = u + 0.05 * e; dn = u - 0.05 * e
        rp = run_many(sp, [(up, proto.n_steps, proto.boundary, s) for s in seeds_for(up, proto.hash(), 12, 3000)], cache=cache)
        rm = run_many(sp, [(dn, proto.n_steps, proto.boundary, s) for s in seeds_for(dn, proto.hash(), 12, 7000)], cache=cache)
        rate = np.linalg.norm(np.mean([standardize(r["psi"], sp) for r in rp], 0) -
                              np.mean([standardize(r["psi"], sp) for r in rm], 0)) / 0.10
        print(f"{eta:.2f}  {phi.mean():.3f}    {N*phi.var():.4f}     {rate:.2f}", flush=True)
    cache.flush()


if __name__ == "__main__":
    main()
