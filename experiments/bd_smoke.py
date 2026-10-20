"""Smoke test: simulate + descriptor + standardization path for both spaces."""
from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from boundary import params as P
from boundary.adapters import run_many, standardize, load_standardizer, seeds_for
from boundary.protocol import Protocol


def check(space_name):
    sp = P.get_space(space_name)
    mu, sig = load_standardizer(sp)
    proto = Protocol(space=space_name, n_steps=sp.base_steps)
    u = np.full(sp.d, 0.5)
    t0 = time.time()
    seeds = seeds_for(u, proto.hash(), 2)
    res = run_many(sp, [(u, proto.n_steps, proto.boundary, s) for s in seeds],
                   max_workers=4, desc=f"smoke-{space_name}")
    psi = np.asarray(res[0]["psi"])
    pt = standardize(psi, sp)
    print(f"[{space_name}] mu/sig dim={mu.shape[0]} psi dim={psi.shape[0]} "
          f"n_particles={res[0]['n_particles']} steps={res[0]['steps_used']} "
          f"wall={time.time()-t0:.1f}s finite={np.isfinite(pt).all()} "
          f"mean|z|={np.mean(np.abs(pt)):.2f} tau={res[0]['tau_int']}")
    assert psi.shape[0] == sp.descr_dim, f"descr dim mismatch {psi.shape[0]} vs {sp.descr_dim}"
    assert np.isfinite(pt).all()


def main():
    check("single")
    check("two")
    print("SMOKE OK")


if __name__ == "__main__":
    main()
