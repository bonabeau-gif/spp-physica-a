"""Smoke test for Module F: trace one boundary from the strongest verified candidate."""
from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys, pickle, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from boundary import params as P
from boundary.protocol import Protocol
from boundary.cache import RunCache
from boundary.continuation import continue_boundary

OUT = ROOT / "results" / "boundary"
STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 10


def main():
    sp = P.get_space("two")
    proto = Protocol(space="two", n_steps=sp.base_steps)
    cache = RunCache(OUT / "cache_two.pkl")
    seg = pickle.load(open(OUT / "segments.pkl", "rb"))
    cand = max(seg["d1_verified"], key=lambda v: v["b"])
    u0 = np.asarray(cand["u"]); n0 = np.asarray(cand["normal"])
    print(f"seed candidate b={cand['b']:.2f}, tracing up to {STEPS} steps", flush=True)
    t0 = time.time()
    tr = continue_boundary(sp, u0, n0, proto, b_min=3.0, ds0=0.03, max_steps=STEPS, cache=cache)
    cache.flush()
    print(f"traced {len(tr.points)} points; terminated_by={tr.terminated_by}; "
          f"sims={tr.total_sims}; wall={time.time()-t0:.0f}s", flush=True)
    if tr.bvals:
        print(f"b range [{min(tr.bvals):.2f}, {max(tr.bvals):.2f}]; "
              f"manifold_dim {tr.manifold_dim}", flush=True)
    pickle.dump({"points": [p.tolist() for p in tr.points],
                 "normals": [n.tolist() for n in tr.normals],
                 "bvals": tr.bvals, "manifold_dim": tr.manifold_dim,
                 "terminated_by": tr.terminated_by, "total_sims": tr.total_sims},
                open(OUT / "trace_test.pkl", "wb"))
    print("TRACE TEST done")


if __name__ == "__main__":
    main()
