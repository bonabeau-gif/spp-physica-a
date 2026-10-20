"""Assessment #9: add a CMA-MAE (pyribs) baseline against vanilla MAP-Elites.

Runs a process-parallel CMA-MAE on the two single-species archives at a matched
budget, recording the coverage / QD-score history, so it can be compared with the
vanilla parallel_map_elites histories from experiments/parallel_archives.py at the
same number of evaluations. CMA-MAE is the covariance-adapted state-of-the-art the
manuscript cites; this answers whether the vanilla emitter is leaving coverage on
the table.
"""
from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"
import sys, time
from functools import partial
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
from spp_unified.parameters import ACTIVE_SINGLE_NAMES, EXPERIMENT_BOUNDS
from parallel_archives import eval_single

from ribs.archives import GridArchive
from ribs.emitters import EvolutionStrategyEmitter
from ribs.schedulers import Scheduler

OUT = ROOT / "results" / "highbudget"
OUT.mkdir(parents=True, exist_ok=True)
BOUNDS = EXPERIMENT_BOUNDS


def _init_worker():
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = "1"


def run_cmame(name, payload, measure_ranges, evaluations=16000, dims=(12, 12),
              emitters=5, batch=15, sigma0=0.12, seed=0, workers=8):
    worker = partial(eval_single, payload)
    dim = len(ACTIVE_SINGLE_NAMES)
    archive = GridArchive(solution_dim=dim, dims=list(dims), ranges=measure_ranges,
                          learning_rate=0.02, threshold_min=0.0, seed=seed)
    result = GridArchive(solution_dim=dim, dims=list(dims), ranges=measure_ranges, seed=seed)
    rng = np.random.default_rng(seed)
    ems = [EvolutionStrategyEmitter(archive, x0=rng.uniform(0.25, 0.75, dim),
                                    sigma0=sigma0 + 0.02 * e, ranker="imp",
                                    selection_rule="mu", restart_rule="basic",
                                    batch_size=batch, seed=seed + e) for e in range(emitters)]
    scheduler = Scheduler(archive, ems, result_archive=result)
    hist = []
    calls = 0
    next_rec = 500
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as ex:
        while calls < evaluations:
            sols = scheduler.ask()
            u = np.clip(np.asarray(sols), 0.0, 1.0)
            xs = [BOUNDS[:, 0] + uu * (BOUNDS[:, 1] - BOUNDS[:, 0]) for uu in u]
            seeds = [seed + 2_000_000 + calls + i for i in range(len(xs))]
            res = list(ex.map(worker, xs, seeds, chunksize=2))
            obj = np.array([r[0] for r in res])
            meas = np.array([r[1] for r in res], dtype=float)
            scheduler.tell(obj, meas)
            calls += len(xs)
            if calls >= next_rec:
                st = result.stats
                hist.append({"evaluations": calls, "coverage": float(st.coverage),
                             "num_elites": int(st.num_elites), "qd_score": float(st.qd_score)})
                next_rec += 500
    return pd.DataFrame(hist)


def main():
    specs = {
        "single_qd_alignment_milling": dict(
            payload={"steps": 180, "boundary": "reflecting", "m0": "phi", "m1": "milling"},
            measure_ranges=[(0, 1), (0, 0.5)], seed=11),
        "single_qd_alignment_clustering": dict(
            payload={"steps": 180, "boundary": "periodic", "m0": "phi", "m1": "clustering"},
            measure_ranges=[(0, 1), (0, 8)], seed=23),
    }
    for name, spec in specs.items():
        t = time.time()
        print(f"[CMA-MAE {name}] starting", flush=True)
        h = run_cmame(name, spec["payload"], spec["measure_ranges"],
                      evaluations=16000, seed=spec["seed"])
        h.to_csv(OUT / f"cmame_{name}_history.csv", index=False)
        print(f"[CMA-MAE {name}] done in {time.time()-t:.0f}s  final coverage={h.coverage.iloc[-1]:.3f}", flush=True)


if __name__ == "__main__":
    main()
