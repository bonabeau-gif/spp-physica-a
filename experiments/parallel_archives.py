"""Run the four 12 x 12 behavior archives at 64,000 evaluations each.

Uses process-parallel batched MAP-Elites with deterministic proposal order.
Run: python experiments/parallel_archives.py --workers 4
"""
from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"
import argparse
import sys
import time
from functools import partial
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from spp_unified.parameters import (
    ACTIVE_SINGLE_NAMES, EXPERIMENT_BOUNDS, TWO_SPECIES_EXTRA_BOUNDS,
    SingleParams, TwoSpeciesParams, SimulationConfig,
)
from spp_unified.model import simulate_single, simulate_two_species
from spp_unified.search import parallel_map_elites

RESULTS = ROOT / "results" / "highbudget"
RESULTS.mkdir(parents=True, exist_ok=True)


# ---- Top-level (picklable) evaluation workers -----------------------------
def _cfg(payload, seed):
    return SimulationConfig(steps=payload["steps"], sample_start=payload["steps"] // 2,
                            boundary=payload["boundary"], seed=int(seed))


def eval_single(payload, x, seed):
    r = simulate_single(SingleParams.from_active_vector(np.asarray(x)), _cfg(payload, seed)).summary
    return r["temporal_stability"], (r[payload["m0"]], r[payload["m1"]]), r


def _two_non(x):
    n = len(ACTIVE_SINGLE_NAMES)
    b = SingleParams.from_active_vector(x[:n])
    return TwoSpeciesParams(base=b, fraction_B=x[n], alignment_symmetric=x[n + 1],
                            alignment_antisymmetric=x[n + 2], force_symmetric=x[n + 3],
                            force_antisymmetric=x[n + 4])


def _two_rec(x):
    n = len(ACTIVE_SINGLE_NAMES)
    b = SingleParams.from_active_vector(x[:n])
    return TwoSpeciesParams(base=b, fraction_B=x[n], alignment_symmetric=x[n + 1],
                            alignment_antisymmetric=0.0, force_symmetric=x[n + 2],
                            force_antisymmetric=0.0)


def eval_two_non(payload, x, seed):
    r = simulate_two_species(_two_non(np.asarray(x)), _cfg(payload, seed)).summary
    return r["temporal_stability"], (r["drift"], r["chirality"]), r


def eval_two_rec(payload, x, seed):
    r = simulate_two_species(_two_rec(np.asarray(x)), _cfg(payload, seed)).summary
    return r["temporal_stability"], (r["drift"], r["chirality"]), r


# ---- Archive definitions (match stage_runner.py) --------------------------
def archive_specs(evals_single, evals_multi):
    rec = np.array([[.2, .8], [-1, 1.5], [-1, 1]])  # reciprocal extra bounds
    return {
        "single_qd_alignment_milling": dict(
            worker=partial(eval_single, {"steps": 180, "boundary": "reflecting",
                                         "m0": "phi", "m1": "milling"}),
            names=ACTIVE_SINGLE_NAMES, bounds=EXPERIMENT_BOUNDS,
            measure_ranges=[(0, 1), (0, .5)], evaluations=evals_single,
            init_random=2000, seed=11),
        "single_qd_alignment_clustering": dict(
            worker=partial(eval_single, {"steps": 180, "boundary": "periodic",
                                         "m0": "phi", "m1": "clustering"}),
            names=ACTIVE_SINGLE_NAMES, bounds=EXPERIMENT_BOUNDS,
            measure_ranges=[(0, 1), (0, 8)], evaluations=evals_single,
            init_random=2000, seed=23),
        "multi_qd_nonreciprocal": dict(
            worker=partial(eval_two_non, {"steps": 200, "boundary": "periodic"}),
            names=ACTIVE_SINGLE_NAMES + TWO_SPECIES_EXTRA_NAMES_NON(),
            bounds=np.vstack([EXPERIMENT_BOUNDS, TWO_SPECIES_EXTRA_BOUNDS]),
            measure_ranges=[(0, 1), (0, 1)], evaluations=evals_multi,
            init_random=3000, seed=31),
        "multi_qd_reciprocal": dict(
            worker=partial(eval_two_rec, {"steps": 200, "boundary": "periodic"}),
            names=ACTIVE_SINGLE_NAMES + ("fraction_B", "alignment_symmetric", "force_symmetric"),
            bounds=np.vstack([EXPERIMENT_BOUNDS, rec]),
            measure_ranges=[(0, 1), (0, 1)], evaluations=evals_multi,
            init_random=3000, seed=37),
    }


def TWO_SPECIES_EXTRA_NAMES_NON():
    return ("fraction_B", "alignment_symmetric", "alignment_antisymmetric",
            "force_symmetric", "force_antisymmetric")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evals", type=int, default=64000, help="evals per single-species archive")
    ap.add_argument("--evals-multi", type=int, default=64000, help="evals per two-species archive")
    ap.add_argument("--workers", type=int, default=9)
    ap.add_argument("--batch", type=int, default=90)
    ap.add_argument("--only", type=str, default="", help="comma list of archive names to run")
    a = ap.parse_args()

    specs = archive_specs(a.evals, a.evals_multi)
    order = ["single_qd_alignment_milling", "single_qd_alignment_clustering",
             "multi_qd_nonreciprocal", "multi_qd_reciprocal"]
    if a.only:
        wanted = set(s.strip() for s in a.only.split(","))
        order = [o for o in order if o in wanted]

    for name in order:
        spec = specs[name]
        t = time.time()
        print(f"[{name}] starting {spec['evaluations']} evals on {a.workers} workers", flush=True)
        res = parallel_map_elites(
            spec["worker"], spec["names"], spec["bounds"], spec["measure_ranges"],
            dims=(12, 12), evaluations=spec["evaluations"], init_random=spec["init_random"],
            sigma=0.12, random_fraction=0.12, seed=spec["seed"],
            batch_size=a.batch, max_workers=a.workers, record_every=500)
        res.elites.to_csv(RESULTS / f"{name}.csv", index=False)
        res.history.to_csv(RESULTS / f"{name}_history.csv", index=False)
        cov = len(res.elites) / 144.0
        print(f"[{name}] DONE in {time.time()-t:.0f}s  elites={len(res.elites)} "
              f"coverage={cov:.3f}  ({time.time()-t:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
