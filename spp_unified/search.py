from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import gc
import os
import ctypes
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd


@dataclass
class QDResult:
    archive: GridArchive
    history: pd.DataFrame


def cma_mae(evaluate: Callable[[np.ndarray, int], tuple[float, tuple[float, float], dict]],
            dim: int, bounds: np.ndarray, measure_ranges: list[tuple[float, float]],
            dims: tuple[int, int] = (12, 12), iterations: int = 20,
            batch_size: int = 12, emitters: int = 3, seed: int = 0,
            learning_rate: float = 0.02) -> QDResult:
    """Run CMA-MAE. evaluate(x, seed) returns objective, two measures, metadata."""
    from ribs.archives import GridArchive
    from ribs.emitters import EvolutionStrategyEmitter
    from ribs.schedulers import Scheduler
    extra_fields = {
        "phi": ((), np.float64), "milling": ((), np.float64),
        "clustering": ((), np.float64), "drift": ((), np.float64),
        "chirality": ((), np.float64), "omega": ((), np.float64),
        "segregation": ((), np.float64),
    }
    archive = GridArchive(solution_dim=dim, dims=dims, ranges=measure_ranges,
                          learning_rate=learning_rate, threshold_min=0.0, seed=seed,
                          extra_fields=extra_fields)
    result = GridArchive(solution_dim=dim, dims=dims, ranges=measure_ranges, seed=seed,
                         extra_fields=extra_fields)
    rng = np.random.default_rng(seed)
    ems = []
    for e in range(emitters):
        x0 = rng.uniform(0.25, 0.75, size=dim)
        sigma = 0.10 + 0.03 * e
        ems.append(EvolutionStrategyEmitter(
            archive, x0=x0, sigma0=sigma, ranker="imp", selection_rule="mu",
            restart_rule="basic",
            batch_size=batch_size, seed=seed + e,
        ))
    scheduler = Scheduler(archive, ems, result_archive=result)
    hist = []
    calls = 0
    for it in range(iterations):
        sols = scheduler.ask()
        obj = np.zeros(len(sols))
        meas = np.zeros((len(sols), 2))
        extra = {k: np.zeros(len(sols)) for k in extra_fields}
        for i, u in enumerate(sols):
            u_clip = np.clip(np.asarray(u), 0.0, 1.0)
            x = bounds[:, 0] + u_clip * (bounds[:, 1] - bounds[:, 0])
            q, m, md = evaluate(x, seed + 100000 * it + i)
            obj[i] = q
            meas[i] = m
            for k in extra:
                extra[k][i] = float(md.get(k, np.nan))
            calls += 1
        scheduler.tell(obj, meas, **extra)
        st = result.stats
        hist.append({
            "iteration": it + 1, "evaluations": calls,
            "coverage": st.coverage, "num_elites": st.num_elites,
            "qd_score": st.qd_score,
        })
    return QDResult(result, pd.DataFrame(hist))


def archive_dataframe(archive: GridArchive, names: tuple[str, ...], bounds: np.ndarray | None = None) -> pd.DataFrame:
    df = archive.data(return_type="pandas")
    if len(df) == 0:
        return df
    sols = np.stack([df[f"solution_{j}"].to_numpy() for j in range(len(names))], axis=1)
    if bounds is not None:
        sols = np.clip(sols, 0.0, 1.0)
        sols = bounds[:, 0] + sols * (bounds[:, 1] - bounds[:, 0])
    out = pd.DataFrame(sols, columns=names)
    out["measure_0"] = df["measures_0"].to_numpy()
    out["measure_1"] = df["measures_1"].to_numpy()
    out["objective"] = df["objective"].to_numpy()
    for k in ("phi", "milling", "clustering", "drift", "chirality", "omega", "segregation"):
        if k in df:
            out[k] = df[k].to_numpy()
    if "index_0" in df:
        out["cell_0"] = df["index_0"].to_numpy()
        out["cell_1"] = df["index_1"].to_numpy()
    elif "index" in df:
        idx = df["index"].to_numpy(dtype=int)
        c0, c1 = np.unravel_index(idx, tuple(int(v) for v in archive.dims))
        out["cell_0"] = c0
        out["cell_1"] = c1
    return out

def _trim_memory():
    """Return freed NumPy allocations to the OS on glibc-based systems."""
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


@dataclass
class MapElitesResult:
    elites: pd.DataFrame
    history: pd.DataFrame


def map_elites(evaluate: Callable[[np.ndarray, int], tuple[float, tuple[float, float], dict]],
                names: tuple[str, ...], bounds: np.ndarray,
                measure_ranges: list[tuple[float, float]], dims: tuple[int, int] = (12, 12),
                evaluations: int = 600, init_random: int = 100, sigma: float = 0.12,
                random_fraction: float = 0.12, seed: int = 0,
                record_every: int = 25) -> MapElitesResult:
    """Lightweight MAP-Elites in a normalized genotype space.

    This implementation avoids the substantial overhead and boundary-resampling
    pathologies of general CMA-based emitters for expensive stochastic simulators.
    Each cell stores the most temporally stable solution found for that behavior.
    """
    rng = np.random.default_rng(seed)
    dim = len(names)
    cells: dict[tuple[int, int], dict] = {}
    hist = []

    def cell_of(m):
        out = []
        for val, (lo, hi), d in zip(m, measure_ranges, dims):
            if not np.isfinite(val) or val < lo or val > hi:
                return None
            z = min(max((val - lo) / (hi - lo), 0.0), np.nextafter(1.0, 0.0))
            out.append(int(z * d))
        return tuple(out)

    for k in range(evaluations):
        if k < init_random or not cells or rng.random() < random_fraction:
            u = rng.random(dim)
        else:
            parent = cells[list(cells.keys())[rng.integers(len(cells))]]["u"]
            # Mix local mutations and occasional larger jumps.
            scale = sigma * (2.0 if rng.random() < 0.15 else 1.0)
            u = np.clip(parent + rng.normal(0.0, scale, size=dim), 0.0, 1.0)
        x = bounds[:, 0] + u * (bounds[:, 1] - bounds[:, 0])
        objective, measures, md = evaluate(x, seed + 1000000 + k)
        cell = cell_of(measures)
        if cell is not None and (cell not in cells or objective > cells[cell]["objective"]):
            cells[cell] = {"u": u.copy(), "x": x.copy(), "objective": float(objective),
                           "measure_0": float(measures[0]), "measure_1": float(measures[1]),
                           "metadata": dict(md)}
        if (k + 1) % record_every == 0 or k == evaluations - 1:
            hist.append({"evaluations": k + 1, "num_elites": len(cells),
                         "coverage": len(cells) / float(dims[0] * dims[1]),
                         "qd_score": float(sum(v["objective"] for v in cells.values()))})
            _trim_memory()

    rows = []
    for (i, j), rec in cells.items():
        row = {n: rec["x"][q] for q, n in enumerate(names)}
        row.update({"measure_0": rec["measure_0"], "measure_1": rec["measure_1"],
                    "objective": rec["objective"], "cell_0": i, "cell_1": j})
        row.update(rec["metadata"])
        rows.append(row)
    return MapElitesResult(pd.DataFrame(rows), pd.DataFrame(hist))


def _limit_blas_threads():
    """Pin per-worker BLAS/OpenMP threads to one so that a process pool of
    simulators does not oversubscribe the cores."""
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[var] = "1"


def parallel_map_elites(worker: Callable[[np.ndarray, int], tuple[float, tuple[float, float], dict]],
                        names: tuple[str, ...], bounds: np.ndarray,
                        measure_ranges: list[tuple[float, float]], dims: tuple[int, int] = (12, 12),
                        evaluations: int = 60000, init_random: int = 6000, sigma: float = 0.12,
                        random_fraction: float = 0.12, seed: int = 0,
                        batch_size: int = 90, max_workers: int = 9,
                        record_every: int = 500) -> MapElitesResult:
    """Process-parallel batched MAP-Elites.

    Numerically equivalent to :func:`map_elites` except that candidates are
    proposed and evaluated in batches so the embarrassingly parallel simulator
    calls can be spread across ``max_workers`` processes. ``worker`` must be a
    top-level (picklable) callable ``worker(x, seed) -> (objective, (m0, m1),
    metadata)`` — use :func:`functools.partial` on a module-level function to
    bind a configuration payload. The per-candidate seed scheme, the archive
    insertion rule (keep the higher objective), the random/mutation proposal
    mix, and the recorded history schema all match :func:`map_elites`, so the
    only difference from the serial routine is the order in which a batch's
    candidates observe the archive state (standard batched MAP-Elites).
    """
    rng = np.random.default_rng(seed)
    dim = len(names)
    cells: dict[tuple[int, int], dict] = {}
    hist = []
    lo_b = bounds[:, 0]
    span_b = bounds[:, 1] - bounds[:, 0]

    def cell_of(m):
        out = []
        for val, (lo, hi), d in zip(m, measure_ranges, dims):
            if not np.isfinite(val) or val < lo or val > hi:
                return None
            z = min(max((val - lo) / (hi - lo), 0.0), np.nextafter(1.0, 0.0))
            out.append(int(z * d))
        return tuple(out)

    def propose(k):
        if k < init_random or not cells or rng.random() < random_fraction:
            return rng.random(dim)
        parent = cells[list(cells.keys())[rng.integers(len(cells))]]["u"]
        scale = sigma * (2.0 if rng.random() < 0.15 else 1.0)
        return np.clip(parent + rng.normal(0.0, scale, size=dim), 0.0, 1.0)

    _limit_blas_threads()
    next_record = record_every
    with ProcessPoolExecutor(max_workers=max_workers, initializer=_limit_blas_threads) as ex:
        k = 0
        while k < evaluations:
            bsize = int(min(batch_size, evaluations - k))
            us = [propose(k + b) for b in range(bsize)]
            xs = [lo_b + u * span_b for u in us]
            seeds = [seed + 1_000_000 + (k + b) for b in range(bsize)]
            results = list(ex.map(worker, xs, seeds, chunksize=4))
            for b, (objective, measures, md) in enumerate(results):
                cell = cell_of(measures)
                if cell is not None and (cell not in cells or objective > cells[cell]["objective"]):
                    cells[cell] = {"u": us[b].copy(), "x": xs[b].copy(),
                                   "objective": float(objective),
                                   "measure_0": float(measures[0]), "measure_1": float(measures[1]),
                                   "metadata": dict(md)}
            k += bsize
            if k >= next_record or k >= evaluations:
                hist.append({"evaluations": k, "num_elites": len(cells),
                             "coverage": len(cells) / float(dims[0] * dims[1]),
                             "qd_score": float(sum(v["objective"] for v in cells.values()))})
                next_record += record_every
                _trim_memory()

    rows = []
    for (i, j), rec in cells.items():
        row = {n: rec["x"][q] for q, n in enumerate(names)}
        row.update({"measure_0": rec["measure_0"], "measure_1": rec["measure_1"],
                    "objective": rec["objective"], "cell_0": i, "cell_1": j})
        row.update(rec["metadata"])
        rows.append(row)
    return MapElitesResult(pd.DataFrame(rows), pd.DataFrame(hist))
