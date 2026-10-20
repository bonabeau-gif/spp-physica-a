from __future__ import annotations
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def run_parallel(worker, tasks, max_workers=4, desc='jobs'):
    # Global cap via SPP_MAX_WORKERS lets callers throttle to share cores with a
    # concurrently running background job without editing each script.
    cap = os.environ.get('SPP_MAX_WORKERS')
    if cap:
        max_workers = min(max_workers, int(cap))
    out = [None] * len(tasks)
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(worker, t): i for i, t in enumerate(tasks)}
        done = 0
        for fut in as_completed(futs):
            i = futs[fut]
            out[i] = fut.result()
            done += 1
            if done % max(1, len(tasks)//10) == 0 or done == len(tasks):
                print(f'{desc}: {done}/{len(tasks)}', flush=True)
    return out


def archive_cells(Z01, objective, dims=(20,20)):
    cells = {}
    for i, (z, q) in enumerate(zip(Z01, objective)):
        c = tuple(np.minimum((np.asarray(z) * np.asarray(dims)).astype(int), np.asarray(dims)-1))
        if c not in cells or q > cells[c][1]:
            cells[c] = (i, float(q))
    return cells


def update_archive(cells, Z01, objective, indices=None, dims=(20,20)):
    if indices is None:
        indices = range(len(Z01))
    for i in indices:
        z, q = Z01[i], objective[i]
        c = tuple(np.minimum((np.asarray(z) * np.asarray(dims)).astype(int), np.asarray(dims)-1))
        if c not in cells or q > cells[c][1]:
            cells[c] = (i, float(q))
    return cells
