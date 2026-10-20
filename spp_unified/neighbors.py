"""Deterministic CPU linked-cell neighbor search (minimum-viable acceleration, no GPU).

The SPP interactions are compactly supported: alignment vanishes beyond ra, the manuscript
pair force is exactly zero beyond r1, overlaps act within 2*r0. So a linked-cell list with
bin width >= the largest active cutoff finds EXACTLY the same within-cutoff pairs as the
dense O(N^2) path -- it is an exact accelerator, not an approximation. Below a small-N or
small-box threshold it falls back to the dense reference automatically.

`cell_list_pairs` returns unique unordered pairs (i<j) within `cutoff`. The physics step
builds alignment / force / overlap accumulations from these pairs (see model.py). Pair
membership is validated for set-equality against the dense reference in the self-test below.
"""
from __future__ import annotations
import numpy as np
from scipy.spatial import cKDTree
from spp_unified.geometry import pairwise_displacements

SMALL_N = 200          # below this, dense all-pairs beats KD-tree build overhead


def dense_pairs(x: np.ndarray, S: float, periodic: bool, cutoff: float):
    """Reference: all unordered pairs (i<j) with separation < cutoff, via dense N x N."""
    _, r = pairwise_displacements(x, S, periodic)
    np.fill_diagonal(r, np.inf)
    ii, jj = np.where(np.triu(r < cutoff, 1))
    return ii.astype(np.int64), jj.astype(np.int64)


def cell_list_pairs(x: np.ndarray, S: float, periodic: bool, cutoff: float):
    """Unordered pairs (i<j) within cutoff via a KD-tree; exact vs dense_pairs.

    Uses scipy cKDTree.query_pairs (C-level). For periodic boundaries the tree's `boxsize`
    gives the exact toroidal minimum-image distance, matching geometry.pairwise_displacements.
    Below SMALL_N the dense path is faster than building a tree.
    """
    n = len(x)
    if n < SMALL_N:
        return dense_pairs(x, S, periodic, cutoff)
    if periodic:
        xw = np.mod(x, S)
        np.clip(xw, 0.0, np.nextafter(S, 0.0), out=xw)   # guard FP coords into [0, S)
        tree = cKDTree(xw, boxsize=S)
    else:
        tree = cKDTree(x)
    pairs = tree.query_pairs(cutoff, output_type="ndarray")   # (M,2), i<j
    if len(pairs) == 0:
        return np.array([], np.int64), np.array([], np.int64)
    ii = pairs[:, 0].astype(np.int64); jj = pairs[:, 1].astype(np.int64)
    srt = np.lexsort((jj, ii))                                # match np.where(triu) order
    return ii[srt], jj[srt]


def sparse_align_force(x, uhead, species, S, periodic, r0, r1, ra, beta,
                       hard_core, alignment_matrix, force_matrix):
    """Cell-list replacement for model.py's dense alignment+force block (lines 78-96).

    Returns (align_dir, force) matching the dense path to floating-point tolerance.
    Pair force base term is symmetric; alignment weights and cross-species force may be
    asymmetric, so each unordered pair contributes to both endpoints with the correctly
    ordered species indices.
    """
    from spp_unified.model import manuscript_pair_force
    n = len(x)
    cutoff = max(ra, r1) * (1.0 + 1e-9)
    ii, jj = cell_list_pairs(x, S, periodic, cutoff)
    avec = uhead.copy()                                  # focal self term (unit weight)
    force = np.zeros((n, 2))
    if len(ii):
        d = x[jj] - x[ii]                                # = x[j]-x[i], matches dense d[i,j]
        if periodic:
            d -= S * np.round(d / S)
        r = np.sqrt(np.einsum("pk,pk->p", d, d))
        # --- alignment (r <= ra, inclusive as in dense) ---
        am = r <= ra
        if am.any():
            iia, jja = ii[am], jj[am]
            wij = alignment_matrix[species[iia], species[jja]]
            wji = alignment_matrix[species[jja], species[iia]]
            np.add.at(avec, iia, wij[:, None] * uhead[jja])
            np.add.at(avec, jja, wji[:, None] * uhead[iia])
        # --- force ---
        unit = np.divide(d, r[:, None], out=np.zeros_like(d), where=r[:, None] > 1e-12)
        fs = manuscript_pair_force(r, r0, r1, beta, hard_core)
        if force_matrix.shape[0] > 1:
            shell = (r > 2.0 * r0) & (r < r1)
            env = np.zeros_like(r)
            env[shell] = 1.0 - (r[shell] - 2.0 * r0) / max(r1 - 2.0 * r0, 1e-12)
            fij = fs + force_matrix[species[ii], species[jj]] * env
            fji = fs + force_matrix[species[jj], species[ii]] * env
        else:
            fij = fji = fs
        np.add.at(force, ii, fij[:, None] * unit)
        np.add.at(force, jj, -(fji[:, None] * unit))
    anorm = np.sqrt(np.sum(avec * avec, axis=1, keepdims=True))
    align_dir = np.divide(avec, anorm, out=np.zeros_like(avec), where=anorm > 1e-12)
    return align_dir, force


def resolve_overlaps_cell(x, S, r0, periodic, iterations=3):
    """Cell-list version of geometry.resolve_overlaps; bit-identical pair order (lexsorted
    == np.where(triu) order), so results match the dense resolver exactly."""
    min_r = 2.0 * r0
    x = x.copy()
    for _ in range(max(0, iterations)):
        ii, jj = cell_list_pairs(x, S, periodic, min_r)
        if len(ii) == 0:
            break
        d = x[jj] - x[ii]
        if periodic:
            d -= S * np.round(d / S)
        rij = np.sqrt(np.sum(d * d, axis=1))
        keep = rij < min_r
        ii, jj, d, rij = ii[keep], jj[keep], d[keep], rij[keep]
        if len(ii) == 0:
            break
        u = np.zeros((len(ii), 2))
        nz = rij > 1e-12
        u[nz] = d[nz] / rij[nz, None]
        u[~nz, 0] = 1.0
        delta = 0.5 * (min_r - rij + 1e-9)[:, None] * u
        corr = np.zeros_like(x)
        np.add.at(corr, ii, -delta)
        np.add.at(corr, jj, delta)
        x += corr
        if periodic:
            x %= S
        else:
            x = np.clip(x, r0, S - r0)
    return x


def _selftest():
    rng = np.random.default_rng(0)
    ok = True
    for periodic in (True, False):
        for N in (250, 600, 1024, 2000):
            for S, cutoff in ((15.4, 0.8), (10.0, 0.5), (8.0, 0.83)):
                x = rng.uniform(0, S, size=(N, 2))
                a = set(map(tuple, np.stack(dense_pairs(x, S, periodic, cutoff), 1)))
                b = set(map(tuple, np.stack(cell_list_pairs(x, S, periodic, cutoff), 1)))
                ncell = int(S // cutoff)
                match = a == b
                ok &= match
                print(f"periodic={periodic!s:5} N={N:5d} S={S:5.1f} cut={cutoff} "
                      f"ncell={ncell:2d} dense={len(a):6d} cell={len(b):6d} "
                      f"{'OK' if match else 'MISMATCH (+%d/-%d)' % (len(b-a), len(a-b))}")
    print("\nALL PAIR-SET TESTS PASS" if ok else "\nFAILED")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if _selftest() else 1)
