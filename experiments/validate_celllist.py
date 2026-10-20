"""Physics-level validation of the cell-list accelerator against the dense path.

Compares, on identical particle states, the per-step alignment direction and force
(and the overlap resolver) computed by the dense O(N^2) expressions in model.py vs the
sparse cell-list functions in neighbors.py. Covers single-species and two-species with
ASYMMETRIC alignment/force matrices (the non-reciprocal case). The dynamics are chaotic,
so trajectory bit-matching is not the test; per-step agreement to ~1e-10 is.
"""
from __future__ import annotations
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from spp_unified.geometry import pairwise_displacements
from spp_unified.model import manuscript_pair_force, beta_from_adhesion
from spp_unified import neighbors as NB


def dense_align_force(x, theta, species, S, periodic, r0, r1, ra, beta, hard_core,
                      alignment_matrix, force_matrix):
    """Verbatim replica of model.py lines 76-96 (dense reference)."""
    d, r = pairwise_displacements(x, S, periodic)
    np.fill_diagonal(r, np.inf)
    uhead = np.column_stack([np.cos(theta), np.sin(theta)])
    weights = alignment_matrix[species[:, None], species[None, :]]
    amask = (r <= ra).astype(float) * weights
    avec = np.einsum("ij,jk->ik", amask, uhead, optimize=False) + uhead
    anorm = np.sqrt(np.sum(avec * avec, axis=1, keepdims=True))
    align_dir = np.divide(avec, anorm, out=np.zeros_like(avec), where=anorm > 1e-12)
    unit = np.divide(d, r[:, :, None], out=np.zeros_like(d),
                     where=np.isfinite(r)[:, :, None] & (r[:, :, None] > 1e-12))
    fscalar = manuscript_pair_force(r, r0, r1, beta, hard_core)
    if force_matrix.shape[0] > 1:
        cross = force_matrix[species[:, None], species[None, :]]
        shell = (r > 2.0 * r0) & (r < r1)
        envelope = np.zeros_like(r)
        envelope[shell] = 1.0 - (r[shell] - 2.0 * r0) / max(r1 - 2.0 * r0, 1e-12)
        fscalar = fscalar + cross * envelope
    force = (fscalar[:, :, None] * unit).sum(axis=1)
    return align_dir, force


def dense_overlaps(x, S, r0, periodic, iters):
    from spp_unified.geometry import resolve_overlaps
    return resolve_overlaps(x.copy(), S, r0, periodic, iters)


def run_case(name, N, S, periodic, two_species, rng):
    r0 = 0.1
    r1 = 5.0 * r0
    ra = 8.0 * r0
    beta = beta_from_adhesion(0.2)
    hard_core = 5.0
    x = rng.uniform(0, S, size=(N, 2))
    theta = rng.uniform(-np.pi, np.pi, size=N)
    uhead = np.column_stack([np.cos(theta), np.sin(theta)])
    if two_species:
        species = (rng.uniform(size=N) < 0.5).astype(np.int8)
        alignment_matrix = np.array([[1.0, 0.7], [0.2, 1.0]])      # asymmetric
        force_matrix = np.array([[0.0, 0.9], [-0.6, 0.0]])          # asymmetric
    else:
        species = np.zeros(N, dtype=np.int8)
        alignment_matrix = np.ones((1, 1))
        force_matrix = np.zeros((1, 1))

    ad, fd = dense_align_force(x, theta, species, S, periodic, r0, r1, ra, beta, hard_core,
                              alignment_matrix, force_matrix)
    asp, fsp = NB.sparse_align_force(x, uhead, species, S, periodic, r0, r1, ra, beta,
                                     hard_core, alignment_matrix, force_matrix)
    e_align = float(np.max(np.abs(ad - asp)))
    e_force = float(np.max(np.abs(fd - fsp)))
    xo_d = dense_overlaps(x, S, r0, periodic, 3)
    xo_s = NB.resolve_overlaps_cell(x, S, r0, periodic, 3)
    e_ovl = float(np.max(np.abs(xo_d - xo_s)))
    print(f"{name:38} align={e_align:.2e} force={e_force:.2e} overlap={e_ovl:.2e}")
    return max(e_align, e_force), e_ovl


def main():
    rng = np.random.default_rng(7)
    tol_phys, tol_ovl = 1e-9, 1e-12
    worst_phys = worst_ovl = 0.0
    for periodic in (True, False):
        tag = "periodic" if periodic else "reflecting"
        for two in (False, True):
            sp = "2species-asym" if two else "1species"
            for N, S in ((1024, 15.4), (2000, 21.5)):
                p, o = run_case(f"{tag}/{sp}/N={N}", N, S, periodic, two, rng)
                worst_phys = max(worst_phys, p); worst_ovl = max(worst_ovl, o)
    print(f"\nworst per-step align/force error: {worst_phys:.2e} (tol {tol_phys:.0e})")
    print(f"worst overlap-resolver error:     {worst_ovl:.2e} (tol {tol_ovl:.0e})")
    ok = worst_phys < tol_phys and worst_ovl < tol_ovl
    print("PASS: cell list matches dense physics" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
