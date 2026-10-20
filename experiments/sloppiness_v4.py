"""v4 T2/T3: recompute the noise-floor with the corrected threshold.

The resolved-direction threshold is the EXPECTED LARGEST SINGULAR VALUE OF THE
RANDOM NOISE matrix E (entry (o,j) ~ N(0, Jfloor[o,j]^2)), estimated by Monte
Carlo, rather than the spectral norm of the deterministic error-scale matrix
Jfloor (which is rank-1 because the per-entry SE depends only on the output row).
We count sv(J) above this threshold at three perturbation scales (T3) and
bootstrap the leading resolved eigenvectors at Delta=0.04.
"""
from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments"))
import torch
torch.set_num_threads(1)
import np_encoder_patch  # noqa: F401
from spp_unified.parameters import ACTIVE_SINGLE_NAMES
from run_sloppiness_expanded import load_encoder, _worker
from build_single_corpus import EXPANDED_BOUNDS
from expanded_common import run_parallel
from spp_unified.learned import normalize_latent

OUT = ROOT / "results" / "expanded"
OUTS = ["phi", "milling", "log_clustering", "nematic", "learned_0", "learned_1", "hexatic", "density_mode_peak"]
SCALES = [0.02, 0.04, 0.08]
N_J = 16
N_FLOOR = 24
NB = 400
MC = 3000
RNG = np.random.default_rng(11)


def yvec(summary, d, hx, pk, ae, lo, hi, scale):
    z = ae.transform(np.asarray([d])); z01, _, _ = normalize_latent(z, lo, hi)
    y = np.array([summary["phi"], summary["milling"], np.log1p(summary["clustering"]), summary["nematic"],
                  z01[0, 0], z01[0, 1], hx, pk])
    return y / scale


def mc_threshold(Jfloor, k=MC, q=None):
    """Expected largest singular value of a random matrix with entrywise std Jfloor."""
    m, n = Jfloor.shape
    vals = np.empty(k)
    for i in range(k):
        E = RNG.standard_normal((m, n)) * Jfloor
        vals[i] = np.linalg.svd(E, compute_uv=False)[0]
    return float(vals.mean()) if q is None else float(np.percentile(vals, q))


def leading_eig(J):
    w, V = np.linalg.eigh(J.T @ J); o = np.argsort(w)[::-1]; return w[o], V[:, o]


def main():
    ae, lo, hi = load_encoder()
    df = pd.read_csv(OUT / "single_behavior_corpus_latent.csv")
    corpus = df.copy(); corpus["log_clustering"] = np.log1p(corpus.clustering)
    scale = corpus[OUTS].std().replace(0, 1).to_numpy()
    reps = pd.read_csv(OUT / "sloppiness_representatives.csv").drop_duplicates("sample_id").reset_index(drop=True)

    rows = []; load_rows = []
    for _, rr in reps.iterrows():
        name, boundary = rr.regime, rr.boundary
        u = np.array([rr[f"u_{j}"] for j in range(9)])
        x0 = EXPANDED_BOUNDS[:, 0] + u * (EXPANDED_BOUNDS[:, 1] - EXPANDED_BOUNDS[:, 0])
        ft = [(x0, boundary, 500000 + (int.from_bytes(name.encode("utf-8"), "little") % 997) * 100 + s) for s in range(N_FLOOR)]
        fr = run_parallel(_worker, ft, max_workers=6, desc=f"floor:{name}")
        sig = np.array([yvec(s, d, hx, pk, ae, lo, hi, scale) for (s, d, hx, pk) in fr]).std(0)

        for delta in SCALES:
            side = {}
            for j in range(9):
                for sgn in (-1, 1):
                    up = u.copy(); up[j] = np.clip(up[j] + sgn * delta, 0, 1)
                    x = EXPANDED_BOUNDS[:, 0] + up * (EXPANDED_BOUNDS[:, 1] - EXPANDED_BOUNDS[:, 0])
                    t = [(x, boundary, 600000 + j * 1000 + (sgn + 1) * 100 + s) for s in range(N_J)]
                    r = run_parallel(_worker, t, max_workers=6, desc=f"J:{name}:{delta}:{j}:{sgn}")
                    side[(j, sgn)] = np.array([yvec(s, d, hx, pk, ae, lo, hi, scale) for (s, d, hx, pk) in r])
            J = np.column_stack([(side[(j, 1)].mean(0) - side[(j, -1)].mean(0)) / (2 * delta) for j in range(9)])
            sv = np.linalg.svd(J, compute_uv=False)
            Jfloor = np.sqrt(2.0) * sig[:, None] / (np.sqrt(N_J) * 2 * delta)   # per-entry SE, shape (8,9)
            thr_corr = mc_threshold(Jfloor)
            thr_old = float(np.linalg.svd(Jfloor, compute_uv=False)[0])
            n_corr = int((sv > thr_corr).sum()); n_old = int((sv > thr_old).sum())
            raw = sv[sv > 1e-8]
            raw_span = float(np.log10((raw.max() / raw.min()) ** 2)) if len(raw) > 1 else 0.0
            rows.append({"regime": name, "delta": delta, "raw_span": raw_span,
                         "n_resolved_corrected": n_corr, "n_resolved_old": n_old,
                         "thr_corrected": thr_corr, "thr_old": thr_old, "sv_max": float(sv.max())})
            print(rows[-1], flush=True)

            if abs(delta - 0.04) < 1e-9:
                # bootstrap loadings for the leading resolved directions (corrected count)
                ndir = max(1, min(n_corr, 3))
                w, V = leading_eig(J); refs = [V[:, k].copy() for k in range(ndir)]
                boot = {k: [] for k in range(ndir)}
                for _b in range(NB):
                    pick = RNG.integers(0, N_J, N_J)
                    Jb = np.column_stack([(side[(j, 1)][pick].mean(0) - side[(j, -1)][pick].mean(0)) / (2 * delta) for j in range(9)])
                    _, Vb = leading_eig(Jb)
                    for k in range(ndir):
                        v = Vb[:, k]; v = v if np.dot(v, refs[k]) >= 0 else -v; boot[k].append(v)
                for k in range(ndir):
                    B = np.array(boot[k]); mn = B.mean(0); lo_ci = np.percentile(B, 5, 0); hi_ci = np.percentile(B, 95, 0)
                    for p, a, l, h in zip(ACTIVE_SINGLE_NAMES, mn, lo_ci, hi_ci):
                        load_rows.append({"regime": name, "direction": k + 1, "resolved": k < n_corr,
                                          "parameter": p, "loading": float(a), "ci_lo": float(l), "ci_hi": float(h)})

    sdf = pd.DataFrame(rows); sdf.to_csv(OUT / "sloppiness_v4_summary.csv", index=False)
    pd.DataFrame(load_rows).to_csv(OUT / "sloppiness_v4_loadings.csv", index=False)

    d04 = sdf[np.isclose(sdf.delta, 0.04)]
    counts04 = dict(zip(d04.regime, d04.n_resolved_corrected))
    # step-size change per rule (min..max resolved count across scales)
    stepchg = {}
    for name, g in sdf.groupby("regime"):
        stepchg[name] = {"min": int(g.n_resolved_corrected.min()), "max": int(g.n_resolved_corrected.max()),
                         "at_0.02": int(g[np.isclose(g.delta, 0.02)].n_resolved_corrected.iloc[0]),
                         "at_0.04": int(g[np.isclose(g.delta, 0.04)].n_resolved_corrected.iloc[0]),
                         "at_0.08": int(g[np.isclose(g.delta, 0.08)].n_resolved_corrected.iloc[0])}
    head = {"resolved_counts_0.04_corrected": {k: int(v) for k, v in counts04.items()},
            "resolved_counts_0.04_old": dict(zip(d04.regime, [int(x) for x in d04.n_resolved_old])),
            "range_corrected": [int(d04.n_resolved_corrected.min()), int(d04.n_resolved_corrected.max())],
            "step_size_change": stepchg}
    json.dump(head, open(OUT / "sloppiness_v4_headline.json", "w"), indent=2)
    print(json.dumps(head, indent=2))


if __name__ == "__main__":
    main()
