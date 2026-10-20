"""Assessment v2 problem #1: recompute local Jacobian eigenvector loadings with
bootstrap confidence intervals, from many seeds, over the FIVE distinct
representative rules (the global-aligned and high-density-aligned representatives
are the same dense-flock rule and are merged). Report leading-eigenvector
loadings only for directions that are resolved above the seed-to-seed noise
floor. This resolves the §7.1/§7.2/Table S2 contradiction where a single rule
was reported with two different leading eigenvectors.
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
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
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
DELTA = 0.04
N_J = 16
N_FLOOR = 24
NB = 400
RNG = np.random.default_rng(7)


def yvec(summary, d, hx, pk, ae, lo, hi, scale):
    z = ae.transform(np.asarray([d]))
    z01, _, _ = normalize_latent(z, lo, hi)
    y = np.array([summary["phi"], summary["milling"], np.log1p(summary["clustering"]),
                  summary["nematic"], z01[0, 0], z01[0, 1], hx, pk])
    return y / scale


def leading_eigvec(J):
    F = J.T @ J
    w, V = np.linalg.eigh(F)
    order = np.argsort(w)[::-1]
    return w[order], V[:, order]


def align_sign(vec, ref):
    return vec if np.dot(vec, ref) >= 0 else -vec


def main():
    ae, lo, hi = load_encoder()
    df = pd.read_csv(OUT / "single_behavior_corpus_latent.csv")
    corpus = df.copy(); corpus["log_clustering"] = np.log1p(corpus.clustering)
    scale = corpus[OUTS].std().replace(0, 1).to_numpy()

    reps = pd.read_csv(OUT / "sloppiness_representatives.csv")
    reps = reps.drop_duplicates(subset="sample_id", keep="first").reset_index(drop=True)  # merge dupes
    print("distinct regimes:", reps.regime.tolist(), flush=True)

    records = []
    summary_rows = []
    for _, rr in reps.iterrows():
        name = rr.regime; boundary = rr.boundary
        u = np.array([rr[f"u_{j}"] for j in range(9)])
        x0 = EXPANDED_BOUNDS[:, 0] + u * (EXPANDED_BOUNDS[:, 1] - EXPANDED_BOUNDS[:, 0])
        # noise floor (per-output std at unperturbed point)
        ftasks = [(x0, boundary, 500000 + (int.from_bytes(name.encode("utf-8"), "little") % 997) * 100 + s) for s in range(N_FLOOR)]
        fres = run_parallel(_worker, ftasks, max_workers=3, desc=f"floor:{name}")
        yf = np.array([yvec(s, d, hx, pk, ae, lo, hi, scale) for (s, d, hx, pk) in fres])
        sig = yf.std(axis=0)  # standardized-output seed std

        # per-seed y at each (param, sign)
        # side_y[j][sign] = array (N_J, 8)
        side_y = {}
        for j in range(9):
            for sign in (-1, 1):
                up = u.copy(); up[j] = np.clip(up[j] + sign * DELTA, 0, 1)
                x = EXPANDED_BOUNDS[:, 0] + up * (EXPANDED_BOUNDS[:, 1] - EXPANDED_BOUNDS[:, 0])
                tasks = [(x, boundary, 600000 + j * 1000 + (sign + 1) * 100 + s) for s in range(N_J)]
                res = run_parallel(_worker, tasks, max_workers=3, desc=f"J:{name}:{j}:{sign}")
                side_y[(j, sign)] = np.array([yvec(s, d, hx, pk, ae, lo, hi, scale) for (s, d, hx, pk) in res])

        def build_J(pick):
            J = np.zeros((len(OUTS), 9))
            for j in range(9):
                yp = side_y[(j, 1)][pick].mean(0)
                ym = side_y[(j, -1)][pick].mean(0)
                J[:, j] = (yp - ym) / (2 * DELTA)
            return J

        allpick = np.arange(N_J)
        J = build_J(allpick)
        w, V = leading_eigvec(J)
        sv = np.sqrt(np.clip(w, 0, None))
        # noise floor on singular values: spectral norm of floor matrix
        Jfloor = np.sqrt(2.0) * sig[:, None] / (np.sqrt(N_J) * 2 * DELTA)
        floor_sv = np.linalg.svd(Jfloor, compute_uv=False)[0]
        n_resolved = int((sv > floor_sv).sum())

        # bootstrap leading eigenvectors (directions 1..min(n_resolved,3))
        ndir = max(1, min(n_resolved, 3))
        refs = [V[:, k].copy() for k in range(ndir)]
        boot = {k: [] for k in range(ndir)}
        for b in range(NB):
            pick = RNG.integers(0, N_J, N_J)
            Jb = build_J(pick)
            _, Vb = leading_eigvec(Jb)
            for k in range(ndir):
                boot[k].append(align_sign(Vb[:, k], refs[k]))
        for k in range(ndir):
            B = np.array(boot[k])
            mean = B.mean(0); lo_ci = np.percentile(B, 5, 0); hi_ci = np.percentile(B, 95, 0)
            for p, pm, pl, ph in zip(ACTIVE_SINGLE_NAMES, mean, lo_ci, hi_ci):
                records.append({"regime": name, "direction": k + 1, "resolved": k < n_resolved,
                                "parameter": p, "loading": float(pm),
                                "ci_lo": float(pl), "ci_hi": float(ph),
                                "ci_excludes_zero": bool(pl > 0 or ph < 0)})
        # top parameters of leading resolved direction (|loading|, CI excludes 0)
        lead = [r for r in records if r["regime"] == name and r["direction"] == 1]
        lead_sorted = sorted(lead, key=lambda r: -abs(r["loading"]))
        top = [f"{r['parameter']} {r['loading']:.2f} [{r['ci_lo']:.2f},{r['ci_hi']:.2f}]"
               for r in lead_sorted[:4]]
        summary_rows.append({"regime": name, "n_resolved": n_resolved,
                             "leading_dir_resolved": bool(n_resolved >= 1),
                             "top_loadings_dir1": "; ".join(top)})
        print(summary_rows[-1], flush=True)

    pd.DataFrame(records).to_csv(OUT / "sloppiness_loadings_bootstrap.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(OUT / "sloppiness_loadings_summary.csv", index=False)
    json.dump(summary_rows, open(OUT / "sloppiness_loadings_headline.json", "w"), indent=2)


if __name__ == "__main__":
    main()
