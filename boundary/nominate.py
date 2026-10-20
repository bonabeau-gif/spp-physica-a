"""Module D: boundary nomination (cheapest first).

D1 archive-pair mining: no new simulation. Over the corpus (which already carries
computed descriptors), find nearby elite pairs with a large descriptor change per unit
parameter distance and emit them as candidate segments.

D2 surrogate-gradient prescreen: no new simulation per candidate. Fit Matern-5/2 GPs
mapping u -> the leading descriptor PCs; the posterior-mean gradient is analytic, so a
large quasi-random sample can be ranked by |grad| for free, clustered, and handed to
the real oracle. The Matern-5/2 gradient is derived here (not available in sklearn):

    dk/du_j = -(5/3) sigma^2 (1 + sqrt5 r) exp(-sqrt5 r) (u_j - x_j) / l_j^2 ,

with r the ARD-scaled distance; the 1/r cancels, so it is smooth at r=0.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import qmc
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

from boundary import params as P
from boundary.adapters import standardize

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "results" / "expanded"
_SQRT5 = np.sqrt(5.0)


def load_corpus(space: P.Space):
    """Return (U unit (n,d), psi_tilde (n,m)) for the space's frozen corpus."""
    if space.name == "two":
        df = pd.read_csv(EXP / "multi_behavior_corpus_latent.csv")
        X = np.load(EXP / "multi_behavior_descriptors.npz")["X"]
    else:
        df = pd.read_csv(EXP / "single_behavior_corpus.csv")
        X = np.load(EXP / "single_behavior_descriptors.npz")["X"]
    phys = np.column_stack([df[P.PLAN_TO_CODE[n]].to_numpy() for n in space.param_order])
    U = np.array([P.to_unit(p, space) for p in phys])
    psi_t = np.array([standardize(x, space) for x in X])
    return U, psi_t


def d1_archive_pairs(U, psi_t, eps_pair=0.25, K=200, dedup=0.15, knn=5):
    """Nearby elite pairs ranked by descriptor change per unit parameter distance.

    A fixed radius eps_pair is too tight for a corpus that is sparse in high-dimensional
    parameter space (126 points in 14-D), so we pair each rule with its k nearest
    neighbours (keeping pairs local *relative to the data*), capped at eps_pair only when
    that still yields enough pairs. The distance cap still matters: without any locality
    the ratio is a chord across the whole space and is meaningless.
    """
    n = len(U)
    D = np.sqrt(((U[:, None, :] - U[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(D, np.inf)
    # adaptive cap: use eps_pair if it admits enough pairs, else the median kNN distance
    knn_d = np.sort(D, axis=1)[:, :knn]
    cap = eps_pair if (D < eps_pair).sum() >= 4 * K else float(np.median(knn_d))
    cand = []
    seen = set()
    for i in range(n):
        for j in np.argsort(D[i])[:knn]:
            a, b = (i, int(j)) if i < j else (int(j), i)
            if (a, b) in seen:
                continue
            du = D[a, b]
            if du < 1e-6 or du > cap:
                continue
            seen.add((a, b))
            ratio = np.linalg.norm(psi_t[a] - psi_t[b]) / du
            cand.append((ratio, a, b))
    cand.sort(reverse=True)
    accepted, mids = [], []
    for ratio, i, j in cand:
        mid = 0.5 * (U[i] + U[j])
        if all(np.linalg.norm(mid - m) >= dedup for m in mids):
            accepted.append({"u_start": U[i], "u_end": U[j], "score": float(ratio),
                             "nominator": "d1"})
            mids.append(mid)
            if len(accepted) >= K:
                break
    return accepted


def fit_pc_gps(U, psi_t, n_pc=10, length_scale=0.28, sigma2=1.0, noise=0.015):
    pca = PCA(n_components=min(n_pc, psi_t.shape[1])).fit(psi_t)
    Y = pca.transform(psi_t)
    d = U.shape[1]
    gps, ystd = [], []
    for c in range(Y.shape[1]):
        kern = (ConstantKernel(sigma2, "fixed") *
                Matern(length_scale=np.full(d, length_scale), length_scale_bounds="fixed", nu=2.5) +
                WhiteKernel(noise, "fixed"))
        gp = GaussianProcessRegressor(kernel=kern, alpha=1e-6, normalize_y=True, optimizer=None)
        gp.fit(U, Y[:, c])
        gps.append(gp)
        ystd.append(float(getattr(gp, "_y_train_std", 1.0)))
    weights = pca.explained_variance_ratio_[:Y.shape[1]]
    # fold PC weight + normalize_y std + dual coeffs into ONE per-training-point vector,
    # since grad is linear in alpha: beta_i = sum_c w_c * ystd_c * alpha_{c,i}
    Xtr = gps[0].X_train_
    beta = np.zeros(len(Xtr))
    for c, gp in enumerate(gps):
        beta += weights[c] * ystd[c] * gp.alpha_.ravel()
    return {"pca": pca, "gps": gps, "Xtr": Xtr, "beta": beta,
            "l": length_scale, "sigma2": sigma2, "weights": weights}


def ghat(model, Uq, batch=4000):
    """|| sum_c w_c grad mu_c(u) || over query points, batched."""
    Xtr = model["Xtr"]; beta = model["beta"]; l = model["l"]; sig2 = model["sigma2"]
    out = np.empty(len(Uq))
    for s in range(0, len(Uq), batch):
        q = Uq[s:s + batch]
        diff = q[:, None, :] - Xtr[None, :, :]            # (b, Ntr, d)
        r = np.sqrt((diff ** 2).sum(-1)) / l              # ARD distance
        coef = -(5.0 / 3.0) * sig2 * (1.0 + _SQRT5 * r) * np.exp(-_SQRT5 * r) / (l ** 2)
        G = (coef[:, :, None] * diff * beta[None, :, None]).sum(1)   # (b, d)
        out[s:s + batch] = np.linalg.norm(G, axis=1)
    return out


def d2_prescreen(model, d, n_sample=100000, top=1000, n_clusters=100, seed=0):
    Uq = qmc.Sobol(d=d, scramble=True, seed=seed).random(n_sample)
    g = ghat(model, Uq)
    idx = np.argsort(g)[::-1][:top]
    top_pts = Uq[idx]
    km = KMeans(n_clusters=min(n_clusters, len(top_pts)), n_init=10, random_state=0).fit(top_pts)
    centers = km.cluster_centers_
    return [{"u": np.clip(c, 0, 1), "nominator": "d2"} for c in centers], {"g_top": float(g[idx[0]]),
            "g_median_sample": float(np.median(g))}
