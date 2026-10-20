"""Assessment point #3: quantify what the learned behavior plane adds beyond the
classical eight-vector (phi, m, log(1+Gamma), S2, psi6, P, c_rho, rho).

Outputs (results/expanded/):
  learned_vs_classical_regression.csv  - linear & nonlinear CV R^2 of z1,z2 on the 8-vector
  learned_vs_classical_pairs.csv        - rule pairs close in all 8 measures but far in z
  dr_baseline_comparison.csv            - trustworthiness / kNN overlap for AE, PCA, UMAP,
                                          diffusion map, supervised-from-classical, w/ error bars
  learned_vs_classical_summary.json
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
OUT = ROOT / "results" / "expanded"

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA
from sklearn.manifold import trustworthiness, SpectralEmbedding
from sklearn.neighbors import NearestNeighbors

RNG = np.random.default_rng(2026)

EIGHT = [("phi", "phi", False), ("m", "milling", False), ("log1p_Gamma", "clustering", True),
         ("S2", "nematic", False), ("psi6", "hexatic", False), ("P", "density_mode_peak", False),
         ("c_rho", "local_density_cv", False), ("rho", "rho", False)]


def knn_overlap(X, Z, k=10):
    nx = NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(return_distance=False)[:, 1:]
    nz = NearestNeighbors(n_neighbors=k + 1).fit(Z).kneighbors(return_distance=False)[:, 1:]
    return float(np.mean([len(set(a) & set(b)) / k for a, b in zip(nx, nz)]))


def build_eight(df):
    cols = {}
    for name, src, islog in EIGHT:
        v = df[src].to_numpy(dtype=float)
        cols[name] = np.log1p(v) if islog else v
    E = np.column_stack([cols[n] for n, _, _ in EIGHT])
    return E, [n for n, _, _ in EIGHT]


def main():
    df = pd.read_csv(OUT / "single_behavior_corpus_latent.csv")
    Z01 = df[["learned_0", "learned_1"]].to_numpy()
    E, names = build_eight(df)
    # standardize the eight-vector
    Es = (E - E.mean(0)) / E.std(0)

    # --- 1. Regression of each latent axis on the eight-vector -------------
    reg_rows = []
    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    for axis in (0, 1):
        y = Z01[:, axis]
        lin = LinearRegression()
        yhat_lin = cross_val_predict(lin, Es, y, cv=kf)
        r2_lin = r2_score(y, yhat_lin)
        rf = RandomForestRegressor(n_estimators=400, max_depth=None, min_samples_leaf=3,
                                   random_state=0, n_jobs=4)
        yhat_rf = cross_val_predict(rf, Es, y, cv=kf)
        r2_rf = r2_score(y, yhat_rf)
        # in-sample linear for reference
        lin.fit(Es, y)
        reg_rows.append({"axis": f"z{axis+1}", "linear_cv_r2": r2_lin,
                         "rf_cv_r2": r2_rf, "linear_insample_r2": lin.score(Es, y),
                         "residual_std_linear": float(np.std(y - yhat_lin)),
                         "residual_std_rf": float(np.std(y - yhat_rf)),
                         "axis_std": float(np.std(y))})
    reg = pd.DataFrame(reg_rows)
    reg.to_csv(OUT / "learned_vs_classical_regression.csv", index=False)

    # --- 2. Pairs close in 8 classical measures but far in latent plane ----
    from scipy.spatial.distance import pdist, squareform
    d8 = squareform(pdist(Es))          # standardized euclidean in classical space
    dz = squareform(pdist(Z01))          # euclidean in the archived unit plane
    n = len(df)
    iu = np.triu_indices(n, k=1)
    d8u, dzu = d8[iu], dz[iu]
    # "close in 8" = below 10th percentile of classical distance; rank those by latent distance
    thr8 = np.quantile(d8u, 0.10)
    mask = d8u <= thr8
    order = np.argsort(-dzu[mask])
    ii, jj = iu[0][mask][order], iu[1][mask][order]
    pair_rows = []
    used = set()
    for a, b in zip(ii, jj):
        if a in used or b in used:
            continue
        used.add(a); used.add(b)
        row = {"id_a": int(df.loc[a, "sample_id"]), "id_b": int(df.loc[b, "sample_id"]),
               "classical_dist": float(d8[a, b]), "latent_dist": float(dz[a, b]),
               "z_a": (float(Z01[a, 0]), float(Z01[a, 1])), "z_b": (float(Z01[b, 0]), float(Z01[b, 1]))}
        for _, src, _ in EIGHT:
            row[f"{src}_a"] = float(df.loc[a, src]); row[f"{src}_b"] = float(df.loc[b, src])
        row["cluster_a"] = int(df.loc[a, "learned_cluster"]); row["cluster_b"] = int(df.loc[b, "learned_cluster"])
        pair_rows.append(row)
        if len(pair_rows) >= 12:
            break
    pd.DataFrame(pair_rows).to_csv(OUT / "learned_vs_classical_pairs.csv", index=False)

    # median latent distance among the classically-closest pairs vs global median
    med_close = float(np.median(dzu[mask]))
    med_all = float(np.median(dzu))

    # --- 3. Dimensionality-reduction baselines (trustworthiness / kNN) -----
    X = np.load(OUT / "single_behavior_descriptors.npz")["X"]
    L = np.load(OUT / "single_behavior_latent.npz")
    train, test = L["train"], L["test"]
    Zae = L["Z01"]  # autoencoder plane (already fit)
    mean = X[train].mean(0); sd = X[train].std(0); sd[sd < 1e-6] = 1
    Xs = (X - mean) / sd

    def boot_trust(Xtest, Ztest, B=300, k=10):
        base = float(trustworthiness(Xtest, Ztest, n_neighbors=k))
        vals = []
        m = len(Xtest)
        for _ in range(B):
            idx = RNG.integers(0, m, m)
            # need unique for neighbor structure; use subsample without replacement of 90%
        # bootstrap via 80% subsamples (trustworthiness needs distinct points)
        for _ in range(B):
            idx = RNG.choice(m, size=int(0.8 * m), replace=False)
            vals.append(float(trustworthiness(Xtest[idx], Ztest[idx], n_neighbors=k)))
        return base, float(np.mean(vals)), float(np.std(vals))

    results = []

    def add(nm, Zfull, stochastic_seeds=None):
        if stochastic_seeds is None:
            t_base, t_mean, t_sd = boot_trust(Xs[test], Zfull[test])
            ko = knn_overlap(Xs[test], Zfull[test], 10)
            results.append({"method": nm, "trustworthiness": t_base, "trust_boot_mean": t_mean,
                            "trust_boot_sd": t_sd, "knn_overlap": ko, "knn_sd": 0.0})

    # Autoencoder (as archived)
    add("autoencoder_2d", Zae)
    # PCA on standardized descriptor
    Zp = np.zeros((len(X), 2)); Zp[:] = PCA(2, random_state=0).fit(Xs[train]).transform(Xs)
    add("pca_2d", Zp)
    # Supervised-from-classical: 2D PCA of the standardized eight classical measures
    Zc = PCA(2, random_state=0).fit(Es[train]).transform(Es)
    add("classical_8vec_pca2d", Zc)
    # Diffusion map / Laplacian eigenmap (spectral embedding) on descriptor
    try:
        Zdm = SpectralEmbedding(n_components=2, affinity="nearest_neighbors",
                                n_neighbors=15, random_state=0).fit_transform(Xs)
        add("diffusion_map_2d", Zdm)
    except Exception as e:
        print("diffusion map failed:", e)
    # UMAP (stochastic) - multiple seeds
    try:
        import umap
        tvals, kvals = [], []
        for s in range(5):
            um = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=s)
            Zu = um.fit(Xs[train]).transform(Xs)
            tvals.append(float(trustworthiness(Xs[test], Zu[test], n_neighbors=10)))
            kvals.append(knn_overlap(Xs[test], Zu[test], 10))
        results.append({"method": "umap_2d", "trustworthiness": float(np.mean(tvals)),
                        "trust_boot_mean": float(np.mean(tvals)), "trust_boot_sd": float(np.std(tvals)),
                        "knn_overlap": float(np.mean(kvals)), "knn_sd": float(np.std(kvals))})
    except Exception as e:
        print("UMAP failed:", e)

    drdf = pd.DataFrame(results)
    drdf.to_csv(OUT / "dr_baseline_comparison.csv", index=False)

    summary = {
        "regression": reg.to_dict("records"),
        "median_latent_dist_classically_close_pairs": med_close,
        "median_latent_dist_all_pairs": med_all,
        "closeness_ratio": med_close / med_all,
        "n_pairs_reported": len(pair_rows),
        "dr_baselines": drdf.to_dict("records"),
    }
    with open(OUT / "learned_vs_classical_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
