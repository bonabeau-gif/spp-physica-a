"""Assessment v2 smaller item: replace the LINEAR classical-measure map in
§5.3 with a NONLINEAR one (UMAP of the eight measures), and evaluate every 2-D
map against BOTH the 51-D descriptor and the 8-vector, to acknowledge that
judging maps only against the descriptor gives the descriptor-trained
autoencoder a built-in advantage.
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
from sklearn.decomposition import PCA
from sklearn.manifold import trustworthiness
import umap

EIGHT = [("phi", False), ("milling", False), ("clustering", True), ("nematic", False),
         ("hexatic", False), ("density_mode_peak", False), ("local_density_cv", False), ("rho", False)]
RNG = np.random.default_rng(3)


def main():
    df = pd.read_csv(OUT / "single_behavior_corpus_latent.csv")
    E = np.column_stack([np.log1p(df[s].to_numpy()) if lg else df[s].to_numpy() for s, lg in EIGHT])
    Es = (E - E.mean(0)) / E.std(0)
    X = np.load(OUT / "single_behavior_descriptors.npz")["X"]
    L = np.load(OUT / "single_behavior_latent.npz")
    train, test = L["train"], L["test"]; Zae = L["Z01"]
    mean = X[train].mean(0); sd = X[train].std(0); sd[sd < 1e-6] = 1
    Xs = (X - mean) / sd

    def boot_trust(ref_test, Z_test, B=300, k=10):
        base = float(trustworthiness(ref_test, Z_test, n_neighbors=k))
        vals = [float(trustworthiness(ref_test[idx], Z_test[idx], n_neighbors=k))
                for idx in (RNG.choice(len(ref_test), int(0.8 * len(ref_test)), replace=False) for _ in range(B))]
        return base, float(np.std(vals))

    # embeddings
    Zc_pca = PCA(2, random_state=0).fit(Es[train]).transform(Es)
    tvals, kv = [], []
    for s in range(5):
        Zc_um = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=s).fit(Es[train]).transform(Es)
        tvals.append(float(trustworthiness(Xs[test], Zc_um[test], n_neighbors=10)))
    # use last umap embedding for the vs-8vector eval and for saving coords
    Zc_um_last = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=0).fit(Es[train]).transform(Es)

    rows = []
    # vs 51-D descriptor
    for name, Z in [("autoencoder_2d", Zae), ("classical_8vec_pca2d", Zc_pca),
                    ("classical_8vec_umap2d", Zc_um_last)]:
        if name == "classical_8vec_umap2d":
            t, tsd = float(np.mean(tvals)), float(np.std(tvals))
        else:
            t, tsd = boot_trust(Xs[test], Z[test])
        rows.append({"reference": "descriptor_51d", "method": name, "trustworthiness": t, "trust_sd": tsd})
    # vs 8-vector (fair to the classical maps)
    for name, Z in [("autoencoder_2d", Zae), ("classical_8vec_pca2d", Zc_pca),
                    ("classical_8vec_umap2d", Zc_um_last)]:
        t, tsd = boot_trust(Es[test], Z[test])
        rows.append({"reference": "classical_8vec", "method": name, "trustworthiness": t, "trust_sd": tsd})

    out = pd.DataFrame(rows)
    out.to_csv(OUT / "classical_nonlinear_map.csv", index=False)
    print(out.round(3).to_string(index=False))
    # save umap coords for the figure
    np.savez_compressed(OUT / "classical_map_coords.npz", pca=Zc_pca, umap=Zc_um_last)
    json.dump(out.to_dict("records"), open(OUT / "classical_nonlinear_map_headline.json", "w"), indent=2)


if __name__ == "__main__":
    main()
