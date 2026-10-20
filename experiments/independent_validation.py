"""Independent representation audit for Sections 5 and S4/S7/S11.

The single-species test corpus has 72 rules and Sobol seed 20260825.
Models are selected using validation MSE, then validation trustworthiness.
Test metrics never participate in selection.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(variable, "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.manifold import trustworthiness
from sklearn.metrics import r2_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler

from build_single_corpus import _worker, make_design
from expanded_common import run_parallel
from spp_unified.learned import BehaviorAutoencoder, _knn_overlap
from spp_unified.parameters import ACTIVE_SINGLE_NAMES

DATA = ROOT / "results" / "expanded"
CLASSICAL = ["phi", "milling", "clustering", "nematic", "hexatic",
             "density_mode_peak", "drift", "chirality"]


def build_test(out, workers=4):
    """Write the complete test design and simulator seed for each rule."""
    design = make_design(n=72, seed=20260825)
    tasks = [(x, boundary, 202608250 + i) for i, (x, _, boundary) in enumerate(design)]
    results = run_parallel(_worker, tasks, max_workers=workers, desc="independent corpus")
    rows, descriptors = [], []
    for i, ((x, band, boundary), (summary, diagnostics, descriptor)) in enumerate(zip(design, results)):
        row = {"sample_id": i, "seed": tasks[i][2],
               "density_band": ["low", "medium", "high"][band], "boundary": boundary}
        row.update(dict(zip(ACTIVE_SINGLE_NAMES, x)))
        row.update(summary)
        row.update(diagnostics)
        rows.append(row)
        descriptors.append(descriptor)
    pd.DataFrame(rows).to_csv(out / "single_test_corpus.csv", index=False)
    np.savez_compressed(out / "single_test_descriptors.npz", X=np.asarray(descriptors, np.float32))


def scores(reference, embedding, reconstruction=None):
    k = min(10, (len(reference) - 1) // 2)
    result = {
        "trustworthiness": float(trustworthiness(reference, embedding, n_neighbors=k)),
        "knn_overlap": _knn_overlap(reference, embedding, k=k),
        "neighbors": k,
    }
    if reconstruction is not None:
        result["standardized_mse"] = float(np.mean((reference - reconstruction) ** 2))
    return result


def train_select(X, train, validation, latent_dim, hidden, epochs, patience, out, label):
    models, rows = [], []
    for seed in range(4):
        model = BehaviorAutoencoder(X.shape[1], latent_dim=latent_dim, hidden=hidden, seed=seed)
        result = model.fit(X, train, validation, max_epochs=epochs, patience=patience)
        rows.append({"seed": seed, "train_mse": result.train_mse,
                     "validation_mse": result.test_mse,
                     "validation_trustworthiness": result.trustworthiness,
                     "validation_knn_overlap": result.knn_overlap, "epochs": result.epochs})
        models.append(model)
    selected = min(range(4), key=lambda i: (rows[i]["validation_mse"],
                                          -rows[i]["validation_trustworthiness"]))
    pd.DataFrame(rows).to_csv(out / f"{label}_selection.csv", index=False)
    models[selected].save(str(out / f"{label}_selected.pt"))
    return models[selected], models


def stability(models, X):
    distances = [pdist(model.transform(X)) for model in models]
    values = [float(spearmanr(distances[i], distances[j]).statistic)
              for i in range(len(models)) for j in range(i + 1, len(models))]
    return {"median": float(np.median(values)), "min": min(values), "max": max(values),
            "pairwise_correlations": values}


def single_audit(out, dimensions):
    df = pd.read_csv(DATA / "single_behavior_corpus.csv")
    X = np.load(DATA / "single_behavior_descriptors.npz")["X"]
    test_df = pd.read_csv(out / "single_test_corpus.csv")
    Xt = np.load(out / "single_test_descriptors.npz")["X"]
    strata = df.density_band.astype(str) + "_" + df.boundary.astype(str)
    train, validation = next(StratifiedShuffleSplit(
        n_splits=1, test_size=.22, random_state=2026).split(X, strata))
    np.savez(out / "single_split.npz", train=train, validation=validation)
    report = {}
    for dim in dimensions:
        selected, models = train_select(X, train, validation, dim, (64, 24), 450, 50,
                                         out, f"single_{dim}d")
        standardized = (Xt - selected.mean) / selected.std
        zt = selected.transform(Xt)
        rec = (selected.reconstruct(Xt) - selected.mean) / selected.std
        report[f"autoencoder_{dim}d"] = {
            "selected_seed": selected.seed, **scores(standardized, zt, rec),
            "test_seed_stability": stability(models, Xt),
            "validation_seed_stability": stability(models, X[validation]),
        }
        np.savez_compressed(out / f"single_{dim}d_test_embedding.npz", Z=zt)
        if dim == 2:
            Xs = (X - selected.mean) / selected.std
            pca = PCA(n_components=2, random_state=0).fit(Xs[train])
            zp = pca.transform(standardized)
            report["descriptor_pca_2d"] = {
                **scores(standardized, zp, pca.inverse_transform(zp)),
                "train_variance_explained": float(pca.explained_variance_ratio_.sum())}
            scaler = StandardScaler().fit(df.loc[train, CLASSICAL])
            c = scaler.transform(df[CLASSICAL])
            ct = scaler.transform(test_df[CLASSICAL])
            cpca = PCA(n_components=2, random_state=0).fit(c[train])
            report["classical_pca_2d"] = scores(standardized, cpca.transform(ct))
            # Fit after fixing the representation. All 360 development rules are eligible.
            target = selected.transform(X)
            regressors = {
                "linear": LinearRegression(),
                "random_forest": RandomForestRegressor(n_estimators=400, random_state=0, n_jobs=1),
                "extra_trees": ExtraTreesRegressor(n_estimators=400, random_state=0, n_jobs=1),
                "hist_gradient_boosting": MultiOutputRegressor(
                    HistGradientBoostingRegressor(random_state=0)),
            }
            report["classical_recoverability"] = {}
            for name, regressor in regressors.items():
                prediction = regressor.fit(c, target).predict(ct)
                report["classical_recoverability"][name] = r2_score(
                    zt, prediction, multioutput="raw_values").tolist()
            archived = np.load(DATA / "single_behavior_latent.npz")["Z"]
            report["archive_coordinate_max_abs_difference"] = float(np.max(np.abs(target - archived)))
    return report


def multi_audit(out, dimensions):
    df = pd.read_csv(DATA / "multi_behavior_corpus_latent.csv")
    X = np.load(DATA / "multi_behavior_descriptors.npz")["X"]
    strata = df.source.astype(str)
    development, test = next(StratifiedShuffleSplit(
        n_splits=1, test_size=.20, random_state=20260825).split(X, strata))
    tr, va = next(StratifiedShuffleSplit(n_splits=1, test_size=.25, random_state=20260826)
                  .split(X[development], strata.iloc[development]))
    train, validation = development[tr], development[va]
    np.savez(out / "multi_split.npz", train=train, validation=validation, test=test)
    report = {"split_sizes": {"train": len(train), "validation": len(validation), "test": len(test)}}
    for dim in dimensions:
        selected, models = train_select(X, train, validation, dim, (80, 28), 500, 60,
                                         out, f"multi_{dim}d")
        standardized = (X[test] - selected.mean) / selected.std
        z = selected.transform(X[test])
        rec = (selected.reconstruct(X[test]) - selected.mean) / selected.std
        report[f"autoencoder_{dim}d"] = {
            "selected_seed": selected.seed, **scores(standardized, z, rec),
            "test_seed_stability": stability(models, X[test])}
    return report


def main():
    import torch
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "independent_validation")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--reuse-corpus", action="store_true")
    parser.add_argument("--kind", choices=["single", "multi", "both"], default="both")
    parser.add_argument("--dimensions", nargs="+", type=int, default=[2, 3, 4, 6])
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    report = {"protocol": {"sobol_seed": 20260825, "simulation_seed_base": 202608250,
                           "selection": "validation MSE, then validation trustworthiness",
                           "test_used_for_selection": False, "torch_version": torch.__version__}}
    if args.kind in ("single", "both"):
        if not args.reuse_corpus:
            build_test(out, args.workers)
        report["single"] = single_audit(out, args.dimensions)
    if args.kind in ("multi", "both"):
        report["multi"] = multi_audit(out, [d for d in args.dimensions if d in (2, 6)])
    (out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
