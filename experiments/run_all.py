from __future__ import annotations
import argparse
import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from spp_unified.parameters import (
    ACTIVE_SINGLE_NAMES, EXPERIMENT_BOUNDS, TWO_SPECIES_EXTRA_BOUNDS,
    SingleParams, TwoSpeciesParams, SimulationConfig,
)
from spp_unified.model import (
    simulate_single, simulate_two_species, manuscript_pair_force, beta_from_adhesion,
)
from spp_unified.evaluation import evaluate_single, evaluate_two_species
from spp_unified.search import map_elites
from spp_unified.sensitivity import one_at_a_time

RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)


def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, default=float))


def active_from_row(row) -> np.ndarray:
    return np.array([row[n] for n in ACTIVE_SINGLE_NAMES], dtype=float)


def run_noise_sweep(cfg, quick=False):
    noise_values = np.linspace(0, 1, 11)
    seeds = range(4 if quick else 8)
    rows = []
    # Vicsek-like limit: no force or gradient, broad alignment neighborhood.
    base = SingleParams(rho=0.15, v0=0.05, inertia=0.3,
                        interaction_range_ratio=5.0, force_strength=0.0,
                        adhesion=0.0, alignment_range_ratio=8.0,
                        alignment_strength=3.5, gradient_strength=0.0)
    for eta in noise_values:
        p = replace(base, noise=float(eta))
        for seed in seeds:
            r = simulate_single(p, cfg.with_seed(1000 + seed)).summary
            rows.append({"noise": eta, "seed": seed, **r})
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "single_noise_sweep.csv", index=False)
    agg = df.groupby("noise").agg(phi=("phi", "mean"), phi_sd=("phi", "std"),
                                  nematic=("nematic", "mean"),
                                  clustering=("clustering", "mean")).reset_index()
    fig, ax = plt.subplots(figsize=(5.3, 3.8))
    ax.errorbar(agg.noise, agg.phi, yerr=agg.phi_sd, marker="o", capsize=3)
    ax.set_xlabel("noise intensity, eta")
    ax.set_ylabel("polar alignment")
    ax.set_ylim(0, 1.03)
    fig.tight_layout(); fig.savefig(FIGURES / "fig_noise_sweep.png", dpi=240); plt.close(fig)
    return df


def run_force_law_figure():
    r0, r1 = 0.1, 1.2
    r = np.linspace(2*r0, r1, 400)
    fig, ax = plt.subplots(figsize=(5.3, 3.8))
    for B in (-0.6, 0.0, 0.6):
        beta = beta_from_adhesion(B)
        f = manuscript_pair_force(r, r0, r1, beta, 50.0)
        ax.plot(r / r0, f, label=f"B={B:g}")
    ax.axhline(0, linewidth=0.8)
    ax.set_xlabel("center distance / r0")
    ax.set_ylabel("pair-force scalar")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(FIGURES / "fig_force_law.png", dpi=240); plt.close(fig)


def simple_de(objective_name, cfg, quick=False, seed=0):
    rng = np.random.default_rng(seed)
    pop_n = 12 if quick else 16
    generations = 5 if quick else 8
    F, CR = 0.45, 0.8
    dim = len(ACTIVE_SINGLE_NAMES)
    U = rng.random((pop_n, dim))
    eval_count = 0

    def eval_u(u, k):
        nonlocal eval_count
        x = EXPERIMENT_BOUNDS[:, 0] + u * (EXPERIMENT_BOUNDS[:, 1] - EXPERIMENT_BOUNDS[:, 0])
        p = SingleParams.from_active_vector(x)
        r = simulate_single(p, cfg.with_seed(seed * 100000 + k + 17)).summary
        eval_count += 1
        if objective_name == "alignment":
            val = r["phi"]
        elif objective_name == "milling":
            val = r["milling"] * (1.0 - r["phi"])
        elif objective_name == "clustering":
            val = min(r["clustering"], 10.0) / 10.0
        else:
            raise ValueError(objective_name)
        return val, r

    vals, metas = [], []
    for i, u in enumerate(U):
        v, m = eval_u(u, i); vals.append(v); metas.append(m)
    vals = np.array(vals)
    history = []
    for g in range(generations + 1):
        bi = int(np.argmax(vals))
        history.append({"generation": g, "evaluations": eval_count,
                        "best_fitness": float(vals[bi]), **metas[bi]})
        if g == generations:
            break
        newU, newV, newM = U.copy(), vals.copy(), list(metas)
        for i in range(pop_n):
            ids = [j for j in range(pop_n) if j != i]
            a, b, c = rng.choice(ids, 3, replace=False)
            mutant = np.clip(U[a] + F * (U[b] - U[c]), 0, 1)
            mask = rng.random(dim) < CR
            mask[rng.integers(dim)] = True
            trial = np.where(mask, mutant, U[i])
            v, m = eval_u(trial, 1000 * (g + 1) + i)
            if v >= vals[i]:
                newU[i], newV[i], newM[i] = trial, v, m
        U, vals, metas = newU, newV, newM
    bi = int(np.argmax(vals))
    xbest = EXPERIMENT_BOUNDS[:, 0] + U[bi] * (EXPERIMENT_BOUNDS[:, 1] - EXPERIMENT_BOUNDS[:, 0])
    out = {"objective": objective_name, "best_fitness": float(vals[bi]),
           "best_parameters": dict(zip(ACTIVE_SINGLE_NAMES, xbest)),
           "best_summary": metas[bi], "evaluations": eval_count}
    pd.DataFrame(history).to_csv(RESULTS / f"de_{objective_name}_history.csv", index=False)
    save_json(RESULTS / f"de_{objective_name}_best.json", out)
    return out


def run_single_qd(search_cfg, quick=False):
    evaluations = 260 if quick else 700
    dims = (10, 10) if quick else (12, 12)

    def eval_mill(x, seed):
        r = simulate_single(SingleParams.from_active_vector(x), search_cfg.with_seed(seed)).summary
        return r["temporal_stability"], (r["phi"], r["milling"]), r

    mill = map_elites(eval_mill, ACTIVE_SINGLE_NAMES, EXPERIMENT_BOUNDS,
                      [(0, 1), (0, 0.5)], dims=dims, evaluations=evaluations,
                      init_random=70 if quick else 160, seed=11)
    mill_df = mill.elites
    mill_df.to_csv(RESULTS / "single_qd_alignment_milling.csv", index=False)
    mill.history.to_csv(RESULTS / "single_qd_alignment_milling_history.csv", index=False)

    periodic_cfg = replace(search_cfg, boundary="periodic")
    def eval_clust(x, seed):
        r = simulate_single(SingleParams.from_active_vector(x), periodic_cfg.with_seed(seed)).summary
        return r["temporal_stability"], (r["phi"], r["clustering"]), r

    clust = map_elites(eval_clust, ACTIVE_SINGLE_NAMES, EXPERIMENT_BOUNDS,
                       [(0, 1), (0, 8.0)], dims=dims, evaluations=evaluations,
                       init_random=70 if quick else 160, seed=23)
    clust_df = clust.elites
    clust_df.to_csv(RESULTS / "single_qd_alignment_clustering.csv", index=False)
    clust.history.to_csv(RESULTS / "single_qd_alignment_clustering_history.csv", index=False)
    return mill_df, clust_df, dims


def plot_archive(df, dims, x_range, y_range, value, xlabel, ylabel, filename):
    arr = np.full(dims, np.nan)
    for _, row in df.iterrows():
        arr[int(row.cell_0), int(row.cell_1)] = row[value]
    fig, ax = plt.subplots(figsize=(5.1, 4.2))
    im = ax.imshow(arr.T, origin="lower", aspect="auto", interpolation="nearest",
                   extent=(x_range[0], x_range[1], y_range[0], y_range[1]))
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    fig.colorbar(im, ax=ax, label=value.replace("_", " "))
    fig.tight_layout(); fig.savefig(FIGURES / filename, dpi=240); plt.close(fig)


def select_single_representatives(mill_df, clust_df):
    reps = {}
    reps["aligned"] = mill_df.loc[mill_df.phi.idxmax()]
    score_m = mill_df.milling * (1.0 - mill_df.phi)
    reps["milling"] = mill_df.loc[score_m.idxmax()]
    reps["clustered"] = clust_df.loc[clust_df.clustering.idxmax()]
    score_d = clust_df.phi + clust_df.milling + np.minimum(clust_df.clustering, 1.0)
    reps["disordered"] = clust_df.loc[score_d.idxmin()]
    return reps


def validate_single_representatives(reps, quick=False):
    seeds = list(range(20 if not quick else 8))
    rows = []
    snapshots = []
    names = []
    for name, row in reps.items():
        p = SingleParams.from_active_vector(active_from_row(row))
        boundary = "reflecting" if name == "milling" else "periodic"
        cfg = SimulationConfig(steps=480 if not quick else 300,
                               sample_start=240 if not quick else 150,
                               sample_every=2, boundary=boundary)
        agg = evaluate_single(p, cfg, seeds)
        rows.append({"state": name, **agg, **{k: getattr(p, k) for k in ACTIVE_SINGLE_NAMES}})
        snapshots.append(simulate_single(p, cfg.with_seed(700 + len(rows)), record_states=True, state_every=12))
        names.append(name)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "single_representative_validation.csv", index=False)

    fig, axes = plt.subplots(1, len(snapshots), figsize=(3.25 * len(snapshots), 3.25))
    for label, res, ax in zip(names, snapshots, np.atleast_1d(axes)):
        x, th = res.states[-1]
        ax.quiver(x[:,0], x[:,1], np.cos(th), np.sin(th), scale=25, width=.005, pivot="mid")
        ax.text(0.03, 0.96, label, transform=ax.transAxes, va="top", fontsize=9)
        ax.set_xlim(0,4); ax.set_ylim(0,4); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout(); fig.savefig(FIGURES / "fig_single_representatives.png", dpi=240); plt.close(fig)
    return df


def run_finite_size(reps, quick=False):
    sizes = [3.2, 4.0, 4.8]
    seeds = range(3 if quick else 6)
    rows = []
    for name in ("aligned", "milling", "clustered"):
        p = SingleParams.from_active_vector(active_from_row(reps[name]))
        boundary = "reflecting" if name == "milling" else "periodic"
        for S in sizes:
            cfg = SimulationConfig(S=S, steps=300 if quick else 420,
                                   sample_start=150 if quick else 210,
                                   boundary=boundary)
            for seed in seeds:
                r = simulate_single(p, cfg.with_seed(3000 + seed)).summary
                rows.append({"state": name, "S": S, "seed": seed, **r})
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "single_finite_size.csv", index=False)
    metric = {"aligned": "phi", "milling": "milling", "clustered": "clustering"}
    fig, ax = plt.subplots(figsize=(5.3, 3.8))
    for state in metric:
        g = df[df.state == state].groupby("S")[metric[state]].agg(["mean", "std"]).reset_index()
        ax.errorbar(g.S, g["mean"], yerr=g["std"], marker="o", capsize=3, label=state)
    ax.set_xlabel("arena size S at fixed rho")
    ax.set_ylabel("state-specific order parameter")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(FIGURES / "fig_finite_size.png", dpi=240); plt.close(fig)
    return df


def run_sensitivity(reps, quick=False):
    all_df = []
    for state in ("aligned", "milling"):
        row = reps[state]
        center = active_from_row(row)
        cfg = SimulationConfig(steps=220 if quick else 300,
                               sample_start=110 if quick else 150,
                               boundary="reflecting" if state == "milling" else "periodic")
        df = one_at_a_time(center, cfg, seeds=(41,42) if quick else (41,42,43),
                           points=5 if quick else 7)
        df.insert(0, "state", state)
        all_df.append(df)
    out = pd.concat(all_df, ignore_index=True)
    out.to_csv(RESULTS / "single_oat_sensitivity.csv", index=False)

    # Plot standardized ranges for the state-defining metric.
    rows = []
    for state, metric in (("aligned", "phi"), ("milling", "milling")):
        sub = out[out.state == state]
        for p, g in sub.groupby("parameter"):
            rows.append({"state": state, "parameter": p,
                         "range": float(g[metric].max() - g[metric].min())})
    imp = pd.DataFrame(rows)
    imp.to_csv(RESULTS / "single_oat_importance.csv", index=False)
    piv = imp.pivot(index="parameter", columns="state", values="range").fillna(0)
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    im = ax.imshow(piv.to_numpy(), aspect="auto")
    ax.set_xticks(range(len(piv.columns)), piv.columns)
    ax.set_yticks(range(len(piv.index)), piv.index)
    fig.colorbar(im, ax=ax, label="range of order parameter")
    fig.tight_layout(); fig.savefig(FIGURES / "fig_sensitivity.png", dpi=240); plt.close(fig)
    return out


def two_nonreciprocal_from_vector(x):
    n = len(ACTIVE_SINGLE_NAMES)
    b = SingleParams.from_active_vector(x[:n])
    return TwoSpeciesParams(base=b, fraction_B=x[n],
                            alignment_symmetric=x[n+1], alignment_antisymmetric=x[n+2],
                            force_symmetric=x[n+3], force_antisymmetric=x[n+4])


def two_reciprocal_from_vector(x):
    n = len(ACTIVE_SINGLE_NAMES)
    b = SingleParams.from_active_vector(x[:n])
    return TwoSpeciesParams(base=b, fraction_B=x[n],
                            alignment_symmetric=x[n+1], alignment_antisymmetric=0.0,
                            force_symmetric=x[n+2], force_antisymmetric=0.0)


def run_multi_qd(search_cfg, quick=False):
    evaluations = 320 if quick else 850
    dims = (10,10) if quick else (12,12)
    non_bounds = np.vstack([EXPERIMENT_BOUNDS, TWO_SPECIES_EXTRA_BOUNDS])
    reciprocal_extra = np.array([[.2,.8],[-1.0,1.5],[-1.0,1.0]])
    rec_bounds = np.vstack([EXPERIMENT_BOUNDS, reciprocal_extra])
    non_names = ACTIVE_SINGLE_NAMES + ("fraction_B", "alignment_symmetric", "alignment_antisymmetric",
                                        "force_symmetric", "force_antisymmetric")
    rec_names = ACTIVE_SINGLE_NAMES + ("fraction_B", "alignment_symmetric", "force_symmetric")

    def eval_non(x, seed):
        r = simulate_two_species(two_nonreciprocal_from_vector(x), search_cfg.with_seed(seed)).summary
        return r["temporal_stability"], (r["drift"], r["chirality"]), r
    non = map_elites(eval_non, non_names, non_bounds, [(0,1),(0,1)], dims=dims,
                     evaluations=evaluations, init_random=90 if quick else 200, seed=31)
    non_df = non.elites
    non_df.to_csv(RESULTS / "multi_qd_nonreciprocal.csv", index=False)
    non.history.to_csv(RESULTS / "multi_qd_nonreciprocal_history.csv", index=False)

    def eval_rec(x, seed):
        r = simulate_two_species(two_reciprocal_from_vector(x), search_cfg.with_seed(seed)).summary
        return r["temporal_stability"], (r["drift"], r["chirality"]), r
    rec = map_elites(eval_rec, rec_names, rec_bounds, [(0,1),(0,1)], dims=dims,
                     evaluations=evaluations, init_random=90 if quick else 200, seed=37)
    rec_df = rec.elites
    rec_df.to_csv(RESULTS / "multi_qd_reciprocal.csv", index=False)
    rec.history.to_csv(RESULTS / "multi_qd_reciprocal_history.csv", index=False)
    return non_df, rec_df, dims


def multi_params_from_row(row):
    x = np.array([row[n] for n in ACTIVE_SINGLE_NAMES] +
                 [row["fraction_B"], row["alignment_symmetric"], row["alignment_antisymmetric"],
                  row["force_symmetric"], row["force_antisymmetric"]])
    return two_nonreciprocal_from_vector(x)


def validate_multi(non_df, quick=False):
    non_df = non_df.copy()
    non_df["chiral_rank"] = non_df.chirality * (0.4 + non_df.rotation_r2 if "rotation_r2" in non_df else 1.0)
    reps = {
        "travelling": non_df.loc[non_df.drift.idxmax()],
        "chiral": non_df.loc[non_df.chirality.idxmax()],
        "segregated": non_df.loc[non_df.segregation.idxmax()],
    }
    seeds = list(range(8 if quick else 20))
    rows = []
    snapshots, labels = [], []
    cfg = SimulationConfig(steps=360 if quick else 600,
                           sample_start=180 if quick else 300,
                           sample_every=2, boundary="periodic")
    for name, row in reps.items():
        p = multi_params_from_row(row)
        agg = evaluate_two_species(p, cfg, seeds)
        rows.append({"state": name, "condition": "nonreciprocal", **agg})
        # Matched reciprocal control preserves all symmetric parameters.
        pc = replace(p, alignment_antisymmetric=0.0, force_antisymmetric=0.0)
        aggc = evaluate_two_species(pc, cfg, seeds)
        rows.append({"state": name, "condition": "matched_reciprocal", **aggc})
        snapshots.append(simulate_two_species(p, cfg.with_seed(900 + len(rows)), record_states=True, state_every=15))
        labels.append(name)
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "multi_representative_validation.csv", index=False)

    fig, axes = plt.subplots(1, len(snapshots), figsize=(3.4*len(snapshots), 3.4))
    for label, res, ax in zip(labels, snapshots, np.atleast_1d(axes)):
        x, th = res.states[-1]
        ax.quiver(x[:,0], x[:,1], np.cos(th), np.sin(th), res.species,
                  scale=25, width=.005, pivot="mid")
        ax.text(0.03, 0.96, label, transform=ax.transAxes, va="top", fontsize=9)
        ax.set_xlim(0,4); ax.set_ylim(0,4); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout(); fig.savefig(FIGURES / "fig_multi_representatives.png", dpi=240); plt.close(fig)
    return out


def plot_method_summary(de_results, single_validation, mill_df, clust_df):
    qd = {
        "alignment": float(max(mill_df.phi.max(), clust_df.phi.max())),
        "milling": float((mill_df.milling * (1-mill_df.phi)).max()),
        "clustering": float(clust_df.clustering.max()/10.0),
    }
    rows = []
    for k, d in de_results.items():
        rows.append({"objective": k, "method": "DE", "best": d["best_fitness"]})
        rows.append({"objective": k, "method": "QD archive", "best": qd[k]})
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "de_vs_qd.csv", index=False)
    piv = df.pivot(index="objective", columns="method", values="best")
    ax = piv.plot(kind="bar", figsize=(5.3,3.8), rot=0)
    ax.set_ylabel("best objective value")
    ax.legend(frameon=False)
    fig = ax.get_figure(); fig.tight_layout(); fig.savefig(FIGURES / "fig_de_vs_qd.png", dpi=240); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    start = time.time()
    search_steps = 120 if args.quick else 180
    cfg_periodic = SimulationConfig(steps=search_steps, sample_start=search_steps//2,
                                    sample_every=2, boundary="periodic")
    cfg_reflect = replace(cfg_periodic, boundary="reflecting")

    run_force_law_figure()
    run_noise_sweep(cfg_periodic, args.quick)

    de_results = {
        "alignment": simple_de("alignment", cfg_periodic, args.quick, 1),
        "milling": simple_de("milling", cfg_reflect, args.quick, 2),
        "clustering": simple_de("clustering", cfg_periodic, args.quick, 3),
    }

    mill_df, clust_df, single_dims = run_single_qd(cfg_reflect, args.quick)
    plot_archive(mill_df, single_dims, (0,1), (0,.5), "objective",
                 "polar alignment", "milling", "fig_archive_alignment_milling.png")
    plot_archive(clust_df, single_dims, (0,1), (0,8), "objective",
                 "polar alignment", "clustering", "fig_archive_alignment_clustering.png")
    reps = select_single_representatives(mill_df, clust_df)
    single_validation = validate_single_representatives(reps, args.quick)
    run_finite_size(reps, args.quick)
    run_sensitivity(reps, args.quick)
    plot_method_summary(de_results, single_validation, mill_df, clust_df)

    multi_cfg = SimulationConfig(steps=140 if args.quick else 220,
                                 sample_start=70 if args.quick else 110,
                                 sample_every=2, boundary="periodic")
    non_df, rec_df, multi_dims = run_multi_qd(multi_cfg, args.quick)
    plot_archive(non_df, multi_dims, (0,1), (0,1), "objective",
                 "net center-of-mass drift", "chiral rotation score",
                 "fig_archive_nonreciprocal.png")
    plot_archive(rec_df, multi_dims, (0,1), (0,1), "objective",
                 "net center-of-mass drift", "chiral rotation score",
                 "fig_archive_reciprocal.png")
    multi_validation = validate_multi(non_df, args.quick)

    summary = {
        "quick": args.quick,
        "elapsed_seconds": time.time() - start,
        "single_alignment_milling_elites": int(len(mill_df)),
        "single_alignment_clustering_elites": int(len(clust_df)),
        "multi_nonreciprocal_elites": int(len(non_df)),
        "multi_reciprocal_elites": int(len(rec_df)),
        "single_max_phi": float(max(mill_df.phi.max(), clust_df.phi.max())),
        "single_max_milling": float(mill_df.milling.max()),
        "single_max_clustering": float(clust_df.clustering.max()),
        "multi_max_drift_nonreciprocal": float(non_df.drift.max()),
        "multi_max_chirality_nonreciprocal": float(non_df.chirality.max()),
        "multi_max_drift_reciprocal": float(rec_df.drift.max()),
        "multi_max_chirality_reciprocal": float(rec_df.chirality.max()),
    }
    save_json(RESULTS / "experiment_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
