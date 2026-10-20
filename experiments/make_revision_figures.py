"""Figures for the revision analyses:
  fig_learned_vs_classical.png   - assessment #3
  fig_champion_per_seed.png      - assessment #7
  fig_chirality_decay.png        - assessment #10
"""
from __future__ import annotations
import ast
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "expanded"
OUT.mkdir(parents=True, exist_ok=True)
E = ROOT / "results" / "expanded"
plt.rcParams.update({"font.size": 9, "axes.splines.top" if False else "axes.grid": False})


def fig_learned_vs_classical():
    reg = pd.read_csv(E / "learned_vs_classical_regression.csv")
    dr = pd.read_csv(E / "dr_baseline_comparison.csv")
    df = pd.read_csv(E / "single_behavior_corpus_latent.csv")
    pairs = pd.read_csv(E / "learned_vs_classical_pairs.csv")

    fig, ax = plt.subplots(1, 3, figsize=(11.5, 3.7))

    # (a) DR baseline trustworthiness vs the 51-D descriptor, with error bars
    order = ["autoencoder_2d", "umap_2d", "diffusion_map_2d", "pca_2d", "classical_8vec_pca2d"]
    labels = {"autoencoder_2d": "autoencoder", "umap_2d": "UMAP", "diffusion_map_2d": "diffusion map",
              "pca_2d": "PCA", "classical_8vec_pca2d": "classical map\n(PCA)",
              "classical_8vec_umap2d": "classical map\n(UMAP)"}
    d = dr.set_index("method").loc[[m for m in order if m in dr.method.values]]
    vals = list(d["trustworthiness"]); errs = list(d["trust_boot_sd"]); labs = list(d.index)
    # insert the nonlinear classical map from classical_nonlinear_map.csv
    cnm_path = E / "classical_nonlinear_map.csv"
    if cnm_path.exists():
        cnm = pd.read_csv(cnm_path)
        row = cnm[(cnm.reference == "descriptor_51d") & (cnm.method == "classical_8vec_umap2d")]
        if len(row):
            vals.append(float(row.trustworthiness.iloc[0])); errs.append(float(row.trust_sd.iloc[0]))
            labs.append("classical_8vec_umap2d")
    xs = np.arange(len(vals))
    colors = ["#2c7fb8", "#41b6c4", "#7fcdbb", "#c7a4cf", "#d95f5f", "#e8896b"]
    ax[0].bar(xs, vals, yerr=errs, capsize=3, color=colors[:len(vals)])
    ax[0].set_xticks(xs); ax[0].set_xticklabels([labels[m] for m in labs], rotation=25, ha="right", fontsize=7.0)
    ax[0].set_ylabel("trustworthiness vs 51-D descriptor")
    ax[0].set_ylim(0.70, 0.93)
    ax[0].set_title("(a) neighborhood preservation", fontsize=9, loc="left")

    # (b) regression R^2 of latent axes on the classical 8-vector
    axb = ax[1]
    w = 0.35
    xs2 = np.arange(len(reg))
    axb.bar(xs2 - w/2, reg["linear_cv_r2"], w, label="linear (CV)", color="#8aa8c8")
    axb.bar(xs2 + w/2, reg["rf_cv_r2"], w, label="random forest (CV)", color="#2c5f8a")
    for i, (_, r) in enumerate(reg.iterrows()):
        axb.text(i, max(r["rf_cv_r2"], r["linear_cv_r2"]) + 0.02,
                 f"resid\n{(1-r['rf_cv_r2'])*100:.0f}%", ha="center", fontsize=7)
    axb.set_xticks(xs2); axb.set_xticklabels(["$z_1$", "$z_2$"])
    axb.set_ylabel("$R^2$ predicting axis from 8-vector")
    axb.set_ylim(0, 1.05); axb.legend(frameon=False, fontsize=7.5, loc="lower right")
    axb.set_title("(b) recoverable from classical measures?", fontsize=9, loc="left")

    # (c) conflated pairs in the learned plane
    axc = ax[2]
    axc.scatter(df["learned_0"], df["learned_1"], s=6, c="#cccccc", edgecolors="none")
    for _, pr in pairs.head(6).iterrows():
        za = ast.literal_eval(pr["z_a"]) if isinstance(pr["z_a"], str) else pr["z_a"]
        zb = ast.literal_eval(pr["z_b"]) if isinstance(pr["z_b"], str) else pr["z_b"]
        axc.plot([za[0], zb[0]], [za[1], zb[1]], "-", color="#d95f5f", lw=1.1)
        axc.scatter([za[0], zb[0]], [za[1], zb[1]], s=18, c="#b2182b", zorder=3)
    axc.set_xlabel("learned measure 1"); axc.set_ylabel("learned measure 2")
    axc.set_title("(c) pairs conflated by classical measures", fontsize=9, loc="left")

    fig.tight_layout()
    fig.savefig(OUT / "fig_learned_vs_classical.png", dpi=240)
    plt.close(fig)
    print("wrote fig_learned_vs_classical.png")


def fig_champion_per_seed():
    ps = pd.read_csv(E / "champion_per_seed.csv")
    summ = pd.read_csv(E / "champion_per_seed_summary.csv")
    order = summ.sort_values("mean")["objective"].tolist()
    flagged = {"milling_vortex", "nematic_lanes", "phase_separation", "chiral_rotation"}
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    rng = np.random.default_rng(0)
    for i, obj in enumerate(order):
        v = ps[ps.objective == obj]["fitness"].to_numpy()
        jit = rng.normal(0, 0.06, len(v))
        col = "#b2182b" if obj in flagged else "#4575b4"
        ax.scatter(np.full(len(v), i) + jit, v, s=16, color=col, alpha=0.7, edgecolors="none")
        ax.plot([i - 0.28, i + 0.28], [np.median(v)] * 2, color="k", lw=1.6)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([o.replace("_", "\n") for o in order], fontsize=7.5)
    ax.set_ylabel("per-seed champion fitness (20 seeds)")
    ax.set_ylim(-0.03, 1.05)
    ax.set_title("Per-seed champion fitness; red = bimodal / seed-fragile objectives (black bar = median)",
                 fontsize=8.5, loc="left")
    fig.tight_layout(); fig.savefig(OUT / "fig_champion_per_seed.png", dpi=240); plt.close(fig)
    print("wrote fig_champion_per_seed.png")


def fig_chirality_decay():
    ts = pd.read_csv(E / "chirality_decay_timeseries.csv")
    nd_path = E / "chirality_decay_ndependence_v2.csv"
    nd = pd.read_csv(nd_path if nd_path.exists() else E / "chirality_decay_ndependence.csv")
    fig, ax = plt.subplots(1, 3, figsize=(11.5, 3.6))
    ax[0].plot(ts.step, ts.abs_omega_mean, color="#2166ac")
    ax[0].fill_between(ts.step, ts.abs_omega_mean - ts.abs_omega_sd, ts.abs_omega_mean + ts.abs_omega_sd,
                       color="#2166ac", alpha=0.2)
    ax[0].set_xlabel("simulation step"); ax[0].set_ylabel(r"$|\omega|(t)$ (rad/step)")
    ax[0].set_ylim(0, None); ax[0].set_title("(a) instantaneous rotation rate", fontsize=9, loc="left")

    ax[1].plot(ts.step, ts.r2_mean, color="#1a9850", label=r"local $R^2$")
    ax[1].plot(ts.step, ts.phi_mean, color="#762a83", label=r"$\phi$")
    ax[1].set_xlabel("simulation step"); ax[1].set_ylabel("value"); ax[1].set_ylim(0, 1.05)
    ax[1].legend(frameon=False, fontsize=8); ax[1].set_title("(b) local linearity and polarization", fontsize=9, loc="left")

    ax[2].errorbar(nd.N, nd.abs_omega_median, yerr=nd.abs_omega_sd, marker="o", capsize=3, color="#b35806")
    ax[2].set_xlabel("particle number $N$"); ax[2].set_ylabel(r"median $|\omega|$ (rad/step)")
    ax[2].set_ylim(0, None); ax[2].set_title("(c) N-dependence of rotation", fontsize=9, loc="left")
    fig.tight_layout(); fig.savefig(OUT / "fig_chirality_decay.png", dpi=240); plt.close(fig)
    print("wrote fig_chirality_decay.png")


if __name__ == "__main__":
    fig_learned_vs_classical()
    fig_champion_per_seed()
    fig_chirality_decay()
