"""Regenerate the 11 main and 19 supplementary figures from bundled results."""
from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "experiments")]
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from spp_unified import SingleParams, SimulationConfig, simulate_single
from spp_unified.parameters import ACTIVE_SINGLE_NAMES

FIG = ROOT / "figures"
OUT = FIG / "reproduced"
DATA = ROOT / "results"
EXP = DATA / "expanded"
REPORT = []


def save(fig, name):
    for extension in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{extension}", dpi=220, bbox_inches="tight")
    plt.close(fig)
    REPORT.append({"figure": name, "source": "bundled numerical results", "generated": True})


def copy_figure(source, name):
    shutil.copy2(FIG / source, OUT / f"{name}.png")
    REPORT.append({"figure": name, "source": f"figures/{source}", "generated": True})


def call(module, *functions):
    mod = importlib.import_module(module)
    for name in functions:
        print(f"Plotting {module}.{name}", flush=True)
        getattr(mod, name)()


def model_diagram():
    fig, ax = plt.subplots(figsize=(6, 5))
    for radius, color, label in ((1.7, "#3574a8", r"$r_a$"), (1.1, "#ad5441", r"$r_1$")):
        ax.add_patch(Circle((0, 0), radius, fill=False, linestyle="--", color=color))
        ax.text(radius / 1.4, radius / 1.4, label, color=color)
    ax.add_patch(Circle((0, 0), .15, color="#303744"))
    for x, y, angle in [(.7, .4, .3), (-.5, .6, .8), (-1.3, -.5, -.2), (1.4, -.8, .5)]:
        ax.add_patch(Circle((x, y), .12, color="#929eaa"))
        ax.arrow(x, y, .32*np.cos(angle), .32*np.sin(angle), width=.015, color="#303744")
    for (dx, dy), color, label in [((.9, 0), "#303744", r"$\epsilon u_i$"),
                                   ((.45, .7), "#3574a8", r"$\alpha a_i$"),
                                   ((-.55, -.45), "#ad5441", r"$f_0 F_i$"),
                                   ((.0, 1.15), "#41846a", r"$\gamma G$")]:
        ax.arrow(0, 0, dx, dy, width=.025, color=color, length_includes_head=True)
        ax.text(dx*1.14, dy*1.14, label, color=color)
    ax.text(-1.7, -1.9, r"Heading = arg(drive) + $\eta\xi_i$,   $\xi_i\sim U[-\pi,\pi]$")
    ax.set(xlim=(-2.1, 2.1), ylim=(-2.1, 2.1), aspect="equal")
    ax.axis("off")
    save(fig, "main_01")


def representatives():
    table = pd.read_csv(DATA / "single_representatives_parameters.csv", index_col=0)
    fig, axes = plt.subplots(1, len(table), figsize=(12, 3))
    for i, ((name, row), ax) in enumerate(zip(table.iterrows(), axes)):
        params = SingleParams.from_active_vector(row[list(ACTIVE_SINGLE_NAMES)].to_numpy(float))
        cfg = SimulationConfig(steps=480, sample_start=240, seed=701+i,
                               boundary="reflecting" if name == "milling" else "periodic")
        result = simulate_single(params, cfg, record_states=True, state_every=12)
        x, theta = result.states[-1]
        ax.quiver(x[:, 0], x[:, 1], np.cos(theta), np.sin(theta), scale=25, width=.005)
        ax.set(title=name, xlim=(0, 4), ylim=(0, 4), aspect="equal", xticks=[], yticks=[])
    save(fig, "main_03")


def independent_panels():
    report = json.loads((DATA / "independent_validation/report.json").read_text())["single"]
    labels = ["AE", "Descriptor\nPCA", "Classical\nPCA"]
    keys = ["autoencoder_2d", "descriptor_pca_2d", "classical_pca_2d"]
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.5), layout="constrained")
    ax[0].bar(labels, [report[k]["trustworthiness"] for k in keys])
    ax[0].set(ylabel="Independent-test trustworthiness", ylim=(0, 1))
    regressors = report["classical_recoverability"]
    for axis, offset in [(0, -.18), (1, .18)]:
        ax[1].bar(np.arange(4)+offset, [v[axis] for v in regressors.values()], .35, label=f"Axis {axis+1}")
    ax[1].set(xticks=range(4), xticklabels=["Linear", "RF", "ExtraTrees", "HistGB"], ylabel="Test R²")
    ax[1].legend(loc="lower right")
    corpus = pd.read_csv(EXP / "single_behavior_corpus_latent.csv").set_index("sample_id")
    pairs = pd.read_csv(EXP / "learned_vs_classical_pairs.csv")
    ax[2].scatter(corpus.learned_0, corpus.learned_1, s=5, color=".8")
    for pair in pairs.head(6).itertuples():
        pair_rows = corpus.loc[[pair.id_a, pair.id_b]]
        ax[2].plot(pair_rows.learned_0, pair_rows.learned_1, "o-", ms=3, color="#b75a4c")
    ax[2].set(xlabel="Learned axis 1", ylabel="Learned axis 2", title="Pairs close in classical space")
    fig.suptitle("Independent representation audit")
    save(fig, "main_06")

    fig, ax = plt.subplots(1, 3, figsize=(11, 3.2))
    for offset, metric in [(-.18, "trustworthiness"), (.18, "knn_overlap")]:
        ax[0].bar(np.arange(2)+offset, [report[k][metric] for k in keys[:2]], .35, label=metric)
    ax[0].set(xticks=[0, 1], xticklabels=labels[:2], ylim=(0, 1))
    ax[0].legend(fontsize=7)
    for i, split in enumerate(("validation", "test")):
        s = report["autoencoder_2d"][f"{split}_seed_stability"]
        ax[1].errorbar(i, s["median"], yerr=[[s["median"]-s["min"]], [s["max"]-s["median"]]], fmt="o", capsize=4)
    ax[1].set(xticks=[0, 1], xticklabels=["Validation", "Test"], ylabel="Latent-distance Spearman r", ylim=(0, 1))
    ax[2].bar(labels[:2], [report[k]["standardized_mse"] for k in keys[:2]])
    ax[2].set(ylabel="Standardized test MSE")
    save(fig, "supp_04")
    dims = [2, 3, 4, 6]
    fig, ax = plt.subplots(1, 3, figsize=(10, 3))
    for a, metric in zip(ax, ("standardized_mse", "trustworthiness", "knn_overlap")):
        a.plot(dims, [report[f"autoencoder_{d}d"][metric] for d in dims], "o-")
        a.set(xlabel="Latent dimensions", ylabel=metric.replace("_", " "), xticks=dims)
    save(fig, "supp_05")


def sensitivity_panels():
    data = pd.read_csv(EXP / "sloppiness_noisefloor_spectra.csv")
    data = data[np.isclose(data.delta, .04)]
    regimes = list(dict.fromkeys(data.regime))
    fig, ax = plt.subplots(figsize=(10, 4))
    for i, regime in enumerate(regimes):
        g = data[data.regime == regime]
        scale = g.eigenvalue.max()
        for row in g.itertuples():
            ax.plot([i-.3, i+.3], [row.eigenvalue/scale]*2,
                    color="#254a70" if row.resolved else "#c5cbd0", lw=2)
    elite = pd.read_csv(EXP / "elite_concentration_eigenvalues.csv").inverse_variance
    for value in elite / elite.max():
        ax.plot([len(regimes)-.3, len(regimes)+.3], [value]*2, color="#b95f3e", lw=2)
    ax.axvspan(len(regimes)-.5, len(regimes)+.5, color=".92", zorder=0)
    ax.set(yscale="log", ylabel="Normalized eigenvalue", xticks=range(len(regimes)+1),
           xticklabels=[r.replace("_", " ") for r in regimes]+["Archive"])
    ax.tick_params(axis="x", rotation=25)
    save(fig, "main_07")
    derivatives = pd.read_csv(EXP / "local_sloppiness_sensitivities.csv")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for regime in regimes:
        g = data[data.regime == regime]
        axes[0].semilogy(g["index"], g.eigenvalue/g.eigenvalue.max(), "o-", ms=3, label=regime)
    axes[0].set(xlabel="Direction", ylabel="Normalized eigenvalue")
    axes[0].legend(fontsize=6)
    derivatives["absolute_derivative"] = derivatives.derivative.abs()
    matrix = derivatives.pivot_table(index="regime", columns="parameter",
                                     values="absolute_derivative", aggfunc="mean")
    image = axes[1].imshow(matrix, aspect="auto", cmap="viridis")
    axes[1].set(xticks=range(len(matrix.columns)), xticklabels=matrix.columns,
                yticks=range(len(matrix)), yticklabels=matrix.index)
    axes[1].tick_params(axis="x", rotation=90, labelsize=6)
    axes[1].tick_params(axis="y", labelsize=6)
    fig.colorbar(image, ax=axes[1], label="Mean absolute standardized derivative")
    axes[2].semilogy(np.arange(1, len(elite)+1), elite/elite.max(), "o-")
    axes[2].set(xlabel="Direction", ylabel="Normalized archive inverse variance")
    fig.tight_layout()
    save(fig, "supp_07")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    model_diagram()
    representatives()
    independent_panels()
    sensitivity_panels()
    call("make_paper_figures", "make_single_method_robustness", "make_chirality_validation")
    call("make_highbudget_figures", "main")
    copy_figure("paper/fig_single_archives.png", "main_02")
    copy_figure("paper/fig_chirality_validation.png", "main_08")
    copy_figure("paper/fig_single_method_robustness.png", "supp_02")
    copy_figure("paper/fig_multi_archives.png", "supp_08")
    copy_figure("paper/fig_archive_coverage.png", "supp_17")
    call("make_expanded_figures", "fig_single_latent", "fig_single_examples", "fig_high_examples",
         "fig_density_dependence", "fig_multi_latent", "fig_multi_examples")
    for source, name in [("density_dependence", "main_04"), ("single_learned_space", "main_05"),
                          ("high_density_examples", "supp_03"), ("single_learned_examples", "supp_06"),
                          ("multi_learned_space", "supp_10"), ("multi_learned_examples", "supp_11")]:
        copy_figure(f"expanded/fig_{source}.png", name)
    call("make_revision_figures", "fig_champion_per_seed", "fig_chirality_decay")
    copy_figure("expanded/fig_champion_per_seed.png", "supp_12")
    copy_figure("expanded/fig_chirality_decay.png", "supp_09")
    mod = importlib.import_module("make_champion_montage")
    mod.montage(EXP / "objective_champions.csv", FIG / "fig_objective_champions.png")
    mod.montage(EXP / "objective_champions_periodic.csv", FIG / "fig_objective_champions_periodic.png", boundary_override="periodic")
    copy_figure("fig_objective_champions.png", "supp_13")
    copy_figure("fig_objective_champions_periodic.png", "supp_14")
    call("finite_size_v2_figure", "main")
    copy_figure("expanded/fig_finite_size.png", "main_10")
    mod = importlib.import_module("nonreciprocity_experiments")
    grid = pd.read_csv(EXP / "nonreciprocity_phase_diagram.csv")
    cut = pd.read_csv(EXP / "nonreciprocity_susceptibility_cut.csv")
    mod.make_figure(grid, grid.alignment_antisymmetric.unique(), grid.force_antisymmetric.unique(), cut)
    copy_figure("expanded/fig_nonreciprocity.png", "main_09")
    mod = importlib.import_module("phase_transition_discovery")
    mod.make_figure({k: pd.read_csv(EXP / f"phase_transition_{k}.csv") for k in ("noise", "density")})
    copy_figure("expanded/fig_phase_transitions.png", "supp_15")
    call("bd_figures_e", "revised_fig28", "characterization_fig")
    copy_figure("fig_boundary_result.png", "main_11")
    copy_figure("fig_boundary_char.png", "supp_16")
    importlib.import_module("make_additional_figures")
    copy_figure("fig_single_repertoire_3d.png", "supp_18")
    copy_figure("fig_single_parameter_profiles.png", "supp_19")
    data = pd.read_csv(DATA / "single_noise_sweep.csv").groupby("noise").phi.agg(["mean", "std"])
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.errorbar(data.index, data["mean"], yerr=data["std"], fmt="o-", capsize=3)
    ax.set(xlabel="Noise η", ylabel="Polar alignment", ylim=(0, 1.05))
    save(fig, "supp_01")
    expected = {f"main_{i:02}" for i in range(1, 12)} | {f"supp_{i:02}" for i in range(1, 20)}
    assert {row["figure"] for row in REPORT} == expected
    (OUT / "generation_report.json").write_text(json.dumps(REPORT, indent=2) + "\n")
    print(f"Generated {len(REPORT)} numbered figures in {OUT}")


if __name__ == "__main__":
    main()
