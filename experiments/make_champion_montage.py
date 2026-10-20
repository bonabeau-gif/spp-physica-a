"""Compact publication montage of objective-champion final states.

Re-simulates each champion in `objective_champions{tag}.csv` at its representative
seed, takes the final configuration, and lays the panels out in a landscape grid
labeled by objective and validated fitness. This is the print-friendly figure for
the manuscript (the animated GIFs remain the interactive artifact).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from spp_unified.parameters import ACTIVE_SINGLE_NAMES, TWO_SPECIES_EXTRA_NAMES, SimulationConfig, SingleParams
from spp_unified.model import simulate_single, simulate_two_species

sys.path.insert(0, str(ROOT / "experiments"))
from discover_and_animate import _two_from_vector, OBJECTIVES  # noqa: E402

TWO_NAMES = ACTIVE_SINGLE_NAMES + TWO_SPECIES_EXTRA_NAMES
ORDER = ["aligned_flock", "milling_vortex", "dense_cluster", "nematic_lanes",
         "hexatic_crystal", "density_waves", "phase_separation", "active_turbulence",
         "travelling_mixture", "chiral_rotation", "species_segregation"]
SPECIES_COLORS = np.array([[0.11, 0.47, 0.86], [0.87, 0.30, 0.15]])


def final_state(objective, row, boundary_override=None):
    spec = OBJECTIVES[objective]
    boundary = boundary_override or spec["boundary"]
    seed = int(row["rep_seed"])
    if spec["kind"] == "single":
        x = np.array([row[n] for n in ACTIVE_SINGLE_NAMES], float)
        cfg = SimulationConfig(S=3.6, r0=0.1, steps=360, sample_start=0, sample_every=2,
                               boundary=boundary, seed=seed)
        r = simulate_single(SingleParams.from_active_vector(x), cfg, record_states=True, state_every=6)
        return r.states[-1], None, cfg.S, spec
    x = np.array([row[n] for n in TWO_NAMES], float)
    cfg = SimulationConfig(S=4.0, r0=0.1, steps=360, sample_start=0, sample_every=2,
                           boundary=boundary, seed=seed)
    r = simulate_two_species(_two_from_vector(x), cfg, record_states=True, state_every=6)
    return r.states[-1], r.species, cfg.S, spec


def montage(csv_path, out_path, boundary_override=None, ncols=4):
    df = pd.read_csv(csv_path).set_index("objective")
    objs = [o for o in ORDER if o in df.index]
    nrows = int(np.ceil(len(objs) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.55 * ncols, 2.7 * nrows))
    axes = np.atleast_2d(axes)
    for ax in axes.ravel():
        ax.axis("off")
    for k, obj in enumerate(objs):
        row = df.loc[obj]
        (x, th), species, S, spec = final_state(obj, row, boundary_override)
        two = spec["kind"] == "two"
        ax = axes[k // ncols, k % ncols]
        ax.set_facecolor("white")
        ax.axis("on"); ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlim(0, S); ax.set_ylim(0, S); ax.set_aspect("equal")
        u, v = np.cos(th), np.sin(th)
        if two and species is not None:
            # Two-species: heading = arrow direction, colour = particle type.
            colors = SPECIES_COLORS[species.astype(int)]
            ax.quiver(x[:, 0], x[:, 1], u, v, color=colors, scale=24, width=0.008, pivot="mid")
        else:
            # Single-species: heading = arrow direction only (as in Figure 4).
            ax.quiver(x[:, 0], x[:, 1], u, v, color="#1a1a1a", scale=24, width=0.006, pivot="mid")
        fit = float(row["val_mean"])
        kindtag = "two-species" if two else "single-species"
        tcol = "#b0451f" if two else "#20406c"
        ax.set_title(f"{obj.replace('_',' ')}  ·  fitness {fit:.2f}\n[{kindtag}]",
                     color=tcol, fontsize=8.7, pad=3)
        # Border colour also encodes the system type.
        for s in ax.spines.values():
            s.set_edgecolor("#d98a5f" if two else "#9fb2cc"); s.set_linewidth(1.6)
    fig.patch.set_facecolor("white")
    legend = [
        Line2D([0], [0], color="#1a1a1a", lw=2, marker=r"$\rightarrow$", ms=11, ls="none",
               label="single-species: dark arrows, heading = arrow direction (blue title/border)"),
        Patch(facecolor=SPECIES_COLORS[0], edgecolor="none", label="two-species type A (blue arrows)"),
        Patch(facecolor=SPECIES_COLORS[1], edgecolor="none", label="two-species type B (red arrows); red title/border"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=1, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(out_path, dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("wrote", out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", type=str, default="")
    ap.add_argument("--periodic-frame", action="store_true",
                    help="render every panel with periodic boundaries")
    args = ap.parse_args()
    suffix = f"_{args.tag}" if args.tag else ""
    csv = ROOT / "results" / "expanded" / f"objective_champions{suffix}.csv"
    out = ROOT / "figures" / f"fig_objective_champions{suffix}.png"
    boundary = "periodic" if args.periodic_frame else None
    montage(csv, out, boundary_override=boundary)


if __name__ == "__main__":
    main()
