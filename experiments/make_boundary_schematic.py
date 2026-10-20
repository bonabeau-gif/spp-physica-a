"""Schematic of the boundary-mapping framework (Section 11.4)."""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
C = {"nom": "#cde6cd", "step": "#d9e8f5", "core": "#f6e2c8", "cat": "#e6d7ef", "gate": "#eeeeee"}


def box(ax, x, y, w, h, text, color, fs=8.3):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.010,rounding_size=0.02",
                                linewidth=1.0, edgecolor="#555", facecolor=color))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, wrap=True)


def arrow(ax, x0, y0, x1, y1, style="-|>", color="#444", lw=1.1, ls="-"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=11,
                                 linewidth=lw, color=color, ls=ls, shrinkA=3, shrinkB=3))


def main():
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    ax.set_xlim(0, 10); ax.set_ylim(0, 7.1); ax.axis("off")

    # nominators (cheapest-first)
    box(ax, 0.15, 5.35, 3.0, 0.85, "Archive-pair mining\n(no new simulation)", C["nom"], 7.8)
    box(ax, 0.15, 4.35, 3.0, 0.85, "GP-gradient prescreen\n(analytic posterior gradient)", C["nom"], 7.6)
    box(ax, 0.15, 3.35, 3.0, 0.85, "Sensitivity-illumination QD\n(noise-corrected elites)", C["nom"], 7.6)
    ax.text(1.65, 6.30, "nominate  (cheapest first)", ha="center", fontsize=8.2, style="italic")

    # pipeline steps
    box(ax, 3.9, 4.35, 1.9, 0.95, "Localize\n1-D scan +\noracle", C["step"], 8.2)
    box(ax, 6.1, 4.35, 1.9, 0.95, "Continue\npredictor–\ncorrector", C["step"], 8.2)
    box(ax, 8.3, 4.35, 1.6, 0.95, "Catalog\ntyped\nboundaries", C["cat"], 8.2)

    # shared core primitive
    box(ax, 3.9, 2.15, 4.1, 1.0, "Sensitivity oracle:  descriptor Jacobian\ntop singular value  vs  Monte-Carlo noise floor\n(multi-scale, §7.2)", C["core"], 7.8)
    # gate
    box(ax, 3.9, 0.9, 4.1, 0.75, "Adaptive run-length gate  (window ≥ 50 τ, §11.2)", C["gate"], 7.8)
    # controls
    box(ax, 8.3, 2.4, 1.6, 0.95, "", C["cat"], 8.2)
    ax.text(9.1, 2.4 + 0.72, "Controls", ha="center", va="center", fontsize=8.2)
    ax.text(9.1, 2.4 + 0.44, r"$J_a=C_a=0$", ha="center", va="center", fontsize=7.8)
    ax.text(9.1, 2.4 + 0.18, r"$f_B=0$", ha="center", va="center", fontsize=7.8)

    # arrows: nominators -> localize
    for yy in (5.77, 4.77, 3.77):
        arrow(ax, 3.15, yy, 3.9, 4.83)
    arrow(ax, 5.8, 4.83, 6.1, 4.83)          # localize -> continue
    arrow(ax, 8.0, 4.83, 8.3, 4.83)          # continue -> catalog
    # oracle feeds localize + continue
    arrow(ax, 4.85, 3.15, 4.85, 4.35, color="#b06a1a")
    arrow(ax, 7.05, 3.15, 7.05, 4.35, color="#b06a1a")
    # gate under oracle
    arrow(ax, 5.95, 1.65, 5.95, 2.15, color="#888")
    # catalog -> controls
    arrow(ax, 9.1, 3.35, 9.1, 3.35+0.0, style="-", color="#555")
    arrow(ax, 9.1, 2.4+0.95, 9.1, 4.35, style="-|>", color="#444")

    ax.text(5.0, 6.85, "Mapping behavioral-boundary manifolds", ha="center", fontsize=10.5, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG / "fig_boundary_framework.png", dpi=260, bbox_inches="tight")
    plt.close(fig)
    print("wrote fig_boundary_framework.png")


if __name__ == "__main__":
    main()
