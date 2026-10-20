"""Overview schematic (Figure 1): the paper's workflow from the simulator through
direct QD, the learned descriptor, surrogate/geometry diagnostics, and the
two-species and directed analyses."""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"

C = {"sim": "#d9e8f5", "qd": "#cde6cd", "learn": "#e6d7ef", "diag": "#f6e2c8", "two": "#f5d0d0"}


def box(ax, x, y, w, h, text, color, fs=8.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.010,rounding_size=0.02",
                                linewidth=1.0, edgecolor="#555", facecolor=color))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, wrap=True)


def arrow(ax, x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=12,
                                 linewidth=1.1, color="#444", shrinkA=2, shrinkB=2))


def main():
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.set_xlim(0, 10); ax.set_ylim(0, 6.4); ax.axis("off")

    box(ax, 0.2, 2.7, 1.9, 1.0, "Self-propelled\nparticle model\n(9 parameters)\n§2", C["sim"], 8.5)
    box(ax, 2.6, 4.2, 2.2, 1.0, "Direct QD archives\nalignment–milling,\n–clustering  §3", C["qd"])
    box(ax, 2.6, 2.7, 2.2, 1.0, "Density-expanded\ncorpus (360 rules,\nρ→0.70)  §4", C["qd"])
    box(ax, 2.6, 1.2, 2.2, 1.0, "Two-species:\nreciprocal / non-\nreciprocal  §8–9", C["two"])

    box(ax, 5.3, 4.2, 2.3, 1.0, "Learned 51-D\ndescriptor → 2-D\nautoencoder  §5", C["learn"])
    box(ax, 5.3, 2.7, 2.3, 1.0, "Surrogate-assisted\nillumination  §6\nSloppiness / Jacobians §7", C["diag"], 8.0)
    box(ax, 5.3, 1.2, 2.3, 1.0, "Directed champions §10\nTransitions (finite-\nsize) §11", C["diag"], 8.0)

    box(ax, 8.1, 2.7, 1.7, 1.0, "Behavioral\nrepertoire\n& geometry", "#eeeeee", 8.5)

    arrow(ax, 2.1, 3.2, 2.6, 4.6); arrow(ax, 2.1, 3.2, 2.6, 3.2); arrow(ax, 2.1, 3.2, 2.6, 1.7)
    arrow(ax, 4.8, 3.2, 5.3, 4.6)   # corpus -> learned
    arrow(ax, 4.8, 4.7, 5.3, 4.7)   # QD -> learned
    arrow(ax, 4.8, 3.2, 5.3, 3.2)   # corpus -> surrogate/sloppiness
    arrow(ax, 4.8, 1.7, 5.3, 1.7)   # two-species -> champions/transitions
    arrow(ax, 6.45, 4.2, 6.45, 3.7) # learned -> surrogate
    arrow(ax, 7.6, 4.7, 8.9, 3.7)
    arrow(ax, 7.6, 3.2, 8.1, 3.2)
    arrow(ax, 7.6, 1.7, 8.9, 2.7)

    ax.text(5.0, 6.05, "Illuminating and learning the SPP behavioral repertoire",
            ha="center", va="center", fontsize=10, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG / "fig_overview_schematic.png", dpi=260, bbox_inches="tight")
    plt.close(fig)
    print("wrote fig_overview_schematic.png")


if __name__ == "__main__":
    main()
