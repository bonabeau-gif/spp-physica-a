"""Eigenvalue barcode for the local behavioral Jacobians.

For each dynamical regime the local sensitivity spectrum (eigenvalues of the
standardized behavior-response Gram matrix) is drawn as a column of horizontal
ticks on a shared logarithmic axis. A wide vertical spread of ticks is the visual
signature of sloppiness: a few stiff eigen-combinations sit near the top and many
sloppy ones trail downward across several decades. The right-hand column shows the
inverse-variance spectrum of the pooled learned-archive elite ensemble, whose
ticks span far less, making explicit that archive covariance is not equivalent to
local stiffness.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "expanded"
FIG = ROOT / "figures" / "expanded"


def main():
    ev = pd.read_csv(OUT / "local_sloppiness_eigenvalues.csv")
    ee = pd.read_csv(OUT / "elite_concentration_eigenvalues.csv")

    regimes = list(dict.fromkeys(ev.regime))
    cols = regimes + ["learned archive"]
    fig, ax = plt.subplots(figsize=(1.15 * len(cols) + 1.6, 5.0))

    def positive_norm(vals, thresh=1e-8):
        # Match the manuscript's nonzero-eigenvalue convention (absolute cutoff),
        # discarding numerically-zero directions of the rank-limited response.
        v = np.asarray(vals, float)
        v = v[np.isfinite(v)]
        v = v[v > thresh]
        return v / v.max()

    spans = {}
    for i, reg in enumerate(regimes):
        y = positive_norm(ev[ev.regime == reg].eigenvalue.to_numpy())
        logy = np.log10(y)
        spans[reg] = float(logy.min())
        for val in logy:
            ax.plot([i - 0.36, i + 0.36], [val, val], color="#1f3b63", lw=2.2, solid_capstyle="round")

    # Learned-archive elite ensemble: inverse variance, normalized like a stiffness.
    iv = positive_norm(ee.inverse_variance.to_numpy())
    logiv = np.log10(iv)
    spans["learned archive"] = float(logiv.min())
    j = len(regimes)
    for val in logiv:
        ax.plot([j - 0.36, j + 0.36], [val, val], color="#b0451f", lw=2.2, solid_capstyle="round")

    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([c.replace("_", " ") for c in cols], rotation=32, ha="right", fontsize=8.5)
    ax.set_ylabel("normalized eigenvalue (log$_{10}$, top = stiffest)")
    ax.set_title("Local sensitivity spectra — sloppiness barcode", fontsize=10)
    ax.axvspan(len(regimes) - 0.5, len(regimes) + 0.5, color="0.92", zorder=0)
    ax.set_ylim(min(spans.values()) - 0.4, 0.85)
    ax.axhline(0, color="0.6", lw=0.7)
    # Annotate the decade span of each column above its stiffest tick.
    for i, c in enumerate(cols):
        ax.text(i, 0.45, f"{abs(spans[c]):.1f} dec", ha="center", va="center",
                fontsize=7.2, color="0.3")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "fig_sloppiness_barcode.png", dpi=260, bbox_inches="tight")
    plt.close(fig)
    print("wrote", FIG / "fig_sloppiness_barcode.png")
    print("decade spans:", {k: round(abs(v), 2) for k, v in spans.items()})


if __name__ == "__main__":
    main()
