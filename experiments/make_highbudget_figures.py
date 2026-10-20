"""Regenerate the archive figures and the coverage/saturation figure from the
high-budget (~64k-evaluation) re-runs in results/highbudget, plus a CMA-MAE
overlay if present. Overwrites the paper figures used by the manuscript:
  figures/paper/fig_single_archives.png
  figures/paper/fig_multi_archives.png
  figures/paper/fig_archive_coverage.png
Prints the new coverage numbers for the manuscript.
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
HB = ROOT / "results" / "highbudget"
FIG = ROOT / "figures" / "paper"
FIG.mkdir(parents=True, exist_ok=True)
DIMS = (12, 12)


def grid(df, value="objective"):
    Z = np.full(DIMS, np.nan)
    for _, r in df.iterrows():
        i, j = int(r.cell_0), int(r.cell_1)
        if 0 <= i < DIMS[0] and 0 <= j < DIMS[1]:
            Z[j, i] = r[value]
    return Z


def single_archives():
    a = pd.read_csv(HB / "single_qd_alignment_milling.csv")
    b = pd.read_csv(HB / "single_qd_alignment_clustering.csv")
    fig, ax = plt.subplots(1, 2, figsize=(8.2, 3.7), constrained_layout=True)
    im = ax[0].imshow(grid(a), origin="lower", extent=(0, 1, 0, .5), aspect="auto", vmin=.80, vmax=1.0)
    im2 = ax[1].imshow(grid(b), origin="lower", extent=(0, 1, 0, 8), aspect="auto", vmin=.80, vmax=1.0)
    ax[0].set(xlabel="polar alignment", ylabel="milling")
    ax[1].set(xlabel="polar alignment", ylabel="clustering")
    for a_, l in zip(ax, "ab"):
        a_.text(-0.12, 1.04, l, transform=a_.transAxes, fontweight="bold", fontsize=11)
    cb = fig.colorbar(im2, ax=ax, shrink=.88, pad=.02); cb.set_label("temporal stability")
    fig.savefig(FIG / "fig_single_archives.png", dpi=300, bbox_inches="tight"); plt.close(fig)
    return len(a) / 144, len(b) / 144


def multi_archives():
    non = pd.read_csv(HB / "multi_qd_nonreciprocal.csv")
    rec = pd.read_csv(HB / "multi_qd_reciprocal.csv")
    fig, ax = plt.subplots(1, 2, figsize=(8.2, 3.7), constrained_layout=True)
    im = ax[0].imshow(grid(rec), origin="lower", extent=(0, 1, 0, 1), aspect="auto", vmin=.80, vmax=1.0)
    im2 = ax[1].imshow(grid(non), origin="lower", extent=(0, 1, 0, 1), aspect="auto", vmin=.80, vmax=1.0)
    ax[0].set(xlabel="net drift", ylabel="chiral rotation", title="reciprocal")
    ax[1].set(xlabel="net drift", ylabel="chiral rotation", title="non-reciprocal")
    for a_, l in zip(ax, "ab"):
        a_.text(-0.12, 1.04, l, transform=a_.transAxes, fontweight="bold", fontsize=11)
    cb = fig.colorbar(im2, ax=ax, shrink=.88, pad=.02); cb.set_label("temporal stability")
    fig.savefig(FIG / "fig_multi_archives.png", dpi=300, bbox_inches="tight"); plt.close(fig)
    return len(rec) / 144, len(non) / 144


def coverage_figure():
    labels = {"single_qd_alignment_milling": "alignment-milling",
              "single_qd_alignment_clustering": "alignment-clustering",
              "multi_qd_nonreciprocal": "non-reciprocal drift-chirality",
              "multi_qd_reciprocal": "reciprocal drift-chirality"}
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    for key, lab in labels.items():
        h = pd.read_csv(HB / f"{key}_history.csv")
        ax.plot(h.evaluations, h.coverage, marker="o", ms=2.5, label=lab)
    # CMA-MAE overlays if available
    for key, lab in [("single_qd_alignment_milling", "CMA-MAE alignment-milling"),
                     ("single_qd_alignment_clustering", "CMA-MAE alignment-clustering")]:
        p = HB / f"cmame_{key}_history.csv"
        if p.exists():
            h = pd.read_csv(p)
            ax.plot(h.evaluations, h.coverage, ls="--", lw=1.2, label=lab)
    ax.axvline(640, color="0.6", ls=":", lw=1)
    ax.text(640, 0.05, " reference\n budget (640)", fontsize=7, color="0.4")
    ax.set_xscale("log")
    ax.set_xlabel("simulator evaluations"); ax.set_ylabel("archive coverage")
    ax.set_ylim(0, 1.02); ax.legend(frameon=False, fontsize=7.5, loc="lower right")
    fig.tight_layout(); fig.savefig(FIG / "fig_archive_coverage.png", dpi=300, bbox_inches="tight"); plt.close(fig)


def main():
    m, c = single_archives()
    r, n = multi_archives()
    coverage_figure()
    print(f"alignment-milling coverage  {m:.3f} ({int(m*144)}/144)")
    print(f"alignment-clustering coverage {c:.3f} ({int(c*144)}/144)")
    print(f"reciprocal coverage {r:.3f} ({int(r*144)}/144)")
    print(f"non-reciprocal coverage {n:.3f} ({int(n*144)}/144)")


if __name__ == "__main__":
    main()
