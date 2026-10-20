from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def archive_heatmap(df: pd.DataFrame, dims: tuple[int, int], value: str, xlabel: str,
                    ylabel: str, out: str, vrange=None):
    a = np.full(dims, np.nan)
    for _, row in df.iterrows():
        if "cell_0" in row and "cell_1" in row:
            a[int(row.cell_0), int(row.cell_1)] = row[value]
    fig, ax = plt.subplots(figsize=(5.2, 4.5))
    im = ax.imshow(a.T, origin="lower", aspect="auto", interpolation="nearest",
                   extent=(0, 1, 0, 1), vmin=None if vrange is None else vrange[0],
                   vmax=None if vrange is None else vrange[1])
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    fig.colorbar(im, ax=ax, label=value)
    fig.tight_layout(); fig.savefig(out, dpi=220); plt.close(fig)


def snapshot_grid(results, titles, S, out, species=False):
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(4*n, 4))
    if n == 1: axes = [axes]
    for ax, res, title in zip(axes, results, titles):
        x, th = res.states[-1]
        u, v = np.cos(th), np.sin(th)
        if species and res.species is not None:
            ax.quiver(x[:,0], x[:,1], u, v, res.species, scale=25, width=.006, pivot="mid")
        else:
            ax.quiver(x[:,0], x[:,1], u, v, scale=25, width=.006, pivot="mid")
        ax.set_xlim(0,S); ax.set_ylim(0,S); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title, fontsize=9)
    fig.tight_layout(); fig.savefig(out, dpi=220); plt.close(fig)
