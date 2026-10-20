"""P1-E5 figure: signed-significance matrix of the density (rho), finite-N (logN), and wall
(boundary B) effects across rule x observable, separating physical density effects from
finite-N and wall effects. Cell = sign(coef) * min(-log10 p, 4); annotated with '*' if p<0.05.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
A = json.load(open(ROOT / "results/boundary/P1E5_decoupling_analysis.json"))["rules"]
rows, rowlab = [], []
for rule in A:
    for obs in A[rule]:
        d = A[rule][obs]
        rows.append([np.sign(d["rho_full"]) * min(-np.log10(max(d["p_rho_full"], 1e-6)), 4),
                     np.sign(d["logN_coef"]) * min(-np.log10(max(d["p_logN"], 1e-6)), 4),
                     np.sign(d["B_coef"]) * min(-np.log10(max(d["p_B"], 1e-6)), 4)])
        rowlab.append(f"{rule.replace('_',' ')} · {obs}")
M = np.array(rows)
P = np.array([[A[r][o][k] for k in ("p_rho_full", "p_logN", "p_B")] for r in A for o in A[r]])

fig, ax = plt.subplots(figsize=(6.6, 8.2))
im = ax.imshow(M, cmap="RdBu_r", vmin=-4, vmax=4, aspect="auto")
ax.set_xticks(range(3)); ax.set_xticklabels(["density ρ", "finite-N logN", "wall B"], fontsize=9)
ax.set_yticks(range(len(rowlab))); ax.set_yticklabels(rowlab, fontsize=7.5)
for i in range(M.shape[0]):
    for j in range(3):
        if P[i, j] < 0.05:
            ax.text(j, i, "✱", ha="center", va="center", fontsize=9, color="k")
ax.set_title("P1-E5: density vs finite-N vs wall effects\n(color = signed −log₁₀p; ✱ = p<0.05)",
             fontsize=10, loc="left")
cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cb.set_label("signed significance", fontsize=8)
fig.tight_layout(); fig.savefig(ROOT / "figures/expanded/fig_p1e5_effects.png", dpi=220)
plt.close(fig)
print("wrote fig_p1e5_effects.png")
