"""Updated coverage-vs-budget figure with the fourth budget (1000) and the located
crossover near 2,400 calls (bootstrap 90% interval), replacing the old three-budget
'helps only at large budgets' figure."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures" / "expanded"
A = json.load(open(ROOT / "results/expanded/surrogate_exp/analysis.json"))
g = A["E2_gaps"]; cx = A["E2_crossover"]
budgets = [30, 300, 1000, 3000]
sur = [g[str(b)]["surrogate_mean"] for b in budgets]
dir_ = [g[str(b)]["direct_mean"] for b in budgets]
rnd = [g[str(b)]["random_mean"] for b in budgets]

fig, ax = plt.subplots(figsize=(5.4, 3.6))
ax.axvspan(cx["ci"][0], cx["ci"][1], color="0.88", zorder=0)
ax.axvline(cx["point"], color="0.55", ls="--", lw=1, zorder=1)
ax.text(cx["point"], 0.045, "crossover\n~2,400 calls", fontsize=7.5, color="0.35", ha="center")
for y, c, m, lab in [(dir_, "#2C6DB2", "o", "direct archive mutation"),
                     (sur, "#C74343", "s", "GP surrogate-assisted"),
                     (rnd, "#6a9e6a", "^", "random sampling")]:
    ax.plot(budgets, y, marker=m, color=c, lw=1.4, ms=5, label=lab)
ax.set_xscale("log"); ax.set_xlabel("additional simulator calls"); ax.set_ylabel("learned-archive coverage")
ax.set_xticks(budgets); ax.set_xticklabels([str(b) for b in budgets])
ax.set_title("Surrogate assistance: a narrow, weak crossover near 2,400 calls", fontsize=9, loc="left")
ax.legend(fontsize=7.5, loc="upper left")
ax.set_ylim(0, 0.85)
fig.tight_layout(); fig.savefig(FIG / "fig_surrogate_scaling.png", dpi=240); plt.close(fig)
print("wrote fig_surrogate_scaling.png; crossover", round(cx["point"]), cx["ci"])
