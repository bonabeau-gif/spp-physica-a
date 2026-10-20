"""P2-E4 noise-cut figure from the confirmatory equilibrated run: (a) susceptibility vs
noise per size; (b) Binder cumulant vs noise per size with the crossing region; (c) self-
averaging (across-seed variance of per-seed mean phi vs N, log-log) off- and on-transition;
(d) susceptibility-peak location eta*(N) with the size-stability band. Finite-size STABILITY
framing; no exponent/thermodynamic-limit claim."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures" / "expanded"; FIG.mkdir(parents=True, exist_ok=True)
grid = pd.read_csv(ROOT / "results/boundary/P2E4_noise_confirm_grid.csv")
head = json.load(open(ROOT / "results/boundary/P2E4_noise_confirm_headline.json"))
Ns = sorted(grid.N.unique())
cmap = plt.cm.viridis(np.linspace(0.1, 0.9, len(Ns)))

fig, ax = plt.subplots(2, 2, figsize=(9.6, 7.2))

# (a) susceptibility
for c, N in zip(cmap, Ns):
    s = grid[grid.N == N].sort_values("eta")
    ax[0, 0].plot(s.eta, s.chi, "o-", color=c, ms=4, lw=1.3, label=f"N={N}")
ax[0, 0].set_xlabel("noise η"); ax[0, 0].set_ylabel("susceptibility  χ = N·Var(φ)")
ax[0, 0].set_yscale("log"); ax[0, 0].legend(fontsize=7.5, ncol=2)
ax[0, 0].set_title("(a) order-parameter fluctuations", fontsize=9, loc="left")

# (b) Binder
for c, N in zip(cmap, Ns):
    s = grid[grid.N == N].sort_values("eta")
    ax[0, 1].plot(s.eta, s.binder, "o-", color=c, ms=4, lw=1.3, label=f"N={N}")
ax[0, 1].set_xlabel("noise η"); ax[0, 1].set_ylabel("Binder cumulant U")
ax[0, 1].set_title("(b) Binder cumulant (crossing = transition)", fontsize=9, loc="left")
ax[0, 1].legend(fontsize=7.5, ncol=2)

# (c) order parameter phi(eta) per size -- the crossover drifts left with N
for c, N in zip(cmap, Ns):
    s = grid[grid.N == N].sort_values("eta")
    ax[1, 0].plot(s.eta, s.phi_mean, "o-", color=c, ms=4, lw=1.3, label=f"N={N}")
ax[1, 0].set_xlabel("noise η"); ax[1, 0].set_ylabel("polar order  ⟨φ⟩")
ax[1, 0].set_title("(c) order parameter: crossover drifts left with N", fontsize=9, loc="left")
ax[1, 0].legend(fontsize=7.5, ncol=2)

# (d) eta*(N)
esN = head["eta_star_by_size"]
ns = sorted(int(n) for n in esN); es = [esN[str(n)] for n in ns]
ax[1, 1].plot(ns, es, "o-", color="k", ms=6, lw=1.4)
band = float(np.median(es))
ax[1, 1].axhspan(band - 0.075, band + 0.075, color="0.85", zorder=0,
                 label="±one grid step (stable band)")
ax[1, 1].set_xscale("log"); ax[1, 1].set_xlabel("N")
ax[1, 1].set_ylabel("susceptibility-peak η*")
ax[1, 1].set_ylim(0.3, 0.85)
ax[1, 1].set_title(f"(d) η*(N): spread {head['eta_star_spread']:.3f}", fontsize=9, loc="left")
ax[1, 1].legend(fontsize=7.5)

fig.tight_layout()
fig.savefig(FIG / "fig_p2e4_noise_scaling.png", dpi=220)
plt.close(fig)
print("wrote fig_p2e4_noise_scaling.png")
