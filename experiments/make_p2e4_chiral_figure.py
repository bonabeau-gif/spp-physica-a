"""P2-E4 reciprocal-to-chiral cut figure: (a) rotation rate |omega| vs antisymmetric coupling
ja per size -- onset near |ja|~1.1, magnitude finite-size-decays at strong coupling;
(b) onset location ja_onset(N) (|omega|=0.05 crossing) with the size-stability band. Contrast
with the noise cut, whose peak location drifts."""
from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures" / "expanded"; FIG.mkdir(parents=True, exist_ok=True)
g = pd.read_csv(ROOT / "results/boundary/P2E4_chiral_cut_grid.csv")
Ns = sorted(g.N.unique()); cmap = plt.cm.plasma(np.linspace(0.1, 0.85, len(Ns)))


def onset_cross(sub, thr=0.05):
    s = sub.sort_values("ja"); ja = s.ja.to_numpy(); om = s.abs_omega_mean.to_numpy()
    for k in range(len(ja) - 1, 0, -1):
        if om[k-1] >= thr > om[k]:
            return float(ja[k] + (thr - om[k]) * (ja[k-1] - ja[k]) / (om[k-1] - om[k]))
    return float("nan")


fig, ax = plt.subplots(1, 2, figsize=(9.4, 3.7))
for c, N in zip(cmap, Ns):
    s = g[g.N == N].sort_values("ja")
    ax[0].plot(-s.ja, s.abs_omega_mean, "o-", color=c, ms=4, lw=1.3, label=f"N={N}")
ax[0].axhline(0.05, color="0.6", ls=":", lw=1)
ax[0].set_xlabel("non-reciprocity  |ja|  (antisymmetric alignment)")
ax[0].set_ylabel("rotation rate  |ω|")
ax[0].set_title("(a) rotation onset near |ja|≈1.1; magnitude decays with N", fontsize=8.5, loc="left")
ax[0].legend(fontsize=7.5, ncol=2)

on = {N: onset_cross(g[g.N == N]) for N in Ns}
ax[1].plot(Ns, [-on[N] for N in Ns], "o-", color="k", ms=6, lw=1.4)
band = float(np.median([-on[N] for N in Ns]))
ax[1].axhspan(band - 0.125, band + 0.125, color="0.85", zorder=0, label="±half grid step")
ax[1].set_xscale("log"); ax[1].set_xlabel("N"); ax[1].set_ylabel("onset  |ja*|  (|ω|=0.05)")
ax[1].set_ylim(0.6, 1.6)
sp = max(-on[N] for N in Ns) - min(-on[N] for N in Ns)
ax[1].set_title(f"(b) onset size-STABLE: |ja*|≈{band:.2f}, spread {sp:.3f}", fontsize=8.5, loc="left")
ax[1].legend(fontsize=7.5)
fig.tight_layout(); fig.savefig(FIG / "fig_p2e4_chiral_cut.png", dpi=220); plt.close(fig)
print(f"wrote fig_p2e4_chiral_cut.png; onset |ja*|~{band:.3f} spread {sp:.3f}")
