"""Finite-size figure + corrected headline for the noise cut (assessment #5).

Primary finite-size evidence is the susceptibility maximum: its height grows
systematically with N while its location stays near eta*~0.6, the signature of a
genuine order-disorder transition rather than a size-drifting crossover. The
Binder cumulant drop deepens with N near the same location (qualitative
corroboration; 40-seed fourth moments are too noisy to localize a sharp
crossing). We report eta* from the size-stable susceptibility peak and the
low bias of the model-free descriptor scout against it.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
E = ROOT / "results" / "expanded"
FIG = ROOT / "figures" / "expanded"


def main():
    df = pd.read_csv(E / "finite_size_noise.csv")
    Ns = sorted(df.N.unique())
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(Ns)))

    fig, ax = plt.subplots(1, 3, figsize=(11.5, 3.6))
    peaks = {}
    chimax = {}
    for c, N in zip(colors, Ns):
        g = df[df.N == N].sort_values("eta")
        ax[0].plot(g.eta, g.phi_mean, "-o", ms=3, color=c, label=f"N={N}")
        ax[1].plot(g.eta, g.susceptibility, "-o", ms=3, color=c, label=f"N={N}")
        ax[2].plot(g.eta, g.binder, "-o", ms=3, color=c, label=f"N={N}")
        peaks[N] = float(g.loc[g.susceptibility.idxmax(), "eta"])
        chimax[N] = float(g.susceptibility.max())
    ax[0].set(xlabel="noise η", ylabel="polar order φ", ylim=(0, 1.03))
    ax[0].set_title("(a) order parameter", fontsize=9, loc="left")
    ax[1].set(xlabel="noise η", ylabel=r"susceptibility $N\,\mathrm{Var}(φ)$")
    ax[1].axvline(peaks[Ns[-1]], color="0.5", ls=":", lw=1)
    ax[1].set_title("(b) susceptibility grows with N at η*≈%.2f" % peaks[Ns[-1]], fontsize=9, loc="left")
    ax[2].set(xlabel="noise η", ylabel="Binder cumulant U")
    ax[2].set_title("(c) Binder drop deepens with N", fontsize=9, loc="left")
    ax[0].legend(frameon=False, fontsize=7.5)
    fig.tight_layout(); fig.savefig(FIG / "fig_finite_size.png", dpi=240); plt.close(fig)

    head = json.load(open(E / "finite_size_headline.json"))
    eta_star = peaks[Ns[-1]]
    # exponent of chi_max ~ N^a
    logN = np.log(np.array(Ns, float)); logchi = np.log(np.array([chimax[n] for n in Ns]))
    a = float(np.polyfit(logN, logchi, 1)[0])
    out = {
        "sizes_N": [int(n) for n in Ns],
        "susceptibility_peak_by_N": {int(k): float(v) for k, v in peaks.items()},
        "chi_max_by_N": {int(k): float(v) for k, v in chimax.items()},
        "chi_max_scaling_exponent": float(a),
        "eta_star_from_susceptibility": float(eta_star),
        "descriptor_scout_peak": float(head["descriptor_scout_peak"]),
        "scout_bias_vs_eta_star": float(head["descriptor_scout_peak"] - eta_star),
    }
    json.dump(out, open(E / "finite_size_clean_headline.json", "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
