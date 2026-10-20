"""Figure for the finite-size equilibration checks (assessment v2 problem #2).
Panels: (a) susceptibility chi(eta) per N; (b) chi_max vs N (log-log) at two
trajectory lengths -> exponent stability; (c) Var_s(phi) vs N at eta*, ordered,
and disordered -> self-averaging off-transition but growth in the ordered phase;
(d) per-seed phi distributions at eta* per N -> unimodal (no coexistence)."""
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
    grid = pd.read_csv(E / "finite_size_v2_grid.csv")
    ps = pd.read_csv(E / "finite_size_v2_perseed.csv")
    g220 = grid[grid.steps == 220]; g440 = grid[grid.steps == 440]
    Ns = sorted(g220.N.unique())
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(Ns)))
    fig, ax = plt.subplots(1, 4, figsize=(14.5, 3.5))

    # (a) susceptibility
    for c, N in zip(colors, Ns):
        s = g220[g220.N == N].sort_values("eta")
        ax[0].plot(s.eta, s.susceptibility, "-o", ms=3, color=c, label=f"N={N}")
    eta_star = float(g220[g220.N == Ns[-1]].sort_values("susceptibility").eta.iloc[-1])
    ax[0].axvline(eta_star, color="0.5", ls=":", lw=1)
    ax[0].set(xlabel="noise η", ylabel=r"susceptibility $N\,\mathrm{Var}(φ)$")
    ax[0].set_title("(a) susceptibility peaks (η*≈%.2f)" % eta_star, fontsize=9, loc="left")
    ax[0].legend(frameon=False, fontsize=7)

    # (b) chi_max(220), chi_max(440) and window-independent amplitude A = 2chi440 - chi220
    c220 = np.array([g220[g220.N == n].susceptibility.max() for n in Ns])
    # chi(440) interpolated at each N's 440-grid peak
    c440, Avals = [], []
    for n in Ns:
        e2 = g220[g220.N == n].sort_values("eta").eta.to_numpy()
        x2 = g220[g220.N == n].sort_values("eta").susceptibility.to_numpy()
        e4 = g440[g440.N == n].sort_values("eta").eta.to_numpy()
        x4 = g440[g440.N == n].sort_values("eta").susceptibility.to_numpy()
        ip = int(np.argmax(x4))
        c440.append(x4[ip]); Avals.append(2 * x4[ip] - float(np.interp(e4[ip], e2, x2)))
    c440 = np.array(c440); Avals = np.array(Avals)
    ax[1].plot(Ns, c220, "o-", color="#2166ac", label=r"$\chi$ (220 steps)")
    ax[1].plot(Ns, c440, "s-", color="#67a9cf", label=r"$\chi$ (440 steps)")
    ax[1].plot(Ns, Avals, "d--", color="#b2182b", label=r"$A=2\chi_{440}-\chi_{220}$")
    ax[1].axhline(0, color="0.6", lw=0.8, zorder=0)
    ax[1].set_xscale("log")
    ax[1].set(xlabel="N", ylabel=r"peak susceptibility")
    ax[1].set_title("(b) peak height: χ vs window-independent A", fontsize=9, loc="left")
    ax[1].legend(frameon=False, fontsize=7.5)

    # (c) Var_s vs N at eta*, ordered (0.30), disordered (0.90)
    def var_at(g, target):
        out = []
        for n in Ns:
            s = g[g.N == n]
            i = (s.eta - target).abs().idxmin()
            out.append(float(s.loc[i, "phi_var"]))
        return out
    for target, lab, col in [(0.30, "ordered (η=0.30)", "#2166ac"),
                             (eta_star, f"near η*≈{eta_star:.2f}", "#b2182b"),
                             (0.90, "disordered (η=0.90)", "#1a9850")]:
        ax[2].loglog(Ns, var_at(g220, target), "-o", ms=3, color=col, label=lab)
    ax[2].set(xlabel="N", ylabel=r"across-seed $\mathrm{Var}(φ)$")
    ax[2].set_title("(c) self-averaging check", fontsize=9, loc="left")
    ax[2].legend(frameon=False, fontsize=7)

    # (d) phi distributions at eta* per N
    eta_grid = float(ps.eta.unique()[np.argmin(np.abs(ps.eta.unique() - eta_star))])
    rng = np.random.default_rng(0)
    for i, N in enumerate(Ns):
        v = ps[(ps.N == N) & (np.isclose(ps.eta, eta_grid))].phi.to_numpy()
        ax[3].scatter(np.full(len(v), i) + rng.normal(0, 0.05, len(v)), v, s=10,
                      color=colors[i], alpha=0.7, edgecolors="none")
        ax[3].plot([i - 0.25, i + 0.25], [np.median(v)] * 2, color="k", lw=1.4)
    ax[3].set_xticks(range(len(Ns))); ax[3].set_xticklabels(Ns)
    ax[3].set(xlabel="N", ylabel=f"φ at η≈{eta_grid:.2f} (per seed)")
    ax[3].set_title("(d) φ distribution at η* (unimodal)", fontsize=9, loc="left")

    fig.tight_layout(); fig.savefig(FIG / "fig_finite_size.png", dpi=240); plt.close(fig)

    head = json.load(open(E / "finite_size_v2_headline.json"))
    print("eta*_220", head["eta_star_220"], "eta*_440", head["eta_star_440"])
    print("chi exponent 220", round(head["chi_exponent_220"], 2), "440", round(head["chi_exponent_440"], 2))
    print("wrote fig_finite_size.png; eta_star=", eta_star)


if __name__ == "__main__":
    main()
