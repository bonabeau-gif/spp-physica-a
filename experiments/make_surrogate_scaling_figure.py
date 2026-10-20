"""Coverage-vs-budget crossover figure (assessment v2, largest gap)."""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
E = ROOT / "results" / "expanded"
FIG = ROOT / "figures" / "expanded"


def main():
    s = pd.read_csv(E / "surrogate_scaling_summary.csv")
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    styles = {"surrogate": ("#b2182b", "o", "GP surrogate"),
              "direct": ("#2166ac", "s", "direct mutation"),
              "random": ("#1a9850", "^", "random")}
    for meth, (c, mk, lab) in styles.items():
        d = s[s.method == meth].sort_values("budget")
        ax.errorbar(d.budget, d.coverage_mean, yerr=d.coverage_sd, marker=mk, color=c,
                    capsize=3, lw=1.6, label=lab)
    ax.set_xscale("log")
    ax.set_xlabel("additional simulator calls")
    ax.set_ylabel("learned-archive coverage")
    ax.set_ylim(0, 0.9)
    ax.axvspan(300, 3000, color="0.9", zorder=0)
    ax.text(950, 0.06, "crossover", fontsize=8, color="0.4", ha="center")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    ax.set_title("Surrogate assistance helps only at large budgets", fontsize=9.5, loc="left")
    fig.tight_layout(); fig.savefig(FIG / "fig_surrogate_scaling.png", dpi=240); plt.close(fig)
    print("wrote fig_surrogate_scaling.png")


if __name__ == "__main__":
    main()
