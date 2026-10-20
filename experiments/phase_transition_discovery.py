"""Locating phase transitions in the high-dimensional SPP parameter space.

A phase diagram of an eleven-parameter model cannot be charted exhaustively. This
script demonstrates a practical, model-agnostic recipe for finding transitions
along informative one-dimensional cuts, and for cross-checking them with three
independent estimators:

  * the fluctuation susceptibility  chi(λ) = N * Var_s O(λ)  of an order parameter
    O across independent seeds, whose peak marks a transition;
  * the Binder cumulant  U(λ) = 1 - <O^4> / (3 <O^2>^2), whose drop/crossing
    locates the same point without a priori knowledge of the ordered phase; and
  * the behavioral-descriptor change rate  ||dΨ/dλ||, a hand-design-free detector
    computed from the 51-dimensional invariant trajectory descriptor, so that
    transitions in states that have no obvious scalar order parameter are still
    exposed.

Two cuts are shown: the noise-driven order-disorder transition of a Vicsek-like
limit, and the density-driven loss of polar order. The same estimators are reused
for the reciprocal->chiral onset in nonreciprocity_experiments.py.
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
sys.path.insert(0, str(ROOT))

from spp_unified.parameters import SingleParams, SimulationConfig
from spp_unified.model import simulate_single
from spp_unified.learned import behavior_descriptor
from experiments.expanded_common import run_parallel

OUT = ROOT / "results" / "expanded"
FIG = ROOT / "figures" / "expanded"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)


def _base(cut, value):
    if cut == "noise":
        return SingleParams(rho=0.15, v0=0.05, inertia=0.3, interaction_range_ratio=5.0,
                            force_strength=0.0, adhesion=0.0, alignment_range_ratio=8.0,
                            alignment_strength=3.5, noise=float(value)), "periodic"
    if cut == "density":
        return SingleParams(rho=float(value), v0=0.05, inertia=0.5, interaction_range_ratio=7.0,
                            force_strength=1.0, adhesion=0.0, alignment_range_ratio=7.0,
                            alignment_strength=2.5, noise=0.10), "periodic"
    raise ValueError(cut)


def _point_task(task):
    """Order-parameter moments (many seeds) + averaged descriptor (few seeds) at one λ."""
    cut, value, seeds, dseeds, steps = task
    p, boundary = _base(cut, value)
    phis = []
    for s in seeds:
        cfg = SimulationConfig(S=3.6, r0=0.1, steps=steps, sample_start=steps // 2,
                               sample_every=2, boundary=boundary, seed=int(s))
        phis.append(simulate_single(p, cfg).summary["phi"])
    phis = np.asarray(phis)
    descs = []
    n = None
    for s in dseeds:
        cfg = SimulationConfig(S=3.6, r0=0.1, steps=steps, sample_start=steps // 2,
                               sample_every=2, boundary=boundary, seed=int(s))
        r = simulate_single(p, cfg, record_states=True, state_every=8)
        descs.append(behavior_descriptor(r.states, cfg, nbins=12, max_frames=10))
        n = r.summary["n"]
    O1 = float(phis.mean()); O2 = float((phis ** 2).mean()); O4 = float((phis ** 4).mean())
    var = max(O2 - O1 * O1, 0.0)
    binder = 1.0 - O4 / (3.0 * O2 * O2) if O2 > 1e-12 else 0.0
    return {"cut": cut, "value": float(value), "n": float(n),
            "phi_mean": O1, "phi_var": var, "susceptibility": float(n) * var,
            "binder": float(binder), "descriptor": np.asarray(descs).mean(axis=0)}


def run_cut(cut, values, quick=False):
    seeds = list(range(8 if quick else 16))
    dseeds = list(range(100, 102 if quick else 103))
    steps = 160 if quick else 220
    tasks = [(cut, round(float(v), 5), seeds, dseeds, steps) for v in values]
    res = run_parallel(_point_task, tasks, max_workers=6, desc=f"transition:{cut}")
    res.sort(key=lambda d: d["value"])
    D = np.vstack([r["descriptor"] for r in res])
    vals = np.array([r["value"] for r in res])
    # Central-difference descriptor change rate, normalized by the descriptor scale.
    rate = np.zeros(len(vals))
    scale = np.linalg.norm(D.std(axis=0)) + 1e-9
    for i in range(1, len(vals) - 1):
        rate[i] = np.linalg.norm(D[i + 1] - D[i - 1]) / (vals[i + 1] - vals[i - 1]) / scale
    rate[0] = rate[1]; rate[-1] = rate[-2]
    df = pd.DataFrame([{k: v for k, v in r.items() if k != "descriptor"} for r in res])
    df["descriptor_rate"] = rate
    df.to_csv(OUT / f"phase_transition_{cut}.csv", index=False)
    return df


LABEL = {"noise": ("noise η", "η"), "density": ("density fraction ρ", "ρ")}


def make_figure(cuts):
    fig, axs = plt.subplots(len(cuts), 3, figsize=(10.2, 3.0 * len(cuts)), constrained_layout=True)
    axs = np.atleast_2d(axs)
    for r, (cut, df) in enumerate(cuts.items()):
        xlab, sym = LABEL[cut]
        star_chi = float(df.loc[df.susceptibility.idxmax(), "value"])
        star_rate = float(df.loc[df.descriptor_rate.idxmax(), "value"])
        a = axs[r, 0]
        a.plot(df.value, df.phi_mean, "o-", color="#2C6DB2")
        a.fill_between(df.value, df.phi_mean - np.sqrt(df.phi_var), df.phi_mean + np.sqrt(df.phi_var),
                       color="#2C6DB2", alpha=0.15)
        a.set(xlabel=xlab, ylabel="polar order φ", ylim=(0, 1.03))
        a.set_title(f"{cut}: order parameter", fontsize=9)
        b = axs[r, 1]
        b.plot(df.value, df.susceptibility, "s-", color="#C74343", label="susceptibility")
        b.axvline(star_chi, color="#C74343", ls=":", lw=1)
        b.set(xlabel=xlab, ylabel="susceptibility $N\\,\\mathrm{Var}(φ)$")
        bb = b.twinx()
        bb.plot(df.value, df.binder, "^--", color="#4a7", ms=3)
        bb.set_ylabel("Binder cumulant U", color="#4a7")
        bb.tick_params(axis="y", labelcolor="#4a7")
        b.set_title(f"transition (χ peak) {sym}*≈{star_chi:.2f}", fontsize=9)
        c = axs[r, 2]
        c.plot(df.value, df.descriptor_rate, "d-", color="#7a4fb5")
        c.axvline(star_rate, color="#7a4fb5", ls=":", lw=1)
        c.set(xlabel=xlab, ylabel="descriptor change rate $\\|dΨ/d%s\\|$" % sym)
        c.set_title(f"model-free detector {sym}*≈{star_rate:.2f}", fontsize=9)
        for ax in (a, b, c):
            ax.spines["top"].set_visible(False)
    fig.savefig(FIG / "fig_phase_transitions.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def main():
    quick = "--quick" in sys.argv
    g = 13 if quick else 21
    cuts = {}
    cuts["noise"] = run_cut("noise", np.linspace(0.0, 1.0, g), quick)
    cuts["density"] = run_cut("density", np.linspace(0.05, 0.70, g), quick)
    make_figure(cuts)
    for cut, df in cuts.items():
        sym = LABEL[cut][1]
        print(f"{cut}: susceptibility peak {sym}*={float(df.loc[df.susceptibility.idxmax(),'value']):.3f}; "
              f"descriptor-rate peak {sym}*={float(df.loc[df.descriptor_rate.idxmax(),'value']):.3f}")
    print("wrote phase_transition_*.csv and fig_phase_transitions.png")


if __name__ == "__main__":
    main()
