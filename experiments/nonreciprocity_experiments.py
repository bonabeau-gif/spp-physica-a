"""Extended non-reciprocity experiments for the two-species model.

The cross-species couplings decompose into symmetric and antisymmetric parts,
    J = [[1, j_s+j_a],[j_s-j_a, 1]],   C = [[0, f_s+f_a],[f_s-f_a, 0]],
where j_a and f_a are the antisymmetric (non-reciprocal) alignment and force
components. A reciprocal mixture has j_a = f_a = 0. These experiments quantify how
non-reciprocity generates persistent rotation:

  1. a two-dimensional (j_a, f_a) phase diagram of chirality, rotation rate,
     drift, and polarization at fixed symmetric couplings, and
  2. a finely sampled one-dimensional cut in j_a with many seeds, from which the
     onset of persistent rotation is located by the peak of the across-seed
     susceptibility chi = N * Var(|omega|) -- the same estimator used for the
     phase-transition analysis.

Outputs land in results/expanded/ and figures/expanded/.
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

from spp_unified.parameters import ACTIVE_SINGLE_NAMES, SingleParams, TwoSpeciesParams, SimulationConfig
from spp_unified.model import simulate_two_species
from experiments.expanded_common import run_parallel

OUT = ROOT / "results" / "expanded"
FIG = ROOT / "figures" / "expanded"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)


def _base_chiral_rule():
    """Base parameters of the strongest chiral rule from the non-reciprocal archive."""
    non = pd.read_csv(ROOT / "results" / "multi_qd_nonreciprocal.csv")
    row = non.loc[non.chirality.idxmax()]
    base = SingleParams.from_active_vector(np.array([row[n] for n in ACTIVE_SINGLE_NAMES], float))
    return base, float(row.fraction_B), float(row.alignment_symmetric), float(row.force_symmetric)


BASE, FRAC_B, J_SYM, F_SYM = _base_chiral_rule()
STEPS = 240
SAMPLE_START = 120


def _measure(ja, fa, seed, steps=STEPS):
    p = TwoSpeciesParams(base=BASE, fraction_B=FRAC_B, alignment_symmetric=J_SYM,
                         alignment_antisymmetric=float(ja), force_symmetric=F_SYM,
                         force_antisymmetric=float(fa))
    cfg = SimulationConfig(S=4.0, r0=0.1, steps=steps, sample_start=steps // 2,
                           sample_every=2, boundary="periodic", seed=int(seed))
    return simulate_two_species(p, cfg).summary


def _grid_task(task):
    ja, fa, seeds = task
    rows = [_measure(ja, fa, s) for s in seeds]
    keys = ["chirality", "abs_omega", "rotation_r2", "drift", "phi", "segregation"]
    out = {"alignment_antisymmetric": ja, "force_antisymmetric": fa}
    for k in keys:
        out[k] = float(np.mean([r[k] for r in rows]))
    out["n"] = float(rows[0]["n"])
    return out


def _cut_task(task):
    ja, seeds = task
    omg = np.array([_measure(ja, 0.0, s)["abs_omega"] for s in seeds])
    chi = np.array([_measure(ja, 0.0, s)["chirality"] for s in seeds])
    n = _measure(ja, 0.0, seeds[0])["n"]
    return {"alignment_antisymmetric": float(ja),
            "abs_omega_mean": float(omg.mean()), "abs_omega_var": float(omg.var()),
            "susceptibility": float(n * omg.var()),
            "chirality_mean": float(chi.mean()), "n": float(n)}


def run_phase_diagram(quick=False):
    g = 7 if quick else 13
    seeds = list(range(6 if quick else 8))
    jas = np.linspace(-1.5, 1.5, g)
    fas = np.linspace(-1.5, 1.5, g)
    tasks = [(round(float(ja), 4), round(float(fa), 4), seeds) for ja in jas for fa in fas]
    res = run_parallel(_grid_task, tasks, max_workers=6, desc="nonrecip:grid")
    df = pd.DataFrame(res)
    df.to_csv(OUT / "nonreciprocity_phase_diagram.csv", index=False)
    return df, jas, fas


def run_susceptibility_cut(quick=False):
    g = 15 if quick else 25
    seeds = list(range(10 if quick else 20))
    jas = np.linspace(-1.6, 1.6, g)
    tasks = [(round(float(ja), 4), seeds) for ja in jas]
    res = run_parallel(_cut_task, tasks, max_workers=6, desc="nonrecip:cut")
    df = pd.DataFrame(res).sort_values("alignment_antisymmetric").reset_index(drop=True)
    df.to_csv(OUT / "nonreciprocity_susceptibility_cut.csv", index=False)
    return df


def make_figure(grid, jas, fas, cut):
    fig, axs = plt.subplots(1, 3, figsize=(10.4, 3.3), constrained_layout=True)

    def heat(ax, col, label, cmap="viridis"):
        A = grid.pivot(index="force_antisymmetric", columns="alignment_antisymmetric", values=col)
        im = ax.imshow(A.to_numpy(), origin="lower", aspect="auto", cmap=cmap,
                       extent=(jas.min(), jas.max(), fas.min(), fas.max()))
        ax.axhline(0, color="w", lw=0.6, ls=":"); ax.axvline(0, color="w", lw=0.6, ls=":")
        ax.set_xlabel("antisymmetric alignment $j_a$")
        ax.set_ylabel("antisymmetric force $f_a$")
        ax.set_title(label, fontsize=9)
        fig.colorbar(im, ax=ax, pad=0.01)

    heat(axs[0], "chirality", "persistent chirality")
    heat(axs[1], "abs_omega", "rotation rate |ω|", cmap="magma")

    ax = axs[2]
    ax.plot(cut.alignment_antisymmetric, cut.abs_omega_mean, "o-", color="#2C6DB2",
            label="|ω| (order parameter)")
    ax.set_xlabel("antisymmetric alignment $j_a$")
    ax.set_ylabel("rotation rate |ω|", color="#2C6DB2")
    ax.tick_params(axis="y", labelcolor="#2C6DB2")
    ax2 = ax.twinx()
    ax2.plot(cut.alignment_antisymmetric, cut.susceptibility, "s--", color="#C74343",
             ms=3, label="susceptibility")
    ax2.set_ylabel("susceptibility  $N\\,\\mathrm{Var}(|ω|)$", color="#C74343")
    ax2.tick_params(axis="y", labelcolor="#C74343")
    jstar = float(cut.loc[cut.susceptibility.idxmax(), "alignment_antisymmetric"])
    ax.axvline(jstar, color="0.4", ls=":", lw=1)
    ax.set_title(f"reciprocal→chiral onset near $j_a$≈{jstar:.2f}", fontsize=9)
    fig.savefig(FIG / "fig_nonreciprocity.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def main():
    quick = "--quick" in sys.argv
    grid, jas, fas = run_phase_diagram(quick)
    cut = run_susceptibility_cut(quick)
    make_figure(grid, jas, fas, cut)
    jstar = float(cut.loc[cut.susceptibility.idxmax(), "alignment_antisymmetric"])
    recip = grid[(grid.alignment_antisymmetric.abs() < 1e-6) & (grid.force_antisymmetric.abs() < 1e-6)]
    print(f"base chiral rule: j_s={J_SYM:.3f} f_s={F_SYM:.3f} frac_B={FRAC_B:.3f}")
    print(f"reciprocal-point chirality (j_a=f_a=0): {float(recip.chirality.mean()):.3f}")
    print(f"max chirality on grid: {float(grid.chirality.max()):.3f}")
    print(f"susceptibility-peak onset j_a* = {jstar:.3f}")
    print("wrote", OUT / "nonreciprocity_phase_diagram.csv", OUT / "nonreciprocity_susceptibility_cut.csv")


if __name__ == "__main__":
    main()
