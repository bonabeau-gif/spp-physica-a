"""Build the REAL Figure 28 from the boundary-mapping campaign outputs.

Panels: (a) traced boundary manifolds projected onto the two parameters along which the
traces spread most; (b) T4 credentialing (|omega| susceptibility cut, peak ~ J_a*=1.33);
(c) T3 credentialing (descriptor-rate ridge in eta, peak ~ 0.5); (d) local manifold
dimension of the traced points. Replaces the schematic fig_boundary_framework.png.
"""
from __future__ import annotations
import sys, json, pickle
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from boundary import params as P

OUT = ROOT / "results" / "boundary"
FIG = ROOT / "figures"
PLAN_LABEL = {"rho": r"$\rho$", "eta": r"$\eta$", "J_a": r"$J_a$", "C_a": r"$C_a$",
              "J_s": r"$J_s$", "C_s": r"$C_s$", "f_B": r"$f_B$", "alpha": r"$\alpha$",
              "v0": r"$v_0$", "eps": r"$\epsilon$", "r1_ratio": r"$r_1$", "f0": r"$f_0$",
              "B": r"$B$", "ra_ratio": r"$r_a$"}


def main():
    sp = P.get_space("two")
    traces = pickle.load(open(OUT / "traces.pkl", "rb"))
    catalog = json.load(open(OUT / "catalog.json"))
    t3 = json.load(open(OUT / "T3_eta.json"))
    t4 = json.load(open(OUT / "T4_ja.json"))

    pts = np.array(traces["points"]); pt_trace = np.array(traces["pt_trace"])
    phys = np.array([P.from_unit(u, sp) for u in pts])
    var = phys.std(0) / (sp.bounds[:, 1] - sp.bounds[:, 0])
    ax_i, ax_j = np.argsort(var)[::-1][:2]
    names = sp.param_order

    fig, axes = plt.subplots(2, 2, figsize=(9.2, 7.4))
    C = plt.cm.viridis(np.linspace(0.1, 0.9, max(1, pt_trace.max() + 1)))

    # (a) traced manifolds
    ax = axes[0, 0]
    for ti in range(pt_trace.max() + 1):
        m = pt_trace == ti
        if m.sum() == 0:
            continue
        ax.plot(phys[m, ax_i], phys[m, ax_j], "-o", ms=4, color=C[ti], lw=1.2,
                label=f"trace {ti} ({m.sum()} pts)")
    ax.set_xlabel(PLAN_LABEL.get(names[ax_i], names[ax_i]))
    ax.set_ylabel(PLAN_LABEL.get(names[ax_j], names[ax_j]))
    ax.set_title(f"(a) Traced boundary manifolds\n({len(pts)} points, {pt_trace.max()+1} traces)", fontsize=10)
    ax.legend(fontsize=6.5, framealpha=0.6)

    # (b) T4 susceptibility cut
    ax = axes[0, 1]
    sc = t4["susceptibility_cut"]
    ja = [r["J_a"] for r in sc]; chi = [r["susceptibility"] for r in sc]
    ax.plot(ja, chi, "s-", color="#C74343", ms=4)
    ax.axvline(1.33, ls=":", color="#555", lw=1)
    ax.annotate(f"$J_a^*$={t4['ja_star_susceptibility']:.2f}\n(target 1.33)",
                xy=(t4["ja_star_susceptibility"], max(chi)), fontsize=8,
                xytext=(0.55, 0.7), textcoords="axes fraction")
    ax.set_xlabel(r"$J_a$ (at $C_a=0$)"); ax.set_ylabel(r"$N\,\mathrm{Var}(|\omega|)$")
    ax.set_title("(b) T4 credentialing:\nreciprocal-to-chiral onset", fontsize=10)

    # (c) T3 descriptor-rate ridge
    ax = axes[1, 0]
    rp = t3["rate_profile"]
    e = [r["eta"] for r in rp]; rr = [r["rate"] for r in rp]
    ax.plot(e, rr, "o-", color="#2C6DB2", ms=4)
    ax.axvspan(0.45, 0.55, color="#cfe0f0", alpha=0.5)
    ax.axvline(t3["eta_star"], ls=":", color="#555", lw=1)
    ax.annotate(rf"$\eta^*$={t3['eta_star']:.2f}", xy=(t3["eta_star"], max(rr)),
                fontsize=8, xytext=(0.6, 0.8), textcoords="axes fraction")
    ax.set_xlabel(r"$\eta$"); ax.set_ylabel("descriptor rate")
    ax.set_title("(c) T3 credentialing:\nnoise order-disorder ridge", fontsize=10)

    # (d) local manifold dimension histogram
    ax = axes[1, 1]
    mdims = [d for tr in traces["traces"] for d in tr["manifold_dim"]]
    if mdims:
        vals, counts = np.unique(mdims, return_counts=True)
        ax.bar(vals, counts, color="#6a51a3", width=0.6)
        ax.set_xticks(range(1, max(vals) + 1))
    ax.set_xlabel("local manifold dimension (PCA, 95% var)")
    ax.set_ylabel("traced points")
    ntypes = len(catalog.get("types", []))
    ax.set_title(f"(d) Boundary thinness\n({ntypes} typed boundaries cataloged)", fontsize=10)

    fig.suptitle("Mapping behavioral-boundary manifolds in the two-species SPP model",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(FIG / "fig_boundary_result.png", dpi=260, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote fig_boundary_result.png; axes=({names[ax_i]},{names[ax_j]}); "
          f"{len(pts)} pts, {ntypes} types")


if __name__ == "__main__":
    main()
