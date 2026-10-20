"""Deliverable figures: revised Figure 28 (E6 intervals + E1 spectra) and the new
framework-characterization figure (E2 offsets, E5 false positives, E3 general-vs-psi_mix,
E4 control matrix)."""
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

OUT = ROOT / "results" / "boundary"; FIG = ROOT / "figures"
LB = {"rho": r"$\rho$", "eta": r"$\eta$", "J_a": r"$J_a$", "C_a": r"$C_a$", "J_s": r"$J_s$",
      "C_s": r"$C_s$", "f_B": r"$f_B$", "alpha": r"$\alpha$", "v0": r"$v_0$", "eps": r"$\epsilon$",
      "r1_ratio": r"$r_1$", "f0": r"$f_0$", "B": r"$B$", "ra_ratio": r"$r_a$"}


def revised_fig28():
    sp = P.get_space("two")
    tr = pickle.load(open(OUT / "traces.pkl", "rb"))
    e26 = json.load(open(OUT / "E2E6_estimators.json"))
    e1 = json.load(open(OUT / "E1_patch.json"))
    pts = np.array(tr["points"]); ptt = np.array(tr["pt_trace"])
    phys = np.array([P.from_unit(u, sp) for u in pts])
    var = phys.std(0) / (sp.bounds[:, 1] - sp.bounds[:, 0]); ai, aj = np.argsort(var)[::-1][:2]
    names = sp.param_order
    fig, ax = plt.subplots(2, 2, figsize=(9.4, 7.6))
    C = plt.cm.viridis(np.linspace(0.1, 0.9, max(1, ptt.max() + 1)))
    # (a) traced manifolds
    a = ax[0, 0]
    for ti in range(ptt.max() + 1):
        m = ptt == ti
        if m.sum(): a.plot(phys[m, ai], phys[m, aj], "-o", ms=4, color=C[ti], lw=1.2)
    a.set_xlabel(LB.get(names[ai], names[ai])); a.set_ylabel(LB.get(names[aj], names[aj]))
    a.set_title(f"(a) Traced boundary manifolds\n({len(pts)} points, {ptt.max()+1} traces)", fontsize=10)
    # (b) J_a susceptibility (C6) with E6 interval
    b = ax[0, 1]; c6 = e26["cuts"]["C6_Ja_chiral"]
    g = np.array(c6["susc_curve"]); b.plot(g[:, 0], g[:, 1], "s-", color="#C74343", ms=3)
    b.axvspan(c6["susc_ci"][0], c6["susc_ci"][1], color="#f0cccc", alpha=.6)
    rate = np.asarray(c6["rate_curve"])
    axis_rate = b.twinx()
    axis_rate.plot(rate[:, 0], rate[:, 1], "o--", color="#2C6DB2", ms=3)
    axis_rate.set_ylabel("Descriptor sensitivity", color="#2C6DB2")
    b.axvline(c6["desc_peak"], ls=":", color="#2C6DB2"); b.set_xlabel(r"$J_a$ (at $C_a=0$)")
    b.set_ylabel(r"$N\,\mathrm{Var}(|\omega|)$")
    b.set_title(f"(b) Reciprocal-to-chiral onset\n$J_a^*$={c6['susc_peak']:.2f} "
                f"[{c6['susc_ci'][0]:.2f},{c6['susc_ci'][1]:.2f}] vs 1.33", fontsize=10)
    # (c) eta: descriptor ridge vs susceptibility (C1) with both peaks
    c = ax[1, 0]; c1 = e26["cuts"]["C1_eta_vicsek"]
    gr = np.array(c1["rate_curve"]); gs = np.array(c1["susc_curve"])
    c.plot(gr[:, 0], gr[:, 1] / gr[:, 1].max(), "o-", color="#2C6DB2", ms=3, label="descriptor rate")
    c.plot(gs[:, 0], gs[:, 1] / gs[:, 1].max(), "s--", color="#C74343", ms=3, label="susceptibility")
    c.axvline(c1["desc_peak"], ls=":", color="#2C6DB2"); c.axvline(c1["susc_peak"], ls=":", color="#C74343")
    c.set_xlabel(r"$\eta$"); c.set_ylabel("normalized"); c.legend(fontsize=7)
    c.set_title(f"(c) Estimator offset (C1)\ndesc {c1['desc_peak']:.2f} vs susc {c1['susc_peak']:.2f}", fontsize=10)
    # (d) E1 predictor vs corrector spectra
    d = ax[1, 1]
    for p in e1["patches"]:
        sp_g = np.array(p["predictor_gate_spectrum"]); sc = np.array(p["corrector_spectrum"])
        col = "#6a51a3" if p["seed"] == "chiral_onset" else "#999999"
        if len(sp_g): d.plot(np.arange(1, len(sp_g) + 1), np.cumsum(sp_g) / sp_g.sum(), "-", color="#bbbbbb", lw=1)
        if len(sc): d.plot(np.arange(1, len(sc) + 1), np.cumsum(sc) / sc.sum(), "-o", color=col, ms=3,
                           label=("chiral (corrector)" if p["seed"] == "chiral_onset" else None))
    d.axhline(0.95, ls=":", color="#555"); d.set_xlabel("PCA component"); d.set_ylabel("cumulative variance")
    d.set_title("(d) Local dimension by patch sampling: predictor (grey)\nvs corrector clouds; chiral only marginally lower", fontsize=9.5)
    d.legend(fontsize=7)
    fig.suptitle("Mapping behavioral-boundary manifolds: implemented and credentialed",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96]); fig.savefig(FIG / "fig_boundary_result.png", dpi=260, bbox_inches="tight")
    plt.close(fig); print("wrote fig_boundary_result.png")


def characterization_fig():
    sp = P.get_space("two")
    e26 = json.load(open(OUT / "E2E6_estimators.json"))
    e5 = json.load(open(OUT / "E5_opchar.json"))
    e3 = json.load(open(OUT / "E3_full.json"))
    e4 = json.load(open(OUT / "E4_controls.json"))
    fig, ax = plt.subplots(2, 2, figsize=(9.4, 7.6))
    # (a) E2 offsets across cuts
    a = ax[0, 0]; cuts = list(e26["cuts"]); off = [e26["cuts"][c]["offset_norm"] for c in cuts]
    ci = [e26["cuts"][c]["offset_norm_ci"] for c in cuts]
    y = np.arange(len(cuts))
    a.errorbar(off, y, xerr=[[o - l[0] for o, l in zip(off, ci)], [l[1] - o for o, l in zip(off, ci)]],
               fmt="o", color="#2C6DB2", capsize=3)
    a.axvline(0, color="#555", lw=1); a.set_yticks(y); a.set_yticklabels([c.split("_")[0] for c in cuts], fontsize=8)
    a.set_xlabel("normalized offset (descriptor - susceptibility)")
    a.set_title(f"(a) E2: estimator offsets scatter in sign\nmean {e26['offset_mean']:+.2f}, std {e26['offset_std']:.2f}", fontsize=10)
    # (b) E5 false-positive summary
    b = ax[0, 1]
    b.bar(["general\noracle", "psi_mix\nnull"], [e5["false_positive_rate_flat"], json.load(open(OUT / "E3_null.json"))["null_fp_rate"]],
          color=["#2C6DB2", "#6a51a3"])
    b.axhline(0.05, ls=":", color="#C74343", label="5% target")
    b.set_ylabel("false-positive rate"); b.legend(fontsize=8)
    b.set_title(f"(b) E5/E3: false-positive rates\ngeneral {e5['false_positive_rate_flat']:.3f} "
                f"(b95={e5['fp_b_p95']:.1f})", fontsize=10)
    # (c) E3 general vs psi_mix mixture boundaries
    c = ax[1, 0]
    catalog = json.load(open(OUT / "manuscript_catalog.json"))["ridges"]
    classes = ["ambiguous", "candidate", "validated_mixture"]
    counts = [sum(row["classification"] == label for row in catalog) for label in classes]
    c.bar(["General\nambiguous", "Mixture\ncandidate", "Mixture\nvalidated"], counts,
          color=["#999999", "#e9b35c", "#2ca25f"])
    c.set_ylabel("Ridges / boundaries")
    c.set_title("(c) Mixture control status (Table S4)", fontsize=10)
    # (d) E4 control matrix
    d = ax[1, 1]
    types = [t["type"] for t in e4["types"]]
    M = np.array([[int(t["C1_reciprocal"]["survives"]), int(t["C2_single_subspace"]["found"]),
                   int(t["C3_species_swap"]["invariant"])] for t in e4["types"]])
    d.imshow(M, cmap="RdYlGn_r", vmin=0, vmax=1, aspect="auto")
    d.set_xticks([0, 1, 2]); d.set_xticklabels(["C1 recip\nsurvives", "C2 single\nfound", "C3 swap\ninvariant"], fontsize=8)
    d.set_yticks(range(len(types))); d.set_yticklabels([t.replace("_", " ") for t in types], fontsize=8)
    for i in range(len(types)):
        for j in range(3):
            d.text(j, i, "Y" if M[i, j] else "N", ha="center", va="center", fontsize=9)
    d.set_title("(d) E4: control battery outcomes", fontsize=10)
    fig.suptitle("Sensitivity-ridge detection and controls", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96]); fig.savefig(FIG / "fig_boundary_char.png", dpi=260, bbox_inches="tight")
    plt.close(fig); print("wrote fig_boundary_char.png")


if __name__ == "__main__":
    revised_fig28()
    characterization_fig()
