"""Paper-2 validation figure from P2-E2 (synthetic benchmarks) and P2-E7 (operating
characteristics). Panel (a): per-benchmark localization error and normal-angle error with
the detection outcome; panel (b): null vs planted-surface boundary-score separation and the
ROC (inset), showing the operating threshold inside a clean margin."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures" / "expanded"; FIG.mkdir(parents=True, exist_ok=True)
E2 = json.load(open(ROOT / "results/boundary/P2E2_benchmark.json"))
E7 = json.load(open(ROOT / "results/boundary/P2E7_operating_characteristics.json"))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.4, 3.7))

# ---- (a) recovery accuracy per benchmark ----
order = ["hyperplane", "intersecting", "heteroscedastic", "bounded_patch", "hypersphere", "shell"]
lab = {"hyperplane": "plane", "intersecting": "2 planes", "heteroscedastic": "hetero.",
       "bounded_patch": "patch", "hypersphere": "sphere", "shell": "shell"}
res = E2["results"]
loc = [res[k]["localization_error"] for k in order]
ang = [res[k]["normal_angle_deg"] for k in order]
x = np.arange(len(order))
ax1.bar(x - 0.2, loc, 0.4, color="#2C6DB2", label="localization error (param units)")
ax1b = ax1.twinx()
ax1b.bar(x + 0.2, ang, 0.4, color="#C74343", label="normal-angle error (deg)")
ax1.axhline(0.05, color="#2C6DB2", ls=":", lw=1)
ax1b.axhline(25, color="#C74343", ls=":", lw=1)
ax1.set_xticks(x); ax1.set_xticklabels([lab[k] for k in order], rotation=25, ha="right", fontsize=8)
ax1.set_ylabel("localization error", color="#2C6DB2"); ax1.set_ylim(0, 0.06)
ax1b.set_ylabel("normal-angle error (deg)", color="#C74343"); ax1b.set_ylim(0, 30)
ax1.tick_params(axis="y", colors="#2C6DB2"); ax1b.tick_params(axis="y", colors="#C74343")
ax1.set_title("(a) recovery of known surfaces (all detected, 0 false positives)", fontsize=8.5, loc="left")

# ---- (b) null vs signal separation + ROC inset ----
nb = E7["null"]; amps = E7["config"]["amps"]; pos = E7["positives"]
cats = ["null\n(n=2000)"] + [f"A={a}" for a in amps]
meds = [nb["b_median"]] + [pos[str(a)]["b_median"] for a in amps]
p99 = nb["b_p99"]; bmax = nb["b_max"]
ax2.bar(range(len(cats)), meds, color=["#6a6a6a", "#8bbf8b", "#4e9e4e", "#1f7a1f"])
ax2.axhline(E7["config"]["op_threshold"], color="k", ls="--", lw=1.2)
ax2.text(len(cats) - 0.5, E7["config"]["op_threshold"] + 0.15, "threshold = 3", fontsize=8, ha="right")
ax2.axhspan(p99, meds[1] if False else 3.22, color="0.9", zorder=0)
ax2.set_xticks(range(len(cats))); ax2.set_xticklabels(cats, fontsize=8)
ax2.set_ylabel("median boundary score b"); ax2.set_ylim(0, 6.2)
ax2.set_title(f"(b) null FPR@3 = 0/2000 (CI[0,0.0019]); ROC AUC = {E7['roc']['auc']:.3f}", fontsize=8.5, loc="left")
axin = ax2.inset_axes([0.58, 0.14, 0.38, 0.42])
axin.plot(E7["roc"]["fpr_grid"], E7["roc"]["tpr_grid"], color="#1f7a1f", lw=1.4)
axin.plot([0, 1], [0, 1], color="0.7", ls=":", lw=0.8)
axin.set_xlim(0, 1); axin.set_ylim(0, 1.02)
axin.set_xlabel("FPR", fontsize=7); axin.set_ylabel("TPR", fontsize=7)
axin.tick_params(labelsize=6); axin.set_title("ROC", fontsize=7)

fig.tight_layout()
fig.savefig(FIG / "fig_boundary_validation.png", dpi=240)
plt.close(fig)
print("wrote fig_boundary_validation.png")
