"""Assessment #5: add finite-size scaling to the noise-driven order-disorder cut.

The single-size susceptibility peak and Binder cumulant do not by themselves
localize a transition; the diagnostic power of the Binder cumulant is the
crossing of curves at different system sizes. We therefore run the Vicsek-like
noise cut at four particle numbers (varying the box at fixed density and fixed
interactions) and locate the transition by the Binder crossing. We also validate
the model-free descriptor-rate scout by comparing its peak with the
finite-size-scaling estimate of the transition.
"""
from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
from spp_unified.parameters import SingleParams, SimulationConfig
from spp_unified.model import simulate_single
from spp_unified.learned import behavior_descriptor
from expanded_common import run_parallel

OUT = ROOT / "results" / "expanded"


def _task(t):
    eta, S, seeds, dseeds, steps = t
    p = SingleParams(rho=0.15, v0=0.05, inertia=0.3, interaction_range_ratio=5.0,
                     force_strength=0.0, adhesion=0.0, alignment_range_ratio=8.0,
                     alignment_strength=3.5, noise=float(eta))
    phis = []
    n = None
    for s in seeds:
        cfg = SimulationConfig(S=S, r0=0.1, steps=steps, sample_start=steps // 2,
                               sample_every=2, boundary="periodic", seed=int(s))
        r = simulate_single(p, cfg)
        phis.append(r.summary["phi"]); n = r.summary["n"]
    phis = np.asarray(phis)
    descs = []
    for s in dseeds:
        cfg = SimulationConfig(S=S, r0=0.1, steps=steps, sample_start=steps // 2,
                               sample_every=2, boundary="periodic", seed=int(s))
        rr = simulate_single(p, cfg, record_states=True, state_every=8)
        descs.append(behavior_descriptor(rr.states, cfg, nbins=12, max_frames=10))
    O2 = float((phis ** 2).mean()); O4 = float((phis ** 4).mean())
    binder = 1.0 - O4 / (3.0 * O2 * O2) if O2 > 1e-12 else 0.0
    return {"eta": float(eta), "S": S, "N": int(n), "phi_mean": float(phis.mean()),
            "phi_var": float(phis.var()), "susceptibility": int(n) * float(phis.var()),
            "binder": float(binder), "descriptor": np.asarray(descs).mean(axis=0)}


def main():
    etas = np.round(np.linspace(0.20, 0.95, 16), 4)
    S_list = [2.6, 3.6, 5.0, 6.6]      # rho=0.15 -> N ~ 29, 56, 108, 189
    seeds = list(range(40)); dseeds = list(range(200, 204)); steps = 220
    tasks = [(e, S, seeds, dseeds, steps) for S in S_list for e in etas]
    res = run_parallel(_task, tasks, max_workers=8, desc="finite-size noise cut")
    rows = [{k: v for k, v in r.items() if k != "descriptor"} for r in res]
    df = pd.DataFrame(rows).sort_values(["N", "eta"]).reset_index(drop=True)
    df.to_csv(OUT / "finite_size_noise.csv", index=False)

    # Binder crossing between the two largest sizes (finite-size estimate of eta_c)
    def crossing(nA, nB):
        a = df[df.N == nA].sort_values("eta"); b = df[df.N == nB].sort_values("eta")
        e = a.eta.to_numpy(); d = a.binder.to_numpy() - np.interp(e, b.eta, b.binder)
        sign = np.sign(d)
        idx = np.where(np.diff(sign) != 0)[0]
        cr = []
        for i in idx:
            # linear interpolation of the zero crossing
            x0, x1 = e[i], e[i + 1]; y0, y1 = d[i], d[i + 1]
            cr.append(float(x0 - y0 * (x1 - x0) / (y1 - y0)))
        return cr

    Ns = sorted(df.N.unique())
    crossings = crossing(Ns[-2], Ns[-1])
    eta_c = float(np.median(crossings)) if crossings else float("nan")

    # descriptor-rate scout at the baseline size
    base = df[df.N == 56].sort_values("eta")
    D = np.vstack([r["descriptor"] for r in res if r["N"] == 56 and r["S"] == 3.6])
    # recompute descriptor rate along eta at S=3.6
    sub = [r for r in res if abs(r["S"] - 3.6) < 1e-9]
    sub.sort(key=lambda r: r["eta"])
    Dv = np.vstack([r["descriptor"] for r in sub]); ev = np.array([r["eta"] for r in sub])
    scale = np.linalg.norm(Dv.std(0)) + 1e-9
    rate = np.zeros(len(ev))
    for i in range(1, len(ev) - 1):
        rate[i] = np.linalg.norm(Dv[i + 1] - Dv[i - 1]) / (ev[i + 1] - ev[i - 1]) / scale
    rate[0] = rate[1]; rate[-1] = rate[-2]
    scout_peak = float(ev[int(np.argmax(rate))])

    # susceptibility peak at largest N
    big = df[df.N == Ns[-1]]
    chi_peak = float(big.loc[big.susceptibility.idxmax(), "eta"])

    headline = {
        "sizes_N": [int(n) for n in Ns],
        "binder_crossing_eta_c": eta_c,
        "binder_crossings_all": crossings,
        "susceptibility_peak_largest_N": chi_peak,
        "descriptor_scout_peak": scout_peak,
        "scout_bias_vs_crossing": scout_peak - eta_c,
    }
    with open(OUT / "finite_size_headline.json", "w") as f:
        json.dump(headline, f, indent=2)
    print(json.dumps(headline, indent=2))


if __name__ == "__main__":
    main()
