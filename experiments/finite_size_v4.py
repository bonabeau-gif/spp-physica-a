"""v4 T6/T7 (+ optional T4 third length): estimate the integrated autocorrelation
time tau of phi within the sampled window at each system size, report the
window/tau factor at the largest N, and add an 880-step run at the largest N over
the peak region to confirm the chi = A + B/T decomposition with three lengths.
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
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments"))
from spp_unified.parameters import SingleParams, SimulationConfig
from spp_unified.model import simulate_single
from expanded_common import run_parallel

OUT = ROOT / "results" / "expanded"
S_LIST = [2.6, 3.6, 5.0, 6.6, 8.6]   # N ~ 29,56,108,189,320
ETA_STAR = 0.50


def _p(eta):
    return SingleParams(rho=0.15, v0=0.05, inertia=0.3, interaction_range_ratio=5.0,
                        force_strength=0.0, adhesion=0.0, alignment_range_ratio=8.0,
                        alignment_strength=3.5, noise=float(eta))


def _tau_task(t):
    S, seed, steps = t
    cfg = SimulationConfig(S=S, r0=0.1, steps=steps, sample_start=steps // 2,
                           sample_every=1, boundary="periodic", seed=int(seed))
    r = simulate_single(_p(ETA_STAR), cfg)
    return S, int(r.summary["n"]), np.asarray(r.timeseries["phi"], float)


def integ_tau(series_list):
    """Integrated autocorrelation time (in samples) averaged over seeds, via the
    automatic-windowing estimator (Sokal)."""
    taus = []
    for x in series_list:
        x = x - x.mean()
        n = len(x)
        if n < 8 or x.std() == 0:
            continue
        ac = np.correlate(x, x, "full")[n - 1:]
        ac = ac / ac[0]
        tau = 1.0
        for k in range(1, n):
            tau += 2 * ac[k]
            if k >= 6 * tau:   # Sokal automatic window
                break
        taus.append(max(tau, 1.0))
    return float(np.mean(taus)), float(np.std(taus))


def _chi_task(t):
    eta, S, steps, nseed = t
    p = _p(eta)
    phis = []
    n = None
    for s in range(nseed):
        cfg = SimulationConfig(S=S, r0=0.1, steps=steps, sample_start=steps // 2,
                               sample_every=2, boundary="periodic", seed=int(s))
        r = simulate_single(p, cfg); phis.append(r.summary["phi"]); n = r.summary["n"]
    phis = np.asarray(phis)
    return {"eta": float(eta), "S": S, "steps": steps, "N": int(n),
            "susceptibility": int(n) * float(phis.var())}


def main():
    # --- tau at each N (record phi series over a 400-step window) ---
    tau_rows = []
    for S in S_LIST:
        tasks = [(S, s, 800) for s in range(16)]
        res = run_parallel(_tau_task, tasks, max_workers=6, desc=f"tau S={S}")
        N = res[0][1]
        tau, tau_sd = integ_tau([r[2] for r in res])   # in samples (sample_every=1 -> steps)
        tau_rows.append({"S": S, "N": N, "tau_steps": round(tau, 1), "tau_sd": round(tau_sd, 1),
                         "window_steps": 110, "window_over_tau": round(110.0 / tau, 2)})
        print(tau_rows[-1], flush=True)
    pd.DataFrame(tau_rows).to_csv(OUT / "finite_size_v4_tau.csv", index=False)

    # --- 880-step run at largest N over the peak region (third length) ---
    etas = np.round(np.linspace(0.40, 0.65, 6), 3)
    tasks = [(e, S_LIST[-1], 880, 48) for e in etas]
    r880 = run_parallel(_chi_task, tasks, max_workers=6, desc="chi 880 largest N")
    pd.DataFrame(r880).to_csv(OUT / "finite_size_v4_chi880.csv", index=False)

    head = {"tau_by_N": {r["N"]: r["tau_steps"] for r in tau_rows},
            "window_over_tau_largest_N": tau_rows[-1]["window_over_tau"],
            "chi880_peak": round(float(max(x["susceptibility"] for x in r880)), 4)}
    json.dump(head, open(OUT / "finite_size_v4_headline.json", "w"), indent=2)
    print(json.dumps(head, indent=2))


if __name__ == "__main__":
    main()
