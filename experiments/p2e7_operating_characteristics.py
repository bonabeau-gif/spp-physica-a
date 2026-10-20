"""P2-E7: expanded null controls and operating characteristics for the boundary oracle.

The plan objects that "reporting 4.5% from 200 trials is not sufficient." Here the SAME
oracle (monkeypatched onto synthetic descriptors via p2e2_benchmark) is run on:
  * a large PURE-NOISE null ensemble (2000 points) -> false-positive rate at the operating
    threshold with a 95% Wilson interval, plus the full null b-score distribution;
  * planted TRUE surfaces at several signal-to-noise levels -> true-positive / false-negative
    rate and a detection-power curve;
  * a threshold sweep giving the full ROC and its AUC (separability of the noise floor).

Reuses p2e2_benchmark's monkeypatched pipeline unchanged (import applies the patch).
"""
from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments"))
import numpy as np
import p2e2_benchmark as BM   # importing applies the descriptor-backend monkeypatch

N_NULL = 2000
N_POS = 400
AMPS = [1.5, 2.5, 4.0]        # signal amplitudes (4.0 is the calibrated on-surface level)
OP_THR = BM.B_MIN             # operating decision threshold (3.0)


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (p, max(0.0, c - h), min(1.0, c + h))


def b_scores(sp, proto, points):
    return np.array([BM.sensitivity(sp, u, proto, h=0.04, n_side=8, n_floor=16).b for u in points])


def plane_bench(A):
    benches = BM.make_benches()
    hp = benches["hyperplane"]
    a = np.asarray(hp.gn(np.full(BM.D, 0.5)), float)
    g1 = hp.g
    sig = lambda u: A * np.tanh(g1(u) / BM.W) * BM.V
    sg = lambda u: BM.SIGMA * np.ones(BM.M)
    return BM.Bench(f"plane_A{A}", sig, sg, g1, lambda u: a, "codim1_flat", hp.onpoints)


def main():
    sp = BM.synth_space()
    proto = BM.Protocol(space="synth", n_steps=1)
    benches = BM.make_benches()

    # ---- null ensemble: 2000 pure-noise points ----
    BM.fit_standardizer(benches["null"])
    rng = np.random.default_rng(2027)
    null_pts = rng.random((N_NULL, BM.D))
    b_null = b_scores(sp, proto, null_pts)
    k_fp = int(np.sum(b_null >= OP_THR))
    fpr, fpr_lo, fpr_hi = wilson(k_fp, N_NULL)
    print(f"NULL n={N_NULL}: FPR@{OP_THR}={fpr:.4f} ({k_fp}/{N_NULL}) 95%CI[{fpr_lo:.4f},{fpr_hi:.4f}] "
          f"| b: med={np.median(b_null):.2f} p95={np.percentile(b_null,95):.2f} max={b_null.max():.2f}",
          flush=True)

    # ---- planted positives at several SNRs -> power curve + false-negative rate ----
    pos = {}
    for A in AMPS:
        pb = plane_bench(A)
        BM.fit_standardizer(pb)
        r = np.random.default_rng(int(A * 1000) + 5)
        on = pb.onpoints(N_POS, r)
        b_pos = b_scores(sp, proto, on)
        k_tp = int(np.sum(b_pos >= OP_THR))
        tpr, tlo, thi = wilson(k_tp, N_POS)
        pos[A] = {"tpr": tpr, "tpr_ci": [tlo, thi], "fnr": 1 - tpr,
                  "b_median": float(np.median(b_pos)), "b_scores": b_pos.tolist()}
        print(f"POS  A={A}: TPR@{OP_THR}={tpr:.3f} CI[{tlo:.3f},{thi:.3f}] FNR={1-tpr:.3f} "
              f"b_med={np.median(b_pos):.1f}", flush=True)

    # ---- ROC/AUC using the strongest planted surface vs the null ----
    b_pos_ref = np.array(pos[AMPS[-1]]["b_scores"])
    thr_grid = np.unique(np.concatenate([[0], np.linspace(0.5, 8, 200), b_null, b_pos_ref]))
    roc = [(float(np.mean(b_null >= t)), float(np.mean(b_pos_ref >= t))) for t in thr_grid]
    roc.sort()
    fx = [p[0] for p in roc]; ty = [p[1] for p in roc]
    auc = float(np.trapz(ty, fx))
    # empirical separation: highest null b vs lowest strong-positive b
    sep = float(np.min(b_pos_ref) - np.percentile(b_null, 99))
    print(f"ROC AUC={auc:.4f} | null p99={np.percentile(b_null,99):.2f} "
          f"pos(A={AMPS[-1]}) min={b_pos_ref.min():.2f} separation={sep:.2f}", flush=True)

    out = {
        "config": {"n_null": N_NULL, "n_pos": N_POS, "amps": AMPS, "op_threshold": OP_THR,
                   "D": BM.D, "M": BM.M, "sigma": BM.SIGMA, "w": BM.W},
        "null": {"n": N_NULL, "fp_count": k_fp, "fpr": fpr, "fpr_ci95": [fpr_lo, fpr_hi],
                 "b_median": float(np.median(b_null)), "b_p95": float(np.percentile(b_null, 95)),
                 "b_p99": float(np.percentile(b_null, 99)), "b_max": float(b_null.max())},
        "positives": {str(A): {k: v for k, v in d.items() if k != "b_scores"} for A, d in pos.items()},
        "roc": {"auc": auc, "n_points": len(roc), "separation_p99null_to_minpos": sep,
                "fpr_grid": fx[::5], "tpr_grid": ty[::5]},
    }
    OUT = ROOT / "results" / "boundary"; OUT.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT / "P2E7_operating_characteristics.json", "w"), indent=2, default=float)
    print("\nwrote P2E7_operating_characteristics.json")


if __name__ == "__main__":
    main()
