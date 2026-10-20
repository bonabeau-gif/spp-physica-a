"""Analysis of the Section 6 surrogate experiments: E1 (6-replicate paired test at 3000),
E2 (crossover across {30,300,1000,3000}), E4 (wall-clock), E5 (calibration at scale).
Re-analysis only; no new simulations."""
from __future__ import annotations
import sys, json, glob
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
import numpy as np
import pandas as pd
from scipy import stats

EXP = ROOT / "results" / "expanded"
SX = EXP / "surrogate_exp"
RNG = np.random.default_rng(20260728)
OUT = {}


def final_cov(hist_glob, budget):
    rows = []
    for f in glob.glob(str(hist_glob)):
        h = pd.read_csv(f); last = h[h.true_calls == budget]
        if len(last):
            rows.append(last.iloc[0][["replicate", "method", "coverage"]])
    return pd.DataFrame(rows)


def piv(df):
    return df.pivot(index="replicate", columns="method", values="coverage").sort_index()


def paired(a, b):
    d = np.asarray(a) - np.asarray(b); n = len(d)
    t, p = stats.ttest_rel(a, b)
    se = d.std(ddof=1) / np.sqrt(n); tc = stats.t.ppf(0.975, n - 1)
    return {"mean": float(d.mean()), "p": float(p), "ci": [float(d.mean() - tc * se), float(d.mean() + tc * se)],
            "n": n, "per_rep": [round(float(x), 3) for x in d]}


def main():
    scal = pd.read_csv(EXP / "surrogate_scaling_history.csv")
    # --- per-budget coverage tables (pool existing reps 0-2 at 3000 with new 3-5) ---
    budgets = {}
    for b in (30, 300):
        f = scal[(scal.budget == b) & (scal.true_calls == b)][["replicate", "method", "coverage"]]
        budgets[b] = piv(f)
    budgets[1000] = piv(final_cov(SX / "hist_b1000_*.csv", 1000))
    old3 = scal[(scal.budget == 3000) & (scal.true_calls == 3000)][["replicate", "method", "coverage"]]
    new3 = final_cov(SX / "hist_b3000_*.csv", 3000)
    p3 = pd.concat([old3, new3]).drop_duplicates(["replicate", "method"])
    budgets[3000] = piv(p3)

    # --- E1: 6-replicate paired test at 3000 ---
    P = budgets[3000]
    OUT["E1_3000"] = {"n": len(P), "means": {m: float(P[m].mean()) for m in P.columns},
                      "sd": {m: float(P[m].std(ddof=1)) for m in P.columns},
                      "surrogate_minus_direct": paired(P["surrogate"], P["direct"]),
                      "per_rep_coverage": P.round(3).to_dict()}
    print("=== E1: 3000-call, %d replicates ===" % len(P))
    print(P.round(3).to_string())
    sd = OUT["E1_3000"]["surrogate_minus_direct"]
    print(f"surrogate-direct: mean={sd['mean']:+.3f} p={sd['p']:.4f} CI={[round(x,3) for x in sd['ci']]} per-rep={sd['per_rep']}")

    # --- E2: gaps + crossover; Bonferroni across the 4 budgets ---
    gaps = {}
    for b, P in budgets.items():
        g = paired(P["surrogate"], P["direct"]); g["p_bonf"] = min(1.0, g["p"] * 4)
        g["surrogate_mean"] = float(P["surrogate"].mean()); g["direct_mean"] = float(P["direct"].mean())
        g["random_mean"] = float(P["random"].mean()); gaps[b] = g
    OUT["E2_gaps"] = gaps
    print("\n=== E2: gap surrogate-direct by budget (Bonferroni x4) ===")
    for b in (30, 300, 1000, 3000):
        g = gaps[b]; print(f"  {b:>4}: {g['mean']:+.3f} CI{[round(x,3) for x in g['ci']]} p_bonf={g['p_bonf']:.3f} (n={g['n']})")

    # crossover: logistic-ish fit of gap vs log10(budget), bootstrap zero-crossing
    lb = np.array([np.log10(b) for b in (30, 300, 1000, 3000)])
    gm = np.array([gaps[b]["mean"] for b in (30, 300, 1000, 3000)])

    def crossing(logb, gap):
        # linear interpolation of the sign change on the log-budget axis
        for i in range(len(logb) - 1):
            if gap[i] < 0 <= gap[i + 1] or gap[i] <= 0 < gap[i + 1]:
                t = -gap[i] / (gap[i + 1] - gap[i])
                return 10 ** (logb[i] + t * (logb[i + 1] - logb[i]))
        return np.nan
    x0 = crossing(lb, gm)
    # bootstrap over replicates within each budget
    boot = []
    for _ in range(2000):
        gb = []
        for b in (30, 300, 1000, 3000):
            P = budgets[b]; idx = RNG.integers(0, len(P), len(P))
            gb.append((P["surrogate"].values[idx] - P["direct"].values[idx]).mean())
        c = crossing(lb, np.array(gb))
        if np.isfinite(c):
            boot.append(c)
    OUT["E2_crossover"] = {"point": float(x0), "ci": [float(np.percentile(boot, 5)), float(np.percentile(boot, 95))],
                           "boot_n": len(boot)}
    print(f"\ncrossover (gap=0): ~{x0:.0f} calls, bootstrap 90% CI [{np.percentile(boot,5):.0f}, {np.percentile(boot,95):.0f}]")

    # --- E4: wall-clock ---
    tim = pd.concat([pd.read_csv(f) for f in glob.glob(str(SX / "timing_*.csv"))], ignore_index=True)
    per = tim.groupby(["budget", "method"]).agg(t_fit=("t_fit", "sum"), t_gen=("t_gen", "sum"),
                                                t_eval=("t_eval", "sum")).reset_index()
    per["total"] = per.t_fit + per.t_gen + per.t_eval
    # cost per simulator call vs particle count (from eval time / batch)
    tim["s_per_call"] = tim.t_eval / 10.0
    cc = tim.dropna(subset=["n_particles_mean"])
    slope = np.polyfit(cc.n_particles_mean, cc.s_per_call, 1)
    OUT["E4"] = {"wallclock_by_budget_method": per.round(1).to_dict(orient="records"),
                 "s_per_call_vs_nparticles_slope_intercept": [float(slope[0]), float(slope[1])],
                 "mean_s_per_call": float(tim.s_per_call.mean())}
    print("\n=== E4: wall-clock (s) by budget/method (t_fit / t_eval / total) ===")
    for r in per[per.budget == 3000].itertuples():
        print(f"  3000 {r.method:>9}: fit={r.t_fit:.0f} eval={r.t_eval:.0f} total={r.total:.0f}")
    # threshold: surrogate wins on wall-clock at 3000 if its total < direct's; overhead ratio
    s3 = per[(per.budget == 3000) & (per.method == "surrogate")].iloc[0]
    d3 = per[(per.budget == 3000) & (per.method == "direct")].iloc[0]
    # surrogate does 3000 sims + fit overhead; extra sim time to justify overhead
    extra_overhead = float(s3.t_fit + s3.t_gen - d3.t_fit - d3.t_gen)
    thresh_s = extra_overhead / 3000.0 + tim.s_per_call.mean()  # break-even s/call
    OUT["E4"]["surrogate_overhead_s_at_3000"] = extra_overhead
    OUT["E4"]["breakeven_s_per_call"] = float(thresh_s)
    print(f"  surrogate overhead at 3000 (fit+acq): {extra_overhead:.0f}s over direct; "
          f"break-even sim cost ~{thresh_s:.2f} s/call (mean actual {tim.s_per_call.mean():.2f})")

    # --- E5: calibration at scale ---
    pr = pd.concat([pd.read_csv(f) for f in glob.glob(str(SX / "preds_*.csv"))], ignore_index=True)
    pr["bin"] = pd.cut(pr.ntrain, [0, 250, 600, 1500, 3200], labels=["~150", "~400", "~1100", "~3100"])
    cal = []
    for bn, g in pr.groupby("bin", observed=True):
        row = {"ntrain_bin": str(bn), "n": len(g)}
        for k, sd in [("z0", "sd_z0"), ("z1", "sd_z1"), ("q", "sd_q")]:
            err = (g[f"actual_{k}"] - g[f"pred_{k}"]).abs()
            row[f"rmse_{k}"] = float(np.sqrt((err ** 2).mean()))
            row[f"cov90_{k}"] = float((err <= 1.645 * g[sd]).mean())
        cal.append(row)
    OUT["E5_calibration"] = cal
    print("\n=== E5: calibration by training size (RMSE, 90% interval coverage) ===")
    for r in cal:
        print(f"  {r['ntrain_bin']:>6} (n={r['n']:>4}): rmse_z0={r['rmse_z0']:.3f} cov90_z0={r['cov90_z0']:.2f} | "
              f"rmse_z1={r['rmse_z1']:.3f} cov90_z1={r['cov90_z1']:.2f} | rmse_q={r['rmse_q']:.4f} cov90_q={r['cov90_q']:.2f}")

    json.dump(OUT, open(SX / "analysis.json", "w"), indent=2, default=float)
    print("\nwrote", SX / "analysis.json")


if __name__ == "__main__":
    main()
