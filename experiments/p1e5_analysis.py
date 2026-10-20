"""P1-E5 analysis: separate density (rho), finite-N (log N), and wall (boundary) effects.

For each rule x observable, fit a NAIVE model y ~ rho and a FULL model
y ~ rho + logN + B + rho:logN, by ordinary least squares with t-test p-values (manual OLS,
no statsmodels dependency). A density effect is called PHYSICAL if the rho coefficient stays
significant and stable when logN and B are added; if it collapses, the apparent density
effect was a finite-N or wall artifact. The boundary coefficient B labels wall vs bulk.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "boundary"
OBS = ["phi", "clustering", "milling"]


def ols(X, y):
    """Return (beta, se, t, p) for design matrix X (n x k, incl. intercept col)."""
    XtX = X.T @ X
    XtXi = np.linalg.pinv(XtX)
    beta = XtXi @ (X.T @ y)
    resid = y - X @ beta
    dof = max(len(y) - X.shape[1], 1)
    sigma2 = float(resid @ resid) / dof
    se = np.sqrt(np.maximum(np.diag(XtXi) * sigma2, 0.0))
    t = np.divide(beta, se, out=np.zeros_like(beta), where=se > 0)
    p = 2 * stats.t.sf(np.abs(t), dof)
    return beta, se, t, p


def main():
    df = pd.read_csv(OUT / "P1E5_decoupling_perseed.csv")
    df["logN"] = np.log(df["N"].to_numpy())
    df["B"] = (df["boundary"] == "reflecting").astype(float)
    out = {"rules": {}, "summary": {}}
    print(f"{'rule':16} {'obs':11} | {'rho_naive(p)':>16} | {'rho_full(p)':>16} | "
          f"{'logN(p)':>14} | {'wall B(p)':>14} | verdict")
    print("-" * 110)
    for rule in sorted(df.rule.unique()):
        g = df[df.rule == rule]
        out["rules"][rule] = {}
        for obs in OBS:
            y = g[obs].to_numpy()
            rho = g.rho.to_numpy(); logN = g.logN.to_numpy(); B = g.B.to_numpy()
            if y.std() < 1e-6:
                continue
            # center continuous predictors so main effects are interpretable and stable
            rc = rho - rho.mean(); lc = logN - logN.mean()
            # naive: y ~ rho
            Xn = np.column_stack([np.ones_like(rho), rc])
            bn, _, _, pn = ols(Xn, y)
            # full (verdict): centered main effects y ~ rho + logN + B
            Xf = np.column_stack([np.ones_like(rho), rc, lc, B])
            bf, sef, tf, pf = ols(Xf, y)
            # interaction check (reported separately, not used for verdict)
            Xi = np.column_stack([np.ones_like(rho), rc, lc, B, rc * lc])
            bi, _, _, pi = ols(Xi, y)
            rho_naive_sig = pn[1] < 0.05
            rho_full_sig = pf[1] < 0.05
            logN_sig = pf[2] < 0.05
            B_sig = pf[3] < 0.05
            # verdict on the density effect
            if rho_naive_sig and rho_full_sig:
                verdict = "density PHYSICAL"
            elif rho_naive_sig and not rho_full_sig:
                verdict = "density -> finite-N/wall artifact"
            elif not rho_naive_sig:
                verdict = "no density effect"
            else:
                verdict = "density emerges w/ controls"
            out["rules"][rule][obs] = {
                "rho_naive": float(bn[1]), "p_rho_naive": float(pn[1]),
                "rho_full": float(bf[1]), "p_rho_full": float(pf[1]),
                "logN_coef": float(bf[2]), "p_logN": float(pf[2]),
                "B_coef": float(bf[3]), "p_B": float(pf[3]),
                "rhoxlogN": float(bi[4]), "p_rhoxlogN": float(pi[4]),
                "verdict": verdict}
            print(f"{rule:16} {obs:11} | {bn[1]:+7.3f}({pn[1]:.3f}) | {bf[1]:+7.3f}({pf[1]:.3f}) | "
                  f"{bf[2]:+6.3f}({pf[2]:.2f}) | {bf[3]:+6.3f}({pf[3]:.2f}) | {verdict}")
    # tallies
    verds = [v[o]["verdict"] for v in out["rules"].values() for o in v]
    from collections import Counter
    out["summary"] = dict(Counter(verds))
    json.dump(out, open(OUT / "P1E5_decoupling_analysis.json", "w"), indent=2)
    print("\nverdict tally:", json.dumps(out["summary"]))
    print("wrote P1E5_decoupling_analysis.json")


if __name__ == "__main__":
    main()
