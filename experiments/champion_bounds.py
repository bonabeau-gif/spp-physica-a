"""Assessment #6: report each directed-search champion's parameter vector and how
many coordinates sit on (within 1% of) a search-box bound. Ceiling champions
(phi=1, D=1 with zero seed variance) may be reporting the domain corner rather
than intrinsic dynamics."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from spp_unified.parameters import (ACTIVE_SINGLE_NAMES, TWO_SPECIES_EXTRA_NAMES,
                                    EXPERIMENT_BOUNDS, TWO_SPECIES_EXTRA_BOUNDS)

DISCOVERY_BOUNDS = np.array([
    [0.05, 0.70], [0.01, 0.12], [0.0, 5.0], [2.1, 14.0], [0.0, 5.0],
    [-0.80, 0.80], [2.1, 14.0], [0.0, 5.0], [0.0, 1.0]], dtype=float)
TWO_BOUNDS = np.vstack([EXPERIMENT_BOUNDS, TWO_SPECIES_EXTRA_BOUNDS])
SINGLE = list(ACTIVE_SINGLE_NAMES)
TWO = list(ACTIVE_SINGLE_NAMES) + list(TWO_SPECIES_EXTRA_NAMES)

TOL = 0.01

for tag, fname in [("default", "objective_champions.csv"), ("periodic", "objective_champions_periodic.csv")]:
    df = pd.read_csv(ROOT / "results" / "expanded" / fname)
    rows = []
    for _, r in df.iterrows():
        if r["kind"] == "single":
            names, bounds = SINGLE, DISCOVERY_BOUNDS
        else:
            names, bounds = TWO, TWO_BOUNDS
        vals = np.array([r[n] for n in names], dtype=float)
        u = (vals - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])
        at_lo = [names[i] for i in range(len(names)) if u[i] <= TOL]
        at_hi = [names[i] for i in range(len(names)) if u[i] >= 1 - TOL]
        n_bound = len(at_lo) + len(at_hi)
        rows.append({
            "objective": r["objective"], "kind": r["kind"],
            "val_mean": round(float(r["val_mean"]), 3), "val_sd": round(float(r["val_sd"]), 3),
            "n_params": len(names), "n_at_bound": n_bound,
            "noise": round(float(r["noise"]), 3),
            "at_lower": ",".join(at_lo), "at_upper": ",".join(at_hi),
        })
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "results" / "expanded" / f"champion_bounds_{tag}.csv", index=False)
    print(f"\n===== {tag} =====")
    print(out.to_string(index=False))
    print(f"mean coords at bound: {out.n_at_bound.mean():.1f} / {out.n_params.iloc[0]}")
