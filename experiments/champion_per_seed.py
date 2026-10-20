"""Assessment #7: per-seed champion fitness distributions.

Re-validates each directed-search champion over 20 seeds using the exact
validation protocol of discover_and_animate.validate_champion (seeds 8000+i,
steps 220 single / 240 two, boundary per objective) and records every per-seed
fitness value, so that bimodality / outright-failing seeds are visible rather
than hidden inside mean +/- SD. Focus objectives: milling_vortex,
phase_separation, chiral_rotation."""
from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from spp_unified.parameters import ACTIVE_SINGLE_NAMES, TWO_SPECIES_EXTRA_NAMES
from experiments.discover_and_animate import OBJECTIVES, eval_single, eval_two

SINGLE = list(ACTIVE_SINGLE_NAMES)
TWO = list(ACTIVE_SINGLE_NAMES) + list(TWO_SPECIES_EXTRA_NAMES)
N_SEEDS = 20


def champion_vector(row):
    if row["kind"] == "single":
        return np.array([row[n] for n in SINGLE], float), "single"
    return np.array([row[n] for n in TWO], float), "two"


def main():
    champs = pd.read_csv(ROOT / "results" / "expanded" / "objective_champions.csv")
    seeds = 8000 + np.arange(N_SEEDS)
    per_seed_rows = []
    summ_rows = []
    for _, row in champs.iterrows():
        obj = row["objective"]
        spec = OBJECTIVES[obj]
        x, kind = champion_vector(row)
        scores = []
        for sd in seeds:
            if kind == "single":
                s = eval_single(x, spec["boundary"], int(sd), steps=220)
            else:
                s = eval_two(x, int(sd), steps=240)
            f = float(spec["fn"](s))
            scores.append(f)
            per_seed_rows.append({"objective": obj, "kind": kind, "seed": int(sd), "fitness": f})
        scores = np.array(scores)
        # simple bimodality / failure diagnostics
        thr = 0.5 * scores.max() if scores.max() > 0 else 0.0
        frac_fail = float(np.mean(scores < thr))
        summ_rows.append({
            "objective": obj, "kind": kind, "n_seeds": N_SEEDS,
            "mean": round(float(scores.mean()), 3), "sd": round(float(scores.std()), 3),
            "median": round(float(np.median(scores)), 3),
            "min": round(float(scores.min()), 3), "max": round(float(scores.max()), 3),
            "q25": round(float(np.quantile(scores, 0.25)), 3),
            "q75": round(float(np.quantile(scores, 0.75)), 3),
            "frac_below_half_max": round(frac_fail, 3),
        })
        print(summ_rows[-1], flush=True)
    pd.DataFrame(per_seed_rows).to_csv(ROOT / "results" / "expanded" / "champion_per_seed.csv", index=False)
    pd.DataFrame(summ_rows).to_csv(ROOT / "results" / "expanded" / "champion_per_seed_summary.csv", index=False)


if __name__ == "__main__":
    main()
