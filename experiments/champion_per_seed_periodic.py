"""20-seed median/IQR for the all-periodic champions, so Table 6 can report
20-seed medians for both the default and all-periodic boundaries (assessment v2
smaller item: drop the disavowed mean+/-sd from Table 6)."""
from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
os.environ["SPP_FORCE_BOUNDARY"] = "periodic"   # must be set before importing discover_and_animate
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
from spp_unified.parameters import ACTIVE_SINGLE_NAMES, TWO_SPECIES_EXTRA_NAMES
from experiments.discover_and_animate import OBJECTIVES, eval_single, eval_two

SINGLE = list(ACTIVE_SINGLE_NAMES)
TWO = list(ACTIVE_SINGLE_NAMES) + list(TWO_SPECIES_EXTRA_NAMES)
N_SEEDS = 20


def main():
    champs = pd.read_csv(ROOT / "results" / "expanded" / "objective_champions_periodic.csv")
    seeds = 8000 + np.arange(N_SEEDS)
    summ = []
    for _, row in champs.iterrows():
        obj = row["objective"]; spec = OBJECTIVES[obj]
        if row["kind"] == "single":
            x = np.array([row[n] for n in SINGLE], float); kind = "single"
        else:
            x = np.array([row[n] for n in TWO], float); kind = "two"
        scores = []
        for sd in seeds:
            s = eval_single(x, "periodic", int(sd), steps=220) if kind == "single" else eval_two(x, int(sd), steps=240)
            scores.append(float(spec["fn"](s)))
        scores = np.array(scores)
        summ.append({"objective": obj, "median": round(float(np.median(scores)), 3),
                     "q25": round(float(np.quantile(scores, .25)), 3),
                     "q75": round(float(np.quantile(scores, .75)), 3)})
        print(summ[-1], flush=True)
    pd.DataFrame(summ).to_csv(ROOT / "results" / "expanded" / "champion_per_seed_periodic_summary.csv", index=False)


if __name__ == "__main__":
    main()
