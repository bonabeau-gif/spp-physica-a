"""Assessment v2 smaller item: extend the chirality |omega|-vs-N check toward the
N~200 used elsewhere (was 10-75). Reuses the strongest non-reciprocal chiral rule
and varies the box (fixed density) so N grows."""
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
sys.path.insert(0, str(ROOT / "experiments"))
from spp_unified.parameters import SimulationConfig
from spp_unified.model import simulate_two_species
from chirality_decay import chiral_rule

OUT = ROOT / "results" / "expanded"


def main():
    p, row = chiral_rule()
    rows = []
    for S in [3.2, 4.0, 5.6, 7.2, 9.0, 11.0, 13.0]:
        oms = []
        n_val = None
        for sd in range(20):
            cfg = SimulationConfig(S=S, steps=800, sample_start=400, sample_every=2,
                                   boundary="periodic", seed=int(sd))
            r = simulate_two_species(p, cfg)
            oms.append(abs(r.summary["abs_omega"])); n_val = r.summary["n"]
        oms = np.array(oms)
        rows.append({"S": S, "N": int(n_val), "abs_omega_mean": float(oms.mean()),
                     "abs_omega_sd": float(oms.std()), "abs_omega_median": float(np.median(oms))})
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(OUT / "chirality_decay_ndependence_v2.csv", index=False)


if __name__ == "__main__":
    main()
