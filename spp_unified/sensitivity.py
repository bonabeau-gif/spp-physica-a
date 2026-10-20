from __future__ import annotations
import numpy as np
import pandas as pd
from .parameters import ACTIVE_SINGLE_NAMES, EXPERIMENT_BOUNDS, SingleParams, SimulationConfig
from .evaluation import evaluate_single


def one_at_a_time(center: np.ndarray, cfg: SimulationConfig, seeds=(101, 102, 103),
                  points: int = 7) -> pd.DataFrame:
    center = np.asarray(center, dtype=float)
    rows = []
    for j, name in enumerate(ACTIVE_SINGLE_NAMES):
        lo, hi = EXPERIMENT_BOUNDS[j]
        for value in np.linspace(lo, hi, points):
            x = center.copy(); x[j] = value
            p = SingleParams.from_active_vector(x)
            m = evaluate_single(p, cfg, seeds)
            rows.append({"parameter": name, "value": value, **m})
    return pd.DataFrame(rows)
