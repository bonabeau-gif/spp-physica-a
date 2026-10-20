"""Frozen experiment configuration for the Section 11.4 experimental plan.

These constants are FROZEN before any experiment runs and must not be re-tuned after
seeing E3 or E5 (Pitfall 1). The config hash is logged in every experiment output; if
b_min ever changes, E5 must be re-run from scratch under the new value and both reported.
"""
from __future__ import annotations
import hashlib
import json

# --- oracle / acceptance ---
B_MIN = 3.0
SCALES = (0.02, 0.04, 0.08)            # multi-scale perturbation half-widths (unit space)
COS_MIN = 0.8                          # normal-agreement threshold across scales
N_SIDE = 8                             # perturbation seeds per side (full oracle)
N_FLOOR = 16                           # floor seeds
MC_Q = 95.0                            # noise-floor percentile

# --- adaptive run-length policy (Module B) ---
MIN_TAU_MULTIPLES = 50.0               # window >= 50 tau_int
BURN_IN_TAU = 10.0                     # burn-in >= 10 tau_int
BASE_STEPS = {"single": 160, "two": 220}
MAX_STEPS = 3520

# --- estimator settings ---
N_SEEDS_SUSC = 24                      # seeds per grid point for susceptibility estimators
N_BOOT = 1000

# --- standardization: frozen autoencoder input stats (per space, in <ae>.stdstats.npz) ---
STANDARDIZATION = "autoencoder ae.mean/ae.std, frozen; single=51D, two=107D"


def config_dict() -> dict:
    return {
        "b_min": B_MIN, "scales": list(SCALES), "cos_min": COS_MIN,
        "n_side": N_SIDE, "n_floor": N_FLOOR, "mc_q": MC_Q,
        "min_tau_multiples": MIN_TAU_MULTIPLES, "burn_in_tau": BURN_IN_TAU,
        "base_steps": BASE_STEPS, "max_steps": MAX_STEPS,
        "n_seeds_susc": N_SEEDS_SUSC, "n_boot": N_BOOT,
        "standardization": STANDARDIZATION,
    }


def config_hash() -> str:
    return hashlib.sha1(json.dumps(config_dict(), sort_keys=True).encode()).hexdigest()[:12]


CONFIG_HASH = config_hash()
