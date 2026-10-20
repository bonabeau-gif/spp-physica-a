"""Thin wrappers over the existing simulator/descriptor/autoencoder.

Design note: the sensitivity oracle works on the standardized RAW descriptor
psi_tilde = (psi - mu)/sigma, where (mu, sigma) are the autoencoder's frozen input
statistics. The forward pass of the autoencoder (the 2-D coordinate z) is NOT needed
to compute sensitivity, so the parallel worker computes only the raw descriptor and
never imports torch. Standardization and the (optional) 2-D encoding happen in the
main process.
"""
from __future__ import annotations
import os
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__import__("os").environ.get("SPP_PROJECT_ROOT", Path(__file__).resolve().parents[1]))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "experiments") not in sys.path:
    sys.path.insert(0, str(ROOT / "experiments"))

from spp_unified.parameters import SingleParams, TwoSpeciesParams, SimulationConfig
from spp_unified.model import simulate_single, simulate_two_species
from spp_unified.learned import behavior_descriptor, species_behavior_descriptor

from boundary import params as P
from boundary.protocol import tau_int, u_hash, derive_seed


def _cfg(space: P.Space, n_steps: int, boundary: str, seed: int) -> SimulationConfig:
    return SimulationConfig(
        S=space.S, r0=space.r0, steps=int(n_steps),
        sample_start=int(round(0.5 * n_steps)), sample_every=space.sample_every,
        boundary=boundary, seed=int(seed), overlap_iterations=3,
    )


def _simulate(space: P.Space, u01, n_steps: int, boundary: str, seed: int):
    x = P.from_unit(np.asarray(u01, float), space)
    cfg = _cfg(space, n_steps, boundary, seed)
    if space.name == "single":
        p = SingleParams.from_active_vector(x)
        res = simulate_single(p, cfg, record_states=True, state_every=space.state_every)
        psi = behavior_descriptor(res.states, cfg, nbins=space.nbins, max_frames=space.max_frames)
    else:
        p = TwoSpeciesParams.from_active_vector(x)
        res = simulate_two_species(p, cfg, record_states=True, state_every=space.state_every)
        psi = species_behavior_descriptor(res.states, res.species, cfg,
                                          nbins=space.nbins, max_frames=space.max_frames)
    return res, cfg, np.asarray(psi, np.float32)


# ---------- picklable worker (top-level) ----------
def sim_worker(task):
    """task = (space_name, u01(list), n_steps, boundary, seed, want_obs2)."""
    space_name, u01, n_steps, boundary, seed, want_obs2 = task
    space = P.get_space(space_name)
    res, cfg, psi = _simulate(space, u01, n_steps, boundary, seed)
    out = {
        "summary": {k: float(v) for k, v in res.summary.items()},
        "psi": psi,
        "n_particles": int(res.summary.get("n", 0)),
        "steps_used": int(cfg.steps),
    }
    # per-observable tau in STEP units (sampled series -> * sample_every)
    tb = {}
    for k in ("phi", "milling"):
        s = res.timeseries.get(k)
        if s is not None and len(s) > 3:
            tb[k] = tau_int(np.asarray(s)) * cfg.sample_every
    out["tau_int"] = tb
    if want_obs2 and space.name == "two":
        from boundary.observables2 import compute_obs2
        out["obs2"] = compute_obs2(res, cfg, space)
    return out


# ---------- batch dispatch with caching ----------
_POOL = {"ex": None, "workers": None}


def _get_pool(max_workers):
    """A single persistent ProcessPoolExecutor reused across all calls, so workers
    import numpy/spp_unified once and stay warm (macOS spawn re-import is ~2s/worker,
    which otherwise dominates when each oracle call dispatches ~29 small batches)."""
    import atexit
    from concurrent.futures import ProcessPoolExecutor
    if _POOL["ex"] is None:
        _POOL["ex"] = ProcessPoolExecutor(max_workers=max_workers)
        _POOL["workers"] = max_workers
        atexit.register(lambda: _POOL["ex"] and _POOL["ex"].shutdown(wait=False, cancel_futures=True))
    return _POOL["ex"]


def _parallel(worker, tasks, max_workers):
    """Quiet persistent-pool map that preserves order and honors SPP_MAX_WORKERS."""
    if not tasks:
        return []
    if max_workers <= 1:
        return [worker(t) for t in tasks]
    ex = _get_pool(max_workers)
    return list(ex.map(worker, tasks, chunksize=1))


def run_many(space: P.Space, points, cache=None, max_workers=6, want_obs2=False, desc="sims"):
    """points: list of (u01, n_steps, boundary, seed). Returns list of result dicts."""
    cap = os.environ.get("SPP_MAX_WORKERS")
    if cap:
        max_workers = min(max_workers, int(cap))

    results = [None] * len(points)
    tasks, task_idx, keys = [], [], [None] * len(points)
    for i, (u01, n_steps, boundary, seed) in enumerate(points):
        key = (u_hash(np.asarray(u01, float)), space.name, boundary, int(n_steps), int(seed), bool(want_obs2))
        keys[i] = key
        if cache is not None and cache.get(key) is not None:
            results[i] = cache.get(key)
        else:
            tasks.append((space.name, list(np.asarray(u01, float)), int(n_steps), boundary, int(seed), bool(want_obs2)))
            task_idx.append(i)
    if tasks:
        fresh = _parallel(sim_worker, tasks, max_workers=max_workers)
        for j, i in enumerate(task_idx):
            results[i] = fresh[j]
            if cache is not None:
                cache.put(keys[i], fresh[j])
    return results


def seeds_for(u01, proto_hash: str, n: int, offset: int = 0):
    uh = u_hash(np.asarray(u01, float))
    return [derive_seed(uh, proto_hash, offset + k) for k in range(n)]


def run_adequate(space: P.Space, u01, proto, keys=("phi", "milling"), cache=None,
                 min_tau_multiples=50.0, n_probe=4):
    """Module B: escalate n_steps x2 until the window is >= min_tau_multiples * tau_max.

    Returns (adequate_protocol, tau_by_key, underresolved_flag). Points that hit
    max_steps without satisfying adequacy are flagged underresolved (to be dropped
    from the catalog, not silently included)."""
    from boundary.protocol import adequate
    p = proto
    while True:
        seeds = seeds_for(u01, p.hash(), n_probe)
        res = run_many(space, [(u01, p.n_steps, p.boundary, s) for s in seeds], cache=cache)
        tau = {}
        for k in keys:
            ts = [r["tau_int"].get(k) for r in res if r["tau_int"].get(k) is not None]
            if ts:
                tau[k] = max(ts)
        ok = adequate(p.n_steps, p.burn_in_frac, tau, list(keys), min_tau_multiples)
        if ok:
            return p, tau, False
        if p.n_steps * 2 > p.max_steps:
            return p, tau, True
        p = p.with_steps(p.n_steps * 2)


# ---------- standardizer (main process; torch only on first load) ----------
_STD_CACHE = {}


def load_standardizer(space: P.Space):
    """(mu, sigma) = frozen autoencoder input statistics. Cached to npy to avoid torch."""
    if space.name in _STD_CACHE:
        return _STD_CACHE[space.name]
    npy = space.ae_path.with_suffix(".stdstats.npz")
    if npy.exists():
        d = np.load(npy)
        mu, sigma = d["mu"].astype(np.float64), d["sigma"].astype(np.float64)
    else:
        import torch  # noqa
        ck = torch.load(space.ae_path, map_location="cpu", weights_only=False)
        mu = np.asarray(ck["mean"], np.float64)
        sigma = np.asarray(ck["std"], np.float64)
        np.savez(npy, mu=mu, sigma=sigma)
    sigma = np.where(sigma < 1e-9, 1e-9, sigma)
    _STD_CACHE[space.name] = (mu, sigma)
    return mu, sigma


def standardize(psi, space: P.Space):
    mu, sigma = load_standardizer(space)
    return (np.asarray(psi, float) - mu) / sigma


def load_encoder(space: P.Space):
    """Return a callable psi_raw(2d)->z01(2d). Needs torch + np_encoder_patch."""
    import torch  # noqa
    import np_encoder_patch  # noqa: F401  (monkeypatches transform; import BEFORE use)
    from spp_unified.learned import BehaviorAutoencoder, normalize_latent
    X = np.load(space.descr_npz)["X"]
    ck = torch.load(space.ae_path, map_location="cpu", weights_only=False)
    hidden = (80, 28) if space.name == "two" else (64, 24)
    ae = BehaviorAutoencoder(X.shape[1], hidden=hidden, seed=int(ck["seed"]))
    ae.model.load_state_dict(ck["state_dict"]); ae.mean = ck["mean"]; ae.std = ck["std"]; ae.model.eval()
    if space.name == "two":
        d = np.load(space.descr_npz); lo, hi = d["lo"], d["hi"]
    else:
        d = np.load(space.latent_npz); lo, hi = d["lo"], d["hi"]

    def encode(psi_raw_2d):
        z = ae.transform(np.asarray(psi_raw_2d, np.float32))
        z01, _, _ = normalize_latent(z, lo, hi)
        return np.asarray(z01, float)

    return encode
