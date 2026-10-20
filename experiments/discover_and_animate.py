"""Expanded behavioral-repertoire discovery + high-fitness GIF generation.

This driver searches the behavioral repertoire along three axes:

  * SIZE   -- larger, higher-density parameter exploration (rho up to 0.70) with
              directed refinement searches per objective, on top of the existing
              archives and the 360-rule corpus.
  * SCOPE  -- a broader catalog of objective functions, including three pattern
              measures the paper computed only as diagnostics (sixfold/hexatic
              bond order, low-k density-mode power, local-density heterogeneity)
              plus two-species drift, chirality, and segregation objectives.
  * PATTERN DISCOVERY -- for every objective it selects the single highest-fitness
              parameter set found anywhere (pooled archives + corpus + new search),
              re-validates it over multiple seeds, and renders an animated GIF.

Run:
    python experiments/discover_and_animate.py            # full
    python experiments/discover_and_animate.py --quick    # fast smoke run
    python experiments/discover_and_animate.py --gifs-only # reuse saved champions
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from spp_unified.parameters import (
    ACTIVE_SINGLE_NAMES, TWO_SPECIES_EXTRA_NAMES, TWO_SPECIES_EXTRA_BOUNDS,
    EXPERIMENT_BOUNDS, SingleParams, TwoSpeciesParams, SimulationConfig,
)
from spp_unified.model import simulate_single, simulate_two_species
from spp_unified.learned import hexatic_order, density_mode_peak, local_density_cv
from spp_unified.animation import render_gif
from expanded_common import run_parallel

RESULTS = ROOT / "results" / "expanded"
GIFS = ROOT / "figures" / "gifs"
RESULTS.mkdir(parents=True, exist_ok=True)
GIFS.mkdir(parents=True, exist_ok=True)

# Broadened single-species bounds: density reaches the dense regime (0.70) so the
# search can find crystalline, banded, and phase-separated states.
DISCOVERY_BOUNDS = np.array([
    [0.05, 0.70], [0.01, 0.12], [0.0, 5.0], [2.1, 14.0], [0.0, 5.0],
    [-0.80, 0.80], [2.1, 14.0], [0.0, 5.0], [0.0, 1.0],
], dtype=float)

TWO_NAMES = ACTIVE_SINGLE_NAMES + TWO_SPECIES_EXTRA_NAMES
TWO_BOUNDS = np.vstack([EXPERIMENT_BOUNDS, TWO_SPECIES_EXTRA_BOUNDS])


# ---------------------------------------------------------------------------
# Extended evaluation: classical order parameters + pattern diagnostics.
# ---------------------------------------------------------------------------

def _pattern_measures(states, S, periodic, n_avg=5):
    """Average sixfold order, density-mode power, and local-density CV over frames."""
    frames = states[-n_avg:]
    hx, dm, cv = [], [], []
    for x, _ in frames:
        hx.append(hexatic_order(x, S, periodic))
        dm.append(density_mode_peak(x, S))
        cv.append(local_density_cv(x, 4 * 0.1, S, periodic))
    return float(np.mean(hx)), float(np.mean(dm)), float(np.mean(cv))


def eval_single(x, boundary, seed, steps=180):
    p = SingleParams.from_active_vector(np.asarray(x, float))
    cfg = SimulationConfig(S=3.6, r0=0.1, steps=steps, sample_start=steps // 2,
                           sample_every=2, boundary=boundary, seed=int(seed))
    r = simulate_single(p, cfg, record_states=True, state_every=6)
    hx, dm, cv = _pattern_measures(r.states, cfg.S, boundary == "periodic")
    s = dict(r.summary)
    s.update(hexatic=hx, density_mode_peak=dm, local_density_cv=cv)
    return s


def _two_from_vector(x):
    n = len(ACTIVE_SINGLE_NAMES)
    b = SingleParams.from_active_vector(np.asarray(x, float)[:n])
    e = np.asarray(x, float)[n:]
    return TwoSpeciesParams(base=b, fraction_B=e[0], alignment_symmetric=e[1],
                            alignment_antisymmetric=e[2], force_symmetric=e[3],
                            force_antisymmetric=e[4])


def eval_two(x, seed, steps=200):
    cfg = SimulationConfig(S=4.0, r0=0.1, steps=steps, sample_start=steps // 2,
                           sample_every=2, boundary="periodic", seed=int(seed))
    r = simulate_two_species(_two_from_vector(x), cfg, record_states=True, state_every=6)
    return dict(r.summary)


# ---------------------------------------------------------------------------
# Objective catalog. Each entry defines a scalar fitness on the summary dict.
# ---------------------------------------------------------------------------

def _clip01(v):
    return float(np.clip(v, 0.0, 1.0))

OBJECTIVES = {
    # name: dict(kind, boundary, fn, color_by, live, blurb)
    "aligned_flock": dict(kind="single", boundary="reflecting",
        fn=lambda s: s["phi"], color_by="heading", live="phi",
        blurb="Maximal polar alignment: a single coherent flock."),
    "milling_vortex": dict(kind="single", boundary="reflecting",
        fn=lambda s: s["milling"] * (1.0 - s["phi"]), color_by="heading", live="milling",
        blurb="Rotating mill: high angular momentum, low net polarization."),
    "dense_cluster": dict(kind="single", boundary="periodic",
        fn=lambda s: _clip01(s["clustering"] / 10.0), color_by="heading", live=None,
        blurb="Compact aggregate far above the uniform-density reference."),
    "nematic_lanes": dict(kind="single", boundary="periodic",
        fn=lambda s: s["nematic"] * (1.0 - s["phi"]), color_by="heading", live="nematic",
        blurb="Nematic order without polarity: counter-propagating lanes."),
    "hexatic_crystal": dict(kind="single", boundary="periodic",
        fn=lambda s: s["hexatic"], color_by="speed_align", live="hexatic",
        blurb="Sixfold bond-orientational (crystalline) order [new pattern]."),
    "density_waves": dict(kind="single", boundary="periodic",
        fn=lambda s: s["density_mode_peak"], color_by="heading", live=None,
        blurb="Strong low-wavenumber density modulation: banding [new pattern]."),
    "phase_separation": dict(kind="single", boundary="periodic",
        fn=lambda s: _clip01(s["local_density_cv"] / 2.0), color_by="heading", live=None,
        blurb="Clusters coexisting with voids: motility-induced phase separation [new pattern]."),
    "active_turbulence": dict(kind="single", boundary="periodic",
        fn=lambda s: (1.0 - s["phi"]) * (1.0 - s["milling"]) * s["temporal_stability"],
        color_by="heading", live=None,
        blurb="Persistently disordered active flow [new pattern]."),
    # Two-species objectives.
    "travelling_mixture": dict(kind="two", boundary="periodic",
        fn=lambda s: s["drift"], color_by="species", live="drift",
        blurb="Coherently translating two-species band."),
    "chiral_rotation": dict(kind="two", boundary="periodic",
        fn=lambda s: s["chirality"], color_by="species", live=None,
        blurb="Persistent global rotation driven by non-reciprocal coupling."),
    "species_segregation": dict(kind="two", boundary="periodic",
        fn=lambda s: s["segregation"], color_by="species", live=None,
        blurb="Demixing of the two particle types."),
}

# Boundary override driven by an environment variable so it survives the 'spawn'
# re-import in worker processes. When SPP_FORCE_BOUNDARY is set, every
# single-species objective uses that boundary (two-species are always periodic).
_FORCE_BOUNDARY = os.environ.get("SPP_FORCE_BOUNDARY")
if _FORCE_BOUNDARY:
    for _spec in OBJECTIVES.values():
        if _spec["kind"] == "single":
            _spec["boundary"] = _FORCE_BOUNDARY

# Suffix appended to output filenames (set from --tag in main); default empty.
FILE_SUFFIX = ""


# ---------------------------------------------------------------------------
# Seed pools from existing results.
# ---------------------------------------------------------------------------

def _load_single_pool():
    frames = []
    for f in ["single_qd_alignment_milling.csv", "single_qd_alignment_clustering.csv"]:
        p = ROOT / "results" / f
        if p.exists():
            frames.append(pd.read_csv(p)[list(ACTIVE_SINGLE_NAMES)])
    corp = RESULTS / "single_behavior_corpus.csv"
    if corp.exists():
        frames.append(pd.read_csv(corp)[list(ACTIVE_SINGLE_NAMES)])
    if not frames:
        return np.zeros((0, len(ACTIVE_SINGLE_NAMES)))
    return pd.concat(frames, ignore_index=True).to_numpy(float)


def _load_two_pool():
    frames = []
    for f in ["multi_qd_nonreciprocal.csv", "multi_qd_reciprocal.csv"]:
        p = ROOT / "results" / f
        if p.exists():
            df = pd.read_csv(p)
            for c in TWO_SPECIES_EXTRA_NAMES:
                if c not in df:
                    df[c] = 0.0
            frames.append(df[list(TWO_NAMES)])
    if not frames:
        return np.zeros((0, len(TWO_NAMES)))
    return pd.concat(frames, ignore_index=True).to_numpy(float)


# ---------------------------------------------------------------------------
# Directed refinement: seed-and-mutate hill climb maximizing one objective.
# ---------------------------------------------------------------------------

def _score_task(task):
    """Module-level, picklable worker for ProcessPoolExecutor.

    task = (objective_name, x_array, seed, steps). Returns scalar fitness.
    """
    objective, x, seed, steps = task
    spec = OBJECTIVES[objective]
    if spec["kind"] == "single":
        s = eval_single(x, spec["boundary"], seed, steps=steps)
    else:
        s = eval_two(x, seed, steps=steps)
    return float(spec["fn"](s))


def _summ_single_task(task):
    x, boundary, seed, steps = task
    return eval_single(x, boundary, seed, steps=steps)


def _summ_two_task(task):
    x, seed, steps = task
    return eval_two(x, seed, steps=steps)


def precompute_pools(quick=False):
    """Simulate each existing candidate once per boundary and cache summaries.

    Returns (single_pool, {'periodic': [summ...], 'reflecting': [summ...]},
             two_pool, [summ...]). Reused across all objectives so the shared
    parameter pool is not re-simulated per objective.
    """
    steps_s = 120 if quick else 180
    steps_t = 140 if quick else 200
    single_pool = _load_single_pool()
    single_summ = {}
    for boundary in ("periodic", "reflecting"):
        tasks = [(single_pool[i], boundary, int(5000 + i), steps_s) for i in range(len(single_pool))]
        single_summ[boundary] = run_parallel(_summ_single_task, tasks, max_workers=6,
                                              desc=f"pool:single:{boundary}") if len(single_pool) else []
    two_pool = _load_two_pool()
    ttasks = [(two_pool[i], int(6000 + i), steps_t) for i in range(len(two_pool))]
    two_summ = run_parallel(_summ_two_task, ttasks, max_workers=6, desc="pool:two") if len(two_pool) else []
    return single_pool, single_summ, two_pool, two_summ


def refine(objective, pools, quick=False):
    single_pool, single_summ, two_pool, two_summ = pools
    spec = OBJECTIVES[objective]
    kind = spec["kind"]
    fn = spec["fn"]
    bounds = DISCOVERY_BOUNDS if kind == "single" else TWO_BOUNDS
    lo, hi = bounds[:, 0], bounds[:, 1]
    dim = bounds.shape[0]
    steps = (120 if quick else 180) if kind == "single" else (140 if quick else 200)

    if kind == "single":
        pool, summ = single_pool, single_summ[spec["boundary"]]
    else:
        pool, summ = two_pool, two_summ

    rng = np.random.default_rng(int.from_bytes(objective.encode("utf-8"), "little") % (2 ** 32))
    pop_n = 12 if quick else 20
    gens = 3 if quick else 6

    # Rank the cached pool by this objective (no new simulations) and seed elites.
    if len(pool):
        pool_scores = np.array([fn(s) for s in summ], float)
        order = np.argsort(-pool_scores)
        elites = pool[order[:max(3, pop_n // 3)]]
    else:
        elites = np.zeros((0, dim))

    U = [(np.asarray(e, float) - lo) / (hi - lo) for e in elites]
    while len(U) < pop_n:
        U.append(rng.random(dim))
    U = np.clip(np.array(U), 0, 1)

    def score_batch(Umat, tag):
        X = lo + Umat * (hi - lo)
        tasks = [(objective, X[i], int(rng.integers(1 << 30)), steps) for i in range(len(X))]
        return np.array(run_parallel(_score_task, tasks, max_workers=6, desc=tag), float)

    vals = score_batch(U, f"{objective}:g0")
    best_u = U[int(np.argmax(vals))].copy()
    best_v = float(vals.max())
    for g in range(1, gens + 1):
        # Gaussian mutations of the current elites, shrinking over generations.
        sigma = 0.16 * (1.0 - 0.5 * g / gens)
        order = np.argsort(-vals)
        parents = U[order[:max(2, pop_n // 3)]]
        children = []
        for _ in range(pop_n):
            base = parents[rng.integers(len(parents))]
            scale = sigma * (2.2 if rng.random() < 0.15 else 1.0)
            children.append(np.clip(base + rng.normal(0, scale, dim), 0, 1))
        C = np.array(children)
        cv = score_batch(C, f"{objective}:g{g}")
        merged = np.vstack([U, C]); mv = np.concatenate([vals, cv])
        keep = np.argsort(-mv)[:pop_n]
        U, vals = merged[keep], mv[keep]
        if vals.max() > best_v:
            best_v = float(vals.max()); best_u = U[int(np.argmax(vals))].copy()
    best_x = lo + best_u * (hi - lo)
    return best_x, best_v


# ---------------------------------------------------------------------------
# Champion validation (multi-seed) and representative-seed selection.
# ---------------------------------------------------------------------------

def validate_champion(objective, x, n_seeds=6, quick=False):
    spec = OBJECTIVES[objective]
    fn = spec["fn"]
    seeds = 8000 + np.arange(3 if quick else n_seeds)
    rows = []
    for sd in seeds:
        if spec["kind"] == "single":
            s = eval_single(x, spec["boundary"], int(sd), steps=140 if quick else 220)
        else:
            s = eval_two(x, int(sd), steps=160 if quick else 240)
        rows.append((int(sd), fn(s), s))
    scores = np.array([r[1] for r in rows])
    # Representative seed = closest to the median score (typical, not cherry-picked max).
    rep = rows[int(np.argmin(np.abs(scores - np.median(scores))))]
    agg = {"mean": float(scores.mean()), "sd": float(scores.std()),
           "median": float(np.median(scores)), "max": float(scores.max()),
           "rep_seed": rep[0], "rep_score": rep[1]}
    return agg, rep[2], rep[0]


# ---------------------------------------------------------------------------
# GIF rendering for a champion.
# ---------------------------------------------------------------------------

def animate_champion(objective, x, seed, quick=False):
    spec = OBJECTIVES[objective]
    steps = 220 if quick else 360
    if spec["kind"] == "single":
        p = SingleParams.from_active_vector(np.asarray(x, float))
        cfg = SimulationConfig(S=3.6, r0=0.1, steps=steps, sample_start=0,
                               sample_every=2, boundary=spec["boundary"], seed=int(seed))
        r = simulate_single(p, cfg, record_states=True, state_every=2)
        species = None
        # simulate_single's summary lacks pattern diagnostics; add them so the
        # pattern objectives (hexatic/density/phase) can be scored for the caption.
        hx, dm, cv = _pattern_measures(r.states, cfg.S, spec["boundary"] == "periodic")
        r.summary.update(hexatic=hx, density_mode_peak=dm, local_density_cv=cv)
    else:
        cfg = SimulationConfig(S=4.0, r0=0.1, steps=steps, sample_start=0,
                               sample_every=2, boundary=spec["boundary"], seed=int(seed))
        r = simulate_two_species(_two_from_vector(x), cfg, record_states=True, state_every=2)
        species = r.species
    live = spec.get("live")
    ms = None
    if live and live in r.timeseries and len(r.timeseries[live]):
        # timeseries is sampled from sample_start; here sample_start=0 so it aligns.
        series = np.asarray(r.timeseries[live], float)
        ms = {live: series}
    val = spec["fn"](r.summary)
    subtitle = f"n={int(r.summary['n'])}  fitness={val:.3f}  seed={seed}"
    out = GIFS / f"{objective}{FILE_SUFFIX}.gif"
    render_gif(r.states, cfg.S, str(out), species=species,
               periodic=(spec["boundary"] == "periodic"),
               title=objective.replace("_", " "), subtitle=spec["blurb"],
               metric_series=ms, color_by=spec["color_by"],
               fps=20, max_frames=140)
    return out, val


# ---------------------------------------------------------------------------
# Orchestration.
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--gifs-only", action="store_true",
                    help="reuse results/expanded/objective_champions.csv, only render GIFs")
    ap.add_argument("--only", type=str, default=None,
                    help="comma-separated subset of objective names")
    ap.add_argument("--periodic", action="store_true",
                    help="force every single-species objective to periodic boundaries")
    ap.add_argument("--tag", type=str, default=None,
                    help="suffix for output files (defaults to 'periodic' with --periodic)")
    args = ap.parse_args()

    global FILE_SUFFIX
    if args.periodic:
        # Set env var *before* any worker pool is spawned so children inherit it
        # and re-apply the override on import; also apply it to this process.
        os.environ["SPP_FORCE_BOUNDARY"] = "periodic"
        for spec in OBJECTIVES.values():
            if spec["kind"] == "single":
                spec["boundary"] = "periodic"
    tag = args.tag if args.tag else ("periodic" if args.periodic else "")
    FILE_SUFFIX = f"_{tag}" if tag else ""

    champ_path = RESULTS / f"objective_champions{FILE_SUFFIX}.csv"
    names = list(OBJECTIVES)
    if args.only:
        names = [n.strip() for n in args.only.split(",") if n.strip() in OBJECTIVES]

    t0 = time.time()
    if args.gifs_only and champ_path.exists():
        champs = pd.read_csv(champ_path)
    else:
        print("precomputing shared candidate pools ...", flush=True)
        pools = precompute_pools(quick=args.quick)
        records = []
        for obj in names:
            t = time.time()
            spec = OBJECTIVES[obj]
            best_x, best_v = refine(obj, pools, quick=args.quick)
            agg, rep_summary, rep_seed = validate_champion(obj, best_x, quick=args.quick)
            print(f"[{obj}] search_fit={best_v:.3f} "
                  f"validated mean={agg['mean']:.3f}+-{agg['sd']:.3f} "
                  f"rep_seed={rep_seed} ({time.time()-t:.0f}s)", flush=True)
            rec = {"objective": obj, "kind": spec["kind"], "boundary": spec["boundary"],
                   "search_fitness": best_v, "rep_seed": rep_seed, **{f"val_{k}": v for k, v in agg.items()}}
            param_names = ACTIVE_SINGLE_NAMES if spec["kind"] == "single" else TWO_NAMES
            rec.update({n: float(best_x[i]) for i, n in enumerate(param_names)})
            for k in ("phi", "nematic", "milling", "clustering", "drift", "chirality",
                      "segregation", "hexatic", "density_mode_peak", "local_density_cv",
                      "temporal_stability", "n"):
                if k in rep_summary:
                    rec[f"m_{k}"] = float(rep_summary[k])
            records.append(rec)
        champs = pd.DataFrame(records)
        champs.to_csv(champ_path, index=False)
        print(f"saved champions -> {champ_path}")

    # Render GIFs.
    manifest = []
    for _, row in champs.iterrows():
        obj = row["objective"]
        if obj not in OBJECTIVES or (args.only and obj not in names):
            continue
        spec = OBJECTIVES[obj]
        param_names = ACTIVE_SINGLE_NAMES if spec["kind"] == "single" else TWO_NAMES
        x = np.array([row[n] for n in param_names], float)
        t = time.time()
        out, val = animate_champion(obj, x, int(row["rep_seed"]), quick=args.quick)
        kb = out.stat().st_size // 1024
        print(f"[gif] {obj:22s} fitness={val:.3f} -> {out.name} ({kb} KB, {time.time()-t:.0f}s)", flush=True)
        manifest.append({"objective": obj, "gif": f"figures/gifs/{out.name}",
                         "fitness": float(val), "blurb": spec["blurb"],
                         "kind": spec["kind"], "size_kb": int(kb)})
    pd.DataFrame(manifest).to_csv(RESULTS / f"gif_manifest{FILE_SUFFIX}.csv", index=False)
    (RESULTS / f"gif_manifest{FILE_SUFFIX}.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nDONE in {time.time()-t0:.0f}s. {len(manifest)} GIFs in {GIFS}")


if __name__ == "__main__":
    main()
