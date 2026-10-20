from __future__ import annotations
import argparse, json, sys, time
from dataclasses import replace
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from spp_unified.parameters import (
    ACTIVE_SINGLE_NAMES, EXPERIMENT_BOUNDS, TWO_SPECIES_EXTRA_BOUNDS,
    SingleParams, TwoSpeciesParams, SimulationConfig,
)
from spp_unified.model import simulate_single, simulate_two_species
from spp_unified.search import map_elites

RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)


def _R():
    from experiments import run_all
    return run_all


def _two_non(x):
    n=len(ACTIVE_SINGLE_NAMES); b=SingleParams.from_active_vector(x[:n])
    return TwoSpeciesParams(base=b,fraction_B=x[n],alignment_symmetric=x[n+1],
                             alignment_antisymmetric=x[n+2],force_symmetric=x[n+3],
                             force_antisymmetric=x[n+4])


def _two_rec(x):
    n=len(ACTIVE_SINGLE_NAMES); b=SingleParams.from_active_vector(x[:n])
    return TwoSpeciesParams(base=b,fraction_B=x[n],alignment_symmetric=x[n+1],
                             alignment_antisymmetric=0.0,force_symmetric=x[n+2],
                             force_antisymmetric=0.0)


def single_mill(quick):
    steps = 120 if quick else 180
    cfg = SimulationConfig(steps=steps, sample_start=steps//2, boundary='reflecting')
    def ev(x, seed):
        r = simulate_single(SingleParams.from_active_vector(x), cfg.with_seed(seed)).summary
        return r['temporal_stability'], (r['phi'], r['milling']), r
    dims=(10,10) if quick else (12,12)
    n=260 if quick else 650
    res=map_elites(ev,ACTIVE_SINGLE_NAMES,EXPERIMENT_BOUNDS,[(0,1),(0,.5)],dims=dims,
                   evaluations=n,init_random=70 if quick else 150,seed=11)
    res.elites.to_csv(RESULTS/'single_qd_alignment_milling.csv',index=False)
    res.history.to_csv(RESULTS/'single_qd_alignment_milling_history.csv',index=False)


def single_clust(quick):
    steps = 120 if quick else 180
    cfg = SimulationConfig(steps=steps, sample_start=steps//2, boundary='periodic')
    def ev(x, seed):
        r = simulate_single(SingleParams.from_active_vector(x), cfg.with_seed(seed)).summary
        return r['temporal_stability'], (r['phi'], r['clustering']), r
    dims=(10,10) if quick else (12,12)
    n=260 if quick else 650
    res=map_elites(ev,ACTIVE_SINGLE_NAMES,EXPERIMENT_BOUNDS,[(0,1),(0,8)],dims=dims,
                   evaluations=n,init_random=70 if quick else 150,seed=23)
    res.elites.to_csv(RESULTS/'single_qd_alignment_clustering.csv',index=False)
    res.history.to_csv(RESULTS/'single_qd_alignment_clustering_history.csv',index=False)


def multi_non(quick):
    steps=140 if quick else 200
    cfg=SimulationConfig(steps=steps,sample_start=steps//2,boundary='periodic')
    bounds=np.vstack([EXPERIMENT_BOUNDS,TWO_SPECIES_EXTRA_BOUNDS])
    names=ACTIVE_SINGLE_NAMES+("fraction_B","alignment_symmetric","alignment_antisymmetric","force_symmetric","force_antisymmetric")
    def ev(x,seed):
        r=simulate_two_species(_two_non(x),cfg.with_seed(seed)).summary
        return r['temporal_stability'],(r['drift'],r['chirality']),r
    dims=(10,10) if quick else (12,12)
    n=320 if quick else 750
    res=map_elites(ev,names,bounds,[(0,1),(0,1)],dims=dims,evaluations=n,
                   init_random=90 if quick else 180,seed=31)
    res.elites.to_csv(RESULTS/'multi_qd_nonreciprocal.csv',index=False)
    res.history.to_csv(RESULTS/'multi_qd_nonreciprocal_history.csv',index=False)


def multi_rec(quick):
    steps=140 if quick else 200
    cfg=SimulationConfig(steps=steps,sample_start=steps//2,boundary='periodic')
    extra=np.array([[.2,.8],[-1,1.5],[-1,1]])
    bounds=np.vstack([EXPERIMENT_BOUNDS,extra])
    names=ACTIVE_SINGLE_NAMES+("fraction_B","alignment_symmetric","force_symmetric")
    def ev(x,seed):
        r=simulate_two_species(_two_rec(x),cfg.with_seed(seed)).summary
        return r['temporal_stability'],(r['drift'],r['chirality']),r
    dims=(10,10) if quick else (12,12)
    n=320 if quick else 750
    res=map_elites(ev,names,bounds,[(0,1),(0,1)],dims=dims,evaluations=n,
                   init_random=90 if quick else 180,seed=37)
    res.elites.to_csv(RESULTS/'multi_qd_reciprocal.csv',index=False)
    res.history.to_csv(RESULTS/'multi_qd_reciprocal_history.csv',index=False)


def core(quick):
    R=_R()
    steps=120 if quick else 180
    cp=SimulationConfig(steps=steps,sample_start=steps//2,boundary='periodic')
    cr=replace(cp,boundary='reflecting')
    R.run_force_law_figure(); R.run_noise_sweep(cp,quick)
    for obj,cfg,seed in [('alignment',cp,1),('milling',cr,2),('clustering',cp,3)]:
        R.simple_de(obj,cfg,quick,seed)


def single_post(quick):
    R=_R()
    mill=pd.read_csv(R.RESULTS/'single_qd_alignment_milling.csv')
    clust=pd.read_csv(R.RESULTS/'single_qd_alignment_clustering.csv')
    dims=(10,10) if quick else (12,12)
    R.plot_archive(mill,dims,(0,1),(0,.5),'objective','polar alignment','milling','fig_archive_alignment_milling.png')
    R.plot_archive(clust,dims,(0,1),(0,8),'objective','polar alignment','clustering','fig_archive_alignment_clustering.png')
    reps=R.select_single_representatives(mill,clust)
    R.validate_single_representatives(reps,quick)
    de={k:json.loads((R.RESULTS/f'de_{k}_best.json').read_text()) for k in ['alignment','milling','clustering']}
    R.plot_method_summary(de,pd.DataFrame(),mill,clust)
    # Persist representatives for downstream stages.
    pd.DataFrame({k:v for k,v in reps.items()}).T.to_csv(R.RESULTS/'single_representatives_parameters.csv')


def single_finite(quick):
    R=_R()
    repsdf=pd.read_csv(R.RESULTS/'single_representatives_parameters.csv',index_col=0)
    reps={k:row for k,row in repsdf.iterrows()}
    R.run_finite_size(reps,quick)


def single_sensitivity(quick):
    R=_R()
    repsdf=pd.read_csv(R.RESULTS/'single_representatives_parameters.csv',index_col=0)
    reps={k:row for k,row in repsdf.iterrows()}
    R.run_sensitivity(reps,quick)


def multi_post(quick):
    R=_R()
    non=pd.read_csv(R.RESULTS/'multi_qd_nonreciprocal.csv')
    rec=pd.read_csv(R.RESULTS/'multi_qd_reciprocal.csv')
    dims=(10,10) if quick else (12,12)
    R.plot_archive(non,dims,(0,1),(0,1),'objective','net center-of-mass drift','chiral rotation score','fig_archive_nonreciprocal.png')
    R.plot_archive(rec,dims,(0,1),(0,1),'objective','net center-of-mass drift','chiral rotation score','fig_archive_reciprocal.png')
    R.validate_multi(non,quick)


def summary(quick):
    R=_R()
    mill=pd.read_csv(R.RESULTS/'single_qd_alignment_milling.csv')
    clust=pd.read_csv(R.RESULTS/'single_qd_alignment_clustering.csv')
    non=pd.read_csv(R.RESULTS/'multi_qd_nonreciprocal.csv')
    rec=pd.read_csv(R.RESULTS/'multi_qd_reciprocal.csv')
    out={
      'quick':quick,
      'single_alignment_milling_elites':len(mill),
      'single_alignment_clustering_elites':len(clust),
      'multi_nonreciprocal_elites':len(non),
      'multi_reciprocal_elites':len(rec),
      'single_max_phi':float(max(mill.phi.max(),clust.phi.max())),
      'single_max_milling':float(mill.milling.max()),
      'single_max_clustering':float(clust.clustering.max()),
      'multi_max_drift_nonreciprocal':float(non.drift.max()),
      'multi_max_chirality_nonreciprocal':float(non.chirality.max()),
      'multi_max_drift_reciprocal':float(rec.drift.max()),
      'multi_max_chirality_reciprocal':float(rec.chirality.max()),
    }
    R.save_json(R.RESULTS/'experiment_summary.json',out)
    print(json.dumps(out,indent=2))

STAGES={'core':core,'single_mill':single_mill,'single_clust':single_clust,
        'single_post':single_post,'single_finite':single_finite,'single_sensitivity':single_sensitivity,
        'multi_non':multi_non,'multi_rec':multi_rec,'multi_post':multi_post,'summary':summary}

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('stage',choices=STAGES); ap.add_argument('--quick',action='store_true')
    a=ap.parse_args(); t=time.time(); STAGES[a.stage](a.quick); print(f'{a.stage} completed in {time.time()-t:.1f}s')
