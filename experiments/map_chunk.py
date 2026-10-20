from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from spp_unified.parameters import ACTIVE_SINGLE_NAMES,EXPERIMENT_BOUNDS,TWO_SPECIES_EXTRA_BOUNDS,SingleParams,TwoSpeciesParams,SimulationConfig
from spp_unified.model import simulate_single,simulate_two_species
from spp_unified.search import map_elites

OUT=ROOT/'results'/'chunks'; OUT.mkdir(parents=True,exist_ok=True)

def two_non(x):
 n=len(ACTIVE_SINGLE_NAMES); b=SingleParams.from_active_vector(x[:n])
 return TwoSpeciesParams(base=b,fraction_B=x[n],alignment_symmetric=x[n+1],alignment_antisymmetric=x[n+2],force_symmetric=x[n+3],force_antisymmetric=x[n+4])
def two_rec(x):
 n=len(ACTIVE_SINGLE_NAMES); b=SingleParams.from_active_vector(x[:n])
 return TwoSpeciesParams(base=b,fraction_B=x[n],alignment_symmetric=x[n+1],alignment_antisymmetric=0,force_symmetric=x[n+2],force_antisymmetric=0)

def setup(kind,quick):
 dims=(10,10) if quick else (12,12)
 if kind=='single_mill':
  steps=120 if quick else 180; cfg=SimulationConfig(steps=steps,sample_start=steps//2,boundary='reflecting')
  names=ACTIVE_SINGLE_NAMES; bounds=EXPERIMENT_BOUNDS; ranges=[(0,1),(0,.5)]
  def ev(x,s):
   r=simulate_single(SingleParams.from_active_vector(x),cfg.with_seed(s)).summary
   return r['temporal_stability'],(r['phi'],r['milling']),r
 elif kind=='single_clust':
  steps=120 if quick else 180; cfg=SimulationConfig(steps=steps,sample_start=steps//2,boundary='periodic')
  names=ACTIVE_SINGLE_NAMES; bounds=EXPERIMENT_BOUNDS; ranges=[(0,1),(0,8)]
  def ev(x,s):
   r=simulate_single(SingleParams.from_active_vector(x),cfg.with_seed(s)).summary
   return r['temporal_stability'],(r['phi'],r['clustering']),r
 elif kind=='multi_non':
  steps=140 if quick else 200; cfg=SimulationConfig(steps=steps,sample_start=steps//2,boundary='periodic')
  names=ACTIVE_SINGLE_NAMES+("fraction_B","alignment_symmetric","alignment_antisymmetric","force_symmetric","force_antisymmetric")
  bounds=np.vstack([EXPERIMENT_BOUNDS,TWO_SPECIES_EXTRA_BOUNDS]); ranges=[(0,1),(0,1)]
  def ev(x,s):
   r=simulate_two_species(two_non(x),cfg.with_seed(s)).summary
   return r['temporal_stability'],(r['drift'],r['chirality']),r
 elif kind=='multi_rec':
  steps=140 if quick else 200; cfg=SimulationConfig(steps=steps,sample_start=steps//2,boundary='periodic')
  names=ACTIVE_SINGLE_NAMES+("fraction_B","alignment_symmetric","force_symmetric")
  bounds=np.vstack([EXPERIMENT_BOUNDS,np.array([[.2,.8],[-1,1.5],[-1,1]])]); ranges=[(0,1),(0,1)]
  def ev(x,s):
   r=simulate_two_species(two_rec(x),cfg.with_seed(s)).summary
   return r['temporal_stability'],(r['drift'],r['chirality']),r
 else: raise ValueError(kind)
 return dims,names,bounds,ranges,ev

if __name__=='__main__':
 ap=argparse.ArgumentParser(); ap.add_argument('kind'); ap.add_argument('run_id',type=int); ap.add_argument('--quick',action='store_true'); ap.add_argument('--evaluations',type=int,default=160)
 a=ap.parse_args(); dims,names,bounds,ranges,ev=setup(a.kind,a.quick)
 seed={'single_mill':11,'single_clust':23,'multi_non':31,'multi_rec':37}[a.kind]+1009*a.run_id
 t=time.time(); res=map_elites(ev,names,bounds,ranges,dims=dims,evaluations=a.evaluations,init_random=min(70,a.evaluations//2),seed=seed,record_every=20)
 res.elites['run_id']=a.run_id
 res.elites.to_csv(OUT/f'{a.kind}_{a.run_id}.csv',index=False)
 res.history['run_id']=a.run_id
 res.history.to_csv(OUT/f'{a.kind}_{a.run_id}_history.csv',index=False)
 print(f'{a.kind} chunk {a.run_id}: {len(res.elites)} elites in {time.time()-t:.1f}s')
