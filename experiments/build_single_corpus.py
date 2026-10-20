from __future__ import annotations
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

import numpy as np
import pandas as pd
from scipy.stats import qmc

from spp_unified.parameters import SingleParams, SimulationConfig, ACTIVE_SINGLE_NAMES
from spp_unified.model import simulate_single
from spp_unified.learned import behavior_descriptor, hexatic_order, density_mode_peak, local_density_cv
from expanded_common import run_parallel

OUT=ROOT/'results'/'expanded'; OUT.mkdir(parents=True,exist_ok=True)

EXPANDED_BOUNDS=np.array([
    [0.05,0.70], [0.01,0.12], [0.0,5.0], [2.1,14.0], [0.0,5.0],
    [-0.80,0.80], [2.1,14.0], [0.0,5.0], [0.0,1.0]
],float)


def _worker(task):
    x, boundary, seed = task
    p=SingleParams.from_active_vector(np.asarray(x,float))
    cfg=SimulationConfig(S=3.2,r0=0.1,steps=160,sample_start=80,sample_every=2,
                         boundary=boundary,seed=int(seed),overlap_iterations=3)
    r=simulate_single(p,cfg,record_states=True,state_every=8)
    desc=behavior_descriptor(r.states,cfg,nbins=12,max_frames=10)
    xf,thf=r.states[-1]
    diag={
        'hexatic':hexatic_order(xf,cfg.S,boundary=='periodic'),
        'density_mode_peak':density_mode_peak(xf,cfg.S),
        'local_density_cv':local_density_cv(xf,4*cfg.r0,cfg.S,boundary=='periodic'),
    }
    return r.summary,diag,desc


def make_design(n=360,seed=20260710):
    # Equal representation of low, medium, and high densities.
    if n%3: raise ValueError('n must be divisible by 3')
    per=n//3
    dens_ranges=[(0.05,0.20),(0.20,0.45),(0.45,0.70)]
    rows=[]
    for b,(lo,hi) in enumerate(dens_ranges):
        m=int(np.ceil(np.log2(per)))
        U=qmc.Sobol(d=9,scramble=True,seed=seed+b).random_base2(m)[:per]
        X=EXPANDED_BOUNDS[:,0]+U*(EXPANDED_BOUNDS[:,1]-EXPANDED_BOUNDS[:,0])
        X[:,0]=lo+U[:,0]*(hi-lo)
        for i,x in enumerate(X):
            boundary='reflecting' if ((i+2*b)%5==0) else 'periodic'
            rows.append((x,b,boundary))
    rng=np.random.default_rng(seed)
    rng.shuffle(rows)
    return rows


def main():
    design=make_design()
    tasks=[(x,boundary,500000+i) for i,(x,band,boundary) in enumerate(design)]
    res=run_parallel(_worker,tasks,max_workers=4,desc='single corpus')
    data=[]; desc=[]
    for i,((x,band,boundary),(summary,diag,d)) in enumerate(zip(design,res)):
        row={'sample_id':i,'density_band':['low','medium','high'][band],'boundary':boundary}
        row.update({k:float(v) for k,v in zip(ACTIVE_SINGLE_NAMES,x)})
        row.update(summary); row.update(diag)
        data.append(row); desc.append(d)
    df=pd.DataFrame(data)
    df.to_csv(OUT/'single_behavior_corpus.csv',index=False)
    np.savez_compressed(OUT/'single_behavior_descriptors.npz',X=np.asarray(desc,np.float32))
    print(df.groupby('density_band')[['rho','n','phi','milling','clustering','hexatic','density_mode_peak']].mean())
    print('descriptor shape',np.asarray(desc).shape)

if __name__=='__main__': main()
