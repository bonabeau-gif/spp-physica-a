from __future__ import annotations
import argparse,sys
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from spp_unified.parameters import ACTIVE_SINGLE_NAMES,EXPERIMENT_BOUNDS,SingleParams,SimulationConfig
from spp_unified.evaluation import evaluate_single
OUT=ROOT/'results'/'sensitivity_chunks';OUT.mkdir(parents=True,exist_ok=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('state',choices=['aligned','milling']);ap.add_argument('parameter',choices=ACTIVE_SINGLE_NAMES);a=ap.parse_args()
 reps=pd.read_csv(ROOT/'results'/'single_representatives_parameters.csv',index_col=0)
 center=np.array([reps.loc[a.state,n] for n in ACTIVE_SINGLE_NAMES],float)
 j=ACTIVE_SINGLE_NAMES.index(a.parameter);lo,hi=EXPERIMENT_BOUNDS[j]
 cfg=SimulationConfig(steps=300,sample_start=150,boundary='reflecting' if a.state=='milling' else 'periodic')
 rows=[]
 for value in np.linspace(lo,hi,7):
  x=center.copy();x[j]=value;p=SingleParams.from_active_vector(x)
  m=evaluate_single(p,cfg,[41,42,43])
  rows.append({'state':a.state,'parameter':a.parameter,'value':value,**m})
 pd.DataFrame(rows).to_csv(OUT/f'{a.state}_{a.parameter}.csv',index=False)
 print(a.state,a.parameter,'done')
