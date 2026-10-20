from __future__ import annotations
import sys
from dataclasses import replace
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from spp_unified.parameters import ACTIVE_SINGLE_NAMES,SingleParams,TwoSpeciesParams,SimulationConfig
from spp_unified.evaluation import evaluate_two_species

def p_non(row):
 b=SingleParams.from_active_vector(np.array([row[n] for n in ACTIVE_SINGLE_NAMES]))
 return TwoSpeciesParams(base=b,fraction_B=row.fraction_B,alignment_symmetric=row.alignment_symmetric,
                         alignment_antisymmetric=row.alignment_antisymmetric,
                         force_symmetric=row.force_symmetric,force_antisymmetric=row.force_antisymmetric)
def p_rec(row):
 b=SingleParams.from_active_vector(np.array([row[n] for n in ACTIVE_SINGLE_NAMES]))
 return TwoSpeciesParams(base=b,fraction_B=row.fraction_B,alignment_symmetric=row.alignment_symmetric,
                         alignment_antisymmetric=0.0,force_symmetric=row.force_symmetric,force_antisymmetric=0.0)
non=pd.read_csv(ROOT/'results'/'multi_qd_nonreciprocal.csv')
rec=pd.read_csv(ROOT/'results'/'multi_qd_reciprocal.csv')
pn=p_non(non.loc[non.chirality.idxmax()])
pm=replace(pn,alignment_antisymmetric=0.0,force_antisymmetric=0.0)
pr=p_rec(rec.loc[rec.chirality.idxmax()])
rows=[]
for steps in [400,800,1200]:
 cfg=SimulationConfig(steps=steps,sample_start=steps//2,sample_every=2,boundary='periodic')
 for label,p in [('nonreciprocal_chiral',pn),('matched_reciprocal',pm),('best_reciprocal_search',pr)]:
  m=evaluate_two_species(p,cfg,list(range(12)))
  rows.append({'condition':label,'steps':steps,**m})
out=pd.DataFrame(rows);out.to_csv(ROOT/'results'/'multi_chirality_horizon.csv',index=False)
fig,ax=plt.subplots(figsize=(5.6,3.9))
for label,g in out.groupby('condition'):
 ax.errorbar(g.steps,g.chirality,yerr=g.chirality_sd,marker='o',capsize=3,label=label.replace('_',' '))
ax.set_xlabel('simulation steps');ax.set_ylabel('persistent chirality score');ax.set_ylim(-.03,1.02);ax.legend(frameon=False,fontsize=8)
fig.tight_layout();fig.savefig(ROOT/'figures'/'fig_chirality_horizon.png',dpi=240);plt.close(fig)
print(out[['condition','steps','chirality','chirality_sd','abs_omega','abs_omega_sd','rotation_r2','drift']].to_string(index=False))
