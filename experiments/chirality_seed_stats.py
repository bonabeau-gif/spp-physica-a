from __future__ import annotations
import sys,json
from dataclasses import replace
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import wilcoxon
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from spp_unified.parameters import ACTIVE_SINGLE_NAMES,SingleParams,TwoSpeciesParams,SimulationConfig
from spp_unified.model import simulate_two_species

def p_non(row):
 b=SingleParams.from_active_vector(np.array([row[n] for n in ACTIVE_SINGLE_NAMES]))
 return TwoSpeciesParams(base=b,fraction_B=row.fraction_B,alignment_symmetric=row.alignment_symmetric,
                         alignment_antisymmetric=row.alignment_antisymmetric,force_symmetric=row.force_symmetric,
                         force_antisymmetric=row.force_antisymmetric)
def p_rec(row):
 b=SingleParams.from_active_vector(np.array([row[n] for n in ACTIVE_SINGLE_NAMES]))
 return TwoSpeciesParams(base=b,fraction_B=row.fraction_B,alignment_symmetric=row.alignment_symmetric,
                         alignment_antisymmetric=0,force_symmetric=row.force_symmetric,force_antisymmetric=0)
non=pd.read_csv(ROOT/'results'/'multi_qd_nonreciprocal.csv');rec=pd.read_csv(ROOT/'results'/'multi_qd_reciprocal.csv')
pn=p_non(non.loc[non.chirality.idxmax()]);pm=replace(pn,alignment_antisymmetric=0,force_antisymmetric=0)
pr=p_rec(rec.loc[rec.chirality.idxmax()])
cfg=SimulationConfig(steps=800,sample_start=400,sample_every=2,boundary='periodic')
rows=[]
for seed in range(20):
 for label,p in [('nonreciprocal_chiral',pn),('matched_reciprocal',pm),('best_reciprocal_search',pr)]:
  r=simulate_two_species(p,cfg.with_seed(seed)).summary
  rows.append({'seed':seed,'condition':label,**r})
df=pd.DataFrame(rows);df.to_csv(ROOT/'results'/'multi_chirality_seed_values.csv',index=False)
piv=df.pivot(index='seed',columns='condition',values='chirality')
absw=df.pivot(index='seed',columns='condition',values='abs_omega')
stats={}
for ctrl in ['matched_reciprocal','best_reciprocal_search']:
 w,p=wilcoxon(piv['nonreciprocal_chiral'],piv[ctrl],alternative='greater')
 wa,pa=wilcoxon(absw['nonreciprocal_chiral'],absw[ctrl],alternative='greater')
 stats[ctrl]={'chirality_w':float(w),'chirality_p':float(p),'abs_omega_w':float(wa),'abs_omega_p':float(pa),
              'median_chirality_non':float(piv['nonreciprocal_chiral'].median()),'median_chirality_control':float(piv[ctrl].median()),
              'median_abs_omega_non':float(absw['nonreciprocal_chiral'].median()),'median_abs_omega_control':float(absw[ctrl].median())}
(ROOT/'results'/'multi_chirality_stats.json').write_text(json.dumps(stats,indent=2))
print(json.dumps(stats,indent=2))
