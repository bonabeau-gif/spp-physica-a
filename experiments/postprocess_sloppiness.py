from __future__ import annotations
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr
from expanded_common import archive_cells
from build_single_corpus import EXPANDED_BOUNDS
from spp_unified.parameters import ACTIVE_SINGLE_NAMES
OUT=ROOT/'results'/'expanded'

def main():
 df=pd.read_csv(OUT/'single_behavior_corpus_latent.csv');periodic=df[df.boundary=='periodic'].copy()
 U=(periodic[list(ACTIVE_SINGLE_NAMES)].to_numpy()-EXPANDED_BOUNDS[:,0])/(EXPANDED_BOUNDS[:,1]-EXPANDED_BOUNDS[:,0])
 outputs=['learned_0','learned_1','phi','milling','clustering']
 imp=[]
 for o in outputs:
  y=np.log1p(periodic[o].to_numpy()) if o=='clustering' else periodic[o].to_numpy()
  tr,te=train_test_split(np.arange(len(U)),test_size=.25,random_state=42)
  m=ExtraTreesRegressor(n_estimators=400,min_samples_leaf=3,max_features=.8,random_state=0,n_jobs=1).fit(U[tr],y[tr])
  pi=permutation_importance(m,U[te],y[te],n_repeats=20,random_state=0,n_jobs=1)
  for p,v,sd in zip(ACTIVE_SINGLE_NAMES,pi.importances_mean,pi.importances_std):imp.append({'output':o,'parameter':p,'importance':float(max(v,0)),'importance_sd':float(sd),'test_r2':float(m.score(U[te],y[te]))})
 impdf=pd.DataFrame(imp);impdf.to_csv(OUT/'global_surrogate_importance.csv',index=False)
 cells=archive_cells(df[['learned_0','learned_1']].to_numpy(),df.temporal_stability.to_numpy(),(20,20));idx=[v[0] for v in cells.values()]
 Ue=(df.loc[idx,list(ACTIVE_SINGLE_NAMES)].to_numpy()-EXPANDED_BOUNDS[:,0])/(EXPANDED_BOUNDS[:,1]-EXPANDED_BOUNDS[:,0])
 C=np.cov(Ue,rowvar=False);w,V=np.linalg.eigh(C);order=np.argsort(w);w=w[order];V=V[:,order]
 pd.DataFrame([{'direction':k+1,'variance':float(v),'inverse_variance':float(1/max(v,1e-12))} for k,v in enumerate(w)]).to_csv(OUT/'elite_concentration_eigenvalues.csv',index=False)
 pd.DataFrame([{'direction':k+1,'parameter':p,'loading':float(V[j,k])} for k in range(9) for j,p in enumerate(ACTIVE_SINGLE_NAMES)]).to_csv(OUT/'elite_concentration_loadings.csv',index=False)
 sens=pd.read_csv(OUT/'local_sloppiness_sensitivities.csv');local=sens.assign(absd=abs(sens.derivative)).groupby('parameter').absd.mean()
 glob=impdf.groupby('parameter').importance.mean()
 rho,pv=spearmanr(local.loc[list(ACTIVE_SINGLE_NAMES)],glob.loc[list(ACTIVE_SINGLE_NAMES)])
 ev=pd.read_csv(OUT/'local_sloppiness_eigenvalues.csv');spans={}
 for name,g in ev.groupby('regime'):
  pos=g.eigenvalue[g.eigenvalue>1e-8];spans[name]=float(np.log10(pos.max()/pos.min())) if len(pos)>1 else 0
 summary={'local_surrogate_rank_spearman':float(rho),'local_surrogate_rank_p':float(pv),'local_spectrum_decades':spans,
  'top_local_parameters':local.sort_values(ascending=False).head(5).index.tolist(),'top_surrogate_parameters':glob.sort_values(ascending=False).head(5).index.tolist(),
  'elite_count':len(idx),'elite_inverse_variance_span_decades':float(np.log10(w[-1]/w[0])),'surrogate_test_r2':impdf.groupby('output').test_r2.first().to_dict()}
 with open(OUT/'sloppiness_summary.json','w') as f:json.dump(summary,f,indent=2)
 print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
