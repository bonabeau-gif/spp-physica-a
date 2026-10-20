from __future__ import annotations
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,pandas as pd,torch
torch.set_num_threads(1);torch.set_num_interop_threads(1)
from scipy.stats import spearmanr
from sklearn.model_selection import StratifiedShuffleSplit
from spp_unified.learned import BehaviorAutoencoder
OUT=ROOT/'results'/'expanded'

def study(kind,corpus,desc,hidden,metrics,test_size,split_seed):
 df=pd.read_csv(OUT/corpus);X=np.load(OUT/desc)['X']
 strata=(df.density_band.astype(str)+'_'+df.boundary.astype(str)) if kind=='single' else df.source.astype(str)
 tr,te=next(StratifiedShuffleSplit(n_splits=1,test_size=test_size,random_state=split_seed).split(X,strata))
 rows=[]
 for ld in [2,3,4,6]:
  ae=BehaviorAutoencoder(X.shape[1],latent_dim=ld,hidden=hidden,seed=2);res=ae.fit(X,tr,te,max_epochs=500,patience=60);Z=ae.transform(X)
  row={'kind':kind,'latent_dim':ld,**res.__dict__}
  for m in metrics:
   rs=[abs(spearmanr(df[m],Z[:,j]).statistic) for j in range(ld)];row[f'max_abs_spearman_{m}']=max(rs)
  rows.append(row);print(row,flush=True)
 return rows
if __name__=='__main__':
 rows=[]
 rows+=study('single','single_behavior_corpus_latent.csv','single_behavior_descriptors.npz',(64,24),['phi','milling','clustering','rho'],.22,2026)
 rows+=study('multi','multi_behavior_corpus_latent.csv','multi_behavior_descriptors.npz',(80,28),['drift','chirality','segregation','phi'],.24,1)
 pd.DataFrame(rows).to_csv(OUT/'latent_dimension_study.csv',index=False)
