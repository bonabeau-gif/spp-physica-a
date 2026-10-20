from __future__ import annotations
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,pandas as pd,torch
torch.set_num_threads(1);torch.set_num_interop_threads(1)
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
from sklearn.model_selection import StratifiedShuffleSplit
from spp_unified.learned import BehaviorAutoencoder
OUT=ROOT/'results'/'expanded'

def run(prefix,corpus_file,desc_file,hidden=(64,24),seeds=4):
 df=pd.read_csv(OUT/corpus_file);npz=np.load(OUT/desc_file);X=npz['X']
 strata=(df.density_band.astype(str)+'_'+df.boundary.astype(str)) if 'density_band' in df else df.source.astype(str)
 tr,te=next(StratifiedShuffleSplit(n_splits=1,test_size=.22 if 'density_band' in df else .24,random_state=2026 if 'density_band' in df else 1).split(X,strata))
 Z=[]
 for s in range(seeds):
  a=BehaviorAutoencoder(X.shape[1],hidden=hidden,seed=s);a.fit(X,tr,te,max_epochs=450 if prefix=='single' else 500,patience=50 if prefix=='single' else 60);Z.append(a.transform(X)[te])
 rows=[]
 for i in range(seeds):
  for j in range(i+1,seeds):
   r,p=spearmanr(pdist(Z[i]),pdist(Z[j]));rows.append({'seed_a':i,'seed_b':j,'distance_spearman':r,'p':p})
 pd.DataFrame(rows).to_csv(OUT/f'{prefix}_latent_seed_stability.csv',index=False)
 return {'median_distance_spearman':float(np.median([r['distance_spearman'] for r in rows])),'min':float(np.min([r['distance_spearman'] for r in rows])),'max':float(np.max([r['distance_spearman'] for r in rows]))}
if __name__=='__main__':
 out={'single':run('single','single_behavior_corpus.csv','single_behavior_descriptors.npz'),
      'multi':run('multi','multi_behavior_corpus_latent.csv','multi_behavior_descriptors.npz',hidden=(80,28))}
 with open(OUT/'latent_seed_stability.json','w') as f:json.dump(out,f,indent=2)
 print(json.dumps(out,indent=2))
