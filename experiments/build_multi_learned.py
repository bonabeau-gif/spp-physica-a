from __future__ import annotations
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,pandas as pd,torch
torch.set_num_threads(1);torch.set_num_interop_threads(1)
from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import silhouette_score
from scipy.stats import spearmanr

from spp_unified.parameters import SingleParams,TwoSpeciesParams,SimulationConfig,ACTIVE_SINGLE_NAMES,TWO_SPECIES_EXTRA_NAMES,EXPERIMENT_BOUNDS,TWO_SPECIES_EXTRA_BOUNDS
from spp_unified.model import simulate_two_species
from spp_unified.learned import species_behavior_descriptor,BehaviorAutoencoder,normalize_latent
from expanded_common import run_parallel
OUT=ROOT/'results'/'expanded';OUT.mkdir(exist_ok=True,parents=True)
NAMES=list(ACTIVE_SINGLE_NAMES)+list(TWO_SPECIES_EXTRA_NAMES)
BOUNDS=np.vstack([EXPERIMENT_BOUNDS,TWO_SPECIES_EXTRA_BOUNDS])

def _params(x):return TwoSpeciesParams.from_active_vector(np.asarray(x,float),reciprocal=False)
def _worker(task):
 x,seed=task;p=_params(x);cfg=SimulationConfig(S=4,r0=.1,steps=220,sample_start=100,sample_every=2,boundary='periodic',seed=int(seed),overlap_iterations=3)
 r=simulate_two_species(p,cfg,record_states=True,state_every=10)
 d=species_behavior_descriptor(r.states,r.species,cfg,nbins=8,max_frames=10)
 return r.summary,d

def archive_vectors():
 non=pd.read_csv(ROOT/'results'/'multi_qd_nonreciprocal.csv');rec=pd.read_csv(ROOT/'results'/'multi_qd_reciprocal.csv')
 # spread selections across archive cells
 non=non.sort_values(['cell_0','cell_1']).iloc[np.linspace(0,len(non)-1,36).astype(int)]
 rec=rec.sort_values(['cell_0','cell_1']).iloc[np.linspace(0,len(rec)-1,30).astype(int)]
 rows=[]
 for _,r in non.iterrows():rows.append((np.array([r.get(n,0.0) for n in NAMES],float),'archive_nonreciprocal'))
 for _,r in rec.iterrows():
  vals=[]
  for n in NAMES:
   vals.append(0.0 if n in ('alignment_antisymmetric','force_antisymmetric') else r.get(n,0.0))
  rows.append((np.array(vals,float),'archive_reciprocal'))
 return rows

def main():
 rows=archive_vectors();rng=np.random.default_rng(7711)
 for reciprocal in (False,True):
  for _ in range(30):
   u=rng.random(len(NAMES));x=BOUNDS[:,0]+u*(BOUNDS[:,1]-BOUNDS[:,0])
   if reciprocal:x[11]=0;x[13]=0
   rows.append((x,'random_reciprocal' if reciprocal else 'random_nonreciprocal'))
 rng.shuffle(rows)
 res=run_parallel(_worker,[(x,1200000+i) for i,(x,s) in enumerate(rows)],max_workers=6,desc='multi learned corpus')
 data=[];desc=[]
 for i,((x,src),(summary,d)) in enumerate(zip(rows,res)):
  row={'sample_id':i,'source':src,'reciprocal':('reciprocal' in src and 'nonreciprocal' not in src)}
  row.update({n:float(v) for n,v in zip(NAMES,x)});row.update(summary);data.append(row);desc.append(d)
 df=pd.DataFrame(data);X=np.asarray(desc,np.float32)
 strata=df.source.astype(str);sss=StratifiedShuffleSplit(n_splits=1,test_size=.24,random_state=1);tr,te=next(sss.split(X,strata))
 mets=[];models=[]
 for seed in range(4):
  ae=BehaviorAutoencoder(X.shape[1],hidden=(80,28),seed=seed);m=ae.fit(X,tr,te,max_epochs=500,patience=60);mets.append({'seed':seed,**m.__dict__});models.append(ae);print(mets[-1])
 met=pd.DataFrame(mets);bi=int(met.test_mse.idxmin());ae=models[bi];Z=ae.transform(X)
 _,lo,hi=normalize_latent(Z[tr]);Z01,_,_=normalize_latent(Z,lo,hi)
 ae.save(str(OUT/'multi_behavior_autoencoder.pt'));np.savez_compressed(OUT/'multi_behavior_descriptors.npz',X=X,Z=Z,Z01=Z01,lo=lo,hi=hi,train=tr,test=te)
 met.to_csv(OUT/'multi_autoencoder_metrics.csv',index=False)
 km=KMeans(n_clusters=10,n_init=50,random_state=0).fit(Z01);df['learned_0']=Z01[:,0];df['learned_1']=Z01[:,1];df['learned_cluster']=km.labels_
 reps=[]
 for c in range(10):
  idx=np.where(km.labels_==c)[0];reps.append(int(idx[np.argmin(((Z01[idx]-km.cluster_centers_[c])**2).sum(1))]))
 df['learned_representative']=False;df.loc[reps,'learned_representative']=True;df.to_csv(OUT/'multi_behavior_corpus_latent.csv',index=False)
 pd.DataFrame({'cluster':range(10),'sample_id':df.loc[reps,'sample_id'].astype(int).values}).to_csv(OUT/'multi_learned_representatives.csv',index=False)
 corr=[]
 for m in ['chirality','drift','segregation','phi','milling','clustering','alignment_antisymmetric','force_antisymmetric']:
  for j in range(2):
   r,p=spearmanr(df[m],Z01[:,j]);corr.append({'metric':m,'axis':j,'spearman_r':r,'p':p})
 pd.DataFrame(corr).to_csv(OUT/'multi_latent_correlations.csv',index=False)
 cs=df.groupby('learned_cluster').agg(count=('sample_id','size'),reciprocal_fraction=('reciprocal','mean'),chirality=('chirality','mean'),drift=('drift','mean'),segregation=('segregation','mean'),phi=('phi','mean'),milling=('milling','mean'),clustering=('clustering','mean')).reset_index();cs.to_csv(OUT/'multi_learned_cluster_summary.csv',index=False)
 summary={'selected_seed':int(met.loc[bi,'seed']),'autoencoder':{k:float(met.loc[bi,k]) for k in ['train_mse','test_mse','trustworthiness','knn_overlap','epochs']},'silhouette_10':float(silhouette_score(Z01,km.labels_)),'n':len(df)}
 with open(OUT/'multi_learned_summary.json','w') as f:json.dump(summary,f,indent=2)
 print(json.dumps(summary,indent=2));print(cs.to_string(index=False))
if __name__=='__main__':main()
