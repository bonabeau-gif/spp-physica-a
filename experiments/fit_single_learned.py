from __future__ import annotations
import sys, json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))

import numpy as np
import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.manifold import trustworthiness
from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors

from spp_unified.learned import BehaviorAutoencoder, normalize_latent

OUT=ROOT/'results'/'expanded'


def knn_overlap(X,Z,k=10):
    nx=NearestNeighbors(n_neighbors=k).fit(X).kneighbors(return_distance=False)
    nz=NearestNeighbors(n_neighbors=k).fit(Z).kneighbors(return_distance=False)
    return float(np.mean([len(set(a)&set(b))/k for a,b in zip(nx,nz)]))


def main():
    df=pd.read_csv(OUT/'single_behavior_corpus.csv')
    X=np.load(OUT/'single_behavior_descriptors.npz')['X']
    strata=df['density_band'].astype(str)+'_'+df['boundary'].astype(str)
    sss=StratifiedShuffleSplit(n_splits=1,test_size=.22,random_state=2026)
    train,test=next(sss.split(X,strata))
    rows=[]; models=[]
    for seed in range(4):
        ae=BehaviorAutoencoder(X.shape[1],seed=seed)
        res=ae.fit(X,train,test,max_epochs=450,patience=50)
        rows.append({'seed':seed,**res.__dict__}); models.append(ae)
        print(rows[-1],flush=True)
    met=pd.DataFrame(rows)
    best_i=int(met.sort_values(['test_mse','trustworthiness'],ascending=[True,False]).index[0])
    ae=models[best_i]
    Z=ae.transform(X)
    _,lo,hi=normalize_latent(Z[train])
    Z01,_,_=normalize_latent(Z,lo,hi)
    ae.save(str(OUT/'single_behavior_autoencoder.pt'))
    np.savez_compressed(OUT/'single_behavior_latent.npz',Z=Z,Z01=Z01,lo=lo,hi=hi,train=train,test=test)
    met.to_csv(OUT/'single_autoencoder_metrics.csv',index=False)

    # PCA baseline on the exact same standardized split.
    mean=X[train].mean(0); sd=X[train].std(0); sd[sd<1e-6]=1
    Xs=(X-mean)/sd
    pca=PCA(n_components=2,random_state=0).fit(Xs[train])
    Zp=pca.transform(Xs)
    pca_metrics={
        'explained_variance_2d':float(pca.explained_variance_ratio_.sum()),
        'trustworthiness_test':float(trustworthiness(Xs[test],Zp[test],n_neighbors=10)),
        'knn_overlap_test':knn_overlap(Xs[test],Zp[test],10),
    }

    km=KMeans(n_clusters=12,n_init=50,random_state=0).fit(Z01)
    df['learned_0']=Z01[:,0]; df['learned_1']=Z01[:,1]; df['learned_cluster']=km.labels_
    # centroid representatives
    reps=[]
    for c in range(12):
        idx=np.where(km.labels_==c)[0]
        d=((Z01[idx]-km.cluster_centers_[c])**2).sum(1)
        reps.append(int(idx[np.argmin(d)]))
    df['learned_representative']=False
    df.loc[reps,'learned_representative']=True
    df.to_csv(OUT/'single_behavior_corpus_latent.csv',index=False)
    pd.DataFrame({'cluster':range(12),'sample_id':df.loc[reps,'sample_id'].astype(int).values}).to_csv(OUT/'single_learned_representatives.csv',index=False)

    metrics=['rho','phi','milling','clustering','nematic','drift','chirality','hexatic','density_mode_peak','local_density_cv']
    corr=[]
    for m in metrics:
        for j in range(2):
            r,p=spearmanr(df[m],Z01[:,j])
            corr.append({'metric':m,'axis':j,'spearman_r':float(r),'p':float(p)})
    pd.DataFrame(corr).to_csv(OUT/'single_latent_correlations.csv',index=False)

    cluster_summary=df.groupby('learned_cluster').agg(
        count=('sample_id','size'),rho=('rho','mean'),phi=('phi','mean'),milling=('milling','mean'),
        clustering=('clustering','mean'),nematic=('nematic','mean'),hexatic=('hexatic','mean'),
        density_mode_peak=('density_mode_peak','mean'),stability=('temporal_stability','mean'),
        high_density_fraction=('density_band',lambda s:float(np.mean(s=='high'))),
        reflecting_fraction=('boundary',lambda s:float(np.mean(s=='reflecting'))),
    ).reset_index()
    cluster_summary.to_csv(OUT/'single_learned_cluster_summary.csv',index=False)

    summary={
        'selected_seed':int(met.loc[best_i,'seed']),
        'autoencoder':{k:float(met.loc[best_i,k]) for k in ['train_mse','test_mse','trustworthiness','knn_overlap','epochs']},
        'pca':pca_metrics,
        'silhouette_12':float(silhouette_score(Z01,km.labels_)),
        'latent_lo':lo.tolist(),'latent_hi':hi.tolist(),
    }
    with open(OUT/'single_learned_summary.json','w') as f: json.dump(summary,f,indent=2)
    print(json.dumps(summary,indent=2))
    print(cluster_summary.to_string(index=False))

if __name__=='__main__':main()
