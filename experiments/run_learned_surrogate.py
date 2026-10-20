from __future__ import annotations
import sys, json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))

import numpy as np
import pandas as pd
import torch
torch.set_num_threads(1); torch.set_num_interop_threads(1)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

from spp_unified.parameters import SingleParams, SimulationConfig, ACTIVE_SINGLE_NAMES
from spp_unified.model import simulate_single
from spp_unified.learned import BehaviorAutoencoder, behavior_descriptor, normalize_latent, hexatic_order, density_mode_peak, local_density_cv
from expanded_common import run_parallel, archive_cells
from build_single_corpus import EXPANDED_BOUNDS

OUT=ROOT/'results'/'expanded'; DIMS=(20,20)


def load_encoder():
    X=np.load(OUT/'single_behavior_descriptors.npz')['X']
    ck=torch.load(OUT/'single_behavior_autoencoder.pt',map_location='cpu',weights_only=False)
    ae=BehaviorAutoencoder(X.shape[1],seed=int(ck['seed']))
    ae.model.load_state_dict(ck['state_dict']); ae.mean=ck['mean']; ae.std=ck['std']; ae.model.eval()
    lat=np.load(OUT/'single_behavior_latent.npz')
    return ae,lat['lo'],lat['hi']


def _worker(task):
    x, seed = task
    p=SingleParams.from_active_vector(np.asarray(x,float))
    cfg=SimulationConfig(S=3.2,r0=.1,steps=160,sample_start=80,sample_every=2,
                         boundary='periodic',seed=int(seed),overlap_iterations=3)
    r=simulate_single(p,cfg,record_states=True,state_every=8)
    d=behavior_descriptor(r.states,cfg,nbins=12,max_frames=10)
    xf,_=r.states[-1]
    diag={'hexatic':hexatic_order(xf,cfg.S,True),'density_mode_peak':density_mode_peak(xf,cfg.S),
          'local_density_cv':local_density_cv(xf,4*cfg.r0,cfg.S,True)}
    return r.summary,diag,d


def eval_batch(U,seeds,ae,lo,hi):
    X=EXPANDED_BOUNDS[:,0]+np.asarray(U)*(EXPANDED_BOUNDS[:,1]-EXPANDED_BOUNDS[:,0])
    res=run_parallel(_worker,[(x,int(s)) for x,s in zip(X,seeds)],max_workers=6,desc='surrogate batch')
    desc=np.asarray([r[2] for r in res],np.float32)
    Z=ae.transform(desc); Z01,_,_=normalize_latent(Z,lo,hi)
    rows=[]
    for x,u,z,(summary,diag,d) in zip(X,U,Z01,res):
        row={k:float(v) for k,v in zip(ACTIVE_SINGLE_NAMES,x)}
        row.update({f'u_{i}':float(v) for i,v in enumerate(u)})
        row.update(summary); row.update(diag); row['learned_0']=float(z[0]);row['learned_1']=float(z[1])
        rows.append(row)
    return pd.DataFrame(rows),desc


def fit_gps(U,Y):
    models=[]
    for k in range(Y.shape[1]):
        kernel=ConstantKernel(1.0,constant_value_bounds='fixed')*Matern(length_scale=np.full(U.shape[1],.28),
                 length_scale_bounds='fixed',nu=2.5)+WhiteKernel(noise_level=.015,noise_level_bounds='fixed')
        gp=GaussianProcessRegressor(kernel=kernel,alpha=1e-5,normalize_y=True,optimizer=None)
        gp.fit(U,Y[:,k]);models.append(gp)
    return models


def predict_gps(models,U):
    mus=[];sds=[]
    for m in models:
        mu,sd=m.predict(U,return_std=True);mus.append(mu);sds.append(sd)
    return np.column_stack(mus),np.column_stack(sds)


def current_cells(df):
    return archive_cells(df[['learned_0','learned_1']].to_numpy(),df['temporal_stability'].to_numpy(),DIMS)


def generate_direct(rng,df,batch=10):
    cells=current_cells(df); elite_idx=[v[0] for v in cells.values()]
    out=[]
    for _ in range(batch):
        if rng.random()<.2 or not elite_idx:
            u=rng.random(9)
        else:
            row=df.iloc[elite_idx[rng.integers(len(elite_idx))]]
            parent=np.array([row[f'u_{i}'] for i in range(9)])
            u=np.clip(parent+rng.normal(0,.10 if rng.random()>.15 else .22,9),0,1)
        out.append(u)
    return np.asarray(out)


def generate_surrogate(rng,df,models,batch=10):
    cells=current_cells(df); elite_idx=[v[0] for v in cells.values()]
    Ur=rng.random((4500,9))
    if elite_idx:
        parents=np.asarray([[df.iloc[i][f'u_{j}'] for j in range(9)] for i in rng.choice(elite_idx,1500,replace=True)])
        Um=np.clip(parents+rng.normal(0,.13,parents.shape),0,1)
        U=np.vstack([Ur,Um])
    else: U=Ur
    mu,sd=predict_gps(models,U)
    z=np.clip(mu[:,:2],0,1); q=mu[:,2]+1.3*sd[:,2]+.20*(sd[:,0]+sd[:,1])
    candidates={}
    for i,zz in enumerate(z):
        c=tuple(np.minimum((zz*np.asarray(DIMS)).astype(int),np.asarray(DIMS)-1))
        bonus=.35 if c not in cells else 0.0
        val=float(q[i]+bonus)
        if c not in candidates or val>candidates[c][0]: candidates[c]=(val,i,c not in cells)
    ranked=sorted(candidates.values(),key=lambda a:(a[2],a[0]),reverse=True)
    chosen=[]
    for _,i,_ in ranked:
        if len(chosen)>=batch: break
        if not chosen or np.min(np.linalg.norm(U[i]-U[np.asarray(chosen)],axis=1))>.12:
            chosen.append(i)
    if len(chosen)<batch:
        chosen.extend([i for _,i,_ in ranked if i not in chosen][:batch-len(chosen)])
    return U[np.asarray(chosen[:batch])],mu[np.asarray(chosen[:batch])],sd[np.asarray(chosen[:batch])]


def add_initial_columns(df):
    U=(df[list(ACTIVE_SINGLE_NAMES)].to_numpy()-EXPANDED_BOUNDS[:,0])/(EXPANDED_BOUNDS[:,1]-EXPANDED_BOUNDS[:,0])
    for i in range(9): df[f'u_{i}']=U[:,i]
    return df


def run_one(rep,method,initial,ae,lo,hi,calls=60,batch=10):
    rng=np.random.default_rng(7000+rep*101+{'surrogate':1,'direct':2,'random':3}[method])
    df=initial.copy().reset_index(drop=True); df['source']='initial';df['round']=0
    histories=[]; preds=[]
    for rd in range(1,calls//batch+1):
        if method=='surrogate':
            U=df[[f'u_{i}' for i in range(9)]].to_numpy()
            Y=df[['learned_0','learned_1','temporal_stability']].to_numpy()
            models=fit_gps(U,Y)
            Unew,mu,sd=generate_surrogate(rng,df,models,batch)
        elif method=='direct':
            Unew=generate_direct(rng,df,batch);mu=sd=None
        else:
            Unew=rng.random((batch,9));mu=sd=None
        nd,_=eval_batch(Unew,[800000+rep*10000+rd*100+i for i in range(batch)],ae,lo,hi)
        nd['source']=method;nd['round']=rd
        if mu is not None:
            for i in range(batch):
                preds.append({'replicate':rep,'round':rd,'actual_z0':nd.iloc[i].learned_0,'actual_z1':nd.iloc[i].learned_1,
                    'actual_q':nd.iloc[i].temporal_stability,'pred_z0':mu[i,0],'pred_z1':mu[i,1],'pred_q':mu[i,2],
                    'sd_z0':sd[i,0],'sd_z1':sd[i,1],'sd_q':sd[i,2]})
        df=pd.concat([df,nd],ignore_index=True)
        cells=current_cells(df)
        histories.append({'replicate':rep,'method':method,'true_calls':rd*batch,'coverage':len(cells)/(DIMS[0]*DIMS[1]),
                          'num_elites':len(cells),'mean_quality':float(np.mean([v[1] for v in cells.values()]))})
        print(rep,method,rd,histories[-1],flush=True)
    return df,pd.DataFrame(histories),pd.DataFrame(preds)


def main():
    ae,lo,hi=load_encoder()
    corpus=add_initial_columns(pd.read_csv(OUT/'single_behavior_corpus_latent.csv'))
    corpus=corpus[corpus.boundary=='periodic'].reset_index(drop=True)
    all_rows=[];all_hist=[];all_pred=[]
    for rep in range(2):
        # Stratified initial design of 120 from the representation corpus.
        rng=np.random.default_rng(1100+rep)
        idx=[]
        for band in ['low','medium','high']:
            pool=np.where(corpus.density_band.values==band)[0]
            idx.extend(rng.choice(pool,40,replace=False).tolist())
        initial=corpus.iloc[idx].copy()
        for method in ['surrogate','direct','random']:
            df,h,p=run_one(rep,method,initial,ae,lo,hi,calls=30)
            df['replicate']=rep;df['method']=method
            all_rows.append(df[df.source!='initial'].copy());all_hist.append(h)
            if len(p):all_pred.append(p)
    pd.concat(all_rows,ignore_index=True).to_csv(OUT/'learned_illumination_evaluations.csv',index=False)
    pd.concat(all_hist,ignore_index=True).to_csv(OUT/'learned_illumination_history.csv',index=False)
    if all_pred: pd.concat(all_pred,ignore_index=True).to_csv(OUT/'learned_surrogate_predictions.csv',index=False)
    hist=pd.concat(all_hist)
    final=hist.sort_values('true_calls').groupby(['replicate','method']).tail(1)
    summ=final.groupby('method').agg(coverage_mean=('coverage','mean'),coverage_sd=('coverage','std'),
        elites_mean=('num_elites','mean'),quality_mean=('mean_quality','mean')).reset_index()
    summ.to_csv(OUT/'learned_illumination_summary.csv',index=False)
    pred=pd.concat(all_pred,ignore_index=True)
    metrics={}
    for k in ['z0','z1','q']:
        err=pred[f'actual_{k}']-pred[f'pred_{k}']
        metrics[f'rmse_{k}']=float(np.sqrt(np.mean(err**2)))
        metrics[f'mae_{k}']=float(np.mean(np.abs(err)))
        metrics[f'coverage90_{k}']=float(np.mean(np.abs(err)<=1.645*pred[f'sd_{k}']))
    with open(OUT/'learned_surrogate_metrics.json','w') as f:json.dump(metrics,f,indent=2)
    print(summ.to_string(index=False));print(metrics)

if __name__=='__main__':main()
