from __future__ import annotations
import sys,json,warnings
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))

import numpy as np
import pandas as pd
import torch
torch.set_num_threads(1);torch.set_num_interop_threads(1)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel,Matern,WhiteKernel
from scipy.stats import spearmanr

from spp_unified.parameters import SingleParams,SimulationConfig,ACTIVE_SINGLE_NAMES
from spp_unified.model import simulate_single
from spp_unified.learned import BehaviorAutoencoder,behavior_descriptor,normalize_latent,hexatic_order,density_mode_peak
from expanded_common import run_parallel,archive_cells
from build_single_corpus import EXPANDED_BOUNDS

OUT=ROOT/'results'/'expanded';DELTA=.04


def load_encoder():
    X=np.load(OUT/'single_behavior_descriptors.npz')['X'];ck=torch.load(OUT/'single_behavior_autoencoder.pt',map_location='cpu',weights_only=False)
    ae=BehaviorAutoencoder(X.shape[1],seed=int(ck['seed']));ae.model.load_state_dict(ck['state_dict']);ae.mean=ck['mean'];ae.std=ck['std'];ae.model.eval()
    lat=np.load(OUT/'single_behavior_latent.npz');return ae,lat['lo'],lat['hi']


def _worker(task):
    x,boundary,seed=task;p=SingleParams.from_active_vector(np.asarray(x,float))
    cfg=SimulationConfig(S=3.2,r0=.1,steps=150,sample_start=75,sample_every=2,boundary=boundary,seed=int(seed),overlap_iterations=3)
    r=simulate_single(p,cfg,record_states=True,state_every=8);d=behavior_descriptor(r.states,cfg)
    xf,_=r.states[-1]
    return r.summary,d,hexatic_order(xf,cfg.S,boundary=='periodic'),density_mode_peak(xf,cfg.S)


def main():
    ae,lo,hi=load_encoder();df=pd.read_csv(OUT/'single_behavior_corpus_latent.csv')
    # Six regimes spanning classical and learned behavior spaces, including high densities.
    picks={
      'aligned':int(df.phi.idxmax()),
      'milling':int((df.milling*(1-df.phi)).idxmax()),
      'clustered':int(df.clustering.idxmax()),
      'high_density_aligned':int(df[df.density_band=='high'].phi.idxmax()),
      'high_density_aggregate':int(df[df.density_band=='high'].clustering.idxmax()),
      'density_wave':int(df.density_mode_peak.idxmax()),
    }
    reps=[]
    for name,i in picks.items():
        row=df.loc[i];u=(row[list(ACTIVE_SINGLE_NAMES)].to_numpy(float)-EXPANDED_BOUNDS[:,0])/(EXPANDED_BOUNDS[:,1]-EXPANDED_BOUNDS[:,0])
        reps.append((name,u,row.boundary,int(row.sample_id)))
    pd.DataFrame([{'regime':n,'sample_id':sid,'boundary':b,**{f'u_{j}':u[j] for j in range(9)}} for n,u,b,sid in reps]).to_csv(OUT/'sloppiness_representatives.csv',index=False)

    tasks=[];meta=[]
    for ri,(name,u,boundary,sid) in enumerate(reps):
      for j in range(9):
       for sign in (-1,1):
        up=u.copy();up[j]=np.clip(up[j]+sign*DELTA,0,1);x=EXPANDED_BOUNDS[:,0]+up*(EXPANDED_BOUNDS[:,1]-EXPANDED_BOUNDS[:,0])
        for seedrep in range(2):
         tasks.append((x,boundary,950000+ri*10000+j*100+seedrep));meta.append((name,j,sign,seedrep))
    res=run_parallel(_worker,tasks,max_workers=6,desc='local sloppiness')
    records=[]
    for m,(summary,d,hx,pk) in zip(meta,res):
      z=ae.transform(np.asarray([d]));z01,_,_=normalize_latent(z,lo,hi);name,j,sign,sr=m
      records.append({'regime':name,'parameter':ACTIVE_SINGLE_NAMES[j],'j':j,'sign':sign,'seedrep':sr,
       'phi':summary['phi'],'milling':summary['milling'],'log_clustering':np.log1p(summary['clustering']),
       'nematic':summary['nematic'],'learned_0':z01[0,0],'learned_1':z01[0,1],'hexatic':hx,'density_mode_peak':pk})
    rdf=pd.DataFrame(records);rdf.to_csv(OUT/'local_sloppiness_raw.csv',index=False)
    outs=['phi','milling','log_clustering','nematic','learned_0','learned_1','hexatic','density_mode_peak']
    corpus=df.copy();corpus['log_clustering']=np.log1p(corpus.clustering)
    scales=corpus[outs].std().replace(0,1).to_numpy()
    sens_rows=[];eig_rows=[];load_rows=[]
    for name,_,_,_ in reps:
      J=np.zeros((len(outs),9))
      for j,p in enumerate(ACTIVE_SINGLE_NAMES):
        g=rdf[(rdf.regime==name)&(rdf.parameter==p)].groupby('sign')[outs].mean()
        J[:,j]=(g.loc[1].to_numpy()-g.loc[-1].to_numpy())/(2*DELTA)/scales
        for oi,o in enumerate(outs):sens_rows.append({'regime':name,'output':o,'parameter':p,'derivative':J[oi,j]})
      F=J.T@J
      w,V=np.linalg.eigh(F);order=np.argsort(w)[::-1];w=w[order];V=V[:,order]
      for k,val in enumerate(w):eig_rows.append({'regime':name,'direction':k+1,'eigenvalue':float(val)})
      for k in range(3):
       for j,p in enumerate(ACTIVE_SINGLE_NAMES):load_rows.append({'regime':name,'direction':k+1,'parameter':p,'loading':float(V[j,k])})
    pd.DataFrame(sens_rows).to_csv(OUT/'local_sloppiness_sensitivities.csv',index=False)
    pd.DataFrame(eig_rows).to_csv(OUT/'local_sloppiness_eigenvalues.csv',index=False)
    pd.DataFrame(load_rows).to_csv(OUT/'local_sloppiness_loadings.csv',index=False)

    # Global ARD length scales from periodic simulations only.
    pdat=df[df.boundary=='periodic'].copy();U=(pdat[list(ACTIVE_SINGLE_NAMES)].to_numpy()-EXPANDED_BOUNDS[:,0])/(EXPANDED_BOUNDS[:,1]-EXPANDED_BOUNDS[:,0])
    Ys={'learned_0':pdat.learned_0.to_numpy(),'learned_1':pdat.learned_1.to_numpy(),'phi':pdat.phi.to_numpy(),
        'milling':pdat.milling.to_numpy(),'log_clustering':np.log1p(pdat.clustering.to_numpy())}
    ard=[]
    for key,y in Ys.items():
      ker=ConstantKernel(1,(1e-2,1e2))*Matern(length_scale=np.ones(9),length_scale_bounds=(.03,30),nu=2.5)+WhiteKernel(.02,(1e-5,.5))
      gp=GaussianProcessRegressor(kernel=ker,normalize_y=True,alpha=1e-6,n_restarts_optimizer=0,random_state=0)
      with warnings.catch_warnings(): warnings.simplefilter('ignore');gp.fit(U,y)
      # Constant*Matern + White
      ls=np.asarray(gp.kernel_.k1.k2.length_scale,float)
      for p,l in zip(ACTIVE_SINGLE_NAMES,ls):ard.append({'output':key,'parameter':p,'length_scale':float(l),'inverse_length_scale':float(1/l)})
    ard_df=pd.DataFrame(ard);ard_df.to_csv(OUT/'global_gp_ard.csv',index=False)

    # Elite-ensemble concentration in the learned archive.
    cells=archive_cells(df[['learned_0','learned_1']].to_numpy(),df.temporal_stability.to_numpy(),(20,20))
    idx=[v[0] for v in cells.values()];Ue=(df.loc[idx,list(ACTIVE_SINGLE_NAMES)].to_numpy()-EXPANDED_BOUNDS[:,0])/(EXPANDED_BOUNDS[:,1]-EXPANDED_BOUNDS[:,0])
    C=np.cov(Ue,rowvar=False);w,V=np.linalg.eigh(C);order=np.argsort(w);w=w[order];V=V[:,order] # smallest = most constrained
    er=[];el=[]
    for k,val in enumerate(w):
      er.append({'direction':k+1,'variance':float(val),'inverse_variance':float(1/max(val,1e-12))})
      for j,p in enumerate(ACTIVE_SINGLE_NAMES):el.append({'direction':k+1,'parameter':p,'loading':float(V[j,k])})
    pd.DataFrame(er).to_csv(OUT/'elite_concentration_eigenvalues.csv',index=False);pd.DataFrame(el).to_csv(OUT/'elite_concentration_loadings.csv',index=False)

    sens=pd.DataFrame(sens_rows);local=sens.assign(absd=lambda d:abs(d.derivative)).groupby('parameter').absd.mean()
    gp_rank=ard_df.groupby('parameter').inverse_length_scale.mean()
    rho,pv=spearmanr(local.loc[list(ACTIVE_SINGLE_NAMES)],gp_rank.loc[list(ACTIVE_SINGLE_NAMES)])
    spans={}
    ev=pd.DataFrame(eig_rows)
    for name,g in ev.groupby('regime'):
      pos=g.eigenvalue[g.eigenvalue>1e-8]
      spans[name]=float(np.log10(pos.max()/pos.min())) if len(pos)>1 else 0
    summary={'local_gp_rank_spearman':float(rho),'local_gp_rank_p':float(pv),'local_spectrum_decades':spans,
      'top_local_parameters':local.sort_values(ascending=False).head(5).index.tolist(),
      'top_gp_parameters':gp_rank.sort_values(ascending=False).head(5).index.tolist(),
      'elite_count':len(idx),'elite_inverse_variance_span_decades':float(np.log10(1/w[0])-np.log10(1/w[-1]))}
    with open(OUT/'sloppiness_summary.json','w') as f:json.dump(summary,f,indent=2)
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
