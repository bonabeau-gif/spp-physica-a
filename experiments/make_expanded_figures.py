from __future__ import annotations
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,pandas as pd,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from spp_unified.parameters import SingleParams,TwoSpeciesParams,SimulationConfig,ACTIVE_SINGLE_NAMES,TWO_SPECIES_EXTRA_NAMES
from spp_unified.model import simulate_single,simulate_two_species

OUT=ROOT/'results'/'expanded';FIG=ROOT/'figures'/'expanded';FIG.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.size':9,'axes.labelsize':9,'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8,'figure.dpi':140,'savefig.dpi':300})


def panel(ax,label):ax.text(-.12,1.05,label,transform=ax.transAxes,fontweight='bold',fontsize=11,va='top')
def clean(ax):ax.spines['top'].set_visible(False);ax.spines['right'].set_visible(False)

def single_state(row,steps=260,seed=42):
 p=SingleParams.from_active_vector(row[list(ACTIVE_SINGLE_NAMES)].to_numpy(float))
 cfg=SimulationConfig(S=3.2,r0=.1,steps=steps,sample_start=steps//2,sample_every=2,boundary=row.boundary,seed=seed,overlap_iterations=3)
 return simulate_single(p,cfg,record_states=True,state_every=max(8,steps//20)),cfg

def multi_state(row,steps=300,seed=42):
 vals=np.array([row[n] for n in list(ACTIVE_SINGLE_NAMES)+list(TWO_SPECIES_EXTRA_NAMES)],float)
 p=TwoSpeciesParams.from_active_vector(vals,reciprocal=False);cfg=SimulationConfig(S=4,r0=.1,steps=steps,sample_start=steps//2,sample_every=2,boundary='periodic',seed=seed,overlap_iterations=3)
 return simulate_two_species(p,cfg,record_states=True,state_every=max(10,steps//20)),cfg

def draw_state(ax,x,th,S,species=None):
 # Heading is shown by arrow direction, matching Figure 4. Single-species states
 # use uniform dark arrows; two-species states colour each arrow by particle type.
 u=np.cos(th);v=np.sin(th)
 if species is None:
  ax.quiver(x[:,0],x[:,1],u,v,color='#1a1a1a',angles='xy',scale_units='xy',scale=17,width=.0035,pivot='mid')
 else:
  colors=np.where(species==0,'#2C6DB2','#C74343')
  ax.quiver(x[:,0],x[:,1],u,v,color=colors,angles='xy',scale_units='xy',scale=18,width=.004,pivot='mid')
 ax.set(xlim=(0,S),ylim=(0,S),xticks=[],yticks=[]);ax.set_aspect('equal');
 for s in ax.spines.values():s.set_linewidth(.7);s.set_color('.35')


def fig_single_latent():
 d=pd.read_csv(OUT/'single_behavior_corpus_latent.csv');fig,axs=plt.subplots(2,2,figsize=(7.0,6.1),constrained_layout=True)
 specs=[('phi','Polar alignment'),('milling','Milling'),('clustering','Clustering'),('rho','Density fraction')]
 for k,(ax,(c,lab)) in enumerate(zip(axs.ravel(),specs)):
  val=np.log1p(d[c]) if c=='clustering' else d[c]
  sc=ax.scatter(d.learned_0,d.learned_1,c=val,s=19,cmap='viridis',alpha=.86,edgecolors='none');ax.set_xlabel('learned measure 1');ax.set_ylabel('learned measure 2');clean(ax);panel(ax,chr(97+k));cb=fig.colorbar(sc,ax=ax,pad=.01);cb.set_label('log(1 + clustering)' if c=='clustering' else lab)
 fig.savefig(FIG/'fig_single_learned_space.png',bbox_inches='tight');plt.close(fig)


def fig_single_examples():
 d=pd.read_csv(OUT/'single_behavior_corpus_latent.csv');r=d[d.learned_representative].sort_values('learned_cluster')
 fig,axs=plt.subplots(3,4,figsize=(8.2,6.5),constrained_layout=True)
 for k,(ax,(_,row)) in enumerate(zip(axs.ravel(),r.iterrows())):
  sim,cfg=single_state(row,steps=240,seed=3100+k);x,th=sim.states[-1];draw_state(ax,x,th,cfg.S)
  ax.text(.02,.98,f"{chr(97+k)}  cluster {int(row.learned_cluster)}",transform=ax.transAxes,va='top',ha='left',fontsize=8,fontweight='bold',bbox=dict(facecolor='white',alpha=.7,edgecolor='none',pad=1.5))
  ax.text(.02,.02,f"ρ={row.rho:.2f}  φ={sim.summary['phi']:.2f}\nm={sim.summary['milling']:.2f}  Γ={sim.summary['clustering']:.2f}",transform=ax.transAxes,va='bottom',fontsize=7,bbox=dict(facecolor='white',alpha=.72,edgecolor='none',pad=1.2))
 fig.savefig(FIG/'fig_single_learned_examples.png',bbox_inches='tight');plt.close(fig)


def choose_high(d):
 h=d[d.density_band=='high']
 picks=[]
 criteria=[h.phi,h.clustering,h.density_mode_peak,h.hexatic,h.nematic*(1-h.phi),-abs(h.clustering-1)-h.phi,h.local_density_cv,h.temporal_stability]
 for s in criteria:
  for idx in s.sort_values(ascending=False).index:
   if idx not in picks: picks.append(idx);break
 return d.loc[picks[:8]]

def fig_high_examples():
 d=pd.read_csv(OUT/'single_behavior_corpus_latent.csv');r=choose_high(d)
 fig,axs=plt.subplots(2,4,figsize=(8.3,4.4),constrained_layout=True)
 for k,(ax,(_,row)) in enumerate(zip(axs.ravel(),r.iterrows())):
  sim,cfg=single_state(row,steps=260,seed=4100+k);x,th=sim.states[-1];draw_state(ax,x,th,cfg.S)
  ax.text(.02,.98,chr(97+k),transform=ax.transAxes,va='top',fontweight='bold',bbox=dict(facecolor='white',alpha=.7,edgecolor='none',pad=1))
  ax.text(.02,.02,f"N={int(sim.summary['n'])}, ρ={row.rho:.2f}\nφ={sim.summary['phi']:.2f}, Γ={sim.summary['clustering']:.2f}",transform=ax.transAxes,va='bottom',fontsize=7,bbox=dict(facecolor='white',alpha=.72,edgecolor='none',pad=1.2))
 r.to_csv(OUT/'high_density_representatives.csv',index=False)
 fig.savefig(FIG/'fig_high_density_examples.png',bbox_inches='tight');plt.close(fig)


def fig_density_dependence():
 d=pd.read_csv(OUT/'single_behavior_corpus_latent.csv');bins=np.arange(.05,.76,.1);d['bin']=pd.cut(d.rho,bins,include_lowest=True);cent=np.array([(x.left+x.right)/2 for x in d['bin'].cat.categories])
 specs=[('phi','polar alignment'),('milling','milling'),('clustering','clustering'),('density_mode_peak','low-k density peak')]
 fig,axs=plt.subplots(2,2,figsize=(7.0,5.2),constrained_layout=True)
 for k,(ax,(c,lab)) in enumerate(zip(axs.ravel(),specs)):
  g=d.groupby('bin',observed=False)[c];q10=g.quantile(.1).to_numpy();q50=g.quantile(.5).to_numpy();q90=g.quantile(.9).to_numpy();ax.fill_between(cent,q10,q90,alpha=.2);ax.plot(cent,q50,'o-',lw=1.5,ms=4);ax.set_xlabel('density fraction ρ');ax.set_ylabel(lab);clean(ax);panel(ax,chr(97+k))
 fig.savefig(FIG/'fig_density_dependence.png',bbox_inches='tight');plt.close(fig)


def initial_cov(rep):
 d=pd.read_csv(OUT/'single_behavior_corpus_latent.csv');d=d[d.boundary=='periodic'].reset_index(drop=True);rng=np.random.default_rng(1100+rep);idx=[]
 for b in ['low','medium','high']:
  pool=np.where(d.density_band.values==b)[0];idx.extend(rng.choice(pool,40,replace=False).tolist())
 z=d.iloc[idx];cells=set()
 for a,b in z[['learned_0','learned_1']].to_numpy():cells.add((min(int(a*20),19),min(int(b*20),19)))
 return len(cells)/400

def fig_surrogate():
 h=pd.read_csv(OUT/'learned_illumination_history.csv');p=pd.read_csv(OUT/'learned_surrogate_predictions.csv');fig,axs=plt.subplots(2,2,figsize=(7.2,5.4),constrained_layout=True)
 for method,g in h.groupby('method'):
  piv=g.pivot(index='true_calls',columns='replicate',values='coverage');x=np.r_[0,piv.index.to_numpy()];vals=np.vstack([[initial_cov(r) for r in piv.columns],piv.to_numpy()]);axs[0,0].plot(x,vals.mean(1),marker='o',label=method);axs[0,0].fill_between(x,vals.min(1),vals.max(1),alpha=.13)
  q=g.pivot(index='true_calls',columns='replicate',values='mean_quality');axs[0,1].plot(q.index,q.mean(1),marker='o',label=method);axs[0,1].fill_between(q.index,q.min(1),q.max(1),alpha=.13)
 axs[0,0].set(xlabel='additional simulator calls',ylabel='learned-archive coverage');axs[0,1].set(xlabel='additional simulator calls',ylabel='mean elite stability')
 for ax in axs[0]:clean(ax);ax.legend(frameon=False);panel(ax,'a' if ax is axs[0,0] else 'b')
 axs[1,0].scatter(p.pred_z0,p.actual_z0,s=16,alpha=.65,label='measure 1');axs[1,0].scatter(p.pred_z1,p.actual_z1,s=16,alpha=.65,label='measure 2');axs[1,0].plot([0,1],[0,1],'k--',lw=1);axs[1,0].set(xlabel='GP prediction',ylabel='verified learned measure');axs[1,0].legend(frameon=False);clean(axs[1,0]);panel(axs[1,0],'c')
 axs[1,1].errorbar(p.pred_q,p.actual_q,xerr=1.645*p.sd_q,fmt='o',ms=3,alpha=.55);lo=min(p.pred_q.min(),p.actual_q.min());hi=max(p.pred_q.max(),p.actual_q.max());axs[1,1].plot([lo,hi],[lo,hi],'k--',lw=1);axs[1,1].set(xlabel='GP-predicted stability',ylabel='verified stability');clean(axs[1,1]);panel(axs[1,1],'d')
 fig.savefig(FIG/'fig_surrogate_benchmark.png',bbox_inches='tight');plt.close(fig)


def fig_sloppiness():
 ev=pd.read_csv(OUT/'local_sloppiness_eigenvalues.csv');s=pd.read_csv(OUT/'local_sloppiness_sensitivities.csv');gi=pd.read_csv(OUT/'global_surrogate_importance.csv');ee=pd.read_csv(OUT/'elite_concentration_eigenvalues.csv');fig,axs=plt.subplots(2,2,figsize=(8.0,6.1),constrained_layout=True)
 for name,g in ev.groupby('regime'):
  y=g.eigenvalue.to_numpy();y=y/max(y.max(),1e-12);axs[0,0].semilogy(g.direction,y,'o-',ms=3,label=name.replace('_',' '))
 axs[0,0].set(xlabel='local eigen-direction',ylabel='normalized eigenvalue',ylim=(1e-8,1.2));axs[0,0].legend(frameon=False,ncol=2,fontsize=6);clean(axs[0,0]);panel(axs[0,0],'a')
 heat=s.assign(absd=abs(s.derivative)).groupby(['regime','parameter']).absd.mean().unstack().reindex(columns=list(ACTIVE_SINGLE_NAMES));im=axs[0,1].imshow(heat.to_numpy(),aspect='auto',cmap='magma');axs[0,1].set_xticks(range(9),[x.replace('_','\n') for x in ACTIVE_SINGLE_NAMES],rotation=45,ha='right',fontsize=6);axs[0,1].set_yticks(range(len(heat)),[x.replace('_',' ') for x in heat.index],fontsize=7);fig.colorbar(im,ax=axs[0,1],pad=.01,label='mean standardized |derivative|');panel(axs[0,1],'b')
 gh=gi.groupby(['output','parameter']).importance.mean().unstack().reindex(columns=list(ACTIVE_SINGLE_NAMES));im=axs[1,0].imshow(gh.to_numpy(),aspect='auto',cmap='viridis');axs[1,0].set_xticks(range(9),[x.replace('_','\n') for x in ACTIVE_SINGLE_NAMES],rotation=45,ha='right',fontsize=6);axs[1,0].set_yticks(range(len(gh)),gh.index,fontsize=7);fig.colorbar(im,ax=axs[1,0],pad=.01,label='permutation importance');panel(axs[1,0],'c')
 axs[1,1].semilogy(ee.direction,ee.inverse_variance,'o-');axs[1,1].set(xlabel='elite-ensemble direction (most constrained first)',ylabel='inverse variance');clean(axs[1,1]);panel(axs[1,1],'d')
 fig.savefig(FIG/'fig_sloppiness_expanded.png',bbox_inches='tight');plt.close(fig)


def fig_multi_latent():
 d=pd.read_csv(OUT/'multi_behavior_corpus_latent.csv');fig,axs=plt.subplots(2,2,figsize=(7,6.0),constrained_layout=True)
 # categorical source
 cmap={'archive_nonreciprocal':'#C74343','random_nonreciprocal':'#E59A9A','archive_reciprocal':'#2C6DB2','random_reciprocal':'#8BB7E3'}
 for src,g in d.groupby('source'):axs[0,0].scatter(g.learned_0,g.learned_1,s=20,c=cmap[src],label=src.replace('_',' '),alpha=.8)
 axs[0,0].legend(frameon=False,fontsize=6);axs[0,0].set(xlabel='learned measure 1',ylabel='learned measure 2');clean(axs[0,0]);panel(axs[0,0],'a')
 for ax,(c,lab) in zip([axs[0,1],axs[1,0],axs[1,1]],[('drift','center-of-mass drift'),('chirality','persistent chirality'),('segregation','species segregation')]):
  sc=ax.scatter(d.learned_0,d.learned_1,c=d[c],s=20,cmap='viridis',alpha=.85);ax.set(xlabel='learned measure 1',ylabel='learned measure 2');clean(ax);fig.colorbar(sc,ax=ax,pad=.01,label=lab)
 panel(axs[0,1],'b');panel(axs[1,0],'c');panel(axs[1,1],'d');fig.savefig(FIG/'fig_multi_learned_space.png',bbox_inches='tight');plt.close(fig)


def fig_multi_examples():
 d=pd.read_csv(OUT/'multi_behavior_corpus_latent.csv');r=d[d.learned_representative].sort_values('learned_cluster');fig,axs=plt.subplots(2,5,figsize=(10,4.3),constrained_layout=True)
 for k,(ax,(_,row)) in enumerate(zip(axs.ravel(),r.iterrows())):
  sim,cfg=multi_state(row,steps=300,seed=5100+k);x,th=sim.states[-1];draw_state(ax,x,th,cfg.S,sim.species)
  ax.text(.02,.98,f"{chr(97+k)}  cluster {int(row.learned_cluster)}",transform=ax.transAxes,va='top',fontsize=7,fontweight='bold',bbox=dict(facecolor='white',alpha=.7,edgecolor='none',pad=1))
  ax.text(.02,.02,f"d={sim.summary['drift']:.2f}  C={sim.summary['chirality']:.2f}\nseg={sim.summary['segregation']:.2f}",transform=ax.transAxes,va='bottom',fontsize=6.5,bbox=dict(facecolor='white',alpha=.72,edgecolor='none',pad=1))
 fig.savefig(FIG/'fig_multi_learned_examples.png',bbox_inches='tight');plt.close(fig)


def fig_learned_validation():
 sm=json.load(open(OUT/'single_learned_summary.json'));mm=json.load(open(OUT/'multi_learned_summary.json'));stab=json.load(open(OUT/'latent_seed_stability.json'));fig,axs=plt.subplots(1,3,figsize=(7.4,2.6),constrained_layout=True)
 labels=['trustworthiness','kNN overlap'];single=[sm['autoencoder']['trustworthiness'],sm['autoencoder']['knn_overlap']];pca=[sm['pca']['trustworthiness_test'],sm['pca']['knn_overlap_test']];x=np.arange(2);w=.35;axs[0].bar(x-w/2,single,w,label='autoencoder');axs[0].bar(x+w/2,pca,w,label='PCA');axs[0].set_xticks(x,labels,rotation=20);axs[0].set_ylim(0,1);axs[0].legend(frameon=False);clean(axs[0]);panel(axs[0],'a')
 axs[1].bar(['single species','two species'],[stab['single']['median_distance_spearman'],stab['multi']['median_distance_spearman']]);axs[1].set_ylim(0,1);axs[1].set_ylabel('median pairwise-distance correlation');axs[1].tick_params(axis='x',rotation=20);clean(axs[1]);panel(axs[1],'b')
 axs[2].bar(['single','multi'],[sm['autoencoder']['test_mse'],mm['autoencoder']['test_mse']]);axs[2].set_ylabel('held-out standardized MSE');clean(axs[2]);panel(axs[2],'c')
 fig.savefig(FIG/'fig_learned_validation.png',bbox_inches='tight');plt.close(fig)


def fig_latent_dimension():
 d=pd.read_csv(OUT/'latent_dimension_study.csv');fig,axs=plt.subplots(1,2,figsize=(7.1,2.8),constrained_layout=True)
 for kind,g in d.groupby('kind'):
  axs[0].plot(g.latent_dim,g.test_mse,'o-',label=kind)
  axs[1].plot(g.latent_dim,g.trustworthiness,'o-',label=kind)
 axs[0].set(xlabel='latent dimensions',ylabel='held-out standardized MSE',xticks=[2,3,4,6]);axs[1].set(xlabel='latent dimensions',ylabel='trustworthiness',xticks=[2,3,4,6],ylim=(.82,1.0))
 for k,ax in enumerate(axs):clean(ax);panel(ax,chr(97+k));ax.legend(frameon=False)
 fig.savefig(FIG/'fig_latent_dimension.png',bbox_inches='tight');plt.close(fig)
if __name__=='__main__':
 fig_single_latent();fig_single_examples();fig_high_examples();fig_density_dependence();fig_surrogate();fig_sloppiness();fig_multi_latent();fig_multi_examples();fig_learned_validation();fig_latent_dimension();print('wrote',FIG)
