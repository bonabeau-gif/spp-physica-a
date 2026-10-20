from __future__ import annotations
from pathlib import Path
import numpy as np,pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
ROOT=Path(__file__).resolve().parents[1];RES=ROOT/'results';FIG=ROOT/'figures'

# 3D single-species repertoire.
a=pd.read_csv(RES/'single_qd_alignment_milling.csv');b=pd.read_csv(RES/'single_qd_alignment_clustering.csv')
d=pd.concat([a,b],ignore_index=True).drop_duplicates(subset=['phi','milling','clustering'])
fig=plt.figure(figsize=(6.3,5.2));ax=fig.add_subplot(111,projection='3d')
sc=ax.scatter(d.phi,d.milling,d.clustering,c=d.objective,s=22,alpha=.8)
ax.set_xlabel('polar alignment');ax.set_ylabel('milling');ax.set_zlabel('clustering')
fig.colorbar(sc,ax=ax,pad=.12,label='temporal stability',shrink=.7)
fig.tight_layout();fig.savefig(FIG/'fig_single_repertoire_3d.png',dpi=260);plt.close(fig)

# Representative normalized parameter profiles.
from spp_unified.parameters import ACTIVE_SINGLE_NAMES,EXPERIMENT_BOUNDS
r=pd.read_csv(RES/'single_representatives_parameters.csv',index_col=0)
X=r[list(ACTIVE_SINGLE_NAMES)].to_numpy(float)
X=(X-EXPERIMENT_BOUNDS[:,0])/(EXPERIMENT_BOUNDS[:,1]-EXPERIMENT_BOUNDS[:,0])
fig,ax=plt.subplots(figsize=(7.0,3.5));im=ax.imshow(X,aspect='auto',vmin=0,vmax=1)
ax.set_yticks(range(len(r.index)),r.index);ax.set_xticks(range(len(ACTIVE_SINGLE_NAMES)),ACTIVE_SINGLE_NAMES,rotation=55,ha='right')
fig.colorbar(im,ax=ax,label='position within search range')
fig.tight_layout();fig.savefig(FIG/'fig_single_parameter_profiles.png',dpi=240);plt.close(fig)

# Side-by-side reciprocal/nonreciprocal archives.
non=pd.read_csv(RES/'multi_qd_nonreciprocal.csv');rec=pd.read_csv(RES/'multi_qd_reciprocal.csv')
fig,axes=plt.subplots(1,2,figsize=(9.2,3.8),sharex=True,sharey=True)
for ax,df,label in zip(axes,[rec,non],['reciprocal','non-reciprocal']):
 arr=np.full((12,12),np.nan)
 for _,row in df.iterrows():arr[int(row.cell_0),int(row.cell_1)]=row.objective
 im=ax.imshow(arr.T,origin='lower',extent=(0,1,0,1),aspect='auto',vmin=.8,vmax=1)
 ax.text(.03,.96,label,transform=ax.transAxes,va='top',fontsize=10)
 ax.set_xlabel('net center-of-mass drift')
axes[0].set_ylabel('persistent chirality score')
fig.colorbar(im,ax=axes.ravel().tolist(),label='temporal stability',shrink=.88)
fig.subplots_adjust(left=.08,right=.90,bottom=.16,top=.96,wspace=.12)
fig.savefig(FIG/'fig_multi_archive_comparison.png',dpi=240);plt.close(fig)

# Seed-level chirality validation.
s=pd.read_csv(RES/'multi_chirality_seed_values.csv')
order=['nonreciprocal_chiral','matched_reciprocal','best_reciprocal_search']
vals=[s[s.condition==q].chirality.to_numpy() for q in order]
fig,ax=plt.subplots(figsize=(5.7,3.8));ax.boxplot(vals,tick_labels=['non-reciprocal\nchiral','matched\nreciprocal','best reciprocal\nsearch elite'],showmeans=True)
ax.set_ylabel('persistent chirality score (800 steps)');ax.set_ylim(-.03,1.03)
fig.tight_layout();fig.savefig(FIG/'fig_chirality_seed_comparison.png',dpi=240);plt.close(fig)

# Coverage histories.
files=[('single_qd_alignment_milling_history.csv','single: alignment-milling'),
       ('single_qd_alignment_clustering_history.csv','single: alignment-clustering'),
       ('multi_qd_nonreciprocal_history.csv','multi: non-reciprocal'),
       ('multi_qd_reciprocal_history.csv','multi: reciprocal')]
fig,ax=plt.subplots(figsize=(5.8,3.8))
for f,label in files:
 h=pd.read_csv(RES/f);ax.plot(h.evaluations,h.coverage,marker='o',label=label)
ax.set_xlabel('simulator evaluations');ax.set_ylabel('archive coverage');ax.set_ylim(0,.5);ax.legend(frameon=False,fontsize=8)
fig.tight_layout();fig.savefig(FIG/'fig_archive_coverage.png',dpi=240);plt.close(fig)
print('additional figures written')
