from __future__ import annotations
import argparse,sys
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
CH=ROOT/'results'/'chunks'; RES=ROOT/'results'
MAP={
 'single_mill':('single_qd_alignment_milling.csv','single_qd_alignment_milling_history.csv'),
 'single_clust':('single_qd_alignment_clustering.csv','single_qd_alignment_clustering_history.csv'),
 'multi_non':('multi_qd_nonreciprocal.csv','multi_qd_nonreciprocal_history.csv'),
 'multi_rec':('multi_qd_reciprocal.csv','multi_qd_reciprocal_history.csv'),
}
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('kind');ap.add_argument('--dims',type=int,default=10);a=ap.parse_args()
 files=sorted(CH.glob(f'{a.kind}_[0-9]*.csv'))
 dfs=[pd.read_csv(f) for f in files if '_history' not in f.name]
 if not dfs: raise SystemExit('no chunks')
 all_df=pd.concat(dfs,ignore_index=True)
 idx=all_df.groupby(['cell_0','cell_1'])['objective'].idxmax()
 merged=all_df.loc[idx].sort_values(['cell_0','cell_1']).reset_index(drop=True)
 out,hout=MAP[a.kind]; merged.to_csv(RES/out,index=False)
 # Cumulative union history at chunk granularity.
 occupied=set(); rows=[]; evals=0
 for f,df in zip(files,dfs):
  evals+=160
  occupied.update(zip(df.cell_0.astype(int),df.cell_1.astype(int)))
  rows.append({'evaluations':evals,'num_elites':len(occupied),'coverage':len(occupied)/(a.dims*a.dims),'qd_score':float(merged[merged.apply(lambda r:(int(r.cell_0),int(r.cell_1)) in occupied,axis=1)].objective.sum())})
 pd.DataFrame(rows).to_csv(RES/hout,index=False)
 print(a.kind,'merged',len(merged),'elites from',len(dfs),'chunks')
