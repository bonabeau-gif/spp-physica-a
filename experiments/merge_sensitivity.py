from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
files=sorted((ROOT/'results'/'sensitivity_chunks').glob('*.csv'))
df=pd.concat([pd.read_csv(f) for f in files],ignore_index=True)
df.to_csv(ROOT/'results'/'single_oat_sensitivity.csv',index=False)
rows=[]
for state,metric in [('aligned','phi'),('milling','milling')]:
 for p,g in df[df.state==state].groupby('parameter'):
  rows.append({'state':state,'parameter':p,'range':float(g[metric].max()-g[metric].min())})
imp=pd.DataFrame(rows);imp.to_csv(ROOT/'results'/'single_oat_importance.csv',index=False)
piv=imp.pivot(index='parameter',columns='state',values='range').fillna(0)
fig,ax=plt.subplots(figsize=(6.2,4.4));im=ax.imshow(piv.to_numpy(),aspect='auto')
ax.set_xticks(range(len(piv.columns)),piv.columns);ax.set_yticks(range(len(piv.index)),piv.index)
fig.colorbar(im,ax=ax,label='range of order parameter');fig.tight_layout();fig.savefig(ROOT/'figures'/'fig_sensitivity.png',dpi=240);plt.close(fig)
print('merged',len(df))
