from __future__ import annotations
import argparse,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--quick',action='store_true');a=ap.parse_args()
 nrun=2 if a.quick else 4; dims=10 if a.quick else 12
 for kind in ['single_mill','single_clust','multi_non','multi_rec']:
  for r in range(nrun):
   cmd=[sys.executable,str(ROOT/'experiments'/'map_chunk.py'),kind,str(r),'--evaluations','160']
   if a.quick:cmd.append('--quick')
   subprocess.run(cmd,cwd=ROOT,check=True)
  subprocess.run([sys.executable,str(ROOT/'experiments'/'merge_chunks.py'),kind,'--dims',str(dims)],cwd=ROOT,check=True)
