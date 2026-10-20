from __future__ import annotations
import argparse, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
STAGES=['core','single_mill','single_clust','single_post','single_finite','single_sensitivity','multi_non','multi_rec','multi_post','summary']
if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--quick',action='store_true'); a=ap.parse_args()
    for stage in STAGES:
        cmd=[sys.executable,str(ROOT/'experiments'/'stage_runner.py'),stage]
        if a.quick: cmd.append('--quick')
        print('RUN',stage,flush=True)
        subprocess.run(cmd,cwd=ROOT,check=True)
