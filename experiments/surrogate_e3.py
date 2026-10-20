"""E3: coordinate-noise-injection at 1000 calls (vs the original 30). Tests whether the
falsification is budget-dependent: does true coverage fall with eps once evaluations stop
binding? Tracks TRUE (exact coords) vs PERCEIVED (noised coords) coverage separately."""
from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import sys, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments"))
import numpy as np, pandas as pd, torch
torch.set_num_threads(1)
import np_encoder_patch  # noqa
from run_learned_surrogate import load_encoder, add_initial_columns
from surrogate_revision import stratified_initial, run_direct_noisy

OUT = ROOT / "results" / "expanded" / "surrogate_exp"
OUT.mkdir(parents=True, exist_ok=True)
CALLS = 1000; EPS = [0.0, 0.15, 0.30]; REPS = 3


def main():
    ae, lo, hi = load_encoder()
    corpus = add_initial_columns(pd.read_csv(ROOT / "results/expanded/single_behavior_corpus_latent.csv"))
    corpus = corpus[corpus.boundary == "periodic"].reset_index(drop=True)
    rows = []
    for eps in EPS:
        for rep in range(REPS):
            f = OUT / f"e3_eps{eps}_rep{rep}.json"
            if f.exists():
                rows.append(json.load(open(f))); print("skip", f.name); continue
            initial = stratified_initial(corpus, rep)
            r = run_direct_noisy(rep, initial, ae, lo, hi, eps, calls=CALLS)
            json.dump(r, open(f, "w"))
            rows.append(r)
            print(f"eps={eps} rep{rep}: true={r['true_coverage']:.3f} perceived={r['perceived_coverage']:.3f}", flush=True)
    df = pd.DataFrame(rows)
    agg = df.groupby("eps").agg(true_mean=("true_coverage", "mean"), true_sd=("true_coverage", "std"),
                                perc_mean=("perceived_coverage", "mean"), perc_sd=("perceived_coverage", "std")).reset_index()
    base = float(agg[agg.eps == 0.0].true_mean.iloc[0])
    out = {"calls": CALLS, "by_eps": agg.round(4).to_dict("records"),
           "true_cov_eps0": base, "true_cov_eps0.30": float(agg[np.isclose(agg.eps, 0.30)].true_mean.iloc[0]),
           "relative_true_drop_at_0.30": float(1 - agg[np.isclose(agg.eps, 0.30)].true_mean.iloc[0] / base),
           "perceived_rises": bool(agg[np.isclose(agg.eps, 0.30)].perc_mean.iloc[0] > base)}
    json.dump(out, open(OUT / "E3_noise1000.json", "w"), indent=2)
    print("\nE3 @1000:", json.dumps(out, indent=2, default=float), flush=True)


if __name__ == "__main__":
    main()
