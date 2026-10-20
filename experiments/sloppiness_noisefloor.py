"""Assessment #4: add a seed-to-seed noise floor to the local sloppiness spectra.

For each of the six representative regimes we (i) evaluate the standardized
behavioral output vector over many replicate seeds at the UNPERTURBED point to
establish a per-output noise floor, (ii) recompute the local Jacobian with many
seeds per perturbation side and at three perturbation scales (+/-0.02, 0.04,
0.08), (iii) propagate the seed noise to a floor on the singular values of J,
and (iv) count only the singular values above that floor. This tests whether the
reported multi-decade spans and sub-leading eigenvectors are resolved or are at
the seed-noise level, and whether the span is a step-size artifact.
"""
from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
import torch
torch.set_num_threads(1)
import np_encoder_patch  # noqa: F401  (numpy forward pass; torch<->numpy2 bridge is broken)
from spp_unified.parameters import ACTIVE_SINGLE_NAMES
from run_sloppiness_expanded import load_encoder, _worker
from build_single_corpus import EXPANDED_BOUNDS
from expanded_common import run_parallel
from spp_unified.learned import normalize_latent

OUT = ROOT / "results" / "expanded"
OUTS = ["phi", "milling", "log_clustering", "nematic", "learned_0", "learned_1", "hexatic", "density_mode_peak"]
SCALES = [0.02, 0.04, 0.08]
N_FLOOR = 24
N_J = 8


def output_vector(summary, d, hx, pk, ae, lo, hi):
    z = ae.transform(np.asarray([d]))
    z01, _, _ = normalize_latent(z, lo, hi)
    return np.array([summary["phi"], summary["milling"], np.log1p(summary["clustering"]),
                     summary["nematic"], z01[0, 0], z01[0, 1], hx, pk])


def main():
    ae, lo, hi = load_encoder()
    df = pd.read_csv(OUT / "single_behavior_corpus_latent.csv")
    corpus = df.copy(); corpus["log_clustering"] = np.log1p(corpus.clustering)
    scale = corpus[OUTS].std().replace(0, 1).to_numpy()

    reps = pd.read_csv(OUT / "sloppiness_representatives.csv")
    rows = []
    for _, rr in reps.iterrows():
        u = np.array([rr[f"u_{j}"] for j in range(9)]); rows.append((rr.regime, u, rr.boundary))

    # ---- 1. baseline noise floor: many seeds at the unperturbed point ----
    floor_records = {}
    for name, u, boundary in rows:
        x = EXPANDED_BOUNDS[:, 0] + u * (EXPANDED_BOUNDS[:, 1] - EXPANDED_BOUNDS[:, 0])
        tasks = [(x, boundary, 500000 + int.from_bytes(name.encode("utf-8"), "little") % 1000 * 100 + s) for s in range(N_FLOOR)]
        res = run_parallel(_worker, tasks, max_workers=8, desc=f"floor:{name}")
        ys = np.array([output_vector(s, d, hx, pk, ae, lo, hi) for (s, d, hx, pk) in res])
        ys_std = ys / scale
        floor_records[name] = ys_std.std(axis=0)   # per-output seed std (standardized)

    # ---- 2. Jacobian at multiple scales with many seeds ----
    summary_rows = []
    eig_rows = []
    for name, u, boundary in rows:
        sig_floor_out = floor_records[name]           # std of standardized outputs
        for delta in SCALES:
            J = np.zeros((len(OUTS), 9))
            for j in range(9):
                sides = {}
                for sign in (-1, 1):
                    up = u.copy(); up[j] = np.clip(up[j] + sign * delta, 0, 1)
                    x = EXPANDED_BOUNDS[:, 0] + up * (EXPANDED_BOUNDS[:, 1] - EXPANDED_BOUNDS[:, 0])
                    tasks = [(x, boundary, 600000 + j * 1000 + (sign + 1) * 100 + s) for s in range(N_J)]
                    res = run_parallel(_worker, tasks, max_workers=8, desc=f"J:{name}:{delta}:{j}:{sign}")
                    ys = np.array([output_vector(s, d, hx, pk, ae, lo, hi) for (s, d, hx, pk) in res])
                    sides[sign] = (ys / scale).mean(axis=0)
                J[:, j] = (sides[1] - sides[-1]) / (2 * delta)
            # Noise floor on J: SE of each derivative from seed noise
            # SE(dy/du) = sqrt(2) * sigma_y / (sqrt(N_J) * 2*delta)
            Jfloor = np.sqrt(2.0) * sig_floor_out[:, None] / (np.sqrt(N_J) * 2 * delta)
            sv = np.linalg.svd(J, compute_uv=False)
            floor_sv = np.linalg.svd(Jfloor, compute_uv=False)[0]  # spectral norm of floor
            resolved = sv[sv > floor_sv]
            raw_pos = sv[sv > 1e-8]
            raw_span = float(np.log10((raw_pos.max() / raw_pos.min()) ** 2)) if len(raw_pos) > 1 else 0.0
            res_span = float(np.log10((resolved.max() / resolved.min()) ** 2)) if len(resolved) > 1 else 0.0
            summary_rows.append({
                "regime": name, "delta": delta,
                "raw_span_decades": raw_span, "resolved_span_decades": res_span,
                "n_singular_total": int(len(sv)), "n_resolved": int(len(resolved)),
                "sv_floor": float(floor_sv), "sv_max": float(sv.max()),
                "sv_min_resolved": float(resolved.min()) if len(resolved) else float("nan"),
            })
            for k, s in enumerate(sv):
                eig_rows.append({"regime": name, "delta": delta, "index": k + 1,
                                 "singular_value": float(s), "eigenvalue": float(s ** 2),
                                 "resolved": bool(s > floor_sv)})
            print(summary_rows[-1], flush=True)

    sdf = pd.DataFrame(summary_rows)
    sdf.to_csv(OUT / "sloppiness_noisefloor_summary.csv", index=False)
    pd.DataFrame(eig_rows).to_csv(OUT / "sloppiness_noisefloor_spectra.csv", index=False)
    pd.DataFrame([{"regime": k, **{f"floor_{o}": v for o, v in zip(OUTS, val)}}
                  for k, val in floor_records.items()]).to_csv(OUT / "sloppiness_noisefloor_levels.csv", index=False)

    # headline: resolved spans at delta=0.04, stability across scales
    head = {}
    for name in floor_records:
        g = sdf[sdf.regime == name]
        head[name] = {"resolved_span_0.04": float(g[g.delta == 0.04].resolved_span_decades.iloc[0]),
                      "raw_span_0.04": float(g[g.delta == 0.04].raw_span_decades.iloc[0]),
                      "resolved_span_range": [float(g.resolved_span_decades.min()), float(g.resolved_span_decades.max())],
                      "n_resolved_0.04": int(g[g.delta == 0.04].n_resolved.iloc[0])}
    with open(OUT / "sloppiness_noisefloor_headline.json", "w") as f:
        json.dump(head, f, indent=2)
    print(json.dumps(head, indent=2))


if __name__ == "__main__":
    main()
