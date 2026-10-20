# Reproducing the paper

Run commands from the repository root after installing `.[analysis,dev]`. Every driver is a Python program; long-running drivers should run in their own processes. `SPP_MAX_WORKERS=4` caps pools created by `expanded_common.run_parallel`; `parallel_archives.py` has its own `--workers` option. Boundary workers also use `SPP_MAX_WORKERS` (see `boundary/adapters.py`).

## Saved results and fresh calculations

`results/highbudget/` contains the four 64,000-evaluation archives. `results/expanded/` contains the 360-rule single-species and 126-rule mixture corpora, descriptors, trained representations, local-response results, and champion validations. `results/boundary/` contains ridge, patch, estimator, and control results. Intermediate small-budget archives in the top-level `results/` are inputs to representative selection and the mixture corpus; they are distinct from the high-budget archives.

`results/independent_validation/` contains a fresh, executed audit with all 72 test rules and simulation seeds, 78/22 development split indices, mixture 60/20/20 split indices, four-seed selection tables, selected checkpoints, and the measured report. Model selection uses validation data only. These fresh measurements are not substituted for manuscript table entries. See `NUMERICS.md` for changes that affect numerical replay.

`figures/manuscript/` preserves the exact embedded manuscript images. `docs/figure_manifest.json` records their captions, hashes, and matching figure assets. The figure-generation command writes newly plotted numbered panels into `figures/reproduced/`; panels based on the fresh independent audit show its measured values.

## Experiment map

| Manuscript analysis | Commands, in execution order | Principal outputs |
|---|---|---|
| Section 2 dynamics and S1 validation | `python experiments/stage_runner.py core` | `results/single_noise_sweep.csv`, DE histories, force-law and noise figures |
| Section 3 classical archives | `python experiments/parallel_archives.py --workers 4` | `results/highbudget/*` |
| S2 CMA-MAE baseline | `python experiments/cmame_baseline.py` | `results/highbudget/cmame_*_history.csv` |
| Small-budget QD comparison | `python experiments/run_chunked_pipeline.py`, then `python experiments/stage_runner.py single_post` | Top-level archive and representative tables |
| Classical finite-size and one-at-a-time checks | `python experiments/stage_runner.py single_finite`, `python experiments/stage_runner.py single_sensitivity` | `single_finite_size.csv`, `single_oat_*` |
| Section 4 density corpus | `python experiments/build_single_corpus.py` | `results/expanded/single_behavior_corpus.csv`, `single_behavior_descriptors.npz` |
| Section 5 learned single-species representation | `python experiments/fit_single_learned.py`, `python experiments/latent_stability.py` | Single model, coordinates, clusters, and selection diagnostics |
| Sections 5/S4/S7/S11 independent audit | `python experiments/independent_validation.py --workers 4` | `outputs/independent_validation/` |
| Section 5 classical-map comparison | `python experiments/learned_vs_classical.py`, `python experiments/classical_nonlinear_map.py` | Within-corpus comparison and selected rule pairs; independent scores are in the audit above |
| Section 6 response spectra | `python experiments/run_sloppiness_expanded.py`, `python experiments/postprocess_sloppiness.py` | Raw local perturbations, response derivatives, eigenvalues, elite concentration |
| Section 6/S5 noise resolution | `python experiments/sloppiness_noisefloor.py`, `python experiments/sloppiness_loadings.py`, `python experiments/sloppiness_v4.py` | Multiscale noise floors, bootstrap loadings, Monte Carlo threshold audit |
| Section 7 reciprocal controls | `python experiments/stage_runner.py multi_post`, `python experiments/validate_chirality_horizon.py`, `python experiments/chirality_seed_stats.py` | Mixture validation, horizon and paired seed tables |
| Section 7.3 antisymmetric coupling plane | `python experiments/nonreciprocity_experiments.py` | `results/expanded/nonreciprocity_*` |
| S6 stationary rotation and size | `python experiments/chirality_decay.py`, `python experiments/chirality_ndep_v2.py` | Windowed rotation and particle-count scans |
| S7 species-aware representation | `python experiments/build_multi_learned.py` | 107-component descriptor, mixture model and clusters |
| S8 directed objectives and animations | `python experiments/discover_and_animate.py`, `python experiments/champion_per_seed.py`, `python experiments/champion_bounds.py` | Eleven champions, per-seed validation, and GIFs |
| S8 periodic-boundary controls | `python experiments/champion_per_seed_periodic.py` | Periodic champion validations |
| S8 champion snapshots | `python experiments/make_champion_montage.py` | Champion configuration montages |
| S9 noise transition and finite windows | `python experiments/finite_size_v2.py`, `python experiments/finite_size_v4.py` | Finite-size scans, 880-step checks, autocorrelation times |
| S9 additional density and adhesion cuts | `python experiments/phase_transition_discovery.py` | `results/expanded/phase_transition_*` |
| Section 8/S10 boundary nomination and continuation | `python experiments/bd_nominate.py`, `python experiments/bd_campaign.py`, `python experiments/bd_campaign2.py`, `python experiments/bd_campaign3.py` | Candidates, traces, catalogs, and cached evaluations |
| S10 patch geometry and estimator offsets | `python experiments/e1_patch.py`, `python experiments/e2e6_estimators.py` | `E1_patch.json`, `E2E6_estimators.json` |
| S10 mixture descriptor, controls, operating characteristics | `python experiments/e3_mixture.py`, `python experiments/e4_controls.py`, `python experiments/e5_opchar.py` | Mixture candidates, reciprocal/single-species/swap controls, false-positive results |
| All numbered figures | `python scripts/reproduce_figures.py` | `figures/reproduced/` and generation report |

`run_pipeline.py` and `run_all.py` provide small-budget integrated runs. They do not replace the 64,000-evaluation production archive driver. The stage named `core` runs DE with 16 particles in the optimizer population and eight post-initialization generations, giving 144 simulator calls per objective. MAP-Elites uses a 12 × 12 archive. The supplied high-budget endpoints have 140/144 alignment–milling and 144/144 alignment–clustering cells.

## Independent representation audit

For the versions used to generate the bundled audit, install `requirements-audit.txt` in a separate Python 3.12 virtual environment. The general installation was also tested with the newer versions recorded in `validation_environment.json`.

```bash
python experiments/independent_validation.py --workers 4 --output outputs/audit
# Reuse exactly those simulated descriptors when rerunning training:
python experiments/independent_validation.py --reuse-corpus --output outputs/audit
```

The test design uses Sobol seed 20260825 and simulation seeds 202608250 through 202608321. The three density strata each contain 24 rules; the boundary allocation follows the corpus generator's deterministic modulo-five rule. The test corpus has 58 periodic and 14 reflecting rules. Single-species model selection uses split seed 2026, four initialization seeds 0–3, and bottlenecks 2, 3, 4, 6. The separate mixture audit uses split seeds 20260825 and 20260826, with 75 training, 25 validation, and 26 test rules. The same split is used at both mixture latent dimensions.

The single-species audit also fits PCA only on the training partition, compares the classical eight-vector, and fits four fixed predictor families to the development corpus. It saves selection records before reporting test scores. The report includes the difference between newly fitted and archived coordinates so platform-dependent training differences are visible.

## Release checks

```bash
python -m pytest -q
python scripts/check_release.py
python -m build
```

Full search campaigns are computational experiments, not part of the fast test suite. The supplied archive trajectories and statistical tables are checked for integrity, while the tests exercise limiting cases, deterministic replay, neighbor geometry, descriptors, optimizer accounting, serialization, and short training.
