# Numerical conventions and replay scope

## Model and sampling

The update is synchronous. Each heading combines heading persistence, normalized local alignment (including self), optional taxis, and pair force. Independent uniform angular noise on [−π, π] is multiplied by `noise`, then the particle moves at speed `v0`. `dt=1` is used in the paper. Pair-force attraction is positive along the focal-to-neighbor displacement; the finite hard-core force is −50 by default. `beta = exp(tan(pi*adhesion/2))` within the experimental bounds.

The API has nine active single-species parameters and five mixture parameters: composition, symmetric/antisymmetric alignment, and symmetric/antisymmetric radial coupling. The optional taxis direction and strength are fixed in the paper's searches. `parameters.py` is the authoritative ordering. The interaction matrices are `[[1, Js+Ja], [Js-Ja, 1]]` and `[[0, Cs+Ca], [Cs-Ca, 0]]`.

The particle count uses `max(8, round(rho * round(sqrt(3)/6 * (S/r0)^2)))`. Recording begins after the first update, at loop indices divisible by `state_every`, plus the last step. The CLI records the corresponding one-based step numbers explicitly. Descriptor extraction uses the last ten recorded states; the final interval can be shorter than the regular recording interval.

Three simultaneous overlap-projection passes follow each update. This finite iteration count does not mathematically guarantee convergence in crowded configurations. The release retains this dynamics and reports the final minimum separation and residual overlap count in CLI outputs. The two-particle projection test checks exact separation and midpoint preservation for an isolated pair; it is not a guarantee for a dense trajectory. Changing the projection count changes the model's numerical trajectory.

Periodic pair distances use the minimum image; milling uses a circular coordinate-wise center. Reflection reverses the normal heading component. Clustering's periodic ideal-gas normalization is valid for radius ≤ S/2; the paper bounds satisfy that condition. The reflecting normalization uses the first branch of the square distance CDF. Mean drift uses accumulated commanded displacement before overlap correction, consistent with the supplied simulator. It therefore measures propulsion transport and should not be interpreted as the corrected geometric center displacement.

## Descriptors and representation

The 51-component descriptor contains 12 transformed radial-distribution bins, 12 radial heading-correlation bins, three nearest-neighbor quantiles, six local/shape/polarization statistics, eight density-spectrum shells, and ten short-time/mobility features. Radial-distribution values are clipped to [0,20] and transformed as `log1p(g)/log(21)` before training. The spectrum is normalized by N²; it therefore differs by a factor N from a structure factor normalized by N. The mixture extension has 107 components.

Distance and heading-correlation features are rotationally invariant. Fourier modes on a square reciprocal lattice and periodic minimum-image geometry preserve the square's rotation symmetries; arbitrary rotations of a fixed periodic square are not an exact symmetry. Species-resolved center separation uses arithmetic species centers followed by a minimum-image difference, and can be sensitive to species crossing a seam. These are implementation conventions relevant to interpreting the learned representation.

Training statistics use only training observations, with standard deviation floored at 1e−6. New training runs use first/99th percentiles of training codes for display normalization. K-nearest-neighbor overlap counts the ten nearest *other* observations: scikit-learn already excludes self when `kneighbors` is called without an explicit query. The release corrects a redundant first-neighbor removal in this calculation.

The checkpoint and coordinate files in `results/expanded/` preserve the supplied fit. The fresh audit is in `results/independent_validation/`. In the tested environment its selected 2D single-species seed is 2, standardized test MSE is about 0.260, and trustworthiness is about 0.884, versus about 0.406 and 0.845 for PCA. These are newly measured results, not an exact replay of the manuscript's 0.250 and 0.890. Differences include the explicitly recorded simulation/split seed choices, corrected neighbor metric, and numerical training environment. The mixture audit uses the split indices saved with this release; its results also differ from the manuscript's table. Consult the JSON report rather than assuming bitwise or table-level equality.

## Search and sensitivity

Batched MAP-Elites proposes against the archive state at the start of each batch. Results are inserted in proposal order, independently of worker completion order. Batch size is part of the experiment protocol: changing it changes the search trajectory. With batch size one, the parallel and serial routines agree in the test suite. String-derived search seeds use stable byte encodings, avoiding Python's randomized string hash.

Boundary calculations use the frozen standardizers and protocol hashes in `boundary/`. Saved catalogs remain associated with the supplied implementation and data. The general-descriptor chiral ridge is ambiguous under the complete control battery. The three reciprocity-dependent demixing candidates lack a completed single-species control; only the two composition-driven mixture boundaries have the full stated classification. Figure generation preserves these distinctions.

The experiment sources include both norm-based and Monte Carlo sensitivity-floor calculations. `sloppiness_noisefloor.py` and `sloppiness_loadings.py` generate the reported noise-resolution and loading tables; `sloppiness_v4.py` supplies the additional estimator audit. Their outputs are separately named and should not be interchanged.

## Storage and environment

CSV/JSON are used for tabular outputs, NPZ for arrays, PyTorch checkpoints for learned models, and pickle for the supplied boundary caches/traces. Load the bundled checkpoints and caches only from a trusted checkout. Run the analysis from the repository, because model/data paths are relative to it. The standalone wheel supports the simulator API; boundary analyses additionally need `SPP_PROJECT_ROOT` pointing to the research checkout if imported from an installed wheel.

`docs/validation_environment.json` records the environment used for release checks. `SHA256SUMS` covers the release files. The full 256,000-call archive campaign and all long-running boundary campaigns were not rerun during packaging; their saved results are bundled, while short numerical tests and the full independent representation audit were executed.
