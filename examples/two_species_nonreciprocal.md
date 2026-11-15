# two_species_nonreciprocal.json and two_species_reciprocal.json

Non-default model (`kind: two`). The non-reciprocal run has antisymmetric alignment coupling
J_a = 1.3; the reciprocal file is its matched control (J_a = C_a = 0, everything else equal,
same seed). Compare them:

```bash
spp-physica-a run examples/two_species_nonreciprocal.json --out runs/nonrec
spp-physica-a run examples/two_species_reciprocal.json --out runs/rec
spp-physica-a compare runs/rec runs/nonrec --json
```

Measured (69 particles, 440 updates, seed 43; model time 0.24 s each, estimate 0.26 s):

| quantity | reciprocal | non-reciprocal |
|---|---|---|
| phi | 0.1171 | 0.1412 |
| milling | 0.1023 | 0.1036 |
| clustering | 3.7526 | 3.0931 |
| segregation | 0.0056 | -0.0076 |
| omega (rad per time unit) | -0.0076 | -0.2065 |
| rotation_r2 | 0.0674 | 0.8303 |
| chirality | 0.0012 | 0.1173 |
| final_overlapping_pairs | 41 | 28 |

Single seeds only; the paper's chirality claims rest on paired multi-seed statistics
(`experiments/chirality_seed_stats.py`).
