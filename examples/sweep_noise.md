# sweep_noise.json

Noise sweep of the Vicsek-like configuration (same base as `typical_single.json`):
6 noise values x 4 seeds = 24 runs, about 3 s in total.

```bash
spp-physica-a validate examples/sweep_noise.json      # estimate 2.9 s for 24 points
spp-physica-a sweep examples/sweep_noise.json --out runs/noise_sweep
spp-physica-a query runs/noise_sweep summary --json  # one row per point
```

Measured mean polar alignment phi over seeds 1000-1003:

| noise | 0.0 | 0.2 | 0.4 | 0.6 | 0.8 | 1.0 |
|---|---|---|---|---|---|---|
| phi | 1.0000 | 0.9225 | 0.7002 | 0.3276 | 0.1229 | 0.1070 |

Per seed, these match the corresponding rows of `results/single_noise_sweep.csv`
(simulation seed 1000+k is `seed` k there) to within 2e-14 for noise 0.0-0.6. At noise 0.8 and
1.0 (disordered regime) individual values differ by up to 0.013: there the dynamics amplify
last-digit floating-point differences (the saved file was produced in another numerical
environment), so only the statistics agree (seed-0..3 means 0.1229 vs 0.1263 at 0.8).
