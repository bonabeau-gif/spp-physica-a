# typical_single.json

The Vicsek-like configuration of the paper's noise sweep (`experiments/run_all.py::run_noise_sweep`,
stage `core`: 180 updates, sampling from loop index 90), at noise 0.2 and simulation seed 1000.
It reproduces the `noise=0.2, seed=0` row of `results/single_noise_sweep.csv`
(phi 0.9239810172434239 there, 0.9239810172434241 here: equal to 15 significant digits).

```bash
spp-physica-a validate examples/typical_single.json   # estimate 0.12 s
spp-physica-a run examples/typical_single.json --out runs/typical
spp-physica-a query runs/typical stats phi --json
```

Measured (model time 0.08 s):

| quantity | value |
|---|---|
| n | 69 |
| phi | 0.9240 |
| nematic | 0.7208 |
| milling | 0.0460 |
| clustering | 1.1748 |
| drift | 0.9178 |
| final_overlapping_pairs | 0 |
