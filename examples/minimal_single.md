# minimal_single.json

Smallest useful run: 37 particles, 60 updates. Finishes in well under a second. Used by the smoke test.

```bash
spp-physica-a validate examples/minimal_single.json
spp-physica-a run examples/minimal_single.json --out runs/minimal
spp-physica-a query runs/minimal summary --json
```

Measured (Python 3.12.11, numpy 2.5.4, macOS arm64; model time 0.03 s, estimate 0.065 s):

| quantity | value |
|---|---|
| n | 37 |
| phi | 0.6470 |
| milling | 0.3076 |
| clustering | 2.6198 |
| rotation_r2 | 0.9349 |
| temporal_stability | 0.8381 |
| final_overlapping_pairs | 5 |

Exact values are stored in `tests/data/minimal_single_summary.json`.
