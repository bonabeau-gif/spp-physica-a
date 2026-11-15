# AGENTS.md

## What this is

A simulator of self-propelled particles in a 2D square box: particles align with neighbors,
repel and attract through a piecewise radial force, and feel angular noise. It produces flocks,
mills, clusters, and disordered states, measured by order parameters (polar alignment `phi`,
milling, clustering, nematic order, rotation). A two-species variant tests reciprocal versus
non-reciprocal cross-species couplings. The paper is Wen and Bonabeau (2026), see `README.md`.

## Install

Python 3.10+ (3.12 tested). From a fresh clone:

```bash
git clone https://github.com/bonabeau-gif/spp-physica-a.git
cd spp-physica-a
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
python -m pip install -e ".[dev]"                        # simulator, CLI, tests
spp-physica-a --version                                  # prints: spp-physica-a 1.0.0
```

`python -m spp_unified` is equivalent to `spp-physica-a`. The paper's learned-representation and
quality-diversity analyses need `.[analysis,dev]` (adds torch, scikit-learn, ribs, umap); the CLI does not.

## The interface

Start with `spp-physica-a describe`. Every subcommand takes `--json` and `--help`.

| Command | What it does |
|---|---|
| `describe [--model M] [--param NAME] [--outputs] [--schema]` | Models, parameters (type, default, unit, range, paper-explored range), constraints, outputs |
| `template [--model single\|two]` | Complete commented config with defaults (`--json`: plain JSON) |
| `validate <config\|->` | All problems at once, plus runtime / memory / output-size estimate. Runs nothing |
| `run <config> --out <dir> [--force]` | One run into a self-describing run directory |
| `sweep <config> --out <dir>` | Cartesian product of the config's `sweep` block, one run dir per point, plus `index.json` |
| `query <run> summary\|outputs\|series NAME\|stats NAME\|at STEP\|final\|diagnostics\|descriptor` | Small JSON answers from a finished run (never reruns) |
| `compare <run> <run> ...` | Parameters that differ and the difference in each summary value |
| `simulate --config F --output D` | Legacy command (writes only `run.json` + `trajectory.npz`); kept unchanged |

Exit codes: `0` success, `1` runtime failure, `2` invalid usage or configuration.
With `--json`, stdout is exactly one JSON document; errors are
`{"ok": false, "errors": [{"field", "value", "reason", "hint"}]}`. Logs go to stderr.

## Worked example

```bash
spp-physica-a template --model single > my.json         # edit values; keep "seed"
spp-physica-a validate examples/typical_single.json --json
spp-physica-a run examples/typical_single.json --out runs/typical --json
spp-physica-a query runs/typical summary --json          # phi = 0.9240, n = 69
spp-physica-a query runs/typical series phi --points 20 --json
```

How does polar order respond to noise? Run the shipped sweep (24 runs, about 3 s):

```bash
spp-physica-a sweep examples/sweep_noise.json --out runs/noise --json
spp-physica-a query runs/noise summary --json            # one row per (noise, seed)
```

In that configuration (see `examples/typical_single.json`), mean `phi` over 4 seeds falls from
1.00 (noise 0) to 0.92 (0.2), 0.70 (0.4), 0.33 (0.6), 0.12 (0.8). There is no grouped-statistics
query yet: average per-point values yourself, and parse JSON, not text.
Other examples, each with a sidecar `.md` of commands and measured values:
`examples/minimal_single.json` (smoke test), `examples/two_species_nonreciprocal.json` and its
matched control `examples/two_species_reciprocal.json`.

## Configuration format

One JSON file (full-line `//` comments allowed). Same format as `configs/*.json`:

```json
{"schema": 1, "kind": "two",
 "parameters": {"rho": 0.15, "noise": 0.05},
 "mixture": {"alignment_antisymmetric": 1.3},
 "simulation": {"steps": 440, "sample_start": 220, "seed": 43},
 "state_every": 10}
```

Omitted keys take registry defaults. `docs/config.schema.json` is the JSON Schema. A `sweep`
block maps dotted names to value lists: `"sweep": {"parameters.noise": [0.2, 0.4], "simulation.seed": [1, 2]}`;
swept values override the base values (which may be omitted). `query <sweep_dir> summary` lists
phi, milling, clustering and nematic per point; use `query <sweep_dir>/point_NNNN ...` for the rest.
No paths, environment variables, or working-directory files influence a run.

## Rules for using it well

- Run `validate` before `run`, and read its `estimate.runtime_s` before large runs or sweeps.
  Cost grows as `steps * n^2`; `n` is set by `rho`, `S` and `r0` (see `describe --param parameters.rho`).
- Always set `simulation.seed`. If omitted, `run` picks one and records it in `config.resolved.json`.
- Use `query`, not the raw `outputs/` files. `summary` values are averages over the sampling
  window `sample_start..steps`, not final values; `stats NAME` gives final and max.
- For statements about behavior, run several seeds (a `sweep` over `simulation.seed`) and compare
  means. Single runs in the disordered regime are chaotic.
- Use a new `--out` directory for each run; `run` refuses a non-empty one unless `--force`.
- Units: lengths in simulation units (box side `S`, radius `r0`), time in updates (`dt=1`),
  `noise` as a fraction of pi, radii as multiples of `r0`. `rho` is a packing fraction, not a count.

## Run directory

`config.resolved.json` (every parameter, reproduces the run), `manifest.json` (status, command,
times, git commit, versions, seed, error), `summary.json` (values with units), `schema.json`
(columns and units of `outputs/`), `log.txt`, and `outputs/` with `timeseries.csv`,
`trajectory.npz`, and the legacy `run.json`. Same config and code give byte-identical
`outputs/` and `summary.json`.

## Verify the installation

```bash
python -m pytest -q tests/test_smoke.py        # validate -> run -> query on the minimal example, ~0.5 s
python -m pytest -q                            # full suite, ~5-15 s (torch/sklearn/ribs tests skip without them)
```

## Known limits

From `docs/NUMERICS.md` and `README.md`:
- Three overlap-projection passes per update do not guarantee zero overlaps in crowded states;
  check `query <run> diagnostics` (`final_overlapping_pairs`).
- Clustering's normalization assumes `r1 <= S/2`; `validate` warns otherwise. Values outside the
  paper's explored ranges (`describe` shows them) are valid but were not studied.
- `drift` uses commanded displacement before overlap correction: propulsion transport, not the
  corrected geometric center displacement.
- Descriptor rotation invariance is exact only for the square's symmetries in periodic boxes.
- The fresh independent audit in `results/independent_validation/` is not an exact replay of
  the manuscript's tables. The general-descriptor chiral ridge is ambiguous under the full controls.
- `adhesion` beyond +/-0.95 is clipped. There are no time-dependent interventions.
- The CLI covers single runs and sweeps. Searches, learned representations, sensitivity and
  boundary campaigns are the experiment drivers in `docs/REPRODUCING.md`.

## For contributors

- Tests: `python -m pytest -q`. Release integrity: `python scripts/check_release.py`
  (after changing tracked files, regenerate checksums with `--write-manifest`).
- Every configurable quantity is registered in `spp_unified/registry.py`. Add new parameters
  there first. `describe`, `template`, `validate`, and the schema are generated from it; then run
  `spp-physica-a describe --schema > docs/config.schema.json`.
  `tests/test_agent_layer.py` checks the registry against the dataclasses in `parameters.py`.
- The CLI layer (`cli.py`, `configuration.py`, `rundir.py`, `queries.py`) only parses, validates,
  calls the library, and writes files. Model logic stays in `model.py`.
