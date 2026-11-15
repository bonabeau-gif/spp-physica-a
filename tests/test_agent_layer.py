"""Tests for the agent-facing layer: registry, CLI, configuration, run directories, queries."""
from dataclasses import fields
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from spp_unified import registry as R
from spp_unified import configuration as C
from spp_unified.cli import main
from spp_unified.parameters import (SingleParams, TwoSpeciesParams, SimulationConfig,
                                    ACTIVE_SINGLE_NAMES, EXPERIMENT_BOUNDS,
                                    TWO_SPECIES_EXTRA_NAMES, TWO_SPECIES_EXTRA_BOUNDS)

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
DATA = Path(__file__).resolve().parent / "data"


def cli(capsys, *argv):
    code = main(list(argv))
    out = capsys.readouterr()
    return code, out.out, out.err


def cli_json(capsys, *argv):
    code, out, err = cli(capsys, *argv, "--json")
    return code, json.loads(out), err


def tiny(seed=3, **sim):
    return {"kind": "single", "parameters": {"rho": 0.06},
            "simulation": {"steps": 24, "sample_start": 10, "seed": seed, **sim}, "state_every": 4}


def write(tmp_path, cfg, name="cfg.json"):
    path = tmp_path / name
    path.write_text(json.dumps(cfg))
    return str(path)


# --- registry -------------------------------------------------------------
def test_registry_defaults_match_dataclasses():
    idx = R.parameter_index()
    for group, cls in (("parameters", SingleParams), ("simulation", SimulationConfig)):
        for f in fields(cls):
            entry = idx[f"{group}.{f.name}"]
            if f.name != "seed":
                assert entry["default"] == f.default, f.name
    for f in fields(TwoSpeciesParams):
        if f.name != "base":
            assert idx[f"mixture.{f.name}"]["default"] == f.default
    registered = {n for n in idx if "." in n}
    expected = ({f"parameters.{f.name}" for f in fields(SingleParams)}
                | {f"simulation.{f.name}" for f in fields(SimulationConfig)}
                | {f"mixture.{f.name}" for f in fields(TwoSpeciesParams) if f.name != "base"})
    assert registered == expected


def test_explored_ranges_match_experiment_bounds():
    idx = R.parameter_index()
    for name, (lo, hi) in zip(ACTIVE_SINGLE_NAMES, EXPERIMENT_BOUNDS):
        assert idx[f"parameters.{name}"]["explored_range"] == [lo, hi]
    for name, (lo, hi) in zip(TWO_SPECIES_EXTRA_NAMES, TWO_SPECIES_EXTRA_BOUNDS):
        assert idx[f"mixture.{name}"]["explored_range"] == [lo, hi]


def test_describe_json_covers_registry(capsys):
    code, doc, _ = cli_json(capsys, "describe")
    assert code == 0
    assert {p["name"] for p in doc["parameters"]} == {p["name"] for p in R.PARAMETERS}
    for p in doc["parameters"]:
        for key in ("type", "default", "unit", "description", "applies_to", "constraints"):
            assert key in p, (p["name"], key)
    assert set(doc["models"]) == {"single", "two"}
    assert {o["name"] for o in doc["outputs"]["summary"]} >= {"phi", "milling", "clustering"}
    assert doc["interventions"] == []
    code, one, _ = cli_json(capsys, "describe", "--param", "parameters.noise")
    assert one["range"] == {"min": 0.0, "max": 1.0}


def test_committed_schema_is_current():
    committed = json.loads((ROOT / "docs" / "config.schema.json").read_text())
    assert committed == json.loads(json.dumps(C.json_schema()))


# --- templates and validation ---------------------------------------------
@pytest.mark.parametrize("model", ["single", "two"])
def test_template_passes_validate(capsys, tmp_path, model):
    code, text, _ = cli(capsys, "template", "--model", model)
    assert code == 0 and "//" in text
    path = tmp_path / "t.json"
    path.write_text(text)
    code, doc, _ = cli_json(capsys, "validate", str(path))
    assert code == 0 and doc["ok"] and doc["kind"] == model
    code, plain, _ = cli_json(capsys, "template", "--model", model)
    assert C.validate(plain) == ([], [])


def test_invalid_config_reports_every_problem(capsys, tmp_path):
    bad = {"kind": "single", "parameters": {"noise": 1.5, "rhoo": 0.1},
           "simulation": {"steps": 50, "sample_start": 80, "seed": 1}}
    code, doc, _ = cli_json(capsys, "validate", write(tmp_path, bad))
    assert code == 2 and doc["ok"] is False
    by_field = {e["field"]: e for e in doc["errors"]}
    assert by_field["parameters.noise"]["reason"] == "out of range"
    assert "rho" in by_field["parameters.rhoo"]["hint"]
    assert "sample_start < simulation.steps" in by_field["simulation.sample_start"]["reason"]
    for e in doc["errors"]:
        assert set(e) == {"field", "value", "reason", "hint"}


def test_validation_type_model_and_placement_errors(tmp_path):
    errs, _ = C.validate({"kind": "single", "mixture": {"fraction_B": 0.2},
                          "simulation": {"steps": 10.5, "boundary": "wall"}})
    fields_ = {e["field"] for e in errs}
    assert {"mixture", "simulation.steps", "simulation.boundary"} <= fields_
    errs, _ = C.validate({"parameters": {"rho": 5.0}})
    assert any("particle_placement" in e["reason"] for e in errs)
    with pytest.raises(ValueError):          # the library agrees placement fails
        from spp_unified import simulate_single
        simulate_single(SingleParams(rho=5.0), SimulationConfig(steps=2, sample_start=0))


def test_run_rejects_bad_config_with_json_error(capsys, tmp_path):
    code, doc, _ = cli_json(capsys, "run", write(tmp_path, {"kind": "three"}), "--out", str(tmp_path / "r"))
    assert code == 2 and doc["errors"][0]["field"] == "kind"
    assert not (tmp_path / "r").exists()
    code, doc, _ = cli_json(capsys, "run", "--out")
    assert code == 2 and doc["ok"] is False


# --- runs -----------------------------------------------------------------
def test_run_directory_contents_and_never_overwrite(capsys, tmp_path):
    cfg = write(tmp_path, tiny())
    out = tmp_path / "run"
    code, doc, _ = cli_json(capsys, "run", cfg, "--out", str(out))
    assert code == 0 and doc["status"] == "completed"
    for name in ("config.resolved.json", "manifest.json", "summary.json", "schema.json", "log.txt",
                 "outputs/run.json", "outputs/trajectory.npz", "outputs/timeseries.csv"):
        assert (out / name).exists(), name
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["status"] == "completed" and manifest["seed"] == 3
    resolved = json.loads((out / "config.resolved.json").read_text())
    assert set(resolved["simulation"]) == {f.name for f in fields(SimulationConfig)}
    code, doc, _ = cli_json(capsys, "run", cfg, "--out", str(out))
    assert code == 2 and "not empty" in doc["errors"][0]["reason"]
    code, _, _ = cli_json(capsys, "run", cfg, "--out", str(out), "--force")
    assert code == 0


def test_same_seed_gives_identical_outputs(capsys, tmp_path):
    cfg = write(tmp_path, tiny(seed=11))
    for name in ("a", "b"):
        assert cli_json(capsys, "run", cfg, "--out", str(tmp_path / name))[0] == 0
    for rel in ("summary.json", "config.resolved.json", "outputs/run.json",
                "outputs/trajectory.npz", "outputs/timeseries.csv"):
        assert (tmp_path / "a" / rel).read_bytes() == (tmp_path / "b" / rel).read_bytes(), rel


def test_resolved_config_reproduces_run(capsys, tmp_path):
    cfg = write(tmp_path, tiny(seed=5))
    cli_json(capsys, "run", cfg, "--out", str(tmp_path / "a"))
    code, _, _ = cli_json(capsys, "run", str(tmp_path / "a" / "config.resolved.json"), "--out", str(tmp_path / "b"))
    assert code == 0
    assert (tmp_path / "a/summary.json").read_bytes() == (tmp_path / "b/summary.json").read_bytes()
    assert (tmp_path / "a/outputs/trajectory.npz").read_bytes() == (tmp_path / "b/outputs/trajectory.npz").read_bytes()


def test_missing_seed_is_chosen_and_recorded(capsys, tmp_path):
    cfg = tiny()
    del cfg["simulation"]["seed"]
    code, doc, err = cli_json(capsys, "run", write(tmp_path, cfg), "--out", str(tmp_path / "r"))
    assert code == 0 and "chose seed" in err
    resolved = json.loads((tmp_path / "r" / "config.resolved.json").read_text())
    assert resolved["simulation"]["seed"] == doc["seed"]
    assert json.loads((tmp_path / "r" / "manifest.json").read_text())["seed_chosen_by_run"] is True


def test_run_matches_legacy_simulate(capsys, tmp_path):
    demo = str(ROOT / "configs" / "single_demo.json")
    assert cli(capsys, "simulate", "--config", demo, "--output", str(tmp_path / "legacy"))[0] == 0
    assert cli_json(capsys, "run", demo, "--out", str(tmp_path / "new"))[0] == 0
    for rel in ("run.json", "trajectory.npz"):
        assert (tmp_path / "legacy" / rel).read_bytes() == (tmp_path / "new" / "outputs" / rel).read_bytes()


def test_sweep_writes_index(capsys, tmp_path):
    cfg = tiny()
    cfg["sweep"] = {"parameters.noise": [0.1, 0.5], "simulation.seed": [1]}
    code, doc, _ = cli_json(capsys, "sweep", write(tmp_path, cfg), "--out", str(tmp_path / "s"))
    assert code == 0 and doc["completed"] == 2
    index = json.loads((tmp_path / "s" / "index.json").read_text())
    assert [r["parameters"]["parameters.noise"] for r in index["runs"]] == [0.1, 0.5]
    code, table, _ = cli_json(capsys, "query", str(tmp_path / "s"), "summary")
    assert len(table["rows"]) == 2
    bad = tiny()
    bad["sweep"] = {"parameters.nosie": [0.1]}
    code, doc, _ = cli_json(capsys, "validate", write(tmp_path, bad, "bad.json"))
    assert code == 2 and "noise" in doc["errors"][0]["hint"]


# --- queries --------------------------------------------------------------
def test_queries_are_compact_and_consistent(capsys, tmp_path):
    out = tmp_path / "run"
    cli_json(capsys, "run", str(EXAMPLES / "two_species_nonreciprocal.json"), "--out", str(out))
    answers = {}
    for q in (["summary"], ["outputs"], ["series", "phi"], ["series", "milling", "--points", "7"],
              ["stats", "phi"], ["at", "300"], ["at", "300", "--time"], ["final"],
              ["diagnostics"], ["descriptor"]):
        code, out_text, _ = cli(capsys, "query", str(out), *q, "--json")
        assert code == 0, q
        assert len(out_text.encode()) < 20_000, q
        answers[" ".join(q)] = json.loads(out_text)
    with np.load(out / "outputs" / "trajectory.npz") as data:
        phi = data["timeseries_phi"]
        assert answers["summary"]["values"]["phi"]["value"] == pytest.approx(float(np.mean(phi)), abs=1e-12)
        assert answers["stats phi"]["max"] == pytest.approx(float(phi.max()))
        assert answers["stats phi"]["final"] == pytest.approx(float(phi[-1]))
    assert answers["series milling --points 7"]["returned"] == 7
    assert "decimation" in answers["series milling --points 7"]["downsampling"]
    assert answers["descriptor"]["length"] == 107
    code, doc, _ = cli_json(capsys, "query", str(out), "series", "phii")
    assert code == 2 and "phi" in doc["errors"][0]["hint"]


def test_compare_reports_parameter_and_summary_differences(capsys, tmp_path):
    a, b = tiny(seed=2), tiny(seed=2)
    b["parameters"]["noise"] = 0.6
    cli_json(capsys, "run", write(tmp_path, a, "a.json"), "--out", str(tmp_path / "a"))
    cli_json(capsys, "run", write(tmp_path, b, "b.json"), "--out", str(tmp_path / "b"))
    code, doc, _ = cli_json(capsys, "compare", str(tmp_path / "a"), str(tmp_path / "b"))
    assert code == 0
    assert [d["name"] for d in doc["parameters_differing"]] == ["parameters.noise"]
    phi = next(m for m in doc["summary"] if m["name"] == "phi")
    assert phi["difference_vs_first"][0] == pytest.approx(phi["values"][1] - phi["values"][0])


# --- legacy baselines -----------------------------------------------------
@pytest.mark.parametrize("config,key", [("single_demo.json", "single_demo"),
                                        ("two_species_demo.json", "two_species_demo")])
def test_legacy_simulate_reproduces_baseline(capsys, tmp_path, config, key):
    baseline = json.loads((DATA / "baseline_demo_summaries.json").read_text())
    code, out, _ = cli(capsys, "simulate", "--config", str(ROOT / "configs" / config),
                       "--output", str(tmp_path / "o"))
    assert code == 0
    summary = json.loads(out)
    for k, v in baseline[key].items():
        assert summary[k] == pytest.approx(v, rel=1e-6, abs=1e-9), k


def test_json_channel_clean_for_every_subcommand(tmp_path):
    """Run through a real subprocess: stdout must be exactly one JSON document."""
    run_dir = tmp_path / "r"
    cfg = write(tmp_path, tiny())
    commands = [["describe"], ["describe", "--model", "two"], ["template"], ["validate", cfg],
                ["run", cfg, "--out", str(run_dir)], ["query", str(run_dir), "summary"],
                ["compare", str(run_dir), str(run_dir)], ["validate", str(tmp_path / "missing.json")],
                ["query", str(tmp_path), "final"], ["bogus"]]
    for argv in commands:
        proc = subprocess.run([sys.executable, "-m", "spp_unified", *argv, "--json"],
                              capture_output=True, text=True, cwd=tmp_path)
        json.loads(proc.stdout)
        assert proc.returncode in (0, 2), (argv, proc.stderr)
