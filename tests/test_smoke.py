"""End-to-end smoke test of the documented agent workflow on the minimal example.

validate -> run -> query summary, through the real command line, checked against
values stored in tests/data/minimal_single_summary.json. Runs in a few seconds.
"""
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


def spp(*argv, cwd):
    proc = subprocess.run([sys.executable, "-m", "spp_unified", *argv, "--json"],
                          capture_output=True, text=True, cwd=cwd, timeout=120)
    return proc.returncode, json.loads(proc.stdout)


def test_minimal_example_end_to_end(tmp_path):
    config = str(ROOT / "examples" / "minimal_single.json")
    code, doc = spp("validate", config, cwd=tmp_path)
    assert code == 0 and doc["ok"] and doc["estimate"]["runtime_s"] < 5
    code, doc = spp("run", config, "--out", "runs/minimal", cwd=tmp_path)
    assert code == 0 and doc["status"] == "completed"
    code, summary = spp("query", "runs/minimal", "summary", cwd=tmp_path)
    assert code == 0
    expected = json.loads((ROOT / "tests" / "data" / "minimal_single_summary.json").read_text())["values"]
    for key, value in expected.items():
        assert summary["values"][key]["value"] == pytest.approx(value, rel=1e-6, abs=1e-9), key
    # The run touched nothing outside its directory.
    assert sorted(p.name for p in tmp_path.iterdir()) == ["runs"]
