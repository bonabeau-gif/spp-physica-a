"""Check package data, compilation, portable paths, and release checksums."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
EXCLUDED = {".git", ".venv", "__pycache__", ".pytest_cache", ".cache",
            "build", "dist", "outputs"}


def release_files():
    for path in sorted(ROOT.rglob("*")):
        parts = path.relative_to(ROOT).parts
        if any(part in EXCLUDED or part.endswith(".egg-info") for part in parts):
            continue
        if path.is_file() and path.name not in ("SHA256SUMS", ".DS_Store") and path.suffix not in (".pyc", ".log"):
            yield path


def checksum(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    import numpy as np
    import pandas as pd
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-manifest", action="store_true",
                        help="Write SHA256SUMS after successful structural checks")
    args = parser.parse_args()
    files = list(release_files())
    errors = []
    for path in files:
        if path.stat().st_size >= 100*1024*1024:
            errors.append(f"File exceeds 100 MiB: {path.relative_to(ROOT)}")
        if path.suffix == ".py":
            text = path.read_text()
            try:
                ast.parse(text, filename=str(path))
            except SyntaxError as exc:
                errors.append(str(exc))
            if ("/" + "Users/") in text or ("/" + "home/") in text:
                errors.append(f"Machine-specific path: {path.relative_to(ROOT)}")
    for kind, count, dim in [("single", 360, 51), ("multi", 126, 107)]:
        path = ROOT / "results" / "expanded" / f"{kind}_behavior_descriptors.npz"
        with np.load(path, allow_pickle=False) as data:
            if data["X"].shape != (count, dim) or not np.isfinite(data["X"]).all():
                errors.append(f"Invalid descriptor corpus: {kind}")
    for name, expected in [("single_qd_alignment_milling", 140), ("single_qd_alignment_clustering", 144)]:
        table = pd.read_csv(ROOT / "results" / "highbudget" / f"{name}.csv")
        if len(table) != expected or table.duplicated(["cell_0", "cell_1"]).any():
            errors.append(f"Archive cell accounting failed: {name}")
    report = json.loads((ROOT / "results/independent_validation/report.json").read_text())
    if report["protocol"]["test_used_for_selection"]:
        errors.append("Independent audit reports test-data selection")
    for kind in ("single", "multi"):
        with np.load(ROOT / f"results/independent_validation/{kind}_split.npz") as split:
            groups = [set(split[key]) for key in split.files]
            if any(a & b for i, a in enumerate(groups) for b in groups[i+1:]):
                errors.append(f"Overlapping split indices: {kind}")
    figures = json.loads((ROOT / "docs/figure_manifest.json").read_text())
    if len(figures) != 30:
        errors.append("Expected 30 manuscript figure records")
    for row in figures:
        if checksum(ROOT / row["asset"]) != row["sha256"]:
            errors.append(f"Manuscript image hash mismatch: {row['figure']}")
    manifest = ROOT / "SHA256SUMS"
    if manifest.exists() and not args.write_manifest:
        recorded = {}
        for line in manifest.read_text().splitlines():
            digest, name = line.split("  ", 1)
            recorded[name] = digest
        current = {str(path.relative_to(ROOT)): checksum(path) for path in files}
        for name in sorted(current.keys() | recorded.keys()):
            if current.get(name) != recorded.get(name):
                errors.append(f"Checksum or inventory mismatch: {name}")
    if errors:
        print("\n".join(errors), file=sys.stderr)
        raise SystemExit(1)
    if args.write_manifest:
        manifest.write_text("".join(f"{checksum(p)}  {p.relative_to(ROOT)}\n" for p in files))
    print(json.dumps({"status": "passed", "files": len(files),
                      "python_files": sum(p.suffix == ".py" for p in files),
                      "manuscript_figures": len(figures),
                      "bytes": sum(p.stat().st_size for p in files)}, indent=2))


if __name__ == "__main__":
    main()
