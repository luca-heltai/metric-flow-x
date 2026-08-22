"""Focused tests for the reproducible WP-09 tutorial tools."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from analyze_y_junction_residuals import analyze  # noqa: E402
from generate_network_diagram import generate  # noqa: E402
from validate_vtk_network import validate  # noqa: E402


NETWORK = ROOT / "tutorials/02_y_junction/network.vtk"


def test_copied_network_valid_and_topology_outputs_are_deterministic(tmp_path: Path) -> None:
    assert validate(NETWORK)["valid"] is True
    svg_a, json_a = tmp_path / "a.svg", tmp_path / "a.json"
    svg_b, json_b = tmp_path / "b.svg", tmp_path / "b.json"
    generate(NETWORK, svg_a, json_a)
    generate(NETWORK, svg_b, json_b)
    assert svg_a.read_bytes() == svg_b.read_bytes()
    assert json_a.read_bytes() == json_b.read_bytes()
    topology = json.loads(json_a.read_text())
    assert [item["id"] for item in topology["vessels"]] == [0, 1, 2]
    assert [item["boundary_id"] for item in topology["points"]] == [0, 255, 1, 2]
    assert 'data-vessel-id="0"' in svg_a.read_text()
    assert 'data-boundary-id="255"' in svg_a.read_text()


def test_residual_metrics_use_named_columns_and_scales(tmp_path: Path) -> None:
    path = tmp_path / "residuals.csv"
    names = [
        "mass_residual", "head_residual_0", "head_residual_1",
        "compatibility_residual_0", "compatibility_residual_1", "compatibility_residual_2",
    ]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=names)
        writer.writeheader()
        writer.writerow({name: "1" for name in names})
        writer.writerow({name: "-2" for name in names})
    result = analyze(path, mass_scale=10, head_scale=2, compatibility_scale=3)
    assert result["metrics"]["mass"]["max_abs"] == 20
    assert result["metrics"]["head"]["max_abs"] == 4
    assert result["metrics"]["compatibility"]["max_abs"] == 6


def test_missing_solver_residual_columns_fails_without_fabrication(tmp_path: Path) -> None:
    path = tmp_path / "state_only.csv"
    path.write_text("time,A,U\n0,1,2\n")
    with pytest.raises(ValueError, match="missing required raw residual columns.*fabricate"):
        analyze(path)
