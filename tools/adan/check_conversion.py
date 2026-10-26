#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Check source topology, metric lengths, radii, R/C and wall-law tangents."""
import argparse
import csv
import json
import math
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import validate_vtk_network as validator
from recover_geometry import DEFAULT_DATA_DIR


def require(condition):
    if not condition:
        raise ValueError("Conversion invariant failed")


def published(area, data, i):
    """Elastic part of ADAVN equation (3), independent of equivalent E."""
    ref_area = data["a_d"][i]
    radius = math.sqrt(ref_area / math.pi)
    h = data["h_wall"][i]
    strain = math.sqrt(area / ref_area) - 1
    ee, ec = data["adavn_E_e"][i], data["adavn_E_c"][i]
    er, e0 = data["adavn_epsilon_r"][i], data["adavn_epsilon_0"][i]
    return data["adavn_p_ref"][i] + math.pi * radius * h / area * (
        ee * strain + ec * er * math.log1p(math.exp((strain - e0) / er))
    )


def check(root=DEFAULT_DATA_DIR):
    vtk = validator.parse_vtk(root / "adan_complete.vtk")
    result = validator.validate_network(vtk)
    require(result["valid"])
    with (root / "source/data_sheet_1.csv").open() as stream:
        rows = [
            [v.strip() for v in r]
            for r in list(csv.reader(stream, delimiter=";"))[1:]
            if r[6].strip() == "1"
        ]
    nodes = sorted({int(r[j]) for r in rows for j in (1, 2)})
    index = {n: i for i, n in enumerate(nodes)}
    max_length_error = 0.0
    for i, row in enumerate(rows):
        first, second = vtk.cells[2 * i : 2 * i + 2]
        require(first[0] == index[int(row[1])] and second[1] == index[int(row[2])])
        require(first[1] == second[0] == len(nodes) + i)
        observed = sum(
            math.dist(vtk.points[a], vtk.points[b]) for a, b in (first, second)
        )
        target = float(row[3]) * 0.01
        max_length_error = max(max_length_error, abs(observed - target) / target)
        require(math.isclose(observed, target, rel_tol=1e-10))
        require(
            math.isclose(
                vtk.cell_arrays["r_in"][2 * i], float(row[4]) * 0.01, rel_tol=1e-14
            )
        )
        require(
            math.isclose(
                vtk.cell_arrays["r_out"][2 * i + 1], float(row[5]) * 0.01, rel_tol=1e-14
            )
        )
        require(vtk.cell_arrays["vessel_id"][2 * i : 2 * i + 2] == [2 * i, 2 * i + 1])
    with (root / "adan_complete.terminals.csv").open() as stream:
        terminals = list(csv.DictReader(stream))
    source = {row[0]: row for row in rows}
    max_r_error = 0.0
    for t in terminals:
        row = source[t["source_name"]]
        p = index[int(t["source_node"])]
        total = vtk.point_arrays["R1"][p] + vtk.point_arrays["R2"][p]
        expected = float(row[9]) * 1e5
        max_r_error = max(max_r_error, abs(total - expected) / expected)
        require(math.isclose(total, expected, rel_tol=1e-14))
        require(
            math.isclose(vtk.point_arrays["C"][p], float(row[10]) * 1e-5, rel_tol=1e-14)
        )
        require(int(vtk.point_arrays["boundary_id"][p]) == int(t["boundary_id"]))
    # Compare the active square-root law at reference area to the value and
    # finite-difference derivative of Eq. (3), omitting only its viscous term.
    max_tangent_error = 0.0
    for i in range(len(vtk.cells)):
        data = vtk.cell_arrays
        ref_area = data["a_d"][i]
        h = data["h_wall"][i]

        require(
            math.isclose(
                data["p_d"][i] + data["p0"][i],
                published(ref_area, data, i),
                rel_tol=1e-13,
            )
        )
        delta = ref_area * 1e-5
        finite_difference = (
            published(ref_area + delta, data, i) - published(ref_area - delta, data, i)
        ) / (2 * delta)
        beta = 4 * math.sqrt(math.pi) * data["E"][i] * h / 3
        active_tangent = beta / (2 * ref_area * math.sqrt(ref_area))
        error = abs(finite_difference - active_tangent) / active_tangent
        max_tangent_error = max(max_tangent_error, error)
        require(error < 2e-8)
    receipt = {
        "valid": True,
        "source_segments_checked": len(rows),
        "terminal_R_C_checked": len(terminals),
        "source_connectivity_and_radii_preserved": True,
        "max_relative_source_length_error": max_length_error,
        "max_relative_total_R_error": max_r_error,
        "max_relative_wall_tangent_finite_difference_error": max_tangent_error,
    }
    (root / "conversion_checks.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    args = parser.parse_args()
    check(args.data_dir)
