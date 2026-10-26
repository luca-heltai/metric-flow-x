#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Independently check SI geometry, radii, physics and provenance of an export."""

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from validate_vtk_network import parse_vtk, validate


def check(path):
    report = json.loads(path.with_suffix(".provenance.json").read_text())
    result = validate(path)
    if not result["valid"]:
        raise ValueError(json.dumps(result))
    if hashlib.sha256(path.read_bytes()).hexdigest() != report["vtk_sha256"]:
        raise ValueError("VTK checksum differs from provenance")
    mesh = parse_vtk(path)
    affine, voxels = report["voxel_to_ras_m"], report["voxel_indices"]
    if len(voxels) != len(mesh.points):
        raise ValueError("Voxel mapping size differs from VTK")
    for voxel, point in zip(voxels, mesh.points):
        expected = [
            sum(affine[d][j] * voxel[j] for j in range(3)) + affine[d][3]
            for d in range(3)
        ]
        if math.dist(expected, point) > 1e-12:
            raise ValueError("VTK point differs from the source affine in metres")
    coefficients = report["physics_si"]
    radii = report["point_radii_m"]
    for i, (a, b) in enumerate(mesh.cells):
        radius = (radii[a] + radii[b]) / 2
        expected = {
            "vessel_id": i,
            "L": math.dist(mesh.points[a], mesh.points[b]),
            "r_d": radius,
            "a_d": math.pi * radius**2,
            "a0": math.pi * radius**2,
            "p0": 0,
            "E": coefficients["young_modulus"],
            "h_wall": radius * coefficients["wall_ratio"],
            "p_d": coefficients["reference_pressure"],
        }
        for name, value in expected.items():
            if not math.isclose(
                mesh.cell_arrays[name][i], value, rel_tol=1e-12, abs_tol=1e-20
            ):
                raise ValueError(f"Inconsistent {name} at cell {i}")
    if len(mesh.cells) != len(mesh.points) - 1:
        raise ValueError("Export is not a connected tree")
    boundary = mesh.point_arrays["boundary_id"]
    if boundary[report["root_point"]] != 0:
        raise ValueError("Root differs from provenance")
    terminals = {item["point"]: item["boundary_id"] for item in report["terminals"]}
    if terminals != {
        i: int(bid) for i, bid in enumerate(boundary) if bid not in (0, 255)
    }:
        raise ValueError("Terminal map differs from VTK")
    for i in terminals:
        for name, value in {
            "R1": 0,
            "R2": coefficients["terminal_resistance"],
            "C": 0,
            "P_out": coefficients["outlet_pressure"],
        }.items():
            if mesh.point_arrays[name][i] != value:
                raise ValueError(f"Inconsistent terminal {name}")
    return {
        "valid": True,
        "points": len(mesh.points),
        "cells": len(mesh.cells),
        "outlets": len(terminals),
        "length_m": sum(mesh.cell_arrays["L"]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("vtk", type=Path)
    args = parser.parse_args()
    try:
        print(json.dumps(check(args.vtk), indent=2))
    except (ValueError, OSError, KeyError, IndexError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
