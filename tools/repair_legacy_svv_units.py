#!/usr/bin/env python3
"""Repair the CGS geometry/flow error in the checksum-pinned legacy svVascularize VTK assets.

These files were generated with centimetre geometry and cm^3/s design flows,
but SI wall constants and pressures. Keep the physical model and topology;
recompute geometry, wall thickness, resistance and compliance in SI.
New tree conversions should use convert_to_standard_vtk.py and its metadata.
"""
from __future__ import annotations

import argparse
import math
import os
import tempfile
from pathlib import Path

from validate_vtk_network import parse_vtk

MARKER = " [legacy svVascularize CGS repaired to SI v1]"


def repair(path: Path) -> bool:
    lines = path.read_text().splitlines()
    if lines[1].endswith(MARKER):
        return False
    mesh = parse_vtk(path)
    cells = mesh.cell_arrays
    points = mesh.point_arrays
    for name in ("a0", "a_d"):
        cells[name] = [value * 1e-4 for value in cells[name]]
    for name in ("L", "r_d", "r_in", "r_out"):
        if name in cells:
            cells[name] = [value * 1e-2 for value in cells[name]]
    # This is the wall law used by the archived converter. Its coefficients,
    # Young's modulus and pressures are already SI, so evaluate at SI radii.
    cells["h_wall"] = [
        radius / young * (6.538312e4 * math.exp(-4.615404e2 * radius) + 2.632577e4)
        for radius, young in zip(cells["r_d"], cells["E"])
    ]
    for name in ("R1", "R2"):
        points[name] = [value * 1e6 for value in points[name]]
    points["C"] = [value * 1e-6 for value in points["C"]]
    degree = [0] * len(mesh.points)
    for first, second in mesh.cells:
        degree[first] += 1
        degree[second] += 1
    terminal_ids = {int(bid) for bid, count in zip(points["boundary_id"], degree) if count == 1}
    interior_id = 300655
    if interior_id in terminal_ids:
        raise ValueError(f"{path}: interior marker would collide with an outlet")
    points["boundary_id"] = [interior_id if count > 1 else int(bid)
                             for bid, count in zip(points["boundary_id"], degree)]

    def scalars(stream, arrays):
        for name, values in arrays.items():
            datatype = "unsigned_int" if name == "boundary_id" else "double"
            stream.write(f"SCALARS {name} {datatype} 1\nLOOKUP_TABLE default\n")
            for value in values:
                stream.write(f"{int(value)}\n" if name == "boundary_id" else f"{value:.15e}\n")

    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(f"{lines[0]}\n{lines[1]}{MARKER}\nASCII\nDATASET UNSTRUCTURED_GRID\n")
            stream.write(f"POINTS {len(mesh.points)} double\n")
            for point in mesh.points:
                stream.write(" ".join(f"{value * 1e-2:.15e}" for value in point) + "\n")
            stream.write(f"CELLS {len(mesh.cells)} {3 * len(mesh.cells)}\n")
            for first, second in mesh.cells:
                stream.write(f"2 {first} {second}\n")
            stream.write(f"CELL_TYPES {len(mesh.cells)}\n")
            stream.write("3\n" * len(mesh.cells))
            stream.write(f"CELL_DATA {len(mesh.cells)}\n")
            scalars(stream, cells)
            stream.write(f"POINT_DATA {len(mesh.points)}\n")
            scalars(stream, points)
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    args = parser.parse_args()
    folder = args.dataset_root / "notebooks/Networks-by-svv/trees"
    for path in sorted(folder.glob("*_adnr*.vtk")):
        if repair(path):
            print(f"Repaired legacy CGS units: {path}")


if __name__ == "__main__":
    main()
