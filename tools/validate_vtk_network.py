#!/usr/bin/env python3
"""Validate the legacy ASCII VTK network input contract.

The solver consumes one-dimensional ``VTK_LINE`` cells and scalar cell/point
arrays.  This source-backed checker deliberately does not use VTK/PyVista so
it can run in a clean checkout and in CI before deal.II is configured.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import deque
from pathlib import Path
from typing import Any, Iterable


CELL_ARRAYS = ("vessel_id", "a0", "E", "h_wall", "p_d", "p0", "L", "a_d", "r_d")
POINT_ARRAYS = ("boundary_id", "R1", "R2", "C", "P_out")
TAPER_ARRAYS = ("r_in", "r_out")
DIRECTIVE_RE = re.compile(r"^(POINTS|CELLS|CELL_TYPES|CELL_DATA|POINT_DATA|SCALARS|FIELD|VECTORS|NORMALS|TEXTURE_COORDINATES|LOOKUP_TABLE)\b")


class VtkInputError(ValueError):
    """A malformed or semantically invalid network input."""


class LegacyVtk:
    def __init__(self, path: Path):
        self.path = path
        self.points: list[tuple[float, float, float]] = []
        self.cells: list[tuple[int, int]] = []
        self.cell_types: list[int] = []
        self.cell_arrays: dict[str, list[float]] = {}
        self.point_arrays: dict[str, list[float]] = {}
        self.cell_array_types: dict[str, str] = {}
        self.point_array_types: dict[str, str] = {}


def _error(path: Path, message: str) -> VtkInputError:
    return VtkInputError(f"{path}: {message}")


def _numeric_values(lines: list[str], index: int, count: int, path: Path, what: str) -> tuple[list[float], int]:
    values: list[float] = []
    while len(values) < count:
        if index >= len(lines):
            raise _error(path, f"{what} has {len(values)} values; expected {count}")
        text = lines[index].strip()
        if not text or text.startswith("#"):
            index += 1
            continue
        if DIRECTIVE_RE.match(text):
            raise _error(path, f"{what} has {len(values)} values; expected {count} before {text.split()[0]}")
        for token in text.split():
            try:
                value = float(token)
            except ValueError as exc:
                raise _error(path, f"non-numeric token {token!r} in {what}") from exc
            if not math.isfinite(value):
                raise _error(path, f"non-finite value {token!r} in {what}")
            values.append(value)
            if len(values) == count:
                break
        index += 1
    return values, index


def _integer(value: float, path: Path, what: str) -> int:
    if not value.is_integer():
        raise _error(path, f"{what} must be integral, got {value:g}")
    return int(value)


def _scalar_array(
    lines: list[str], index: int, count: int, path: Path, section: str
) -> tuple[str, str, list[float], int]:
    fields = lines[index].strip().split()
    if len(fields) < 3 or fields[0] != "SCALARS":
        raise _error(path, f"expected SCALARS declaration in {section}")
    name, datatype = fields[1], fields[2]
    components = 1
    if len(fields) >= 4:
        try:
            components = int(fields[3])
        except ValueError as exc:
            raise _error(path, f"invalid component count for array {name!r}") from exc
    if components < 1:
        raise _error(path, f"array {name!r} has invalid component count {components}")
    index += 1
    while index < len(lines) and not lines[index].strip():
        index += 1
    if index >= len(lines) or not lines[index].strip().startswith("LOOKUP_TABLE"):
        raise _error(path, f"array {name!r} is missing LOOKUP_TABLE")
    index += 1
    values, index = _numeric_values(lines, index, count * components, path, f"{section} array {name!r}")
    if components != 1:
        raise _error(path, f"array {name!r} has {components} components; only scalar arrays are supported")
    return name, datatype, values, index


def parse_vtk(path: str | Path) -> LegacyVtk:
    path = Path(path)
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise VtkInputError(f"{path}: cannot read file: {exc}") from exc
    if len(lines) < 4:
        raise _error(path, "file is too short for a legacy VTK header")
    if not lines[0].strip().lower().startswith("# vtk datafile version"):
        raise _error(path, "missing legacy VTK version header")
    if lines[2].strip().upper() != "ASCII":
        raise _error(path, "only legacy ASCII VTK is supported")
    if not lines[3].strip().upper().startswith("DATASET UNSTRUCTURED_GRID"):
        raise _error(path, "expected DATASET UNSTRUCTURED_GRID")

    result = LegacyVtk(path)
    index = 4
    while index < len(lines):
        text = lines[index].strip()
        if not text or text.startswith("#"):
            index += 1
            continue
        fields = text.split()
        directive = fields[0]
        if directive == "POINTS":
            if len(fields) < 3:
                raise _error(path, "malformed POINTS declaration")
            try:
                count = int(fields[1])
            except ValueError as exc:
                raise _error(path, "POINTS count is not an integer") from exc
            values, index = _numeric_values(lines, index + 1, 3 * count, path, "POINTS")
            result.points = [tuple(values[3 * i : 3 * i + 3]) for i in range(count)]  # type: ignore[list-item]
            continue
        if directive == "CELLS":
            if len(fields) < 3:
                raise _error(path, "malformed CELLS declaration")
            try:
                count, total = int(fields[1]), int(fields[2])
            except ValueError as exc:
                raise _error(path, "CELLS counts are not integers") from exc
            values, index = _numeric_values(lines, index + 1, total, path, "CELLS")
            ints = [_integer(v, path, "cell connectivity") for v in values]
            cursor = 0
            for cell_index in range(count):
                if cursor >= len(ints):
                    raise _error(path, f"CELLS ended before cell {cell_index}")
                n_vertices = ints[cursor]
                cursor += 1
                if n_vertices < 0 or cursor + n_vertices > len(ints):
                    raise _error(path, f"invalid connectivity count for cell {cell_index}: {n_vertices}")
                vertices = ints[cursor : cursor + n_vertices]
                cursor += n_vertices
                if n_vertices != 2:
                    raise _error(path, f"cell {cell_index} is not a line (has {n_vertices} vertices)")
                result.cells.append((vertices[0], vertices[1]))
            if cursor != len(ints):
                raise _error(path, "CELLS total does not match the declared cell count")
            continue
        if directive == "CELL_TYPES":
            if len(fields) < 2:
                raise _error(path, "malformed CELL_TYPES declaration")
            count = int(fields[1])
            values, index = _numeric_values(lines, index + 1, count, path, "CELL_TYPES")
            result.cell_types = [_integer(v, path, "cell type") for v in values]
            continue
        if directive in ("CELL_DATA", "POINT_DATA"):
            if len(fields) < 2:
                raise _error(path, f"malformed {directive} declaration")
            try:
                count = int(fields[1])
            except ValueError as exc:
                raise _error(path, f"{directive} count is not an integer") from exc
            section = "cell" if directive == "CELL_DATA" else "point"
            arrays = result.cell_arrays if section == "cell" else result.point_arrays
            types = result.cell_array_types if section == "cell" else result.point_array_types
            index += 1
            while index < len(lines):
                text = lines[index].strip()
                if not text or text.startswith("#"):
                    index += 1
                    continue
                if text.startswith("CELL_DATA") or text.startswith("POINT_DATA") or text.startswith("FIELD"):
                    break
                if not text.startswith("SCALARS"):
                    raise _error(path, f"unsupported {section} data directive {text.split()[0]!r}")
                name, datatype, values, index = _scalar_array(lines, index, count, path, section)
                if name in arrays:
                    raise _error(path, f"duplicate {section} array {name!r}")
                arrays[name] = values
                types[name] = datatype
            continue
        # Ignore legacy metadata that is not part of the network contract, but
        # reject unsupported data arrays in a data section above.
        index += 1

    if not result.points:
        raise _error(path, "missing POINTS")
    if not result.cells:
        raise _error(path, "missing CELLS or network has no cells")
    if len(result.cell_types) != len(result.cells):
        raise _error(path, f"CELL_TYPES length {len(result.cell_types)} does not match CELLS length {len(result.cells)}")
    return result


def _check_finite_positive(values: Iterable[float], name: str, path: Path) -> None:
    for i, value in enumerate(values):
        if not math.isfinite(value) or value <= 0:
            raise _error(path, f"{name}[{i}] must be finite and positive, got {value!r}")


def _same_per_vessel(values: list[float], vessel_ids: list[int], name: str, path: Path) -> None:
    seen: dict[int, float] = {}
    for i, (vid, value) in enumerate(zip(vessel_ids, values)):
        if vid in seen and not math.isclose(value, seen[vid], rel_tol=1e-10, abs_tol=1e-14):
            raise _error(path, f"{name} differs between cells sharing vessel_id {vid} (cell {i})")
        seen.setdefault(vid, value)


def validate_network(parsed: LegacyVtk) -> dict[str, Any]:
    path = parsed.path
    n_points, n_cells = len(parsed.points), len(parsed.cells)
    errors: list[str] = []
    # Keep semantic diagnostics separate so JSON callers can report all graph
    # defects in one invocation instead of fixing them one at a time.
    diagnostics: list[dict[str, Any]] = []

    if any(cell_type != 3 for cell_type in parsed.cell_types):
        bad = [i for i, cell_type in enumerate(parsed.cell_types) if cell_type != 3]
        errors.append(f"cell types must be VTK_LINE (3); invalid cells: {bad}")
    edge_set: set[tuple[int, int]] = set()
    adjacency = [[] for _ in range(n_points)]
    for i, (a, b) in enumerate(parsed.cells):
        if not (0 <= a < n_points and 0 <= b < n_points):
            errors.append(f"cell {i} references a point outside 0..{n_points - 1}: ({a}, {b})")
            continue
        if a == b or math.dist(parsed.points[a], parsed.points[b]) <= 1e-14:
            diagnostics.append({"kind": "zero-length", "cell": i, "points": [a, b]})
        edge = (min(a, b), max(a, b))
        if edge in edge_set:
            diagnostics.append({"kind": "duplicate-edge", "cell": i, "points": list(edge)})
        edge_set.add(edge)
        adjacency[a].append(b)
        adjacency[b].append(a)

    # Every point belongs to the same graph component.  This also reports an
    # unused point as disconnected rather than silently accepting it.
    visited: set[int] = set()
    queue: deque[int] = deque([0]) if n_points else deque()
    while queue:
        point = queue.popleft()
        if point in visited:
            continue
        visited.add(point)
        queue.extend(neighbor for neighbor in adjacency[point] if neighbor not in visited)
    if len(visited) != n_points:
        diagnostics.append({"kind": "disconnected", "points": sorted(set(range(n_points)) - visited)})

    for kind in ("cell", "point"):
        arrays = parsed.cell_arrays if kind == "cell" else parsed.point_arrays
        expected = CELL_ARRAYS if kind == "cell" else POINT_ARRAYS
        for name in expected:
            if name not in arrays:
                errors.append(f"missing required {kind} array {name!r}")
            elif len(arrays[name]) != (n_cells if kind == "cell" else n_points):
                errors.append(f"{kind} array {name!r} has length {len(arrays[name])}; expected {n_cells if kind == 'cell' else n_points}")
        present = [name for name in arrays if name in expected]
        if len(present) != len(set(present)):
            errors.append(f"duplicate {kind} array names")

    for name in TAPER_ARRAYS:
        present = name in parsed.cell_arrays
        if present and len(parsed.cell_arrays[name]) != n_cells:
            errors.append(f"optional taper array {name!r} has the wrong length")
    if ("r_in" in parsed.cell_arrays) != ("r_out" in parsed.cell_arrays):
        errors.append("taper arrays must be supplied as a complete r_in/r_out pair")

    if errors:
        return {"valid": False, "file": str(path), "errors": errors, "diagnostics": diagnostics}

    vessel_ids = [_integer(value, path, "vessel_id") for value in parsed.cell_arrays["vessel_id"]]
    if any(value < 0 for value in vessel_ids):
        errors.append("vessel_id values must be nonnegative")
    expected_ids = set(range(max(vessel_ids) + 1)) if vessel_ids else set()
    if set(vessel_ids) != expected_ids:
        errors.append(f"vessel_id values must be contiguous starting at 0; got {sorted(set(vessel_ids))}")
    # Geometric/material scales must be positive.  Pressures are allowed to
    # be zero (the 37-segment source uses gauge pressure) but must be finite.
    for name in ("a0", "E", "h_wall", "L", "a_d", "r_d"):
        _check_finite_positive(parsed.cell_arrays[name], name, path)
    for name in ("p_d", "p0"):
        if any(not math.isfinite(value) for value in parsed.cell_arrays[name]):
            errors.append(f"{name} values must be finite")
    for name in CELL_ARRAYS[1:]:
        _same_per_vessel(parsed.cell_arrays[name], vessel_ids, name, path)
    if "r_in" in parsed.cell_arrays:
        _check_finite_positive(parsed.cell_arrays["r_in"], "r_in", path)
        _check_finite_positive(parsed.cell_arrays["r_out"], "r_out", path)
        _same_per_vessel(parsed.cell_arrays["r_in"], vessel_ids, "r_in", path)
        _same_per_vessel(parsed.cell_arrays["r_out"], vessel_ids, "r_out", path)

    boundary_ids = [_integer(value, path, "boundary_id") for value in parsed.point_arrays["boundary_id"]]
    if any(value < 0 for value in boundary_ids):
        errors.append("boundary_id values must be nonnegative")
    degrees = [len(neighbors) for neighbors in adjacency]
    if boundary_ids.count(0) != 1:
        errors.append(f"boundary_id must contain exactly one inflow id 0; found {boundary_ids.count(0)}")
    positive_boundary_ids = [value for value in boundary_ids if value > 0 and value != 255]
    if len(positive_boundary_ids) != len(set(positive_boundary_ids)):
        errors.append("terminal boundary IDs must be unique")
    for point, boundary_id in enumerate(boundary_ids):
        if boundary_id == 255 and degrees[point] <= 1:
            errors.append(f"boundary_id 255 marks point {point}, which is not an interior junction/point")
        elif boundary_id != 255 and degrees[point] != 1:
            errors.append(f"boundary point {point} has id {boundary_id} but graph degree {degrees[point]}; endpoints must be degree one")
    inflow = boundary_ids.index(0)
    if degrees[inflow] != 1:
        errors.append(f"inflow boundary point {inflow} is not a graph endpoint")

    for name in ("R1", "R2", "C", "P_out"):
        values = parsed.point_arrays[name]
        for point, value in enumerate(values):
            if not math.isfinite(value):
                errors.append(f"{name}[{point}] must be finite")
            # P_out is a reference pressure and may be populated globally in
            # legacy assets; the circuit coefficients must be zero off-terminal.
            if name != "P_out" and boundary_ids[point] in (0, 255) and abs(value) > 1e-14:
                errors.append(f"{name}[{point}] must be zero for non-terminal boundary id {boundary_ids[point]}")
    for point, boundary_id in enumerate(boundary_ids):
        if boundary_id > 0 and boundary_id != 255:
            r1 = parsed.point_arrays["R1"][point]
            r2 = parsed.point_arrays["R2"][point]
            capacitance = parsed.point_arrays["C"][point]
            if r1 < 0:
                errors.append(f"R1 at terminal boundary {boundary_id} must be nonnegative")
            if r2 <= 0:
                errors.append(f"R2 at terminal boundary {boundary_id} must be positive")
            if capacitance < 0:
                errors.append(f"C at terminal boundary {boundary_id} must be nonnegative")
            if capacitance > 0 and r1 <= 0:
                errors.append(f"RCR terminal boundary {boundary_id} requires R1 > 0 when C > 0")
            if capacitance <= 0 and r1 > 0:
                errors.append(f"single-resistance terminal boundary {boundary_id} requires R1 = 0 when C = 0")

    if diagnostics:
        errors.extend(f"{item['kind']} diagnostic: {item}" for item in diagnostics)
    if errors:
        return {"valid": False, "file": str(path), "errors": errors, "diagnostics": diagnostics}
    return {
        "valid": True,
        "file": str(path),
        "points": n_points,
        "cells": n_cells,
        "vessels": len(expected_ids),
        "terminal_boundary_ids": sorted(set(positive_boundary_ids)),
        "diagnostics": [],
    }


def validate(path: str | Path) -> dict[str, Any]:
    try:
        return validate_network(parse_vtk(path))
    except (OSError, VtkInputError, ValueError) as exc:
        return {"valid": False, "file": str(path), "errors": [str(exc)], "diagnostics": []}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("vtk", type=Path, nargs="+", help="legacy ASCII VTK network file(s)")
    args = parser.parse_args(argv)
    results = [validate(path) for path in args.vtk]
    output: Any = results[0] if len(results) == 1 else {"valid": all(item["valid"] for item in results), "files": results}
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0 if output["valid"] else 1


if __name__ == "__main__":
    sys.exit(main())
