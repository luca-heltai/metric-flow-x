#!/usr/bin/env python3
"""Generate deterministic SVG and JSON topology views of a legacy ASCII VTK network.

The diagram is derived from the VTK point/cell connectivity; it is not a hand-drawn
illustration.  Validation is performed before either output is written.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from validate_vtk_network import parse_vtk, validate_network


def _layout(parsed: Any) -> list[tuple[float, float]]:
    """Return a stable 2-D layout from the source coordinates.

    The source is already a planar Y junction.  We preserve its x/y geometry,
    normalize it to a fixed canvas, and use point index as the tie breaker for
    degenerate coordinates.
    """
    xy = [(float(point[0]), float(point[1])) for point in parsed.points]
    min_x, max_x = min(x for x, _ in xy), max(x for x, _ in xy)
    min_y, max_y = min(y for _, y in xy), max(y for _, y in xy)
    span_x = max(max_x - min_x, 1e-15)
    span_y = max(max_y - min_y, 1e-15)
    width, height, margin = 800.0, 420.0, 55.0
    usable_w, usable_h = width - 2 * margin, height - 2 * margin
    # Keep the topology visually legible while remaining deterministic.
    scale = min(usable_w / span_x, usable_h / span_y)
    offset_x = margin + (usable_w - scale * span_x) / 2.0
    offset_y = margin + (usable_h - scale * span_y) / 2.0
    return [
        (offset_x + scale * (x - min_x), height - (offset_y + scale * (y - min_y)))
        for x, y in xy
    ]


def build_topology(
    parsed: Any,
    vtk_path: Path,
    validation: dict[str, Any],
    *,
    source: str | None = None,
    schema: str = "blood-flow.y-junction.topology.v1",
) -> dict[str, Any]:
    """Build JSON-safe topology and provenance data from validated parser output.

    ``source`` and ``schema`` are optional metadata overrides so the same
    connectivity-derived generator can be used by tutorials other than the
    original Y junction.  The default source remains the historical Y-junction
    value for compatibility with existing generated artifacts.
    """
    source_bytes = vtk_path.read_bytes()
    vessel_ids = [int(value) for value in parsed.cell_arrays["vessel_id"]]
    boundary_ids = [int(value) for value in parsed.point_arrays["boundary_id"]]
    return {
        "schema": schema,
        "source": source if source is not None else "notebooks/bifurcation_network.vtk",
        "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "validation": {
            "valid": bool(validation["valid"]),
            "points": int(validation["points"]),
            "cells": int(validation["cells"]),
            "vessels": int(validation["vessels"]),
            "terminal_boundary_ids": list(validation["terminal_boundary_ids"]),
        },
        "points": [
            {"id": index, "xyz": [float(value) for value in point], "boundary_id": boundary_ids[index]}
            for index, point in enumerate(parsed.points)
        ],
        "vessels": [
            {
                "id": vessel_ids[index],
                "cell": index,
                "points": [int(parsed.cells[index][0]), int(parsed.cells[index][1])],
            }
            for index in range(len(parsed.cells))
        ],
    }


def _svg(parsed: Any, topology: dict[str, Any]) -> str:
    positions = _layout(parsed)
    lines: list[str] = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<svg xmlns="http://www.w3.org/2000/svg" width="800" height="420" viewBox="0 0 800 420">',
        '  <title>Y-junction network topology</title>',
        '  <desc>Generated from legacy ASCII VTK connectivity; vessel and boundary identifiers are labels.</desc>',
        '  <rect width="800" height="420" fill="white"/>',
    ]
    # Cells are emitted in source order, making output byte-for-byte stable.
    for vessel in topology["vessels"]:
        start, end = vessel["points"]
        x1, y1 = positions[start]
        x2, y2 = positions[end]
        lines.append(
            f'  <line x1="{x1:.6f}" y1="{y1:.6f}" x2="{x2:.6f}" y2="{y2:.6f}" '
            f'stroke="#245a9b" stroke-width="6" data-vessel-id="{vessel["id"]}"/>'
        )
        lines.append(
            f'  <text x="{(x1 + x2) / 2:.6f}" y="{(y1 + y2) / 2 - 8:.6f}" '
            f'font-family="sans-serif" font-size="14" text-anchor="middle">vessel {vessel["id"]}</text>'
        )
    for point in topology["points"]:
        x, y = positions[point["id"]]
        boundary = point["boundary_id"]
        lines.append(f'  <circle cx="{x:.6f}" cy="{y:.6f}" r="9" fill="#d65f5f" data-boundary-id="{boundary}"/>')
        lines.append(
            f'  <text x="{x + 14:.6f}" y="{y - 12:.6f}" font-family="sans-serif" font-size="13">'
            f'point {point["id"]}; boundary {boundary}</text>'
        )
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def generate(
    vtk: str | Path,
    svg: str | Path,
    topology: str | Path,
    *,
    source: str | None = None,
    schema: str = "blood-flow.y-junction.topology.v1",
) -> None:
    vtk_path = Path(vtk)
    parsed = parse_vtk(vtk_path)
    validation = validate_network(parsed)
    if not validation.get("valid", False):
        errors = "; ".join(validation.get("errors", ["unknown validation error"]))
        raise ValueError(f"network validation failed for {vtk_path}: {errors}")
    topology_data = build_topology(parsed, vtk_path, validation, source=source, schema=schema)
    Path(svg).write_text(_svg(parsed, topology_data), encoding="utf-8")
    Path(topology).write_text(json.dumps(topology_data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("vtk", type=Path)
    parser.add_argument("--svg", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True, dest="topology")
    parser.add_argument(
        "--source",
        help="source path to record in topology metadata (defaults to the historical Y-junction path)",
    )
    parser.add_argument("--schema", default="blood-flow.y-junction.topology.v1")
    args = parser.parse_args(argv)
    try:
        generate(args.vtk, args.svg, args.topology, source=args.source, schema=args.schema)
    except (OSError, ValueError, KeyError) as exc:
        print(f"generate_network_diagram.py: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
