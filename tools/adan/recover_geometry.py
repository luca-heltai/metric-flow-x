#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Create a CSV-only metric embedding or recover atlas centerline samples.

This reads data, never executes downloaded content. Surface-derived centerlines
are approximations to the unpublished splines, not the original spline files.
"""
import argparse
import bisect
import csv
import heapq
import json
import math
import re
from collections import defaultdict
from pathlib import Path

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "build/adan"


def read_rows(root=DEFAULT_DATA_DIR, include_veins=False):
    with (root / "source/data_sheet_1.csv").open() as stream:
        data = list(csv.reader(stream, delimiter=";"))[1:]
    rows = []
    names = set()
    for row in data:
        if len(row) != 15:
            raise ValueError("Expected the 15-column ADAVN supplementary CSV")
        row = [field.strip() for field in row]
        if row[6] not in ("0", "1"):
            raise ValueError("Vessel type must be 0 (vein) or 1 (artery)")
        if row[6] == "0" and not include_veins:
            continue
        # The full supplement repeats some venous names (including split
        # Starling-resistor segments). Row number, not name, is their identity.
        if row[0] in names and not include_veins:
            raise ValueError("Duplicate source vessel name: " + row[0])
        names.add(row[0])
        if int(row[1]) < 0 or int(row[2]) < 0 or row[1] == row[2]:
            raise ValueError("Invalid source node IDs")
        for column in (3, 4, 5):
            value = float(row[column])
            if not math.isfinite(value) or value <= 0:
                raise ValueError("Lengths and radii must be finite and positive")
        rows.append(row)
    if not rows:
        raise ValueError("No arterial segments in source table")
    return rows


def surface_centers(path):
    tokens = path.read_text().split()
    if "ASCII" not in tokens[:10] or tokens[tokens.index("DATASET") + 1] != "POLYDATA":
        raise ValueError(f"{path}: expected ASCII POLYDATA")
    k = tokens.index("POINTS")
    count = int(tokens[k + 1])
    points = [
        tuple(map(float, tokens[k + 3 + 3 * i : k + 6 + 3 * i])) for i in range(count)
    ]
    seen = set()
    first_repeat = len(points)
    for i, point in enumerate(points):
        if point in seen:
            first_repeat = i
            break
        seen.add(point)
    if first_repeat % 12 or first_repeat < 24:
        raise ValueError(
            f"{path.name}: first repeated vertex {first_repeat}; not complete 12-point rings"
        )
    centers = []
    radii = []
    for i in range(0, first_repeat, 12):
        ring = points[i : i + 12]
        center = tuple(sum(p[j] for p in ring) / 12 for j in range(3))
        distances = [math.dist(p, center) for p in ring]
        if max(distances) - min(distances) > max(0.002, 0.01 * sum(distances) / 12):
            raise ValueError(f"{path.name}: noncircular ring at vertex {i}")
        centers.append(center)
        radii.append(sum(distances) / 12)
    return centers, radii


def sample_curve(curve, fraction):
    lengths = [0.0]
    for a, b in zip(curve, curve[1:]):
        lengths.append(lengths[-1] + math.dist(a, b))
    target = fraction * lengths[-1]
    i = min(bisect.bisect_right(lengths, target) - 1, len(curve) - 2)
    t = (target - lengths[i]) / (lengths[i + 1] - lengths[i])
    return tuple(a + t * (b - a) for a, b in zip(curve[i], curve[i + 1]))


def recover(root=DEFAULT_DATA_DIR):
    rows = read_rows(root)
    groups = defaultdict(list)
    for row in rows:
        groups[re.sub(r"_\d+$", "", row[0])].append(row)
    candidates = {}
    incident = defaultdict(list)
    diagnostics = []
    for name, group in groups.items():
        curve, radii = surface_centers(
            root / "source/surfaces" / ("SW_" + name + ".vtk")
        )
        total = sum(float(r[3]) for r in group)
        nodes = [int(group[0][1])] + [int(r[2]) for r in group]
        if any(int(a[2]) != int(b[1]) for a, b in zip(group, group[1:])):
            raise ValueError(name + ": source segments do not form an ordered chain")
        fractions = [0.0]
        for row in group:
            fractions.append(fractions[-1] + float(row[3]) / total)
        candidates[name] = [
            dict(zip(nodes, (sample_curve(curve, min(1.0, f)) for f in fractions))),
            dict(zip(nodes, (sample_curve(curve, max(0.0, 1 - f)) for f in fractions))),
        ]
        for node in nodes:
            incident[node].append(name)
        diagnostics.append(
            {
                "name": name,
                "source_length_cm": total,
                "surface_centerline_length_cm": sum(
                    math.dist(a, b) for a, b in zip(curve, curve[1:])
                ),
                "rings": len(curve),
                "proximal_surface_radius_cm": radii[0],
                "distal_surface_radius_cm": radii[-1],
            }
        )
    # Shared nodes determine surface direction globally. A greedy first visit
    # can misorient long chains and move whole branches far from the atlas.
    from recover_full import orient
    flips, optimization = orient(candidates, incident)
    points = {}
    disagreements = []
    for node, names in incident.items():
        estimates = [candidates[name][flips[name]][node] for name in names]
        points[node] = tuple(
            sum(p[j] for p in estimates) / len(estimates) * 0.01 for j in range(3)
        )
        spread = max(math.dist(a, b) for a in estimates for b in estimates)
        disagreements.append({"node": node, "spread_cm": spread, "arteries": names})
    report = {
        "mode": "anatomical",
        "orientation_optimization": optimization,
        "arteries": diagnostics,
        "orientation_reversed": [name for name in flips if flips[name]],
        "node_disagreements": sorted(
            disagreements, key=lambda x: x["spread_cm"], reverse=True
        ),
        "points_m": points,
    }
    (root / "geometry_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        "Recovered",
        len(groups),
        "arteries,",
        len(points),
        "nodes; worst node disagreement [cm]:",
    )
    for item in report["node_disagreements"][:10]:
        print(item)
    return rows, points, report


def schematic(root=DEFAULT_DATA_DIR):
    """Embed the CSV metric graph without using any atlas surface geometry."""
    rows = read_rows(root)
    adjacency = defaultdict(list)
    for row in rows:
        a, b, length = int(row[1]), int(row[2]), float(row[3]) * 0.01
        if not length > 0:
            raise ValueError("Nonpositive source length")
        adjacency[a].append((b, length))
        adjacency[b].append((a, length))
    roots = {
        int(row[j])
        for row in rows
        for j in (1, 2)
        if len(adjacency[int(row[j])]) == 1 and row[9] == "-"
    }
    if len(roots) != 1:
        raise ValueError("Expected one inlet")
    inlet = roots.pop()
    distance = {inlet: 0.0}
    queue = [(0.0, inlet)]
    while queue:
        d, a = heapq.heappop(queue)
        if d > distance[a]:
            continue
        for b, length in adjacency[a]:
            candidate = d + length
            if candidate < distance.get(b, float("inf")):
                distance[b] = candidate
                heapq.heappush(queue, (candidate, b))
    if len(distance) != len(adjacency):
        raise ValueError("Disconnected arterial graph")
    smallest = min(float(row[3]) * 0.01 for row in rows)
    # Shortest-path distance is 1-Lipschitz on every edge. Scaling it by
    # 0.9, with bounded transverse offsets, keeps each chord below its
    # prescribed length. The converter then restores exact edge lengths.
    amplitude = 0.01 * smallest
    points = {
        node: (
            0.9 * d,
            amplitude * math.sin(node * 2.399963229728653),
            amplitude * math.cos(node * 2.399963229728653),
        )
        for node, d in distance.items()
    }
    report = {
        "mode": "schematic",
        "points_m": points,
        "geometry": "CSV-only abstract metric embedding; no anatomical coordinates or atlas data.",
    }
    (root / "geometry_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        "Schematic metric graph:", len(points), "nodes;", len(rows), "source segments"
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--mode", choices=["schematic", "anatomical"], default="schematic"
    )
    args = parser.parse_args()
    if args.mode == "anatomical":
        recover(args.data_dir)
    else:
        schematic(args.data_dir)
