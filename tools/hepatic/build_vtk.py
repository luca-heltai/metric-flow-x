#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Convert one segmented hepatic vascular tree to a MetricFlowX SI line mesh."""

import argparse
import hashlib
import json
import sys
from collections import deque
from itertools import product
from pathlib import Path

import nibabel as nib
import numpy as np
import pydicom
from scipy import ndimage
from skimage.morphology import skeletonize

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from validate_vtk_network import validate


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_mask(path, label, length_unit=None):
    """Return a binary mask and voxel-center affine in RAS metres."""
    path = Path(path)
    if not path.is_dir():
        image = nib.load(path)
        data = np.asanyarray(image.dataobj)
        unit = length_unit or image.header.get_xyzt_units()[0]
        scales = {"meter": 1.0, "mm": 1e-3, "micron": 1e-6}
        if unit not in scales:
            raise ValueError("Unknown NIfTI length unit: supply --length-unit")
        affine = image.affine.copy()
        affine[:3, :] *= scales[unit]
        hashes = {path.name: sha256(path)}
    else:
        if length_unit not in (None, "mm"):
            raise ValueError("DICOM patient coordinates are always millimetres")
        files = sorted(p for p in path.iterdir() if p.is_file())
        slices = [pydicom.dcmread(p) for p in files]
        if len(slices) < 2:
            raise ValueError("A DICOM mask directory needs at least two slices")
        first = slices[0]
        orientation = np.asarray(first.ImageOrientationPatient, dtype=float)
        normal = np.cross(orientation[:3], orientation[3:])
        order = np.argsort([np.dot(s.ImagePositionPatient, normal) for s in slices])
        slices = [slices[i] for i in order]
        positions = np.array([s.ImagePositionPatient for s in slices], dtype=float)
        step = positions[1] - positions[0]
        for s in slices:
            if (
                s.SeriesInstanceUID != first.SeriesInstanceUID
                or not np.allclose(s.ImageOrientationPatient, orientation)
                or not np.allclose(s.PixelSpacing, first.PixelSpacing)
                or (s.Rows, s.Columns) != (first.Rows, first.Columns)
            ):
                raise ValueError("Mixed or inconsistent DICOM mask series")
        if not np.allclose(np.diff(positions, axis=0), step, atol=1e-4):
            raise ValueError("Nonuniform or duplicate DICOM slice positions")
        data = np.stack([s.pixel_array for s in slices], axis=2)
        affine = np.eye(4)
        affine[:3, 0] = orientation[3:] * float(first.PixelSpacing[0])
        affine[:3, 1] = orientation[:3] * float(first.PixelSpacing[1])
        affine[:3, 2] = step
        affine[:3, 3] = positions[0]
        affine = np.diag([-1e-3, -1e-3, 1e-3, 1]) @ affine
        hashes = {p.name: sha256(p) for p in files}
    if data.ndim != 3 or not np.isfinite(data).all():
        raise ValueError("Expected a finite three-dimensional segmentation")
    mask = data == label
    if not mask.any():
        raise ValueError(f"Label {label} is absent from {path}")
    spacing = np.linalg.norm(affine[:3, :3], axis=0)
    if (
        not np.isfinite(affine).all()
        or np.any(spacing <= 0)
        or not np.allclose(
            (affine[:3, :3] / spacing).T @ (affine[:3, :3] / spacing),
            np.eye(3),
            atol=1e-5,
        )
    ):
        raise ValueError("Singular or sheared voxel affine is unsupported")
    return mask, affine, hashes


def extract_graph(mask, affine, component=None, break_cycles=False):
    """Keep one component, thin in 3D, and build a conservative voxel graph."""
    labels, count = ndimage.label(mask, structure=np.ones((3, 3, 3)))
    sizes = np.bincount(labels.ravel())[1:]
    ranking = np.argsort(-sizes, kind="stable") + 1
    if count > 1 and component is None:
        raise ValueError(
            f"Mask has {count} components; select --component largest "
            "or a zero-based size rank explicitly"
        )
    rank = 0 if component in (None, "largest") else int(component)
    if not 0 <= rank < count:
        raise ValueError("Component rank outside the available range")
    selected = labels == ranking[rank]
    indices = np.argwhere(selected)
    low, high = indices.min(axis=0), indices.max(axis=0) + 1
    cropped = np.pad(selected[tuple(slice(a, b) for a, b in zip(low, high))], 1)
    spacing = np.linalg.norm(affine[:3, :3], axis=0)
    distance = ndimage.distance_transform_edt(cropped, sampling=spacing)
    voxels = np.argwhere(skeletonize(cropped, method="lee"))
    if len(voxels) < 2:
        raise ValueError("Selected component has no usable centerline")
    radii = distance[tuple(voxels.T)]
    voxels = voxels + low - 1
    points = nib.affines.apply_affine(affine, voxels)
    lookup = {tuple(v): i for i, v in enumerate(voxels)}
    adjacency = [set() for _ in voxels]
    for i, v in enumerate(voxels):
        for delta in product((-1, 0, 1), repeat=3):
            j = lookup.get(tuple(v + delta))
            if j is not None and j != i:
                adjacency[i].add(j)
    # 26-neighbour graphs contain triangles at bends. Drop a diagonal only
    # when a common voxel provides two strictly shorter neighbouring edges.
    edges, shortcuts = [], 0
    for a, neighbors in enumerate(adjacency):
        for b in sorted(neighbors):
            if a >= b:
                continue
            length = np.linalg.norm(points[a] - points[b])
            if any(
                max(
                    np.linalg.norm(points[a] - points[c]),
                    np.linalg.norm(points[b] - points[c]),
                )
                < length * (1 - 1e-8)
                for c in neighbors & adjacency[b]
            ):
                shortcuts += 1
            else:
                edges.append((float(length), a, b))
    # Kruskal is deterministic. Cutting remaining loops is opt-in because a
    # segmentation bridge and a real vascular connection cannot be distinguished.
    parent = list(range(len(points)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    tree, removed = [], []
    for length, a, b in sorted(edges):
        ra, rb = find(a), find(b)
        if ra == rb:
            removed.append([a, b])
        else:
            parent[ra] = rb
            tree.append((a, b))
    if len(tree) != len(points) - 1:
        raise ValueError("Skeleton is disconnected after thinning")
    if removed and not break_cycles:
        raise ValueError(
            f"Skeleton has {len(removed)} cycle edges; inspect the "
            "segmentation or explicitly use --break-cycles"
        )
    report = {
        "component_sizes_voxels": [int(sizes[i - 1]) for i in ranking],
        "selected_component_rank": rank,
        "discarded_mask_voxels": int(mask.sum() - selected.sum()),
        "local_diagonal_shortcuts_removed": shortcuts,
        "cycle_edges_removed": removed,
        "voxel_indices": voxels.tolist(),
    }
    return points, radii, tree, report


def orient_tree(points, radii, edges, root_point=None, root_largest_radius=False):
    adjacency = [[] for _ in points]
    for a, b in edges:
        adjacency[a].append(b)
        adjacency[b].append(a)
    endpoints = [i for i, n in enumerate(adjacency) if len(n) == 1]
    if len(endpoints) < 2:
        raise ValueError("Tree has fewer than two endpoints")
    if root_point is not None:
        root = min(endpoints, key=lambda i: np.linalg.norm(points[i] - root_point))
    elif root_largest_radius:
        root = max(endpoints, key=lambda i: (radii[i], -i))
    else:
        raise ValueError("Choose --root-point or --root-largest-radius")
    visited, oriented, queue = {root}, [], deque([root])
    while queue:
        a = queue.popleft()
        for b in sorted(adjacency[a]):
            if b not in visited:
                visited.add(b)
                oriented.append((a, b))
                queue.append(b)
    return root, sorted(set(endpoints) - {root}), oriented


def write_network(path, points, radii, edges, root, outlets, physics):
    lengths = [float(np.linalg.norm(points[a] - points[b])) for a, b in edges]
    radius = [(float(radii[a]) + float(radii[b])) / 2 for a, b in edges]
    area = [float(np.pi * r * r) for r in radius]
    n_cells, n_points = len(edges), len(points)
    # Unique cell/material IDs are required by build_global_vessel_data(),
    # which indexes property arrays by vessel_id, rather than first cell index.
    cell_data = {
        "vessel_id": list(range(n_cells)),
        "a0": area,
        "a_d": area,
        "r_d": radius,
        "L": lengths,
        "E": [physics["young_modulus"]] * n_cells,
        "h_wall": [physics["wall_ratio"] * r for r in radius],
        "p_d": [physics["reference_pressure"]] * n_cells,
        "p0": [0.0] * n_cells,
    }
    point_data = {k: [0.0] * n_points for k in ("R1", "R2", "C", "P_out")}
    point_data["boundary_id"] = [255] * n_points
    point_data["boundary_id"][root] = 0
    terminal_map = []
    for ordinal, point in enumerate(outlets, 1):
        bid = ordinal if ordinal < 255 else ordinal + 1
        point_data["boundary_id"][point] = bid
        point_data["R2"][point] = physics["terminal_resistance"]
        point_data["P_out"][point] = physics["outlet_pressure"]
        terminal_map.append({"point": point, "boundary_id": bid})
    with path.open("w") as stream:
        stream.write(
            "# vtk DataFile Version 3.0\nHepatic mask centerlines; "
            "RAS metres; see companion provenance JSON\nASCII\n"
            "DATASET UNSTRUCTURED_GRID\n"
        )
        stream.write(f"POINTS {n_points} double\n")
        for p in points:
            stream.write(" ".join(f"{x:.17g}" for x in p) + "\n")
        stream.write(f"CELLS {n_cells} {3 * n_cells}\n")
        for a, b in edges:
            stream.write(f"2 {a} {b}\n")
        stream.write(f"CELL_TYPES {n_cells}\n" + "3\n" * n_cells)
        for kind, arrays, count in (
            ("CELL", cell_data, n_cells),
            ("POINT", point_data, n_points),
        ):
            stream.write(f"{kind}_DATA {count}\n")
            for name, values in arrays.items():
                stream.write(f"SCALARS {name} double 1\nLOOKUP_TABLE default\n")
                stream.write("\n".join(f"{v:.17g}" for v in values) + "\n")
    result = validate(path)
    if not result["valid"]:
        path.unlink()
        raise ValueError(json.dumps(result))
    return terminal_map


def build(args):
    if args.output.suffix != ".vtk":
        raise ValueError("Output must have a .vtk suffix")
    physics = {
        key: getattr(args, key)
        for key in (
            "young_modulus",
            "wall_ratio",
            "reference_pressure",
            "terminal_resistance",
            "outlet_pressure",
        )
    }
    if not all(np.isfinite(v) for v in physics.values()):
        raise ValueError("Physical parameters must be finite")
    if any(
        physics[k] <= 0 for k in ("young_modulus", "wall_ratio", "terminal_resistance")
    ):
        raise ValueError("Modulus, wall ratio and resistance must be positive")
    if args.root_point is not None and not np.isfinite(args.root_point).all():
        raise ValueError("Root coordinates must be finite")
    mask, affine, hashes = load_mask(args.mask, args.label, args.length_unit)
    points, radii, edges, report = extract_graph(
        mask, affine, args.component, args.break_cycles
    )
    root, outlets, edges = orient_tree(
        points, radii, edges, args.root_point, args.root_largest_radius
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    terminals = write_network(args.output, points, radii, edges, root, outlets, physics)
    report.update(
        {
            "source": str(args.mask),
            "source_file_sha256": hashes,
            "source_url": args.source_url,
            "source_license": args.source_license,
            "label": args.label,
            "network_kind": args.network_kind,
            "voxel_to_ras_m": affine.tolist(),
            "point_radii_m": radii.tolist(),
            "root_point": root,
            "root_selection": "nearest endpoint to RAS metres"
            if args.root_point
            else "largest-radius endpoint heuristic",
            "requested_root_point_m": args.root_point,
            "terminals": terminals,
            "physics_si": physics,
            "vtk_sha256": sha256(args.output),
            "algorithm": "Lee 3D thinning; EDT radius; local shortcut removal; rooted spanning tree",
            "software": {
                name: __import__(name).__version__
                for name in ("numpy", "scipy", "skimage", "nibabel", "pydicom")
            },
            "limitations": [
                "Voxel skeleton, not a subvoxel anatomical centerline.",
                "Radii are distances to background voxel centres, biased by resolution.",
                "Wall and terminal coefficients are user assumptions, not image measurements.",
                "Venous exports use the existing elastic solver law, without collapse.",
                "Boundary 0 is the mathematical root; hepatic venous drainage needs adapted boundary conditions.",
            ],
        }
    )
    args.output.with_suffix(".provenance.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                "vtk": str(args.output),
                "points": len(points),
                "cells": len(edges),
                "outlets": len(outlets),
                "discarded_mask_voxels": report["discarded_mask_voxels"],
                "cycle_edges_removed": len(report["cycle_edges_removed"]),
            }
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mask", type=Path, help="NIfTI label image or one DICOM mask-series directory"
    )
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--label", type=int, required=True, help="Select exactly one anatomical class"
    )
    parser.add_argument(
        "--network-kind",
        choices=("artery", "portal", "hepatic", "unclassified"),
        required=True,
    )
    parser.add_argument("--length-unit", choices=("meter", "mm", "micron"))
    parser.add_argument(
        "--component", help="largest, or zero-based rank by descending voxel count"
    )
    parser.add_argument(
        "--break-cycles",
        action="store_true",
        help="Explicitly allow a minimum spanning tree",
    )
    root = parser.add_mutually_exclusive_group(required=True)
    root.add_argument(
        "--root-point",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        help="RAS metres; choose nearest endpoint",
    )
    root.add_argument(
        "--root-largest-radius",
        action="store_true",
        help="Exploratory heuristic; inspect before simulation",
    )
    for flag, help_text in {
        "young-modulus": "Wall Young modulus [Pa]",
        "wall-ratio": "Wall thickness / lumen radius [dimensionless]",
        "reference-pressure": "Wall reference pressure p_d [Pa]",
        "terminal-resistance": "Resistance R2 per terminal, not whole-organ resistance [Pa s/m3]",
        "outlet-pressure": "Terminal gauge/reference pressure [Pa]",
    }.items():
        parser.add_argument("--" + flag, type=float, required=True, help=help_text)
    parser.add_argument("--source-url", required=True)
    parser.add_argument("--source-license", required=True)
    args = parser.parse_args()
    try:
        build(args)
    except (ValueError, OSError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
