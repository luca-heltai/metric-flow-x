#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build a metric-preserving arterial ADAVN input for MetricFlowX (SI units).

The default geometry is a nonanatomical CSV-only metric embedding. Optional
atlas-derived geometry uses separately licensed surfaces supplied with permission.
Each CSV segment becomes two straight cells, with a unique material ID per cell
as required by MetricFlowX's property-array indexing and taper interpolation.
"""
import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

from recover_geometry import DEFAULT_DATA_DIR, read_rows, sample_curve, surface_centers


def wall(radius_m):
    """Müller et al. 2023, sections 2.4.1 and 2.4.2, arterial wall only."""
    r_cm = 100 * radius_m
    weights = (
        (0.85, 0.05, 0.10)
        if radius_m > 0.0018
        else ((0.65, 0.20, 0.15) if radius_m >= 0.0007 else (0.45, 0.0, 0.55))
    )
    ee, ec, km = 4e5 * weights[0], 1e8 * weights[1], 3.6e4 * weights[2]
    h = radius_m * (
        0.2802 * math.exp(-5.053 * r_cm) + 0.1324 * math.exp(-0.1114 * r_cm)
    )
    eps0, epsr, pref = 0.25, 0.05, 1e4
    f0 = ec * epsr * math.log1p(math.exp(-eps0 / epsr))
    tangent_factor = 0.5 * (ee + ec / (1 + math.exp(eps0 / epsr))) - f0
    # Match pressure and dP/dA of the elastic part at A=A_ref. The active
    # square-root law uses beta=4 sqrt(pi) E h / 3. Viscoelasticity is inactive.
    elastic_E = 1.5 * tangent_factor
    elastic_pd = pref + h / radius_m * f0
    return {
        "E": elastic_E,
        "h_wall": h,
        "p_d": elastic_pd,
        "adavn_E_e": ee,
        "adavn_E_c": ec,
        "adavn_K_m": km,
        "adavn_epsilon_0": eps0,
        "adavn_epsilon_r": epsr,
        "adavn_p_ref": pref,
    }


def reconcile_lengths(rows, points):
    initial = dict(points)
    edges = [(int(r[1]), int(r[2]), float(r[3]) * 0.01) for r in rows]
    # A straight chord cannot exceed the desired centerline length. Project
    # only those constraints, retaining the anatomical surface embedding.
    for iteration in range(10000):
        worst = 0.0
        for a, b, length in edges:
            delta = tuple(points[b][j] - points[a][j] for j in range(3))
            distance = math.sqrt(sum(d * d for d in delta))
            target = length * (1 - 1e-9)
            if distance > target:
                excess = distance - target
                worst = max(worst, excess / length)
                shift = tuple(0.5 * excess * d / distance for d in delta)
                points[a] = tuple(points[a][j] + shift[j] for j in range(3))
                points[b] = tuple(points[b][j] - shift[j] for j in range(3))
        if worst < 1e-11:
            break
    else:
        raise ValueError("Length constraints did not converge")
    displacements = sorted(math.dist(points[n], initial[n]) for n in points)
    return {
        "projection_iterations": iteration + 1,
        "max_node_displacement_m": displacements[-1],
        "median_node_displacement_m": displacements[len(displacements) // 2],
    }


def midpoint(a, b, desired_length, hint):
    delta = tuple(y - x for x, y in zip(a, b))
    distance = math.sqrt(sum(d * d for d in delta))
    if distance <= 1e-14 or distance > desired_length * (1 + 1e-10):
        raise ValueError("Invalid segment chord")
    unit = tuple(d / distance for d in delta)
    center = tuple((x + y) / 2 for x, y in zip(a, b))
    radial = tuple(x - y for x, y in zip(hint, center))
    axial = sum(x * y for x, y in zip(radial, unit))
    radial = tuple(radial[j] - axial * unit[j] for j in range(3))
    norm = math.sqrt(sum(x * x for x in radial))
    if norm < 1e-12:
        axis = min(range(3), key=lambda j: abs(unit[j]))
        radial = tuple(
            (1.0 if j == axis else 0.0) - unit[axis] * unit[j] for j in range(3)
        )
        norm = math.sqrt(sum(x * x for x in radial))
    height = 0.5 * math.sqrt(max(0.0, desired_length**2 - distance**2))
    return tuple(center[j] + height * radial[j] / norm for j in range(3))


def write_vtk(path, points, cells, cell_data, point_data, anatomical=False):
    with path.open("w") as stream:
        attribution = (
            "ADAVN arterial subset; Muller/Watanabe/Toro/Feijoo/Blanco 2023; "
            "doi:10.3389/fphys.2023.1162391; modified SI metric graph; "
        )
        license_notice = (
            "geometry: separate HeMoLab terms, see provenance"
            if anatomical
            else "CC BY 4.0 https://creativecommons.org/licenses/by/4.0/"
        )
        stream.write(
            "# vtk DataFile Version 3.0\n"
            + attribution
            + license_notice
            + "\nASCII\nDATASET UNSTRUCTURED_GRID\n"
        )
        stream.write(f"POINTS {len(points)} double\n")
        for p in points:
            stream.write(" ".join(f"{v:.16e}" for v in p) + "\n")
        stream.write(f"CELLS {len(cells)} {3*len(cells)}\n")
        for a, b in cells:
            stream.write(f"2 {a} {b}\n")
        stream.write(f"CELL_TYPES {len(cells)}\n" + "3\n" * len(cells))
        for kind, data, count in [
            ("CELL", cell_data, len(cells)),
            ("POINT", point_data, len(points)),
        ]:
            stream.write(f"{kind}_DATA {count}\n")
            for name, values in data.items():
                if len(values) != count or not all(math.isfinite(v) for v in values):
                    raise ValueError("Invalid array " + name)
                stream.write(f"SCALARS {name} double 1\nLOOKUP_TABLE default\n")
                for value in values:
                    stream.write(f"{value:.16e}\n")


def build(output, rho, p_out, root=DEFAULT_DATA_DIR):
    if not math.isfinite(rho) or rho <= 0 or not math.isfinite(p_out):
        raise ValueError("Density must be finite and positive; P_out must be finite")
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = read_rows(root)
    geometry = json.loads((root / "geometry_report.json").read_text())
    nodes = {int(k): tuple(v) for k, v in geometry["points_m"].items()}
    adjustment = reconcile_lengths(rows, nodes)
    node_ids = sorted(nodes)
    point_index = {n: i for i, n in enumerate(node_ids)}
    points = [nodes[n] for n in node_ids]
    degrees = Counter(int(row[i]) for row in rows for i in (1, 2))
    node_row = {
        int(row[i]): row for row in rows for i in (1, 2) if degrees[int(row[i])] == 1
    }
    roots = [node for node, row in node_row.items() if row[9] == "-"]
    if len(roots) != 1:
        raise ValueError("Expected one inlet without terminal resistance")
    inlet = roots[0]
    hints = {}
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[re.sub(r"_\d+$", "", row[0])].append((i, row))
    reversed_names = set(geometry.get("orientation_reversed", []))
    for name, group in groups.items():
        if geometry.get("mode") == "schematic":
            for i, row in group:
                a, b = nodes[int(row[1])], nodes[int(row[2])]
                hints[i] = tuple((x + y) / 2 for x, y in zip(a, b))
            continue
        curve, _ = surface_centers(root / "source/surfaces" / ("SW_" + name + ".vtk"))
        total = sum(float(row[3]) for _, row in group)
        fraction = 0.0
        for i, row in group:
            half_fraction = (fraction + 0.5 * float(row[3])) / total
            if name in reversed_names:
                half_fraction = 1 - half_fraction
            hints[i] = tuple(v * 0.01 for v in sample_curve(curve, half_fraction))
            fraction += float(row[3])
    cells = []
    cell_data = {
        name: []
        for name in [
            "vessel_id",
            "a0",
            "a_d",
            "E",
            "h_wall",
            "p_d",
            "p0",
            "L",
            "r_d",
            "r_in",
            "r_out",
            "source_segment_id",
            "source_region",
            "source_perforator",
            "adavn_E_e",
            "adavn_E_c",
            "adavn_K_m",
            "adavn_epsilon_0",
            "adavn_epsilon_r",
            "adavn_p_ref",
        ]
    }
    cell_mapping = []
    max_error = 0.0
    for i, row in enumerate(rows):
        a, b = int(row[1]), int(row[2])
        length = float(row[3]) * 0.01
        middle = midpoint(nodes[a], nodes[b], length, hints[i])
        middle_index = len(points)
        points.append(middle)
        radii = [
            float(row[4]) * 0.01,
            0.5 * (float(row[4]) + float(row[5])) * 0.01,
            float(row[5]) * 0.01,
        ]
        for half, pair in enumerate(
            [(point_index[a], middle_index), (middle_index, point_index[b])]
        ):
            vid = len(cells)
            cells.append(pair)
            r_in, r_out = radii[half : half + 2]
            radius = 0.5 * (r_in + r_out)
            area = math.pi * radius**2
            values = dict(
                wall(radius),
                vessel_id=vid,
                a0=area,
                a_d=area,
                p0=0.0,
                L=length / 2,
                r_d=radius,
                r_in=r_in,
                r_out=r_out,
                source_segment_id=i,
                source_region=int(row[7]),
                source_perforator=int(row[8]),
            )
            for name, array in cell_data.items():
                array.append(values[name])
            actual_length = math.dist(points[pair[0]], points[pair[1]])
            max_error = max(max_error, abs(actual_length - length / 2) / (length / 2))
        cell_mapping.append(
            {
                "source_segment_id": i,
                "source_name": row[0],
                "cell_ids": [2 * i, 2 * i + 1],
                "source_nodes": [a, b],
                "source_length_m": length,
            }
        )
    point_data = {name: [0.0] * len(points) for name in ["R1", "R2", "C", "P_out"]}
    point_data["boundary_id"] = [255.0] * len(points)
    point_data["boundary_id"][point_index[inlet]] = 0.0
    terminals = []
    next_bid = 1
    for node in sorted(node_row):
        if node == inlet:
            continue
        if next_bid == 255:
            next_bid += 1
        row = node_row[node]
        p = point_index[node]
        total_r, compliance = float(row[9]) * 1e5, float(row[10]) * 1e-5
        if (
            not math.isfinite(total_r)
            or total_r <= 0
            or not math.isfinite(compliance)
            or compliance < 0
        ):
            raise ValueError("Invalid terminal R/C at " + row[0])
        # The terminal may be the first CSV node: connectivity orientation
        # is not assumed to equal the physiological flow direction.
        r = float(row[4 if int(row[1]) == node else 5]) * 0.01
        w = wall(r)
        c = math.sqrt(2 * w["E"] * w["h_wall"] / (3 * rho * r))
        r1 = rho * c / (math.pi * r * r) if compliance > 0 else 0.0
        if r1 >= total_r:
            raise ValueError("Characteristic impedance exceeds total R at " + row[0])
        for name, value in [
            ("R1", r1),
            ("R2", total_r - r1),
            ("C", compliance),
            ("P_out", p_out),
            ("boundary_id", next_bid),
        ]:
            point_data[name][p] = value
        terminals.append(
            {
                "boundary_id": next_bid,
                "source_node": node,
                "source_name": row[0],
                "R1_Pa_s_m3": r1,
                "R2_Pa_s_m3": total_r - r1,
                "C_m3_Pa": compliance,
                "P_out_Pa": p_out,
                "source_total_R_Pa_s_m3": total_r,
                "vascular_territory": row[11],
                "source_first_connection": row[12],
                "source_second_connection": row[13],
            }
        )
        next_bid += 1
    write_vtk(
        output,
        points,
        cells,
        cell_data,
        point_data,
        geometry.get("mode") == "anatomical",
    )
    output.with_suffix(".mapping.json").write_text(
        json.dumps(cell_mapping, indent=2) + "\n"
    )
    with output.with_suffix(".terminals.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(terminals[0]))
        writer.writeheader()
        writer.writerows(terminals)
    source_metadata = json.loads(Path(__file__).with_name("sources.json").read_text())
    manifest = {
        "source_article_doi": "10.3389/fphys.2023.1162391",
        "source_csv_url": source_metadata["csv"]["url"],
        "original_publisher_url": source_metadata["csv"]["publisher_url"],
        "source_csv_sha256": hashlib.sha256(
            (root / "source/data_sheet_1.csv").read_bytes()
        ).hexdigest(),
        "vtk_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "units": "SI: m, m2, Pa, Pa s/m3, m3/Pa, Pa s",
        "anatomical_artery_names": len(groups),
        "source_arterial_segments": len(rows),
        "source_nodes": len(nodes),
        "vtk_cells": len(cells),
        "vtk_points": len(points),
        "terminals": len(terminals),
        "inlet_source_node": inlet,
        "inlet_boundary_id": 0,
        "interior_boundary_id": 255,
        "density_kg_m3": rho,
        "max_relative_cell_length_error": max_error,
        "geometry_adjustment": adjustment,
        "geometry": (
            "CSV-only nonanatomical metric embedding; two-line exact-length reconstruction."
            if geometry.get("mode") == "schematic"
            else "Reconstructed from surface rings, shared-node averaging, chord constraints and two-line exact-length reconstruction. Original splines unavailable."
        ),
        "geometry_mode": geometry.get("mode", "anatomical"),
        "surface_source": (
            None
            if geometry.get("mode") == "schematic"
            else "http://hemolab.lncc.br/adan-web/"
        ),
        "source_data_license": "CC-BY-4.0",
        "source_data_license_url": "https://creativecommons.org/licenses/by/4.0/",
        "attribution": "Müller, Watanabe, Toro, Feijóo and Blanco (2023), doi:10.3389/fphys.2023.1162391. Modified: arterial extraction, SI conversion, metric embedding and adapted wall/terminal laws.",
        "active_wall_law": "Existing square-root law matched to elastic ADAVN pressure and tangent at reference area; viscoelastic term inactive.",
        "future_wall_arrays": [
            "adavn_E_e",
            "adavn_E_c",
            "adavn_K_m",
            "adavn_epsilon_0",
            "adavn_epsilon_r",
            "adavn_p_ref",
        ],
        "terminal_adaptation": "Original total R and C retained. R1=rho*c/A; R2=R_total-R1. C=0 uses R1=0. Fixed P_out replaces source venous/heart coupling.",
        "limitations": [
            "Arterial subset of ADAVN 2023, not the original 2015 version.",
            "Wall coefficients are local cell-center values; radius-dependent mixture groups can change along a source segment.",
            "Reconstructed centerlines retain CSV lengths but not original spline curvature or exact node positions.",
            "Coronary/extracranial/intracranial boundary coupling replaced by fixed-pressure RCR terminals.",
            "No physiological validation or full-network time integration performed.",
        ],
    }
    if geometry.get("mode") == "anatomical":
        manifest["source_data_license"] = (
            "CSV: CC-BY-4.0; surface-derived geometry: separate HeMoLab terms, written permission required."
        )
    output.with_suffix(".provenance.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(manifest, indent=2))
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--density", type=float, default=1040.0)
    parser.add_argument(
        "--p-out", type=float, default=0.0, help="Fixed downstream pressure [Pa]"
    )
    args = parser.parse_args()
    if not args.density > 0 or not math.isfinite(args.p_out):
        parser.error("Density must be positive and downstream pressure finite")
    build(
        args.output or args.data_dir / "adan_complete.vtk",
        args.density,
        args.p_out,
        args.data_dir,
    )
