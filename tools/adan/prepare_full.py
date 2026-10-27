#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Prepare all ADAVN vessels and the coupled, reference-geometry mean model.

No atlas geometry is used.  Source row numbers are zero based; the paper's
one-based vessel index 816 corresponds to source row 815.
"""
import argparse
import csv
import hashlib
import heapq
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from build_vtk import midpoint, wall, write_vtk
from recover_geometry import DEFAULT_DATA_DIR, read_rows


def terminal_node(row, degree):
    ends = [int(row[j]) for j in (1, 2) if degree[int(row[j])] == 1]
    if len(ends) != 1:
        raise ValueError(f"Expected one terminal endpoint: {row[0]}")
    return ends[0]


def endpoint_radius(row, node):
    return float(row[4 if int(row[1]) == node else 5]) * 0.01


def tube_resistance(row):
    """Exact integral for a linearly tapered Poiseuille tube, SI units.

    The paper uses 0.001 Pa s in arterial perforators and 0.004 elsewhere.
    """
    length, a, b = (float(row[j]) * 0.01 for j in (3, 4, 5))
    mu = 0.001 if row[6] == "1" and row[8] == "1" else 0.004
    return 8 * mu * length * (a*a + a*b + b*b) / (3 * math.pi * a**3 * b**3)


def venous_compartments(rows, wave_speed):
    """Residual systemic compliance and proximal venous impedance, §2.4.5.

    The general-bed reduction is also used for coronary terminals: myocardial
    forcing and layered coronary storage remain outside this tutorial model.
    """
    degree = Counter(int(row[j]) for row in rows for j in (1, 2))
    veins = {row[12]: i for i, row in enumerate(rows)
             if row[6] == "0" and row[12] != "-"}
    shares = dict.fromkeys(veins.values(), 0.)
    for row in rows:
        if row[6] != "1" or row[9] == "-":
            continue
        destinations = [veins[code] for code in row[12:14] if code != "-"]
        weights = [endpoint_radius(rows[j], terminal_node(rows[j], degree))**3
                   for j in destinations]
        for j, weight in zip(destinations, weights):
            shares[j] += float(row[10])*1e-5*weight/sum(weights)
    volume = sum(math.pi*(float(row[3])*.01)*sum(
        (float(row[4])*.01)**(2-k)*(float(row[5])*.01)**k for k in range(3))/3
        for row in rows if row[6] == "0")
    residual = 146e-6/133.322435415-volume/(1040*wave_speed**2)
    if residual <= 0 or sum(shares.values()) <= 0:
        raise ValueError("Nonpositive residual venous compliance or arterial shares")
    result = {}
    for j in sorted(shares):
        node = terminal_node(rows[j], degree)
        radius = endpoint_radius(rows[j], node)
        result[j] = dict(node=node, resistance=1040*wave_speed/(math.pi*radius**2),
                         compliance=residual*shares[j]/sum(shares.values()))
    return result


def hydraulic_graph(rows, venous_wave_speed=2.):
    degree = Counter(int(row[j]) for row in rows for j in (1, 2))
    arterial = [i for i, row in enumerate(rows) if row[6] == "1"]
    venous = [i for i, row in enumerate(rows) if row[6] == "0"]
    veins = {}
    for i in venous:
        row = rows[i]
        if row[12] != "-":
            if row[12] in veins:
                raise ValueError("Repeated venous connection code " + row[12])
            terminal_node(row, degree)
            veins[row[12]] = i
    roots = [terminal_node(rows[i], degree) for i in arterial
             if rows[i][9] == "-" and any(degree[int(rows[i][j])] == 1 for j in (1, 2))]
    drains = [terminal_node(rows[i], degree) for i in venous
              if rows[i][9] == "-" and any(degree[int(rows[i][j])] == 1 for j in (1, 2))]
    if len(roots) != 1 or not drains:
        raise ValueError("Expected one arterial inlet and venous central drains")
    edges = [dict(node0=int(row[1]), node1=int(row[2]),
                  resistance_Pa_s_m3=tube_resistance(row),
                  source_row=i, draining_source_row=-1, kind="tube")
             for i, row in enumerate(rows)]
    next_node = max(degree)+1
    venous_nodes = {}
    for j, compartment in venous_compartments(rows, venous_wave_speed).items():
        venous_nodes[j] = next_node
        edges.append(dict(node0=next_node, node1=compartment["node"],
                          resistance_Pa_s_m3=compartment["resistance"],
                          source_row=j, draining_source_row=-1, kind="venous_proximal"))
        next_node += 1
    for i in arterial:
        row = rows[i]
        if row[9] == "-":
            continue
        source = terminal_node(row, degree)
        codes = [code for code in row[12:14] if code != "-"]
        if not codes or len(codes) != len(set(codes)):
            raise ValueError("Invalid arterio-venous connections: " + row[0])
        destinations = [veins[code] for code in codes]
        weights = [endpoint_radius(rows[j], terminal_node(rows[j], degree))**3
                   for j in destinations]
        total_r = float(row[9]) * 1e5
        if not math.isfinite(total_r) or total_r <= 0:
            raise ValueError("Invalid peripheral resistance: " + row[0])
        # Section 2.4.5: 15% before the arterial compartment, 85% in
        # the equivalent downstream resistor. The static graph represents
        # the compartment explicitly, including crossflow between drains.
        fraction = .15 if float(row[10]) > 0 else 0.
        compartment = source
        if fraction:
            compartment = next_node
            next_node += 1
            edges.append(dict(node0=source, node1=compartment,
                              resistance_Pa_s_m3=fraction*total_r,
                              source_row=i, draining_source_row=-1, kind="proximal"))
        # Parallel conductances are weighted by cubed venous endpoint radii.
        for j, weight in zip(destinations, weights):
            edges.append(dict(node0=compartment,
                              node1=venous_nodes[j],
                              resistance_Pa_s_m3=(1-fraction)*total_r * sum(weights) / weight,
                              source_row=i, draining_source_row=j, kind="connection"))
    return edges, roots[0], drains, degree


def metric_geometry(rows, root, drains):
    """Embed each 1D component, preserving source lengths with two-line bends."""
    adjacency = defaultdict(list)
    for row in rows:
        a, b, length = int(row[1]), int(row[2]), float(row[3]) * 0.01
        adjacency[a].append((b, length))
        adjacency[b].append((a, length))
    amplitude = 0.01 * min(float(row[3]) * 0.01 for row in rows)
    points = {}
    for component, start in enumerate([root] + sorted(drains)):
        if start in points:
            raise ValueError("Two central endpoints belong to the same 1D component")
        distance = {start: 0.0}
        queue = [(0.0, start)]
        while queue:
            d, a = heapq.heappop(queue)
            if d > distance[a]:
                continue
            for b, length in adjacency[a]:
                if d + length < distance.get(b, math.inf):
                    distance[b] = d + length
                    heapq.heappush(queue, (d + length, b))
        for node, d in distance.items():
            points[node] = (0.9 * d,
                            0.1 * component + amplitude * math.sin(node * 2.399963229728653),
                            amplitude * math.cos(node * 2.399963229728653))
    if set(points) != set(adjacency):
        raise ValueError("Unanchored 1D component in source")
    return points


def write_csv(path, records):
    with path.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def prepare(root, output, venous_wave_speed=2.):
    if not math.isfinite(venous_wave_speed) or venous_wave_speed <= 0:
        raise ValueError("Venous reference wave speed must be finite and positive")
    source = root / "source/data_sheet_1.csv"
    metadata = json.loads(Path(__file__).with_name("sources.json").read_text())
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != metadata["csv"]["sha256"]:
        raise ValueError("Full ADAVN preparation requires the checksum-pinned source CSV")
    rows = read_rows(root, include_veins=True)
    edges, inlet, drains, degree = hydraulic_graph(rows, venous_wave_speed)
    nodes = metric_geometry(rows, inlet, drains)
    output.mkdir(parents=True, exist_ok=True)
    node_ids = sorted(nodes)
    index = {node: i for i, node in enumerate(node_ids)}
    # The native graph indexes nodes consecutively. Keep original IDs in all
    # generated CSV files; the published source happens to be consecutive.
    if node_ids != list(range(len(node_ids))):
        raise ValueError("Source node IDs must be consecutive from zero")
    points = [nodes[node] for node in node_ids]
    cells, records = [], []
    max_length_error = 0.
    for i, row in enumerate(rows):
        a, b = int(row[1]), int(row[2])
        length = float(row[3]) * 0.01
        hint = tuple((x+y)/2 for x, y in zip(nodes[a], nodes[b]))
        mid = len(points)
        points.append(midpoint(nodes[a], nodes[b], length, hint))
        radii = [float(row[4])*.01,
                 (float(row[4])+float(row[5]))*.005, float(row[5])*.01]
        for half, pair in enumerate([(index[a], mid), (mid, index[b])]):
            cells.append(pair)
            r_in, r_out = radii[half:half+2]
            radius = (r_in+r_out)/2
            properties = wall(radius)
            properties.update(adavn_wall_type=1, adavn_venous_K=0.,
                              adavn_venous_m=0., adavn_venous_n=0.)
            if row[6] == "0":
                # The source table does not supply c_ven. Use an explicit
                # constant reference c, rather than inventing c_min/c_max
                # values for the paper's radius-dependent equation (44).
                dural = "sinus" in row[0] and "coronary_sinus" not in row[0]
                m, n = (.5, 0.) if dural else (10., -1.5)
                properties.update(h_wall=0.1*radius,
                                  E=1.5*1040*venous_wave_speed**2/0.1,
                                  p_d=666.666, adavn_wall_type=3 if dural else 2,
                                  adavn_p_ref=666.666,
                                  adavn_venous_K=1040*venous_wave_speed**2/(m-n),
                                  adavn_venous_m=m, adavn_venous_n=n,
                                  adavn_K_m=70.898*radius,
                                  adavn_E_e=0., adavn_E_c=0.,
                                  adavn_epsilon_0=0., adavn_epsilon_r=0.)
            record = dict(properties)
            record.update(vessel_id=len(cells)-1, a0=math.pi*radius**2,
                          a_d=math.pi*radius**2, p0=0., L=length/2,
                          r_d=radius, r_in=r_in, r_out=r_out,
                          source_row=i, source_vessel_type=int(row[6]),
                          source_region=int(row[7]), source_perforator=int(row[8]),
                          viscosity=.001 if row[6] == "1" and row[8] == "1" else .004)
            records.append(record)
            error = abs(math.dist(points[pair[0]], points[pair[1]]) - length/2)/(length/2)
            max_length_error = max(max_length_error, error)
    cell_data = {key: [row[key] for row in records] for key in records[0]}
    point_data = {key: [0.]*len(points) for key in ("R1", "R2", "C", "P_out")}
    point_data["boundary_id"] = [255.]*len(points)
    point_data["source_node_id"] = node_ids + [-1.] * len(rows)
    boundary = 0
    boundary_ids = {inlet: 0}
    for node in node_ids:
        if degree[node] != 1:
            continue
        if node == inlet:
            point_data["boundary_id"][index[node]] = 0.
            continue
        boundary += 1
        if boundary == 255:
            boundary += 1
        point_data["boundary_id"][index[node]] = boundary
        boundary_ids[node] = boundary
        point_data["R2"][index[node]] = 1e5 if node in drains else 1e9
        point_data["P_out"][index[node]] = 623.949 if node in drains else 0.
    for row in rows:
        if row[6] == "1" and row[9] != "-":
            node = terminal_node(row, degree)
            total_r = float(row[9])*1e5
            fraction = .15 if float(row[10]) > 0 else 0.
            point_data["R1"][index[node]] = fraction*total_r
            point_data["R2"][index[node]] = (1-fraction)*total_r
            point_data["C"][index[node]] = float(row[10])*1e-5
    compartments = venous_compartments(rows, venous_wave_speed)
    for data in compartments.values():
        point_data["R1"][index[data["node"]]] = data["resistance"]
        point_data["C"][index[data["node"]]] = data["compliance"]
    write_vtk(output / "adavn_full.vtk", points, cells, cell_data, point_data,
              description="ADAVN full arterial-venous metric graph")
    write_csv(output / "edges.csv", edges)
    write_csv(output / "connections.csv", [dict(boundary0=boundary_ids[terminal_node(rows[edge["source_row"]], degree)],
               boundary1=boundary_ids[terminal_node(rows[edge["draining_source_row"]], degree)], resistance=edge["resistance_Pa_s_m3"])
               for edge in edges if edge["kind"] == "connection"])
    write_csv(output / "outlets.csv", [dict(node=node) for node in sorted(drains)])
    mapping = [dict(source_row=i, paper_vessel_index=i+1, name=row[0],
                    node0=int(row[1]), node1=int(row[2]),
                    boundary0=boundary_ids.get(int(row[1]), 255),
                    boundary1=boundary_ids.get(int(row[2]), 255),
                    length_m=float(row[3])*.01, radius0_m=float(row[4])*.01,
                    radius1_m=float(row[5])*.01, vessel_type=int(row[6]),
                    region=int(row[7]), perforator=int(row[8]),
                    compliance_m3_Pa=float(row[10])*1e-5 if row[10] != "-" else 0.,
                    territory=row[11], connections=row[12:14])
               for i, row in enumerate(rows)]
    (output / "mapping.json").write_text(json.dumps(mapping, indent=2)+"\n")
    manifest = dict(source_doi="10.3389/fphys.2023.1162391", source_sha256=digest,
                    license="CC-BY-4.0", license_url=metadata["article_license_url"],
                    attribution="Müller, Watanabe, Toro, Feijóo and Blanco (2023). Changes: full CSV extraction, SI conversion, metric embedding, resistive mean-flow reduction.",
                    arterial_source_rows=sum(row[6] == "1" for row in rows),
                    venous_source_rows=sum(row[6] == "0" for row in rows),
                    source_nodes=len(node_ids), vtk_points=len(points), vtk_cells=len(cells),
                    inlet_node=inlet, central_drain_nodes=sorted(drains),
                    peripheral_connections=sum(edge["kind"] == "connection" for edge in edges),
                    hydraulic_nodes=max(max(edge["node0"], edge["node1"]) for edge in edges)+1,
                    max_relative_cell_length_error=max_length_error,
                    geometry="Nonanatomical metric embedding of four source 1D components; peripheral graph edges carry no spatial geometry.",
                    duplicates_in_source_names=len(rows)-len(set(row[0] for row in rows)),
                    mean_model="Reference-geometry Poiseuille vessels, 15% source peripheral R before each arterial compartment, remaining 85% split in parallel by draining-vein endpoint radius cubed, venous endpoint impedances, prescribed CO and right-atrial reservoir pressure.",
                    omitted_physics=["compliant vessel area response", "heart and pulmonary circulation", "pulsatile inertia", "peripheral storage", "coronary intramyocardial forcing", "intracranial-pressure feedback", "valves and Starling resistors"],
                    hdg_model="ADAVN mixed elastic walls, source arterial compliances, proximal R1=15% of source total R, remaining R coupled to residual venous compartments and impedance-matched 1D venous ports. Prescribed inflow and three right-atrial reservoir outlets; the transient tutorial uses a smooth 0.8 s pulse with mean 96.36 ml/s.",
                    hdg_venous_reference_wave_speed_m_s=venous_wave_speed,
                    residual_venous_compliance_m3_Pa=sum(data["compliance"] for data in compartments.values()),
                    venous_compartments=len(compartments),
                    coronary_reduction="General-bed terminal model also used for coronary ends; no layered myocardial model.",
                    hdg_dural_classification="Venous name contains sinus, excluding coronary_sinus; explicit approximation, not an additional source-table label.",
                    hdg_omissions=["Kelvin-Voigt wall viscosity", "radius-dependent venous reference wave speed", "heart/pulmonary closed loop", "coronary bed forcing", "ICP feedback", "venous valves/Starling resistors"])
    manifest["generator_sha256"] = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                       for name in ("prepare_full.py", "build_vtk.py", "recover_geometry.py")}
    manifest["sha256"] = {name: hashlib.sha256((output/name).read_bytes()).hexdigest()
                          for name in ("adavn_full.vtk", "edges.csv", "outlets.csv", "mapping.json", "connections.csv")}
    (output / "provenance.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(json.dumps(manifest, indent=2))
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=Path("build/adavn_full"))
    parser.add_argument("--venous-wave-speed", type=float, default=2.,
                        help="Explicit constant c_ven for the HDG model [m/s]; does not affect the resistive mean calculation")
    args = parser.parse_args()
    prepare(args.data_dir, args.output_dir, args.venous_wave_speed)
