#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Check the native mean solve, compare MAP and write the complete result VTK."""
import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

from build_vtk import write_vtk

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from validate_vtk_network import parse_vtk

MMHG_PA = 133.322435415


def partial_resistance_fraction(r0, r1, fraction):
    """Integral of 1/r(s)^4 on [0,fraction] divided by its full integral."""
    middle = r0 + fraction*(r1-r0)
    return fraction * (r0*r0+r0*middle+middle*middle) * r1**3 / (
        middle**3 * (r0*r0+r0*r1+r1*r1))


def analyze(root):
    mapping = json.loads((root/"mapping.json").read_text())
    provenance = json.loads((root/"provenance.json").read_text())
    for name, digest in provenance["sha256"].items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest() != digest:
            raise ValueError("Prepared input has changed: " + name)
    with (root/"nodes.csv").open() as stream:
        nodes = {int(row["node"]): row for row in csv.DictReader(stream)}
    with (root/"flows.csv").open() as stream:
        flows = {int(row["edge"]): float(row["flow_m3_s"]) for row in csv.DictReader(stream)}
    with (root/"edges.csv").open() as stream:
        edges = list(csv.DictReader(stream))
    if set(nodes) != set(range(provenance["hydraulic_nodes"])) or set(flows) != set(range(len(edges))):
        raise ValueError("Native output does not cover every graph node and edge")
    pressures = {node: float(row["pressure_Pa"]) for node, row in nodes.items()}
    if not all(math.isfinite(x) for x in list(pressures.values())+list(flows.values())):
        raise ValueError("Nonfinite result")
    log_lines = (root/"mean.log").read_text().splitlines()
    metrics = json.loads(next(line.removeprefix("MEAN_FLOW ") for line in log_lines
                             if line.startswith("MEAN_FLOW ")))
    # Independently reconstruct nodal conservation from the exported edge flows.
    balance = dict.fromkeys(nodes, 0.)
    for i, edge in enumerate(edges):
        balance[int(edge["node0"])] += flows[i]
        balance[int(edge["node1"])] -= flows[i]
    interior_error = max(abs(q) for node, q in balance.items()
                         if node != provenance["inlet_node"] and node not in provenance["central_drain_nodes"])
    inflow = metrics["inflow_m3_s"]
    if abs(balance[provenance["inlet_node"]]-inflow) > 1e-8*abs(inflow) or interior_error > 1e-8*abs(inflow):
        raise ValueError("Independent conservation check failed")
    if abs(metrics["outflow_m3_s"]-inflow) > 1e-8*abs(inflow):
        raise ValueError("Inlet/outlet balance failed")
    mesh = parse_vtk(root/"adavn_full.vtk")
    p = [pressures[int(node)] if node >= 0 else 0.
         for node in mesh.point_arrays["source_node_id"]]
    for i, row in enumerate(mapping):
        pa, pb = pressures[row["node0"]], pressures[row["node1"]]
        p[provenance["source_nodes"]+i] = pa + (pb-pa)*partial_resistance_fraction(
            row["radius0_m"], row["radius1_m"], .5)
    mesh.point_arrays["mean_pressure_Pa"] = p
    mesh.cell_arrays["flow_m3_s"] = [flows[int(i)] for i in mesh.cell_arrays["source_row"]]
    mesh.cell_arrays["mean_velocity_m_s"] = [q/a for q, a in zip(
        mesh.cell_arrays["flow_m3_s"], mesh.cell_arrays["a_d"])]
    result_vtk = root/"arteries_veins_mean.vtk"
    write_vtk(result_vtk, mesh.points, mesh.cells, mesh.cell_arrays, mesh.point_arrays,
              anatomical=provenance.get("geometry_mode") == "anatomical",
              description="ADAVN full arterial-venous resistive mean-flow result")
    # Article vessel indexes are one based. Do not search by duplicated names.
    aorta = mapping[815]
    if aorta["name"] != "A_aortic_arch_C_815":
        raise ValueError("Unexpected identity for paper vessel index 816")
    prediction = p[provenance["source_nodes"]+815]/MMHG_PA
    report = dict(model=provenance["mean_model"], source_doi=provenance["source_doi"],
                  imposed_inputs=dict(cardiac_output_ml_s=inflow*1e6,
                                      right_atrial_reservoir_pressure_mmHg=623.949/MMHG_PA),
                  MAP=dict(paper_vessel_index=816, source_row=815, source_name=aorta["name"],
                           paper_table3_mmHg=92.58, predicted_mmHg=prediction,
                           relative_difference=(prediction-92.58)/92.58),
                  independent_max_interior_balance_m3_s=interior_error,
                  native_metrics=metrics, omitted_physics=provenance["omitted_physics"],
                  result_vtk_sha256=hashlib.sha256(result_vtk.read_bytes()).hexdigest(),
                  interpretation="Reduced-model comparison to a published mean value; not reproduction of the complete closed-loop pulsatile model or its waveforms.")
    (root/"report.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("build/adavn_full"))
    args = parser.parse_args()
    analyze(args.data_dir)
