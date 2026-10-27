#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Export a completed native HDG run onto the connected full-network VTK.

Requires numpy and meshio. The original graph connectivity and source fields
are retained; result CELL_DATA are native patch endpoint averages. The native
VTU files retain the discontinuous fields and should be used for detailed work.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import meshio
import numpy as np

from build_vtk import write_vtk
from analyze_full import MMHG_PA
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from validate_vtk_network import parse_vtk


def export(root, run, output, expected_final_time=8.):
    source = parse_vtk(root/"adavn_full.vtk")
    provenance = json.loads((root/"provenance.json").read_text())
    stationary = (run/"stationary_statistics.json").exists()
    if stationary:
        receipt = json.loads((run/"stationary_statistics.json").read_text())
        if receipt["weighted_newton_defect"] >= 1 or receipt["derivative_norm"] != 0:
            raise ValueError("Stationary run has not passed its native residual check")
    elif (run/"ida_statistics.csv").exists():
        with (run/"ida_statistics.csv").open() as stream:
            receipts = list(csv.DictReader(stream))
        if len(receipts) != 1:
            raise ValueError("Use a fresh directory with exactly one completed integration receipt")
        receipt = receipts[0]
        if int(receipt["n_steps"]) < 1:
            raise ValueError("No completed time integration")
    else:
        raise ValueError("A completed native run receipt is required; partial/graph-only output is not accepted")
    records = ET.parse(run/"adavn_full.pvd").getroot().findall(".//DataSet")
    final = max(records, key=lambda item: float(item.attrib["timestep"]))
    if not math.isfinite(expected_final_time) or expected_final_time <= 0 or not math.isclose(
            float(final.attrib["timestep"]), expected_final_time, rel_tol=1e-10, abs_tol=1e-10):
        raise ValueError("Native output did not reach the expected final time")
    parallel_file = run/final.attrib["file"]
    pieces = [parallel_file.parent/piece.attrib["Source"]
              for piece in ET.parse(parallel_file).getroot().findall(".//Piece")]
    samples = {i: [] for i in range(len(source.cells))}
    minimum_area = math.inf
    for path in pieces:
        mesh = meshio.read(path)
        for name in ("area", "velocity", "pressure", "A_hat", "U_hat", "vessel_id"):
            if not np.isfinite(mesh.point_data[name]).all():
                raise ValueError("Nonfinite native result: " + name)
        minimum_area = min(minimum_area, float(mesh.point_data["area"].min()))
        for block in mesh.cells:
            if block.type != "line":
                raise ValueError("Expected linear VTK patches along the 1D network")
            for pair in block.data:
                vid = int(round(float(mesh.point_data["vessel_id"][pair[0]])))
                a, b = (np.array(source.points[index]) for index in source.cells[vid])
                direction = b-a
                orientation = 1. if np.dot(mesh.points[pair[1]]-mesh.points[pair[0]], direction) > 0 else -1.
                for index in pair:
                    xi = float(np.dot(mesh.points[index]-a, direction)/np.dot(direction, direction))
                    samples[vid].append((xi, {name: float(mesh.point_data[name][index])
                                             for name in ("area", "velocity", "pressure", "A_hat", "U_hat")}, orientation))
    if minimum_area <= 0 or not all(samples.values()):
        raise ValueError("The result must cover every source cell with positive area")
    arrays = {name: [] for name in ("area_m2", "velocity_m_s", "pressure_Pa",
                                    "flow_m3_s", "trace_flow_start_m3_s", "trace_flow_end_m3_s")}
    for vid in range(len(source.cells)):
        ordered = sorted(samples[vid], key=lambda entry: entry[0])
        first, last = ordered[0], ordered[-1]
        if abs(first[0]) > .02 or abs(last[0]-1) > .02:
            raise ValueError("Native geometry does not span source cell " + str(vid))
        ends = (first, last)
        arrays["area_m2"].append(sum(item[1]["area"] for item in ends)/2)
        arrays["velocity_m_s"].append(sum(item[1]["velocity"]*item[2] for item in ends)/2)
        arrays["pressure_Pa"].append(sum(item[1]["pressure"] for item in ends)/2)
        arrays["flow_m3_s"].append(sum(item[1]["area"]*item[1]["velocity"]*item[2] for item in ends)/2)
        for name, item in zip(("trace_flow_start_m3_s", "trace_flow_end_m3_s"), ends):
            arrays[name].append(item[1]["A_hat"]*item[1]["U_hat"]*item[2])
    source.cell_arrays.update(arrays)
    write_vtk(output, source.points, source.cells, source.cell_arrays, source.point_arrays,
              description="ADAVN full arterial-venous native HDG result; endpoint averages")
    probes = []
    probe_csv = run/"probes.csv"
    if not probe_csv.exists():
        probe_csv = run/"vessel_probes.csv"
    if probe_csv.exists():
        with probe_csv.open() as stream:
            table = list(csv.DictReader(stream))
        for vid in (1630, 1631):
            probes.append([row for row in table if int(row["vessel_id"]) == vid][-1])
        aortic_pressure = sum(float(row["pressure_Pa"]) for row in probes)/2/MMHG_PA
    else:
        # Historical receipts from before the consolidated midpoint output.
        for vid in (1630, 1631):
            with (run/f"HDG_IDA_Vessel_{vid}.csv").open() as stream:
                probes.append(list(csv.DictReader(stream))[-1])
        aortic_pressure = sum(float(row["P_dynpcm2"])/10 for row in probes)/2/MMHG_PA
    report = dict(time=float(final.attrib["timestep"]), native_receipt=receipt,
                  cells=len(source.cells), source_arterial_rows=provenance["arterial_source_rows"],
                  source_venous_rows=provenance["venous_source_rows"],
                  minimum_native_area_m2=minimum_area,
                  aortic_pressure_mmHg=aortic_pressure,
                  pressure_sampling="Mean of probes at the two half-segment midpoints of paper vessel 816 (source row 815).",
                  model=provenance["hdg_model"], omitted_physics=provenance["hdg_omissions"],
                  vtk_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
                  source_csv_sha256=provenance["source_sha256"],
                  native_vtu_sha256={path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in pieces})
    if stationary:
        report["MAP_comparison"] = dict(paper_table3_mmHg=92.58,
            relative_difference=(aortic_pressure-92.58)/92.58,
            interpretation="Open-loop stationary pressure compared to the published closed-loop cardiac-cycle mean; not waveform reproduction.")
    (output.with_suffix(".report.json")).write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("build/adavn_full"))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("build/adavn_full/arteries_veins_hdg.vtk"))
    parser.add_argument("--expected-final-time", type=float, default=8.,
                        help="Require this completed final time [s], default ten 0.8 s cycles")
    args = parser.parse_args()
    export(args.data_dir, args.run_dir, args.output, args.expected_final_time)
