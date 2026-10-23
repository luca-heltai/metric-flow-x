#!/usr/bin/env python3
"""Compare the single-vessel inflow regression in the solver's physical units.

Only scalar global IDA tolerances and the ASCII single-vessel VTK fixture are
supported. Adaptive IDA grids/counters and normalized mass imbalance are
reported as diagnostics, not compared with state-unit error weights.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

TIME_BOUND = 1e-14
CASES = ("analytic-1", "analytic-2", "csv-1", "csv-2")
PHYSICAL = {"HDG_IDA_Vessel_0.csv", "terminal_0.csv", "terminal_1.csv", "mass_conservation.csv", "network_terminals.csv"}
DIAGNOSTIC = {"ida_internal_steps.csv", "ida_statistics.csv", "ida_step_history.csv"}


def finite(value, context="number"):
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("Nonfinite " + context)
    return result


def file_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_vtk(path):
    """Read only the known one-cell, two-point, scalar-only VTK layout."""
    lines = path.read_text(encoding="ascii").splitlines()
    if len(lines) < 4 or lines[2].strip() != "ASCII" or lines[3].strip() != "DATASET UNSTRUCTURED_GRID":
        raise ValueError("Unsupported VTK header")
    tokens = iter(" ".join(lines[4:]).split())
    def take():
        try:
            return next(tokens)
        except StopIteration as error:
            raise ValueError("Truncated VTK fixture") from error
    def expect(value):
        if take() != value:
            raise ValueError("Unsupported VTK layout, expected " + value)
    expect("POINTS"); expect("2")
    if take() not in {"float", "double"}:
        raise ValueError("Unsupported VTK point type")
    points = [[finite(take(), "VTK coordinate") for _ in range(3)] for _ in range(2)]
    for token in ("CELLS", "1", "3", "2", "0", "1", "CELL_TYPES", "1", "3", "CELL_DATA", "1"):
        expect(token)
    cells, point_data = {}, {}
    current, count = cells, 1
    for token in tokens:
        if token == "POINT_DATA" and current is cells:
            expect("2"); current, count = point_data, 2
            continue
        if token != "SCALARS":
            raise ValueError("Unsupported VTK scalar layout")
        name, kind, components = take(), take(), take()
        if name in current or kind not in {"int", "float", "double"} or components != "1":
            raise ValueError("Unsupported or duplicate VTK scalar")
        expect("LOOKUP_TABLE"); expect("default")
        current[name] = [finite(take(), "VTK scalar") for _ in range(count)]
    if set(cells) != {"vessel_id", "a0", "E", "h_wall", "p_d", "p0", "L", "a_d", "r_d"} or set(point_data) != {"boundary_id", "R1", "R2", "C", "P_out"}:
        raise ValueError("Unsupported VTK fixture fields")
    if cells["vessel_id"] != [0.] or point_data["boundary_id"] != [0., 1.]:
        raise ValueError("Unsupported vessel/boundary numbering")
    material = {key: values[0] for key, values in cells.items()}
    if any(material[key] <= 0 for key in ("E", "h_wall", "a_d", "L")):
        raise ValueError("Nonpositive material property")
    length = finite(math.dist(points[0], points[1]), "segment length")
    if abs(length - material["L"]) > 32 * math.ulp(1.) * max(1., length):
        raise ValueError("VTK length differs from segment geometry")
    material["beta"] = finite((4 * math.sqrt(math.pi) / 3) * material["E"] * material["h_wall"], "stiffness")
    if material["beta"] <= 0:
        raise ValueError("Nonpositive representable stiffness")
    material["model"] = {"points": points, "cells": cells, "point_data": point_data}
    material["path"] = str(path)
    material["sha256"] = file_sha(path)
    return material


def read_parameters(case):
    path = case / "last_used_parameters.prm"
    stack, values = [], {}
    for raw in path.read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("subsection "):
            stack.append(line[len("subsection "):].strip())
        elif line == "end":
            if not stack:
                raise ValueError("Unbalanced parameter subsections")
            stack.pop()
        elif line.startswith("set ") and "=" in line:
            name, value = line[4:].split("=", 1)
            key = tuple(stack) + (name.strip(),)
            if key in values:
                raise ValueError("Duplicate effective parameter")
            values[key] = value.strip()
        else:
            raise ValueError("Unsupported effective parameter syntax")
    if stack:
        raise ValueError("Unbalanced parameter subsections")
    flow = "MetricFlowSystem<1, 3>"
    if values[(flow, "Use componentwise absolute tolerances")].lower() != "false":
        raise ValueError("Componentwise IDA tolerances are unsupported")
    prefix = (flow, "IDA parameters")
    solver = {"/".join(key[2:]): value for key, value in values.items() if key[:2] == prefix}
    atol = finite(solver["Error control/Absolute error tolerance"], "IDA atol")
    rtol = finite(solver["Error control/Relative error tolerance"], "IDA rtol")
    if atol <= 0 or rtol < 0:
        raise ValueError("Invalid IDA tolerances")
    mesh = Path(values[(flow, "Vtk file path for mesh input")])
    if not mesh.is_absolute():
        mesh = case / mesh
    waveform = values.get((flow, "Inflow CSV file"), "")
    waveform_path = Path(waveform)
    if not waveform_path.is_absolute():
        waveform_path = case / waveform_path
    return {"parameters": values, "inflow_sha256": file_sha(waveform_path) if waveform else None, "atol": atol, "rtol": rtol, "solver": solver, "material": read_vtk(mesh), "path": str(path), "sha256": file_sha(path)}


def expected_output_times(solver):
    start = finite(solver["Initial time"], "initial time")
    stop = finite(solver["Final time"], "final time")
    interval = finite(solver["Time interval between each output"], "output interval")
    if interval <= 0 or stop <= start:
        raise ValueError("Unsupported output time grid")
    ratio = (stop-start)/interval
    count = round(ratio)
    if count < 1 or abs(ratio-count) > 64*math.ulp(1.)*max(1.,abs(ratio)):
        raise ValueError("Unsupported nonintegral output time grid")
    return [start+i*interval for i in range(count+1)]


def read_csv(path):
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        fields, rows = reader.fieldnames, list(reader)
    if not fields or len(fields) != len(set(fields)):
        raise ValueError("Invalid CSV schema: " + str(path))
    for row in rows:
        if set(row) != set(fields) or any(v is None for v in row.values()):
            raise ValueError("Incomplete CSV row: " + str(path))
        for key, value in row.items():
            if key != "label" and not key.endswith("_label"):
                finite(value, str(path) + ":" + key)
    return fields, rows


def state_bound(before, after, atol, rtol):
    return finite(atol + rtol * max(abs(before), abs(after)), "state error bound")


def pressure_bound(area_before, area_after, area_tolerance, material):
    if min(area_before, area_after) <= 0:
        raise ValueError("Nonpositive area")
    return finite(material["beta"] / (2 * material["a_d"] * math.sqrt(min(area_before, area_after))) * area_tolerance, "pressure error bound")


def flow_bound(ab, aa, ub, ua, area_tolerance, velocity_tolerance):
    return finite(abs((ub + ua) / 2) * area_tolerance + ((ab + aa) / 2) * velocity_tolerance, "flow error bound")


def area_from_pressure(pressure, material):
    branch = math.sqrt(material["a_d"]) + material["a_d"] * (pressure - material["p0"] - material["p_d"]) / material["beta"]
    if not math.isfinite(branch) or branch <= 0:
        raise ValueError("Invalid positive tube-law branch")
    return finite(branch * branch, "reconstructed area")


def record_metric(metrics, name, before, after, budget):
    error = finite(abs(before - after), "physical difference")
    budget = finite(budget, "physical error bound")
    if budget < 0:
        raise ValueError("Negative physical error bound")
    normalized = error / budget if budget else (0. if error == 0 else None)
    row = metrics.setdefault(name, {"max_normalized_error": 0., "max_abs_error": 0., "checks": 0, "changed": 0, "accepted": True})
    row["checks"] += 1
    row["changed"] += before != after
    row["max_abs_error"] = max(row["max_abs_error"], error)
    if normalized is None or not math.isfinite(normalized):
        row["max_normalized_error"] = None
        row["accepted"] = False
    else:
        if row["max_normalized_error"] is not None:
            row["max_normalized_error"] = max(row["max_normalized_error"], normalized)
        row["accepted"] &= normalized < 1


def check_times(left, right, key):
    if len(left) != len(right):
        raise ValueError("Physical output row counts differ")
    for a, b in zip(left, right):
        if abs(float(a[key]) - float(b[key])) > TIME_BOUND:
            raise ValueError("Fixed physical output times differ")


def diagnostics(left, right):
    result = {"before_rows": len(left), "after_rows": len(right), "fields": {}}
    if not left or not right:
        return result
    for key in left[0]:
        pairs = [(finite(a[key]), finite(b[key])) for a, b in zip(left, right)]
        changed = sum(a != b for a, b in pairs)
        if changed or len(left) != len(right):
            result["fields"][key] = {"changed_aligned_rows": changed, "max_abs_difference_on_aligned_rows": max((finite(abs(a-b), "diagnostic difference") for a, b in pairs), default=0.)}
    return result


def compare_case(before, after):
    cb, ca = read_parameters(before), read_parameters(after)
    if cb["material"]["model"] != ca["material"]["model"]:
        raise ValueError("Solver inputs or physical VTK properties differ")
    if cb["inflow_sha256"] != ca["inflow_sha256"]:
        raise ValueError("Inflow CSV waveform bytes differ")
    flow = "MetricFlowSystem<1, 3>"
    parameters = []
    for case in (cb, ca):
        values = case["parameters"].copy()
        values.pop((flow, "Output directory"), None)
        # Paths may differ only after their physical inputs have been verified.
        values[(flow, "Vtk file path for mesh input")] = "verified VTK model"
        if (flow, "Inflow CSV file") in values:
            values[(flow, "Inflow CSV file")] = case["inflow_sha256"]
        parameters.append(values)
    if parameters[0] != parameters[1]:
        raise ValueError("Solver inputs or effective physical parameters differ")
    material, atol, rtol = cb["material"], cb["atol"], cb["rtol"]
    expected_times = expected_output_times(cb["solver"])
    files_b = {p.name for p in (before/"output").glob("*.csv")} - {"scaling_timings.csv"}
    files_a = {p.name for p in (after/"output").glob("*.csv")} - {"scaling_timings.csv"}
    if files_b != PHYSICAL | DIAGNOSTIC or files_a != files_b:
        raise ValueError("Unsupported physical/diagnostic CSV file set")
    tables, diagnostic = {}, {}
    for name in sorted(files_b):
        fields_b, left = read_csv(before/"output"/name)
        fields_a, right = read_csv(after/"output"/name)
        if fields_b != fields_a:
            raise ValueError("CSV schemas differ: " + name)
        if name in DIAGNOSTIC:
            if not left or not right or (name == "ida_statistics.csv" and (len(left) != 1 or len(right) != 1)):
                raise ValueError("Incomplete IDA diagnostic output")
            if name == "ida_step_history.csv":
                check_times(left, right, "output_time")
                if len(left) != len(expected_times):
                    raise ValueError("Incomplete fixed-output history")
                for rows in (left, right):
                    if any(abs(float(row["output_time"])-time) > TIME_BOUND for row,time in zip(rows,expected_times)):
                        raise ValueError("History output times differ from the solver grid")
            diagnostic[name] = diagnostics(left, right)
        else:
            tables[name] = (fields_b, left, right)
    fields, left, right = tables["network_terminals.csv"]
    if fields != ["boundary_id", "vessel_id", "label", "R1", "R2", "C", "P_out"] or len(left) != 1 or left != right:
        raise ValueError("Terminal metadata differs or is unsupported")
    outlet = left[0]
    if outlet["boundary_id"] != "1" or outlet["vessel_id"] != "0" or any(float(outlet[k]) < 0 for k in ["R1", "C"]) or float(outlet["R2"]) <= 0:
        raise ValueError("Unsupported RCR terminal metadata")
    metrics = {}
    fields, left, right = tables["HDG_IDA_Vessel_0.csv"]
    core = ["time_s", "P_dynpcm2", "Q_cm3ps", "A_cm2", "U_cmps"]
    if fields != core + ["boundary_1_label"]:
        raise ValueError("Unsupported midpoint output schema")
    check_times(left, right, "time_s")
    if len(left) != len(expected_times):
        raise ValueError("Incomplete midpoint physical output")
    for rows in (left, right):
        if any(abs(float(row["time_s"])-time) > TIME_BOUND for row,time in zip(rows,expected_times)):
            raise ValueError("Output times differ from the unchanged solver grid")
    reference_times = ([r["time_s"] for r in left], [r["time_s"] for r in right])
    for b, a in zip(left, right):
        if b["boundary_1_label"] != a["boundary_1_label"] or b["boundary_1_label"] != outlet["label"]:
            raise ValueError("Midpoint boundary labels differ")
        ab, aa = float(b["A_cm2"])*1e-4, float(a["A_cm2"])*1e-4
        ub, ua = float(b["U_cmps"])*.01, float(a["U_cmps"])*.01
        at, ut = state_bound(ab, aa, atol, rtol), state_bound(ub, ua, atol, rtol)
        record_metric(metrics, "area", ab, aa, at)
        record_metric(metrics, "velocity", ub, ua, ut)
        record_metric(metrics, "pressure", float(b["P_dynpcm2"])*.1, float(a["P_dynpcm2"])*.1, pressure_bound(ab, aa, at, material))
        record_metric(metrics, "flow", float(b["Q_cm3ps"])*1e-6, float(a["Q_cm3ps"])*1e-6, flow_bound(ab, aa, ub, ua, at, ut))
    terminal_flow_budgets = [0.] * len(left)
    for boundary in (0, 1):
        fields, left, right = tables[f"terminal_{boundary}.csv"]
        if fields != ["time_s", "pressure_Pa", "flow_m3_s", "boundary_id", "vessel_id", "label"]:
            raise ValueError("Unsupported terminal output schema")
        check_times(left, right, "time_s")
        if len(left) != len(terminal_flow_budgets):
            raise ValueError("Terminal and midpoint sample counts differ")
        for i, (b, a) in enumerate(zip(left, right)):
            for row in (b, a):
                if row["boundary_id"] != str(boundary) or row["vessel_id"] != "0":
                    raise ValueError("Terminal identifiers differ")
            if b["label"] != a["label"] or (boundary == 1 and b["label"] != outlet["label"]):
                raise ValueError("Terminal labels differ")
            if any(abs(float(row["time_s"])-float(reference_times[k][i])) > TIME_BOUND for k, row in enumerate((b, a))):
                raise ValueError("Terminal and midpoint output times differ")
            pb, pa = float(b["pressure_Pa"]), float(a["pressure_Pa"])
            qb, qa = float(b["flow_m3_s"]), float(a["flow_m3_s"])
            ab, aa = area_from_pressure(pb, material), area_from_pressure(pa, material)
            ub, ua = qb/ab, qa/aa
            at, ut = state_bound(ab, aa, atol, rtol), state_bound(ub, ua, atol, rtol)
            qt = flow_bound(ab, aa, ub, ua, at, ut)
            terminal_flow_budgets[i] += qt
            record_metric(metrics, "terminal_area", ab, aa, at)
            record_metric(metrics, "terminal_velocity", ub, ua, ut)
            record_metric(metrics, "pressure", pb, pa, pressure_bound(ab, aa, at, material))
            record_metric(metrics, "flow", qb, qa, qt)
            if boundary == 1 and float(outlet["C"]) > 0:
                pcb, pca = pb-float(outlet["R1"])*qb, pa-float(outlet["R1"])*qa
                record_metric(metrics, "reconstructed_Pc", pcb, pca, state_bound(pcb, pca, atol, rtol))
    fields, left, right = tables["mass_conservation.csv"]
    if fields != ["t", "total_volume", "dVdt", "net_flux", "imbalance", "rel_imbalance"]:
        raise ValueError("Unsupported mass output schema")
    check_times(left, right, "t")
    if len(left) != len(terminal_flow_budgets):
        raise ValueError("Mass and terminal sample counts differ")
    volume_budgets = []
    for i, (b, a) in enumerate(zip(left, right)):
        if any(abs(float(row["t"])-float(reference_times[k][i])) > TIME_BOUND for k, row in enumerate((b, a))):
            raise ValueError("Mass and midpoint output times differ")
        vb, va = float(b["total_volume"]), float(a["total_volume"])
        if min(vb, va) <= 0:
            raise ValueError("Nonpositive total volume")
        vt = material["L"]*atol + rtol*max(abs(vb),abs(va))
        volume_budgets.append(vt)
        record_metric(metrics, "volume", vb, va, vt)
        if i == 0:
            if float(b["dVdt"]) != 0 or float(a["dVdt"]) != 0:
                raise ValueError("Initial volume derivative must be zero")
            derivative_budget = 0.
        else:
            dtb = float(b["t"])-float(left[i-1]["t"])
            dta = float(a["t"])-float(right[i-1]["t"])
            if min(dtb,dta) <= 0:
                raise ValueError("Physical output times are not increasing")
            delta_volume = max(abs(vb-float(left[i-1]["total_volume"])),abs(va-float(right[i-1]["total_volume"])))
            derivative_budget = (vt+volume_budgets[i-1])/min(dtb,dta) + delta_volume*abs(dtb-dta)/(dtb*dta)
        record_metric(metrics, "dVdt", float(b["dVdt"]), float(a["dVdt"]), derivative_budget)
        record_metric(metrics, "net_flux", float(b["net_flux"]), float(a["net_flux"]), terminal_flow_budgets[i])
        record_metric(metrics, "imbalance", float(b["imbalance"]), float(a["imbalance"]), derivative_budget+terminal_flow_budgets[i])
    diagnostic["rel_imbalance"] = diagnostics([{ "value": r["rel_imbalance"]} for r in left], [{"value": r["rel_imbalance"]} for r in right])
    return {"accepted": all(v["accepted"] for v in metrics.values()), "atol": atol, "rtol": rtol, "physical_rows": sum(len(t[1]) for t in tables.values()), "metrics": metrics, "diagnostics": diagnostic, "parameters_sha256": {"before": cb["sha256"], "after": ca["sha256"]}, "fixture_sha256": {"before": cb["material"]["sha256"], "after": ca["material"]["sha256"]}}


def compare(before, after):
    expected = {str(Path(case)/"output"/name) for case in CASES for name in PHYSICAL | DIAGNOSTIC}
    for root in (before, after):
        observed = {str(p.relative_to(root)) for p in root.rglob("*.csv")
                    if p.parent.name == "output" and p.name != "scaling_timings.csv"}
        if observed != expected:
            raise ValueError("Incomplete or unexpected regression output set")
    cases = {case: compare_case(before/case, after/case) for case in CASES}
    maxima = {}
    for result in cases.values():
        for name, row in result["metrics"].items():
            value = row["max_normalized_error"]
            maxima[name] = None if value is None or (name in maxima and maxima[name] is None) else max(maxima.get(name,0.),value)
    return {"accepted": all(c["accepted"] for c in cases.values()), "policy": "Actual global IDA component weights and dimensionally propagated derived-output bounds; strict normalized error<1", "fixed_output_time_bound_s": TIME_BOUND, "compared_csv_files": 32, "excluded_execution_timing_csv": "scaling_timings.csv", "max_weighted_errors": maxima, "cases": cases}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True, help="Predeclared policy receipt; no tolerance overrides")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = compare(args.before, args.after)
    except (ValueError, KeyError, OSError, ArithmeticError) as error:
        report = {"accepted": False, "failure": str(error)}
    report["policy_receipt_sha256"] = file_sha(args.policy)
    report["tool_sha256"] = file_sha(Path(__file__))
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(json.dumps({k:v for k,v in report.items() if k!="cases"},allow_nan=False))
    return 0 if report["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
