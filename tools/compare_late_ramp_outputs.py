#!/usr/bin/env python3
"""Strict late-window physical comparison for the declared one-cell ramp run."""
import argparse
import importlib.util
import json
import math
from pathlib import Path

FLOW = "MetricFlowSystem<1, 3>"
RAMP = ("Tolerance ramp enabled", "Tolerance ramp initial relative factor",
        "Tolerance ramp initial absolute factor", "Tolerance ramp hold duration",
        "Tolerance ramp end time")


def load_physics(path, policy):
    import hashlib
    if hashlib.sha256(path.read_bytes()).hexdigest() != policy["physical_comparator_sha256"]:
        raise ValueError("Frozen physical comparator differs")
    spec = importlib.util.spec_from_file_location("frozen_physical_comparator", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def checked_parameters(case, policy, physics):
    c = physics.read_parameters(case)
    values = c["parameters"]
    expected = {
        ("Functions", "Inflow function"): policy["waveform"],
        (FLOW, "Initial state source"): "steady network",
        (FLOW, "Initial condition policy"): "native",
        (FLOW, "Use componentwise absolute tolerances"): "false",
        (FLOW, "Finite element degree"): "1",
        (FLOW, "Number of global refinement"): "0",
        (FLOW, "Number of refinement cycles"): "1",
        (FLOW, "Outlet boundary condition type"): "RCR",
        (FLOW, "Terminal parameter source"): "Parameters",
        (FLOW, "Write terminal CSV files"): "true",
        (FLOW, "Numerical flux type"): policy["numerical_flux"],
    }
    for key, expected_value in expected.items():
        if values[key] != expected_value:
            raise ValueError("Unsupported parameter " + "/".join(key))
    if values.get((FLOW, "Terminal parameters CSV file"), "") or values.get((FLOW, "Inflow CSV file"), ""):
        raise ValueError("Only the declared analytic inflow and parameter terminals are supported")
    numeric = {
        (FLOW, "Maximum IDA internal steps per output"): policy["maximum_ida_internal_steps_per_output"],
        ("Metric Flow Parameters", "Density (rho)"): policy["physical"]["rho"],
        ("Metric Flow Parameters", "Viscosity coefficient (mu)"): policy["physical"]["mu"],
        ("Metric Flow Parameters", "Profile constant for friction term (xi)"): policy["physical"]["xi"],
        (FLOW, "Default terminal R1"): policy["physical"]["R1"],
        (FLOW, "Default terminal R2"): policy["physical"]["R2"],
        (FLOW, "Default terminal C"): policy["physical"]["C"],
        (FLOW, "Default terminal P_out"): policy["physical"]["P_out"],
        (FLOW, "Output time interval"): policy["file_interval"],
        (FLOW, "Theta (penalty parameter)"): policy["theta"],
        (FLOW, "Theta Boundary (stability parameter)"): policy["theta_boundary"],
        (FLOW, RAMP[1]): policy["ramp"]["relative_factor"],
        (FLOW, RAMP[2]): policy["ramp"]["absolute_factor"],
        (FLOW, RAMP[3]): policy["ramp"]["hold_duration"],
        (FLOW, RAMP[4]): policy["ramp"]["end_time"],
        (FLOW, "IDA parameters", "Initial time"): policy["initial_time"],
        (FLOW, "IDA parameters", "Final time"): policy["final_time"],
        (FLOW, "IDA parameters", "Time interval between each output"): policy["callback_interval"],
        (FLOW, "IDA parameters", "Running parameters", "Initial step size"): policy["initial_step_size"],
        (FLOW, "IDA parameters", "Running parameters", "Minimum step size"): policy["minimum_step_size"],
        (FLOW, "IDA parameters", "Running parameters", "Maximum order of BDF"): policy["maximum_bdf_order"],
    }
    for key, expected_value in numeric.items():
        if physics.finite(values[key]) != expected_value:
            raise ValueError("Changed declared parameter " + "/".join(key))
    is_base = (c["atol"], c["rtol"]) == (policy["base_atol"], policy["base_rtol"])
    is_reference = (c["atol"], c["rtol"]) == (policy["reference_atol"], policy["reference_rtol"])
    enabled = values[FLOW, RAMP[0]]
    if not (is_base or is_reference) or enabled not in ("true", "false") or (is_reference and enabled != "false"):
        raise ValueError("Undeclared tolerance/ramp configuration")
    normalized = values.copy()
    normalized.pop((FLOW, "Output directory"))
    normalized[FLOW, "Vtk file path for mesh input"] = c["material"]["sha256"]
    for name in ("Absolute error tolerance", "Relative error tolerance"):
        normalized.pop((FLOW, "IDA parameters", "Error control", name))
    normalized.pop((FLOW, RAMP[0]))
    c.update(normalized_parameters=normalized, reference=is_reference, ramp_enabled=enabled == "true")
    return c


def compare(before, after, policy, physics):
    cb, ca = checked_parameters(before, policy, physics), checked_parameters(after, policy, physics)
    if cb["normalized_parameters"] != ca["normalized_parameters"] or cb["material"]["model"] != ca["material"]["model"]:
        raise ValueError("Effective physical/solver inputs differ outside declared tolerance/ramp keys")
    if ca["reference"]:
        raise ValueError("Candidate must use the declared final production tolerances")
    step = policy["file_interval"]
    expected = [policy["initial_time"] + i * step for i in range(round((policy["final_time"]-policy["initial_time"])/step)+1)]
    grid_bound = policy["nominal_output_grid_roundoff_bound_s"]
    schemas = {
        "network_terminals.csv": ["boundary_id", "vessel_id", "label", "R1", "R2", "C", "P_out"],
        "HDG_IDA_Vessel_0.csv": ["time_s", "P_dynpcm2", "Q_cm3ps", "A_cm2", "U_cmps", "boundary_1_label"],
        "mass_conservation.csv": ["t", "total_volume", "dVdt", "net_flux", "imbalance", "rel_imbalance"],
        **{f"terminal_{bid}.csv": ["time_s", "pressure_Pa", "flow_m3_s", "boundary_id", "vessel_id", "label"] for bid in (0, 1)},
    }
    tables = {}
    for name in sorted(physics.PHYSICAL):
        fb, left = physics.read_csv(before/"output"/name)
        fa, right = physics.read_csv(after/"output"/name)
        if fb != fa or fb != schemas[name]:
            raise ValueError("CSV schemas differ or are unsupported: " + name)
        if name == "network_terminals.csv":
            if left != right or len(left) != 1:
                raise ValueError("Terminal metadata differ")
        else:
            time_key = "t" if name == "mass_conservation.csv" else "time_s"
            physics.check_times(left, right, time_key)
            if len(left) != len(expected):
                raise ValueError("Missing fixed physical output rows: " + name)
            for rows in (left, right):
                if any(abs(float(row[time_key])-t) > grid_bound for row, t in zip(rows, expected)):
                    raise ValueError("Output differs from the declared fixed grid: " + name)
        tables[name] = (fb, left, right)
    meta = tables["network_terminals.csv"][1][0]
    for key, value in policy["physical"].items():
        if key in ("R1", "R2", "C", "P_out") and physics.finite(meta[key]) != value:
            raise ValueError("Terminal output parameters differ")
    if meta["boundary_id"] != "1" or meta["vessel_id"] != "0" or meta["label"] != "boundary_1":
        raise ValueError("Unsupported terminal identifiers")
    material = cb["material"]
    atol, rtol = policy["base_atol"], policy["base_rtol"]
    metrics = {}
    start, stop = policy["late_window"]
    indices = [i for i, t in enumerate(expected) if start <= t <= stop]
    record = lambda name, b, a, bound: physics.record_metric(metrics, name, b, a, bound)
    _, left, right = tables["HDG_IDA_Vessel_0.csv"]
    volume_budgets = []
    for i, (b, a) in enumerate(zip(left, right)):
        if b["boundary_1_label"] != meta["label"] or a["boundary_1_label"] != meta["label"]:
            raise ValueError("Midpoint terminal labels differ")
        ab, aa = float(b["A_cm2"])*1e-4, float(a["A_cm2"])*1e-4
        ub, ua = float(b["U_cmps"])*.01, float(a["U_cmps"])*.01
        if min(ab, aa) <= 0:
            raise ValueError("Nonpositive midpoint area")
        at, ut = physics.state_bound(ab, aa, atol, rtol), physics.state_bound(ub, ua, atol, rtol)
        volume_budgets.append(material["L"] * at)
        if i in indices:
            record("area", ab, aa, at)
            record("velocity", ub, ua, ut)
            record("pressure", float(b["P_dynpcm2"])*.1, float(a["P_dynpcm2"])*.1, physics.pressure_bound(ab, aa, at, material))
            record("flow", float(b["Q_cm3ps"])*1e-6, float(a["Q_cm3ps"])*1e-6, physics.flow_bound(ab, aa, ub, ua, at, ut))
    terminal_budgets = [0.] * len(expected)
    for bid in (0, 1):
        _, left, right = tables[f"terminal_{bid}.csv"]
        for i, (b, a) in enumerate(zip(left, right)):
            for row in (b, a):
                if row["boundary_id"] != str(bid) or row["vessel_id"] != "0":
                    raise ValueError("Terminal identifiers differ")
            if b["label"] != a["label"]:
                raise ValueError("Terminal labels differ")
            pb, pa = float(b["pressure_Pa"]), float(a["pressure_Pa"])
            qb, qa = float(b["flow_m3_s"]), float(a["flow_m3_s"])
            ab, aa = physics.area_from_pressure(pb, material), physics.area_from_pressure(pa, material)
            ub, ua = qb/ab, qa/aa
            at, ut = physics.state_bound(ab, aa, atol, rtol), physics.state_bound(ub, ua, atol, rtol)
            qt = physics.flow_bound(ab, aa, ub, ua, at, ut)
            terminal_budgets[i] += qt
            if i in indices:
                record("terminal_area", ab, aa, at)
                record("terminal_velocity", ub, ua, ut)
                record("terminal_pressure", pb, pa, physics.pressure_bound(ab, aa, at, material))
                record("terminal_flow", qb, qa, qt)
                if bid == 1:
                    pcb, pca = pb-float(meta["R1"])*qb, pa-float(meta["R1"])*qa
                    record("reconstructed_Pc", pcb, pca, physics.state_bound(pcb, pca, atol, rtol))
    _, left, right = tables["mass_conservation.csv"]
    midpoint = tables["HDG_IDA_Vessel_0.csv"]
    for i, (b, a) in enumerate(zip(left, right)):
        vb, va = float(b["total_volume"]), float(a["total_volume"])
        if min(vb, va) <= 0:
            raise ValueError("Nonpositive physical volume")
        for row, probe in ((b, midpoint[1][i]), (a, midpoint[2][i])):
            v = float(row["total_volume"])
            exact = material["L"] * float(probe["A_cm2"]) * 1e-4
            if abs(v-exact) > 64*math.ulp(1.)*max(abs(v), abs(exact)):
                raise ValueError("Only one-cell degree-one volume reconstruction is supported")
        if i in indices:
            vt = volume_budgets[i]
            dtb, dta = float(b["t"])-float(left[i-1]["t"]), float(a["t"])-float(right[i-1]["t"])
            if min(dtb, dta) <= 0:
                raise ValueError("Nonincreasing mass output times")
            dv = max(abs(vb-float(left[i-1]["total_volume"])), abs(va-float(right[i-1]["total_volume"])))
            derivative_bound = (vt+volume_budgets[i-1])/min(dtb, dta) + dv*abs(dtb-dta)/(dtb*dta)
            record("volume", vb, va, vt)
            record("dVdt", float(b["dVdt"]), float(a["dVdt"]), derivative_bound)
            record("net_flux", float(b["net_flux"]), float(a["net_flux"]), terminal_budgets[i])
            record("imbalance", float(b["imbalance"]), float(a["imbalance"]), derivative_bound+terminal_budgets[i])
    return {"accepted": all(row["accepted"] for row in metrics.values()), "window": [start, stop], "late_samples": len(indices), "atol": atol, "rtol": rtol, "baseline_is_tighter_reference": cb["reference"], "candidate_ramp_enabled": ca["ramp_enabled"], "metrics": metrics, "parameters_sha256": {"before": cb["sha256"], "after": ca["sha256"]}, "fixture_sha256": cb["material"]["sha256"], "diagnostics_only": policy["diagnostics_only"]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--before", type=Path, required=True)
    p.add_argument("--after", type=Path, required=True)
    p.add_argument("--policy", type=Path, default=Path(__file__).with_name("ida_ramp_policy.json"))
    p.add_argument("--physics-helper", type=Path, default=Path(__file__).with_name("compare_inflow_trajectories.py"))
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    policy = json.loads(args.policy.read_text())
    physics = load_physics(args.physics_helper, policy)
    try:
        report = compare(args.before, args.after, policy, physics)
    except (ValueError, KeyError, OSError, ArithmeticError) as error:
        report = {"accepted": False, "failure": str(error)}
    report["policy_sha256"] = physics.file_sha(args.policy)
    report["validator_sha256"] = physics.file_sha(Path(__file__))
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps({k:v for k,v in report.items() if k != "metrics"}, allow_nan=False))
    return 0 if report["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
