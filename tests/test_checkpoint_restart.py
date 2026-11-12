#!/usr/bin/env python3
"""Run standalone checkpoint/restart in fresh processes without original inputs."""
import argparse
import csv
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile


def run(command, folder, environment, label, expect_success=True):
    result = subprocess.run(command, cwd=folder, env=environment,
                            capture_output=True, text=True, timeout=180)
    (folder / f"{label}.log").write_text(result.stdout + result.stderr)
    if expect_success:
        assert result.returncode == 0, result.stdout + result.stderr
    else:
        assert result.returncode != 0, "Incompatible restart unexpectedly succeeded"
    return result


def read_rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def check_residuals(history):
    rows = read_rows(history / "checkpoint_residuals.csv")
    # These are fixture-specific raw equation bounds, not universal error
    # weights: components have different units. Observable continuation below
    # is the numerical accuracy check against an uninterrupted integration.
    bounds = {"area_residual_linfty": 1e-9,
              "velocity_residual_linfty": 1e-5,
              "area_trace_residual_linfty": 1e-6,
              "velocity_trace_residual_linfty": 1e-8,
              "capacitor_residual_linfty": 1e-9}
    assert rows
    for row in rows:
        assert abs(float(row["time_s"]) - float(row["ida_internal_time_s"])) < 1e-15
        for field, bound in bounds.items():
            value = float(row[field])
            assert math.isfinite(value) and value <= bound, (field, value, bound)
    return rows, {field: max(float(row[field]) for row in rows) for field in bounds}


def parameters(mesh, inflow, output, checkpoints):
    return f"""subsection Metric Flow Parameters
  set Viscosity coefficient (mu) = 0.004
end
subsection MetricFlowSystem<1, 3>
  set Finite element degree = 1
  set Number of global refinement = 2
  set Number of refinement cycles = 1
  set Vtk file path for mesh input = {mesh}
  set Inflow CSV file = {inflow}
  set Output directory = {output}
  set Checkpoint directory = {checkpoints}
  set Output time interval = 0.002
  set Initial condition policy = native
  set Initial consistency tolerance = 1e-6
  set Use direct solver = true
  set Verbosity (console depth) = 0
  set Numerical flux type = HLL
  set Outlet boundary condition type = RCR
  set Output terminal CSV file = terminals.csv
  set Use componentwise absolute tolerances = true
  set Area absolute tolerance fraction = 1e-8
  set Velocity absolute tolerance = 1e-8
  set Capacitor pressure absolute tolerance = 1e-3
  set Maximum IDA internal steps per output = 10000
  subsection IDA parameters
    set Initial time = 0
    set Final time = 0.008
    set Time interval between each output = 0.002
    subsection Error control
      set Absolute error tolerance = 1e-10
      set Relative error tolerance = 1e-8
    end
    subsection Running parameters
      set Initial step size = 1e-6
      set Minimum step size = 0
      set Maximum step size = 0.0002
      set Maximum number of nonlinear iterations = 10
      set Maximum order of BDF = 3
    end
  end
end
"""


def validate(executable, source, destination, ranks, mpiexec):
    folder = destination / f"ranks-{ranks}"
    folder.mkdir(parents=True, exist_ok=False)
    original = folder / "original-inputs"
    original.mkdir()
    mesh = original / "vessel.vtk"
    shutil.copyfile(source / "parameters/single_vessel.vtk", mesh)
    inflow = original / "inflow.csv"
    inflow.write_text("time_s,flow_m3_s\n0,1e-7\n0.008,2e-7\n0.02,3e-7\n")
    output = folder / "reference-output"
    checkpoints = folder / "checkpoints"
    prm = original / "input.prm"
    prm.write_text(parameters(mesh, inflow, output, checkpoints))
    prefix = [mpiexec, "-np", str(ranks)] if ranks > 1 else []
    environment = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    run(prefix + [str(executable), str(prm)], folder, environment, "reference")
    reference_residuals, maximum_residuals = check_residuals(checkpoints)
    assert len(reference_residuals) == 5

    quiet_output, quiet_history = folder / "without-vtk-output", folder / "without-vtk-history"
    quiet_prm = original / "without_vtk.prm"
    quiet_prm.write_text(parameters(mesh, inflow, quiet_output, quiet_history).replace(
        "  set Finite element degree = 1", "  set Write solution output = false\n  set Finite element degree = 1"))
    run(prefix + [str(executable), str(quiet_prm)], folder, environment, "without-vtk")
    assert len([path for path in quiet_history.iterdir() if path.is_dir()]) == 5
    assert not list(quiet_output.rglob("*.vtu"))
    check_residuals(quiet_history)

    history = {float.fromhex(path.name.removeprefix("checkpoint-")): path
               for path in checkpoints.iterdir() if path.is_dir()}
    assert len(history) == 5, history
    selected_time = min(history, key=lambda value: abs(value - .004))
    assert math.isclose(selected_time, .004, rel_tol=0., abs_tol=1e-16)
    assert all((path / "COMPLETE").is_file() for path in history.values())
    selected_name = history[selected_time].name

    # A relocated archive is the only source of numerical/configuration inputs
    # available to the independent restart process. Even the standalone receipt
    # is removed, so accidentally reading it cannot hide a dependency.
    shutil.rmtree(original)
    (folder / "last_used_parameters.prm").unlink(missing_ok=True)
    relocated = folder / "relocated-history"
    checkpoints.rename(relocated)
    restored_output = folder / "restored-output"
    selected = relocated / selected_name
    run(prefix + [str(executable), "--restart", str(selected),
                  "--final-time", "0.008", "--output-directory", str(restored_output),
                  "--checkpoint-directory", str(folder / "continued-checkpoints")],
        folder, environment, "restart")

    restored_residuals, _ = check_residuals(folder / "continued-checkpoints")
    expected_residual = min(reference_residuals,
                            key=lambda row: abs(float(row["time_s"]) - selected_time))
    for field in maximum_residuals:
        assert math.isclose(float(restored_residuals[0][field]), float(expected_residual[field]),
                            rel_tol=1e-8, abs_tol=1e-12), field

    # Resetting IDA discards its BDF history, so this is a numerical continuation
    # comparison, not a bitwise comparison. Bounds allow accumulated error from
    # two integrations at rtol=1e-8 and the declared component absolute scales.
    # Pressure/flow are nonlinear observables rather than IDA unknowns.
    tolerances = {"area_m2": 1e-10, "velocity_m_s": 1e-6,
                  "pressure_Pa": 1e-2, "flow_m3_s": 1e-10}
    reference = read_rows(output / "vessel_probes.csv")
    restored = read_rows(restored_output / "vessel_probes.csv")
    assert len(restored) == 3, restored
    maximum = {key: 0. for key in tolerances}
    for actual in restored:
        time = float(actual["time_s"])
        expected = min(reference, key=lambda row: abs(float(row["time_s"]) - time))
        assert abs(float(expected["time_s"]) - time) < 1e-15
        assert expected["vessel_id"] == actual["vessel_id"]
        for key, absolute in tolerances.items():
            a, b = float(actual[key]), float(expected[key])
            assert math.isfinite(a) and math.isfinite(b), (key, a, b)
            weight = absolute + 1e-6 * max(abs(a), abs(b))
            maximum[key] = max(maximum[key], abs(a - b) / weight)
            assert abs(a - b) <= weight, (ranks, time, key, a, b, weight)
    terminal_rows = read_rows(output / "terminals.csv")
    assert max(abs(float(row["flow_m3_s"])) for row in terminal_rows) > 1e-8
    restored_terminals = read_rows(restored_output / "terminals.csv")
    for actual in restored_terminals:
        time, boundary = float(actual["time_s"]), actual["boundary_id"]
        expected = min((row for row in terminal_rows if row["boundary_id"] == boundary),
                       key=lambda row: abs(float(row["time_s"]) - time))
        assert abs(float(expected["time_s"]) - time) < 1e-15
        for key in ("pressure_Pa", "flow_m3_s"):
            a, b = float(actual[key]), float(expected[key])
            assert math.isfinite(a) and math.isfinite(b)
            assert abs(a - b) <= tolerances[key] + 1e-6 * max(abs(a), abs(b)), (
                ranks, time, boundary, key, a, b)
    if ranks > 1:
        incompatible = run([str(executable), "--restart", str(selected)], folder,
                           environment, "incompatible-ranks", expect_success=False)
        assert "MPI size" in incompatible.stdout + incompatible.stderr
    print(f"Fresh-process checkpoint/restart ranks={ranks}: PASSED; "
          f"maximum normalized probe errors={maximum}; raw residual maxima={maximum_residuals}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path,
                        default=os.environ.get("METRIC_FLOW_X_EXECUTABLE"))
    parser.add_argument("--source-dir", type=Path,
                        default=Path(__file__).resolve().parents[1])
    parser.add_argument("--workdir", type=Path)
    parser.add_argument("--ranks", type=int, action="append")
    parser.add_argument("--mpiexec", default="mpirun")
    arguments = parser.parse_args()
    if arguments.executable is None:
        parser.error("--executable or METRIC_FLOW_X_EXECUTABLE is required")
    executable = arguments.executable.resolve()
    source = arguments.source_dir.resolve()
    ranks = arguments.ranks or [1]
    assert all(value > 0 for value in ranks)
    if arguments.workdir:
        destination = arguments.workdir.resolve()
        destination.mkdir(parents=True, exist_ok=True)
        for count in ranks:
            validate(executable, source, destination, count, arguments.mpiexec)
    else:
        with tempfile.TemporaryDirectory(prefix="metric-flow-checkpoint-") as temporary:
            for count in ranks:
                validate(executable, source, Path(temporary), count, arguments.mpiexec)


if __name__ == "__main__":
    main()
