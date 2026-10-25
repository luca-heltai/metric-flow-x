#!/usr/bin/env python3
"""Check graph-only and HDG CalcIC-only runs, including their physical output."""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET


def rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def validate(executable, source_dir, destination, mpi):
    destination.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                       OMPI_ALLOW_RUN_AS_ROOT="1", OMPI_ALLOW_RUN_AS_ROOT_CONFIRM="1",
                       OMPI_MCA_rmaps_base_oversubscribe="1",
                       OMPI_MCA_hwloc_base_binding_policy="none")
    launcher = shutil.which("mpirun")
    if mpi and launcher is None:
        raise RuntimeError("MPI stationary tests require mpirun")
    cases = [(mode, c, refinement, 1, "steady network")
             for mode in ("steady network", "steady HDG")
             for c in (0., 1e-8) for refinement in (0, 1)]
    cases += [(mode, 1e-8, 0, 1, "rest")
              for mode in ("steady network", "steady HDG")]
    if mpi:
        cases += [(mode, 1e-8, 0, size, "steady network")
                  for mode in ("steady network", "steady HDG") for size in (2, 4)]
    for number, (mode, capacitance, refinement, size, seed) in enumerate(cases):
        folder = destination / str(number)
        folder.mkdir(exist_ok=True)
        output = folder / "output"
        # Time-dependent inflow is frozen at t0. Final time equals t0: neither
        # mode may integrate. Invalid transient-only controls are irrelevant.
        parameters = f"""subsection Metric Flow Parameters
  set Viscosity coefficient (mu) = 0.004
  set Profile constant for friction term (xi) = 2
end
subsection Functions
  set Inflow function = 1e-6*(1+t)
end
subsection MetricFlowSystem<1, 3>
  set Run mode = {mode}
  set Initial state source = {seed}
  set Finite element degree = 2
  set Number of global refinement = {refinement}
  set Vtk file path for mesh input = {source_dir / 'parameters/single_vessel.vtk'}
  set Outlet boundary condition type = RCR
  set Terminal parameter source = Parameters
  set Default terminal R1 = 1e7
  set Default terminal R2 = 1e8
  set Default terminal C = {capacitance}
  set Default terminal P_out = 1000
  set Output terminal CSV file = stationary_traces.csv
  set Use componentwise absolute tolerances = true
  set Area absolute tolerance fraction = 1e-10
  set Velocity absolute tolerance = 1e-10
  set Capacitor pressure absolute tolerance = 1e-6
  set Use direct solver = {'true' if size == 1 else 'false'}
  set Output directory = {output}
  set Output filename = stationary
  set Maximum IDA internal steps per output = 0
  set Tolerance ramp enabled = true
  set Verbosity (console depth) = 0
  subsection IDA parameters
    set Initial time = 0.17
    set Final time = 0.17
    subsection Error control
      set Relative error tolerance = 1e-7
    end
    subsection Initial condition correction parameters
      set Maximum number of nonlinear iterations = 15
    end
  end
end
"""
        (folder / "input.prm").write_text(parameters)
        command = [str(executable), "input.prm"]
        if size > 1:
            command = [launcher, "-n", str(size)] + command
        result = subprocess.run(command, cwd=folder, env=environment,
                                stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=120)
        (folder / "run.log").write_text(result.stdout + result.stderr)
        assert result.returncode == 0, (mode, seed, size, result.stdout + result.stderr)
        assert "IDASolve" not in result.stdout + result.stderr
        pvd = ET.parse(output / "stationary.pvd")
        datasets = pvd.findall(".//DataSet")
        assert len(datasets) == 1 and float(datasets[0].attrib["timestep"]) == .17
        pvtu = output / datasets[0].attrib["file"]
        pieces = ET.parse(pvtu).findall(".//Piece")
        assert pieces and all((pvtu.parent / p.attrib["Source"]).is_file() for p in pieces)
        for name in ("ida_statistics.csv", "ida_step_history.csv",
                     "ida_internal_steps.csv", "initialization_statistics.json",
                     "mass_conservation.csv"):
            assert not (output / name).exists(), name
        q = 1.17e-6
        outlet_pressure = 1000. + 1.1e8 * q
        resistance = 8. * .004 * .24137 * math.pi / 4.5239e-4**2
        terminal_rows = rows(output / "stationary_traces.csv")
        assert [int(row["boundary_id"]) for row in terminal_rows] == [0, 1]
        assert not list(output.glob("terminal_*.csv"))
        for boundary in (0, 1):
            terminal = [row for row in terminal_rows if int(row["boundary_id"]) == boundary]
            assert len(terminal) == 1
            row = terminal[0]
            assert float(row["time_s"]) == .17
            pressure = float(row["pressure_Pa"])
            flux = float(row["flow_m3_s"])
            assert math.isfinite(pressure) and math.isfinite(flux)
            if mode == "steady network":
                expected = outlet_pressure + (resistance * q if boundary == 0 else 0.)
                assert abs(pressure - expected) < 1e-6, (pressure, expected)
                assert abs(flux - (-q if boundary == 0 else q)) < 1e-16
            elif boundary == 0:
                assert abs(flux + q) < 1e-12, flux
            else:
                assert abs(pressure - (1000. + 1.1e8 * flux)) < 1e-3
        vessel = rows(output / "HDG_IDA_Vessel_0.csv")
        assert len(vessel) == 1 and float(vessel[0]["time_s"]) == .17
        for key in ("P_dynpcm2", "Q_cm3ps", "A_cm2", "U_cmps"):
            assert math.isfinite(float(vessel[0][key]))
        assert float(vessel[0]["A_cm2"]) > 0.
        if mode == "steady network":
            assert abs(float(vessel[0]["P_dynpcm2"]) / 10. -
                       (outlet_pressure + .5 * resistance * q)) < 1e-6
            assert abs(float(vessel[0]["Q_cm3ps"]) * 1e-6 - q) < 1e-16
            assert not (output / "stationary_statistics.json").exists()
        else:
            stats = json.loads((output / "stationary_statistics.json").read_text())
            assert stats["model"] == "steady HDG"
            assert stats["time"] == .17 and stats["ida_steps"] == 0
            assert stats["derivative_norm"] == 0
            assert 0 <= stats["weighted_newton_defect"] < 1
        print(f"{mode}, C={capacitance}, refinement={refinement}, MPI{size}, seed={seed}: PASSED", flush=True)
    # Graph failures and invalid mode selection must not emit a solution.
    for name, bad_parameters in (
            ("invalid_graph", parameters.replace("Run mode = steady HDG", "Run mode = steady network")
             .replace("Viscosity coefficient (mu) = 0.004", "Viscosity coefficient (mu) = 0")),
            ("invalid_mode", parameters.replace(f"Run mode = {mode}", "Run mode = unknown"))):
        folder = destination / name
        folder.mkdir(exist_ok=True)
        bad_output = folder / "output"
        (folder / "input.prm").write_text(bad_parameters.replace(str(output), str(bad_output)))
        result = subprocess.run([str(executable), "input.prm"], cwd=folder,
                                env=environment, stdin=subprocess.DEVNULL,
                                capture_output=True, text=True, timeout=60)
        (folder / "run.log").write_text(result.stdout + result.stderr)
        assert result.returncode != 0, name
        assert not (bad_output / "stationary.pvd").exists(), name
    print(f"Stationary graph/HDG modes: {len(cases)} cases and 2 rejection checks passed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", required=True, type=Path)
    parser.add_argument("--source-dir", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--workdir", type=Path)
    parser.add_argument("--mpi", action="store_true")
    args = parser.parse_args()
    if args.workdir:
        validate(args.executable.resolve(), args.source_dir.resolve(), args.workdir.resolve(), args.mpi)
    else:
        with tempfile.TemporaryDirectory(prefix="metric-flow-stationary-") as work:
            validate(args.executable.resolve(), args.source_dir.resolve(), Path(work), args.mpi)


if __name__ == "__main__":
    main()
