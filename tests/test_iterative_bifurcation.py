#!/usr/bin/env python3
"""Advance a coarse RCR bifurcation with the distributed iterative solver."""
import argparse
import csv
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile


def validate(executable, source_dir, destination):
    destination.mkdir(parents=True, exist_ok=True)
    parameter_file = destination / "input.prm"
    output = destination / "output"
    parameter_file.write_text(f"""subsection MetricFlowSystem<1, 3>
  set Finite element degree = 1
  set Number of global refinement = 0
  set Number of refinement cycles = 1
  set Vtk file path for mesh input = {source_dir / 'parameters/aortic.vtk'}
  set Numerical flux type = HLL
  set Outlet boundary condition type = RCR
  set Output directory = {output}
  set Output filename = solution
  set Problem constants = 1.0
  set Theta Boundary (stability parameter) = 0.5
  set Theta (penalty parameter) = 0.5
  set Use Riemann Invariants = true
  set Use direct solver = false
  set Verbosity (console depth) = 0
  set Initial state source = steady network
  set Initial condition policy = native
  set Use componentwise absolute tolerances = false
  subsection IDA parameters
    set Final time = 0.002
    set Initial time = 0
    set Time interval between each output = 0.001
    subsection Error control
      set Absolute error tolerance = 1e-12
      set Relative error tolerance = 1e-7
    end
    subsection Running parameters
      set Initial step size = 1e-3
      set Maximum number of nonlinear iterations = 15
      set Maximum order of BDF = 3
      set Minimum step size = 1e-8
    end
  end
end
subsection Metric Flow Parameters
  set Density (rho) = 1060
  set Profile constant for friction term (xi) = 9
  set Reflection coefficient at outflow boundary (Rt) = 0
  set Tube law exponent (m) = 0.5
  set Viscosity coefficient (mu) = 0.004
end
subsection Functions
  set Inflow function = 1e-6
end
""")
    launcher = shutil.which("mpirun")
    if not launcher:
        raise RuntimeError("This distributed solver regression requires mpirun")
    environment = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                       OMPI_ALLOW_RUN_AS_ROOT="1", OMPI_ALLOW_RUN_AS_ROOT_CONFIRM="1")
    result = subprocess.run([launcher, "-n", "2", str(executable), str(parameter_file)],
                            cwd=destination, env=environment, capture_output=True,
                            text=True, timeout=90)
    (destination / "run.log").write_text(result.stdout + result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
    with (output / "mass_conservation.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 3
    assert all(abs(float(row["t"]) - expected) < 1e-14
               for row, expected in zip(rows, (0., .001, .002)))
    assert all(math.isfinite(float(value)) for row in rows for value in row.values())
    assert all(float(row["total_volume"]) > 0 for row in rows)
    with (output / "ida_statistics.csv").open() as stream:
        statistics = list(csv.DictReader(stream))
    assert len(statistics) == 1 and int(statistics[0]["n_steps"]) > 0
    print("Distributed iterative RCR bifurcation advances with finite output: PASSED")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", required=True, type=Path)
    parser.add_argument("--source-dir", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--workdir", type=Path)
    args = parser.parse_args()
    if args.workdir:
        validate(args.executable.resolve(), args.source_dir.resolve(), args.workdir.resolve())
    else:
        with tempfile.TemporaryDirectory() as folder:
            validate(args.executable.resolve(), args.source_dir.resolve(), Path(folder))


if __name__ == "__main__":
    main()
