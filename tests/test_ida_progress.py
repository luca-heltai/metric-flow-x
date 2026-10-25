#!/usr/bin/env python3
"""Verify console progress against accepted-step history and native step bounds."""
import argparse
import csv
import math
import os
from pathlib import Path
import subprocess
import tempfile


def records(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def validate(executable, source, destination):
    destination.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    template = f"""subsection Metric Flow Parameters
  set Viscosity coefficient (mu) = 0
end
subsection Functions
  set Inflow function = 0
end
subsection MetricFlowSystem<1, 3>
  set Vtk file path for mesh input = {source / 'parameters/single_vessel.vtk'}
  set Number of global refinement = 0
  set Outlet boundary condition type = Reflection
  set Output time interval = 0.0625
  set Output filename = solution
  set Verbosity (console depth) = 0
  subsection IDA parameters
    set Initial time = 0
    set Final time = 0.125
    set Time interval between each output = 0.03125
    subsection Running parameters
      set Initial step size = 0.03125
      set Minimum step size = 0
      set Maximum step size = MAXIMUM
    end
  end
end
"""
    for name, maximum, printing, initial in (
            ("unlimited", 0., True, 0.), ("capped", 1/256, True, 0.),
            ("quiet", 1/256, False, 0.), ("large_time", 1/256, True, 100.)):

        folder = destination / name
        folder.mkdir(exist_ok=True)
        output = folder / "output"
        parameters = template.replace("MAXIMUM", str(maximum)).replace(
            "  set Output filename", f"  set Output directory = {output}\n"
            f"  set Print current statistics = {'true' if printing else 'false'}\n  set Output filename")
        parameters = parameters.replace("set Initial time = 0\n", f"set Initial time = {initial}\n")
        parameters = parameters.replace("set Final time = 0.125\n", f"set Final time = {initial + .125}\n")
        (folder / "input.prm").write_text(parameters)
        result = subprocess.run([str(executable), "input.prm"], cwd=folder, env=environment,
                                stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=90)
        (folder / "run.log").write_text(result.stdout + result.stderr)
        assert result.returncode == 0, result.stdout + result.stderr
        accepted = records(output / "ida_internal_steps.csv")
        history = records(output / "ida_step_history.csv")
        final = records(output / "ida_statistics.csv")[-1]
        nsteps = int(final["n_steps"])
        assert len(accepted) == nsteps and [int(row["step"]) for row in accepted] == list(range(1, nsteps+1))
        hs = [abs(float(row["h"])) for row in accepted]
        if maximum:
            assert max(hs) <= maximum * (1 + 1e-6), hs
            assert nsteps >= 32, nsteps
        else:
            assert max(hs) > 1/256, hs
        table = [[cell.strip() for cell in line.strip().split("|")[1:-1]]
                 for line in result.stdout.splitlines() if line.startswith("|") and
                 line.split("|")[1].strip()[:1].isdigit()]
        assert len(table) == (5 if printing else 0), table
        assert len(history) == 3, history
        if not printing:
            continue
        assert table[0][2:5] == ["-", "-", "-"] and table[0][6] == "0/0", table[0]
        assert [float(row[0]) for row in table] == [initial + t for t in (0., .03125, .0625, .09375, .125)]
        # Table precision is deliberately compact; CSV preserves more digits.
        last = table[-1]
        assert math.isclose(float(last[3]), min(hs), rel_tol=1e-3)
        assert math.isclose(float(last[4]), max(hs), rel_tol=1e-3)
        columns = {6: "steps_cumulative", 7: "nonlin_iters_cumulative",
                   8: "jacobian_evals_cumulative", 9: "residual_evals_cumulative",
                   10: "linear_solves_cumulative", 11: "linear_iters_cumulative",
                   12: "err_test_fails_cumulative", 13: "nonlin_fails_cumulative"}
        for column, key in columns.items():
            previous = 0
            for row in table:
                delta, total = map(int, row[column].split("/"))
                assert delta == total - previous and total >= previous
                previous = total
            assert previous == int(history[-1][key]), (key, previous, history[-1])
        assert int(last[6].split("/")[1]) == nsteps
        assert int(last[7].split("/")[1]) == int(final["n_nonlin_iters"])
        assert int(last[12].split("/")[1]) == int(final["n_err_test_fails"])
        assert int(last[13].split("/")[1]) == int(final["n_nonlin_conv_fails"])
        assert math.isclose(float(history[-1]["min_step_size"]), min(hs), rel_tol=2e-6)
        assert math.isclose(float(history[-1]["max_step_size"]), max(hs), rel_tol=2e-6)
        assert all(row["step_extrema_available"] == "1" for row in history[1:])
        assert history[0]["step_extrema_available"] == "0"
        assert int(last[8].split("/")[1]) == int(history[-1]["lin_setups_cumulative"])
        assert int(last[8].split("/")[1]) > 0 and int(last[10].split("/")[1]) > 0

    # A demanding startup succeeds without an hmin, and must fail rather than
    # quietly taking smaller steps when hmin is explicitly set too large.
    startup = template.replace("= Reflection", "= RCR").replace(
        "  set Outlet boundary condition type", "  set Terminal parameter source = Parameters\n"
        "  set Default terminal R1 = 1e7\n  set Default terminal R2 = 1e8\n"
        "  set Default terminal C = 1e-8\n  set Default terminal P_out = 1000\n"
        "  set Use componentwise absolute tolerances = true\n"
        "  set Area absolute tolerance fraction = 1e-10\n"
        "  set Velocity absolute tolerance = 1e-10\n"
        "  set Capacitor pressure absolute tolerance = 1e-6\n"
        "  set Maximum IDA internal steps per output = 10000\n"
        "  set Outlet boundary condition type")
    startup = startup.replace("Inflow function = 0", "Inflow function = 1e-6")
    startup = startup.replace("= 0.125", "= 0.01").replace("= 0.03125", "= 0.005").replace("= 0.0625", "= 0")
    startup = startup.replace("Initial step size = 0.005", "Initial step size = 0.001").replace("MAXIMUM", "0")
    for minimum in (0., .001):
        folder = destination / f"minimum_{minimum}"
        folder.mkdir(exist_ok=True)
        output = folder / "output"
        parameters = startup.replace("Minimum step size = 0", f"Minimum step size = {minimum}").replace(
            "  set Output filename", f"  set Output directory = {output}\n  set Output filename")
        (folder / "input.prm").write_text(parameters)
        result = subprocess.run([str(executable), "input.prm"], cwd=folder, env=environment,
                                stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=90)
        (folder / "run.log").write_text(result.stdout + result.stderr)
        if minimum == 0.:
            assert result.returncode == 0, result.stdout + result.stderr
            hs = [float(row["h"]) for row in records(output / "ida_internal_steps.csv")]
            assert min(hs) < .001
        else:
            assert result.returncode != 0 and "SUNDIALS IDA reported error -" in result.stderr
            assert "IDASolve" in result.stderr and not (output / "ida_statistics.csv").exists()
            assert len(records(output / "mass_conservation.csv")) == 1
    print("IDA aligned progress, exact accepted-step extrema, native hmin/hmax and throttled file cadence: PASSED")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", required=True, type=Path)
    parser.add_argument("--source-dir", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--workdir", type=Path)
    args = parser.parse_args()
    if args.workdir:
        validate(args.executable.resolve(), args.source_dir.resolve(), args.workdir.resolve())
    else:
        with tempfile.TemporaryDirectory(prefix="metric-flow-ida-progress-") as work:
            validate(args.executable.resolve(), args.source_dir.resolve(), Path(work))


if __name__ == "__main__":
    main()
