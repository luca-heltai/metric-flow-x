#!/usr/bin/env python3
"""Verify parameter-controlled CSV paths and disabled output during a full solve."""
import argparse
import csv
import os
from pathlib import Path
import subprocess
import tempfile


DEFAULTS = {
    "network terminals": "network_terminals.csv",
    "mass conservation": "mass_conservation.csv",
    "IDA statistics": "ida_statistics.csv",
    "IDA step history": "ida_step_history.csv",
    "IDA internal steps": "ida_internal_steps.csv",
    "scaling timings": "scaling_timings.csv",
    "flow regime": "flow_regime.csv",
}


def validate(executable, source, destination):
    environment = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    baseline = None
    for mode in ("defaults", "custom", "disabled"):
        folder = destination / mode
        folder.mkdir(parents=True)
        output = folder / "output"
        names = dict(DEFAULTS)
        overrides = ""
        if mode == "custom":
            names = {key: "nested/" + value for key, value in names.items()}
            names["flow regime"] = str(folder / "absolute/regime.csv")
        elif mode == "disabled":
            names = {key: "" for key in names}
        if mode != "defaults":
            overrides = "\n".join(f"  set Output {key} CSV file = {name}" for key, name in names.items())
        parameters = f"""subsection Metric Flow Parameters
  set Viscosity coefficient (mu) = 0
end
subsection Functions
  set Inflow function = 0
end
subsection MetricFlowSystem<1, 3>
  set Vtk file path for mesh input = {source / 'parameters/single_vessel.vtk'}
  set Number of global refinement = 0
  set Outlet boundary condition type = Reflection
  set Output directory = {output}
  set Output time interval = 0.0625
  set Write flow regime diagnostics = true
  set Output terminal CSV file =
{overrides}
  subsection IDA parameters
    set Initial time = 0
    set Final time = 0.125
    set Time interval between each output = 0.03125
    subsection Running parameters
      set Initial step size = 0.00390625
      set Maximum step size = 0.00390625
    end
  end
end
"""
        (folder / "input.prm").write_text(parameters)
        run = subprocess.run([str(executable), "input.prm"], cwd=folder,
                             env=environment, stdin=subprocess.DEVNULL,
                             capture_output=True, text=True, timeout=90)
        (folder / "run.log").write_text(run.stdout + run.stderr)
        assert run.returncode == 0, run.stdout + run.stderr
        table = [line for line in run.stdout.splitlines() if line.startswith("|") and
                 line.split("|")[1].strip()[:1].isdigit()]
        assert len(table) == 5, table
        assert float(table[-1].split("|")[1]) == .125
        # Disabling diagnostics must preserve the actual integration and counters.
        if baseline is None:
            baseline = [line.split("|")[2:-2] for line in table]
        assert [line.split("|")[2:-2] for line in table] == baseline
        expected = {output / "HDG_IDA_Vessel_0.csv"}
        for key, name in names.items():
            if not name:
                continue
            path = output / name
            assert path.is_file(), path
            expected.add(path)
            with path.open() as stream:
                rows = list(csv.DictReader(stream))
            assert rows, (key, path)
            if key == "IDA statistics":
                assert int(rows[-1]["n_steps"]) >= 32, rows
            elif key == "IDA internal steps":
                assert float(rows[-1]["t"]) >= .125 - 1e-12, rows[-1]
        assert set(folder.rglob("*.csv")) == expected
    print("CSV default, custom, absolute and disabled filenames: PASSED")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, required=True)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="metric-flow-csv-filenames-") as directory:
        validate(args.executable.resolve(), Path(__file__).resolve().parents[1], Path(directory))


if __name__ == "__main__":
    main()
