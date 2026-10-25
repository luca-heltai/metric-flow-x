#!/usr/bin/env python3
"""Advance the coupled ADAN network with the distributed iterative solver."""
import argparse
import csv
import json
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
    text = (source_dir / "tutorials/04_adan56/parameters.prm.in").read_text()
    text = text.replace("@SOURCE_DIR@", str(source_dir))
    import re
    for key, value in {
        "Final time": "0.002",
        "Use direct solver": "false",
        "Output directory": str(output),
    }.items():
        text, count = re.subn(rf"(?m)^(\s*set {re.escape(key)}\s*=).*", rf"\g<1> {value}", text)
        assert count == 1
    text = text.replace("subsection MetricFlowSystem<1, 3>", """subsection MetricFlowSystem<1, 3>
  set Output time interval = 0.002
  set Initial condition policy = native
  set Initial state source = rest
  set Use componentwise absolute tolerances = false""")
    parameter_file.write_text(text)
    launcher = shutil.which("mpirun")
    if not launcher:
        raise RuntimeError("This distributed solver regression requires mpirun")
    environment = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                       OMPI_ALLOW_RUN_AS_ROOT="1", OMPI_ALLOW_RUN_AS_ROOT_CONFIRM="1",
                       OMPI_MCA_rmaps_base_oversubscribe="1",
                       OMPI_MCA_hwloc_base_binding_policy="none")
    result = subprocess.run([launcher, "-n", "2", str(executable), str(parameter_file)],
                            cwd=destination, env=environment, capture_output=True,
                            text=True, timeout=90)
    (destination / "run.log").write_text(result.stdout + result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
    with (output / "mass_conservation.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 2
    assert all(abs(float(row["t"]) - expected) < 1e-14
               for row, expected in zip(rows, (0., .002)))
    assert all(math.isfinite(float(value)) for row in rows for value in row.values())
    assert all(float(row["total_volume"]) > 0 for row in rows)
    with (output / "ida_statistics.csv").open() as stream:
        statistics = list(csv.DictReader(stream))
    assert len(statistics) == 1 and int(statistics[0]["n_steps"]) > 0
    initialization = json.loads((output / "initialization_statistics.json").read_text())
    assert initialization["native_pair_verified"]
    assert initialization["ic_residual_calls"] == 0
    print("Distributed iterative ADAN network advances with finite output: PASSED")


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
