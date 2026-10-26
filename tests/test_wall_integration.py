#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Smoke-test optional ADAVN walls through the standalone IDA/output path."""
import argparse
import csv
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--mpi", action="store_true")
    args = parser.parse_args()
    executable = args.executable.resolve()
    launcher = shutil.which("mpirun")
    if args.mpi and launcher is None:
        raise RuntimeError("Wall integration MPI tests require mpirun")
    environment = dict(
        os.environ,
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        OMPI_ALLOW_RUN_AS_ROOT="1",
        OMPI_ALLOW_RUN_AS_ROOT_CONFIRM="1",
        OMPI_MCA_rmaps_base_oversubscribe="1",
        OMPI_MCA_hwloc_base_binding_policy="none",
    )
    results = {}
    with tempfile.TemporaryDirectory(prefix="metric-flow-wall-") as temporary:
        for model in ("arterial", "venous"):
            original = (
                executable.parent / "parameters" / f"wall_{model}.prm"
            ).read_text()
            for viscous in (False, True):
                for ranks in ((1, 2) if args.mpi else (1,)):
                    folder = Path(temporary) / f"{model}-{viscous}-mpi{ranks}"
                    folder.mkdir()
                    text = original.replace(
                        "set Inflow function = 0.0", "set Inflow function = 1e-7"
                    )
                    if not viscous:
                        text = text.replace(
                            "Viscoelastic wall = true", "Viscoelastic wall = false"
                        )
                        text = text.replace(
                            "Initial condition policy = IDA",
                            "Initial condition policy = native",
                        )
                    parameters = folder / "input.prm"
                    parameters.write_text(text)
                    command = [str(executable), str(parameters)]
                    if ranks > 1:
                        command = [launcher, "-n", str(ranks)] + command
                    result = subprocess.run(
                        command,
                        cwd=folder,
                        env=environment,
                        capture_output=True,
                        text=True,
                        timeout=90,
                    )
                    if result.returncode:
                        raise RuntimeError(
                            f"{model}, viscous={viscous}, MPI{ranks}:\n"
                            + result.stdout[-6000:]
                            + result.stderr[-6000:]
                        )
                    with (folder / "HDG_IDA_Vessel_0.csv").open() as stream:
                        records = list(csv.DictReader(stream))
                    assert len(records) >= 2, records
                    final = records[-1]
                    assert math.isclose(
                        float(final["time_s"]), 0.001, abs_tol=1.0e-12
                    ), final
                    for key, value in final.items():
                        if not key.endswith("_label"):
                            assert math.isfinite(float(value)), final
                    assert float(final["A_cm2"]) > 0.0, final
                    results[model, viscous, ranks] = float(final["P_dynpcm2"])
        for model in ("arterial", "venous"):
            assert (
                abs(results[model, True, 1] - results[model, False, 1]) > 1.0e-7
            ), results
            if args.mpi:
                for viscous in (False, True):
                    assert math.isclose(
                        results[model, viscous, 1],
                        results[model, viscous, 2],
                        rel_tol=2.0e-6,
                        abs_tol=1.0e-5,
                    ), results
    print(
        f"ADAVN elastic/viscoelastic IDA integration and pressure output: {len(results)} passed"
    )


if __name__ == "__main__":
    main()
