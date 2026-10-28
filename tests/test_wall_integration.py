#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Smoke-test optional ADAVN walls through the standalone IDA/output path."""
import argparse
import csv
import math
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--mpi", action="store_true")
    parser.add_argument("--mpi-ranks", type=int, nargs="+", default=[1, 2, 4])
    args = parser.parse_args()
    if 1 not in args.mpi_ranks or any(n < 1 for n in args.mpi_ranks):
        parser.error("--mpi-ranks requires rank 1 and positive counts")
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
                executable.parent / "tests" / "parameters" / f"wall_{model}.prm"
            ).read_text()
            for viscous in (False, True):
                cases = [("native", n) for n in (args.mpi_ranks if args.mpi else (1,))]
                if viscous:
                    cases += [("IDA", 1)]
                for policy, ranks in cases:
                    folder = Path(temporary) / f"{model}-{viscous}-{policy}-mpi{ranks}"
                    folder.mkdir()
                    text = original.replace(
                        "set Inflow function = 0.0", "set Inflow function = 1e-7"
                    )
                    text = text.replace(
                        "Initial condition policy = native",
                        f"Initial condition policy = {policy}",
                    )
                    if not viscous:
                        text = text.replace(
                            "Viscoelastic wall = true", "Viscoelastic wall = false"
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
                    diagnostics = json.loads(
                        (folder / "initialization_statistics.json").read_text()
                    )
                    if policy == "native":
                        assert diagnostics["native_pair_verified"] is True, diagnostics
                        for key in (
                            "ic_residual_calls",
                            "ic_jacobian_setups",
                            "ic_linear_solves",
                        ):
                            assert diagnostics[key] == 0, diagnostics
                        assert diagnostics["algebraic_defect"] <= 0.01, diagnostics
                        assert diagnostics["differential_defect"] <= 0.01, diagnostics
                    with (folder / "vessel_probes.csv").open() as stream:
                        records = [row for row in csv.DictReader(stream) if int(row["vessel_id"]) == 0]
                    assert len(records) >= 2, records
                    final = records[-1]
                    assert math.isclose(
                        float(final["time_s"]), 0.001, abs_tol=1.0e-12
                    ), final
                    for key, value in final.items():
                        if not key.endswith("_label"):
                            assert math.isfinite(float(value)), final
                    assert float(final["area_m2"]) > 0.0, final
                    results[model, viscous, policy, ranks] = float(final["pressure_Pa"])
        for model in ("arterial", "venous"):
            assert (
                abs(
                    results[model, True, "native", 1]
                    - results[model, False, "native", 1]
                )
                > 1.0e-7
            ), results
            assert math.isclose(
                results[model, True, "native", 1],
                results[model, True, "IDA", 1],
                rel_tol=2.0e-6,
                abs_tol=1.0e-5,
            ), results
            if args.mpi:
                for viscous in (False, True):
                    for ranks in args.mpi_ranks:
                        if ranks == 1:
                            continue
                        assert math.isclose(
                            results[model, viscous, "native", 1],
                            results[model, viscous, "native", ranks],
                            rel_tol=2.0e-6,
                            abs_tol=1.0e-5,
                        ), results
    print(
        f"ADAVN native/IDA initialization, MPI integration and pressure output: {len(results)} passed"
    )


if __name__ == "__main__":
    main()
