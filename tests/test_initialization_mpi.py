#!/usr/bin/env python3
"""Exercise consistent-pair and tolerance callbacks across MPI partitions."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, required=True)
    args = parser.parse_args()
    build = args.build_dir.resolve()
    launcher = shutil.which("mpirun")
    if launcher is None:
        raise RuntimeError("Initialization MPI regressions require mpirun")
    environment = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                       OMPI_ALLOW_RUN_AS_ROOT="1", OMPI_ALLOW_RUN_AS_ROOT_CONFIRM="1",
                       OMPI_MCA_rmaps_base_oversubscribe="1",
                       OMPI_MCA_hwloc_base_binding_policy="none")
    cases = {"consistent_initialization": (2, 4),
             "nonfinite_initialization": (2, 4),
             "ida_error_guard": (2,),
             "ida_statistics": (2, 4),
             "terminal_exports": (2, 4),
             "csv_output": (2, 4),
             "tolerance_ramp_ida": (2, 4),
             "iterative_preconditioner": (2, 4)}
    count = 0
    with tempfile.TemporaryDirectory(prefix="metric-flow-initialization-mpi-") as work:
        for name, ranks in cases.items():
            candidates = sorted((build / "tests").glob(f"test_{name}.*"))
            executables = [folder / folder.name for folder in candidates
                           if folder.is_dir() and (folder / folder.name).is_file()]
            if not executables:
                raise RuntimeError(f"No built executable for test_{name}")
            for executable in executables:
                for size in ranks:
                    folder = Path(work) / f"{executable.name}-mpi{size}"
                    folder.mkdir()
                    log = folder / "run.log"
                    with log.open("w") as stream:
                        result = subprocess.run([launcher, "-n", str(size), str(executable)],
                                                cwd=folder, env=environment, stdout=stream,
                                                stderr=subprocess.STDOUT, timeout=180)
                    if result.returncode:
                        raise RuntimeError(f"{executable.name}, MPI{size}:\n" + log.read_text()[-12000:])
                    count += 1
    print(f"Consistent-state, empty-rank, weight-callback, progress and iterative MPI regressions: {count} passed")


if __name__ == "__main__":
    main()
