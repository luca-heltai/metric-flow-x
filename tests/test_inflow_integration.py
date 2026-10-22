#!/usr/bin/env python3
"""Compare analytic/CSV inflow and actual terminal traces at one/two MPI ranks."""
import argparse
import csv
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def validate(executable, destination, mpi):
    destination.mkdir(parents=True, exist_ok=True)
    source = (ROOT / "parameters/constant.prm.in").read_text().replace("@SOURCE_DIR@", str(ROOT))
    source = source.replace("= 4\n", "= 3\n").replace("= 1.1\n", "= 0.01\n")
    source = source.replace("= 0.005\n", "= 0.001\n").replace("= 1e-06", "= 1e-12")
    source = source.replace("= 1e-05", "= 1e-07").replace("= HLL\n", "= HLL_HDG\n")
    source = source.replace("set Inflow function = 0.0", "set Inflow function = 1e-8 - 2e-6*abs(t-0.005)")
    waveform = destination / "inflow.csv"
    waveform.write_text('"note","flow_m3_s","time_s"\n'
                        '"start, zero",0,0\n'
                        '"peak\nannotation",1e-8,0.005\n'
                        'end,0,0.01\n')
    traces = {}
    environment = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                       OMPI_ALLOW_RUN_AS_ROOT="1", OMPI_ALLOW_RUN_AS_ROOT_CONFIRM="1")
    ranks_to_check = (1, 2) if mpi else (1,)
    launcher = shutil.which("mpirun") if mpi else None
    if mpi and not launcher:
        raise RuntimeError("MPI validation requested but mpirun is unavailable")
    for ranks in ranks_to_check:
        for kind in ("analytic", "csv"):
            folder = destination / f"{kind}-{ranks}"
            folder.mkdir(exist_ok=True)
            text = source
            if ranks > 1:
                text = text.replace("set Use direct solver            = true", "set Use direct solver            = false")
            extra = f"  set Write terminal CSV files = true\n  set Output directory = {folder / 'output'}\n"
            if kind == "csv":
                extra += f"  set Inflow CSV file = {waveform}\n"
            text = text.replace("  set Output filename", extra + "  set Output filename")
            (folder / "parameters.prm").write_text(text)
            command = ([launcher, "-n", str(ranks)] if launcher else []) + [str(executable), "parameters.prm"]
            result = subprocess.run(command, cwd=folder, env=environment,
                                    capture_output=True, text=True, timeout=90)
            (folder / "run.log").write_text(result.stdout + result.stderr)
            if result.returncode:
                raise RuntimeError(f"{kind}, ranks={ranks}: {result.stdout}\n{result.stderr}")
            files = sorted((folder / "output").glob("terminal_*.csv"))
            assert [path.name for path in files] == ["terminal_0.csv", "terminal_1.csv"]
            traces[kind, ranks] = {}
            for path in files:
                with path.open() as stream:
                    rows = list(csv.DictReader(stream))
                assert len(rows) == 11
                boundary = int(path.stem.removeprefix("terminal_"))
                assert all(row["label"] == f"boundary_{boundary}" for row in rows)
                numeric = [{key: float(value) for key, value in row.items() if key != "label"} for row in rows]
                assert all(math.isfinite(value) for row in numeric for value in row.values())
                assert abs(numeric[-1]["time_s"] - 0.01) < 1e-14
                traces[kind, ranks][path.name] = numeric
            inlet = traces[kind, ranks]["terminal_0.csv"]
            error = max(abs(row["flow_m3_s"] + max(0, 1e-8 - 2e-6 * abs(row["time_s"] - 0.005))) for row in inlet)
            assert error < 5e-11, (kind, ranks, error)
            assert max(abs(row["flow_m3_s"]) for row in inlet) > 0.99e-8
    reference = traces["analytic", 1]
    for key, case in traces.items():
        for terminal, rows in case.items():
            for actual, expected in zip(rows, reference[terminal]):
                assert abs(actual["time_s"] - expected["time_s"]) < 1e-14
                assert abs(actual["flow_m3_s"] - expected["flow_m3_s"]) < 1e-12, key
                assert abs(actual["pressure_Pa"] - expected["pressure_Pa"]) < 1e-3, key
    # Reject insufficient coverage before the runner creates terminal files.
    bad = destination / "uncovered"
    bad.mkdir(exist_ok=True)
    csv_path = bad / "short.csv"
    output_path = bad / "output"
    csv_path.write_text("time_s,flow_m3_s\n0,0\n0.005,1e-8\n")
    extra = f"  set Inflow CSV file = {csv_path}\n  set Write terminal CSV files = true\n  set Output directory = {output_path}\n"
    (bad / "parameters.prm").write_text(source.replace("  set Output filename", extra + "  set Output filename"))
    result = subprocess.run([str(executable), "parameters.prm"], cwd=bad,
                            env=environment, capture_output=True, text=True, timeout=30)
    assert result.returncode != 0
    assert "extrapolation" in result.stdout + result.stderr
    assert not list(output_path.glob("terminal_*.csv"))
    print(f"Analytic/CSV inflow and terminal traces agree for MPI ranks {ranks_to_check}; inlet flow is nonzero.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--workdir", type=Path)
    parser.add_argument("--mpi", action="store_true")
    args = parser.parse_args()
    if args.workdir:
        validate(args.executable.resolve(), args.workdir.resolve(), args.mpi)
    else:
        with tempfile.TemporaryDirectory() as folder:
            validate(args.executable.resolve(), Path(folder), args.mpi)


if __name__ == "__main__":
    main()
