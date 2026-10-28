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


def validate(executable, destination, mpi, initial_time=0.0):
    destination.mkdir(parents=True, exist_ok=True)
    source = (ROOT / "tests/parameters/constant.prm.in").read_text().replace("@SOURCE_DIR@", str(ROOT))
    source = source.replace("= 4\n", "= 3\n").replace("= 1.1\n", "= 0.01\n")
    source = source.replace("= 0.005\n", "= 0.001\n").replace("= 1e-06", "= 1e-12")
    source = source.replace("= 1e-05", "= 1e-07").replace("= HLL\n", "= HLL_HDG\n")
    final_time = initial_time + 0.01
    peak_time = initial_time + 0.005
    source = source.replace("set Initial time                      = 0\n",
                            f"set Initial time                      = {initial_time}\n")
    source = source.replace("= 0.01\n", f"= {final_time}\n")
    source = source.replace("set Inflow function = 0.0",
                            f"set Inflow function = 1e-8 - 2e-6*abs(t-{peak_time})")
    waveform = destination / "inflow.csv"
    waveform.write_text('"note","flow_m3_s","time_s"\n'
                        f'"start, zero",0,{initial_time}\n'
                        f'"peak\nannotation",1e-8,{peak_time}\n'
                        f'end,0,{final_time}\n')
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
            extra = f"  set Output terminal CSV file = terminals.csv\n  set Output directory = {folder / 'output'}\n"
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
            # The standalone driver must produce the diagnostics used by new
            # paper runs, in addition to the terminal and visualization files.
            with (folder / "output/mass_conservation.csv").open() as stream:
                mass = list(csv.DictReader(stream))
            assert len(mass) == 11
            assert abs(float(mass[0]["t"]) - initial_time) < 1e-14
            assert abs(float(mass[-1]["t"]) - final_time) < 1e-14
            assert all(math.isfinite(float(value)) for row in mass for value in row.values())
            assert all(float(row["total_volume"]) > 0 for row in mass)
            with (folder / "output/ida_statistics.csv").open() as stream:
                statistics = list(csv.DictReader(stream))
            assert len(statistics) == 1
            summary = {key: float(value) for key, value in statistics[0].items()}
            assert all(math.isfinite(value) for value in summary.values())
            assert summary["n_steps"] > 0
            assert summary["wall_time"] > 0
            assert sum(summary[f"bdf{order}_steps"] for order in range(1, 6)) == summary["n_steps"]
            with (folder / "output/ida_step_history.csv").open() as stream:
                history = list(csv.DictReader(stream))
            # Overlapping driver hooks must not duplicate history records or
            # consume interval counters before the corresponding output row.
            assert len(history) == len(mass) == summary["n_outputs"]
            assert all(abs(float(record["output_time"]) - float(row["t"])) < 1e-14
                       for record, row in zip(history, mass))
            assert sum(int(row["steps_interval"]) for row in history) == summary["n_steps"]
            assert sum(int(row["nonlin_iters_interval"]) for row in history) == summary["n_nonlin_iters"]
            with (folder / "output/scaling_timings.csv").open() as stream:
                timings = list(csv.DictReader(stream))
            assert timings
            assert all(int(row["n_mpi_processes"]) == ranks for row in timings)
            assert all(math.isfinite(float(row["wall_time"])) and float(row["wall_time"]) >= 0
                       for row in timings)
            assert not list((folder / "output").glob("terminal_*.csv"))
            with (folder / "output/terminals.csv").open(newline="") as stream:
                terminal_rows = list(csv.DictReader(stream))
            assert len(terminal_rows) == 22
            assert [(float(row["time_s"]), int(row["boundary_id"])) for row in terminal_rows] == sorted(
                (float(row["time_s"]), int(row["boundary_id"])) for row in terminal_rows)
            traces[kind, ranks] = {}
            for boundary in (0, 1):
                rows = [row for row in terminal_rows if int(row["boundary_id"]) == boundary]
                assert len(rows) == 11
                assert all(row["label"] == f"boundary_{boundary}" for row in rows)
                numeric = [{key: float(value) for key, value in row.items() if key != "label"} for row in rows]
                assert all(math.isfinite(value) for row in numeric for value in row.values())
                assert abs(numeric[-1]["time_s"] - final_time) < 1e-14
                traces[kind, ranks][f"terminal_{boundary}.csv"] = numeric
            inlet = traces[kind, ranks]["terminal_0.csv"]
            error = max(abs(row["flow_m3_s"] + max(0, 1e-8 - 2e-6 * abs(row["time_s"] - peak_time))) for row in inlet)
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
    extra = f"  set Inflow CSV file = {csv_path}\n  set Output terminal CSV file = terminals.csv\n  set Output directory = {output_path}\n"
    (bad / "parameters.prm").write_text(source.replace("  set Output filename", extra + "  set Output filename"))
    result = subprocess.run([str(executable), "parameters.prm"], cwd=bad,
                            env=environment, capture_output=True, text=True, timeout=30)
    assert result.returncode != 0
    assert "extrapolation" in result.stdout + result.stderr
    assert not list(output_path.glob("terminal_*.csv"))
    assert not (output_path / "terminals.csv").exists()
    print(f"Analytic/CSV inflow and terminal traces agree for MPI ranks {ranks_to_check}; inlet flow is nonzero.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--workdir", type=Path)
    parser.add_argument("--mpi", action="store_true")
    args = parser.parse_args()
    if args.workdir:
        validate(args.executable.resolve(), args.workdir.resolve(), args.mpi)
        validate(args.executable.resolve(), args.workdir.resolve() / "nonzero-time", args.mpi, 0.02)
    else:
        with tempfile.TemporaryDirectory() as folder:
            validate(args.executable.resolve(), Path(folder), args.mpi)
            validate(args.executable.resolve(), Path(folder) / "nonzero-time", args.mpi, 0.02)


if __name__ == "__main__":
    main()
