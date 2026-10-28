#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Measure native assembly scaling on the full coupled arterial-venous graph."""
import argparse
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
from pathlib import Path


def benchmark(executable, parameters, output, launcher, launcher_args,
              ranks, repeats, iterations):
    executable, parameters, output = (p.resolve() for p in (executable, parameters, output))
    output.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                       MKL_NUM_THREADS="1", DEAL_II_NUM_THREADS="1")
    metadata = dict(host=platform.node(), platform=platform.platform(),
                    slurm_job_id=os.environ.get("SLURM_JOB_ID"),
                    executable=str(executable), parameters=str(parameters),
                    executable_sha256=hashlib.sha256(executable.read_bytes()).hexdigest(),
                    library_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                    for p in executable.parent.glob("libmetric_flow_x*") if p.is_file()},
                    parameter_sha256=hashlib.sha256(parameters.read_bytes()).hexdigest(),
                    launcher=launcher, launcher_args=launcher_args,
                    threads_per_rank=1, iterations=iterations, repeats=repeats,
                    jacobian_iterations_per_batch=max(1, iterations//10),
                    synthetic_state=dict(cell_and_trace_area="1.01 * vessel midpoint reference area",
                                         cell_and_trace_velocity_m_s=.1,
                                         capacitor_pressure_Pa=0., all_derivatives=.02,
                                         evaluation_time_s=.1),
                    cpu_info=subprocess.run(["lscpu", "--json"], check=True,
                                            capture_output=True, text=True).stdout)
    cache = executable.parent/"CMakeCache.txt"
    if cache.exists():
        keys = ("CMAKE_BUILD_TYPE", "CMAKE_CXX_COMPILER", "CMAKE_CXX_FLAGS",
                "deal.II_DIR", "METRIC_FLOW_X_LA_BACKEND")
        metadata["cmake_configuration"] = {
            line.split(":", 1)[0]: line.split("=", 1)[1]
            for line in cache.read_text().splitlines()
            if "=" in line and line.split(":", 1)[0] in keys}
    metadata["allocated_cpu_affinity"] = sorted(os.sched_getaffinity(0))
    input_files = {}
    for line in parameters.read_text().splitlines():
        if "=" in line:
            key, value = (s.strip() for s in line.split("=", 1))
            if key in ("set Vtk file path for mesh input", "set Terminal connections CSV file"):
                path = Path(value).resolve()
                input_files[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    metadata["input_sha256"] = input_files
    samples = []
    for repeat in range(repeats):
        order = ranks if repeat % 2 == 0 else ranks[::-1]
        for count in order:
            log = output/f"assembly_mpi{count}_repeat{repeat}.log"
            command = [launcher, *launcher_args, "-n", str(count), str(executable),
                       str(parameters), str(iterations)]
            print("Running", " ".join(command), flush=True)
            with log.open("w") as stream:
                subprocess.run(command, check=True, stdout=stream,
                               stderr=subprocess.STDOUT, env=environment)
            result = json.loads(next(line.removeprefix("BENCHMARK ")
                                     for line in log.read_text().splitlines()
                                     if line.startswith("BENCHMARK ")))
            if result["mpi_ranks"] != count:
                raise ValueError("MPI launcher did not start the requested rank count")
            result.update(repeat=repeat, log=log.name)
            samples.append(result)
    norms = [result["residual_l2"] for result in samples]
    if not all(math.isfinite(x) for x in norms) or max(norms)-min(norms) > 1e-10*max(norms):
        raise ValueError("Residual norms disagree between MPI layouts")
    summaries = {}
    for count in ranks:
        selected = [row for row in samples if row["mpi_ranks"] == count]
        summaries[str(count)] = {key: statistics.median(row[key] for row in selected)
                                for key in ("residual_median_s", "jacobian_median_s",
                                            "derivative_median_s", "startup_s")}
    baseline = summaries[str(min(ranks))]
    for count in ranks:
        for operation in ("residual", "jacobian"):
            key = operation+"_median_s"
            speedup = baseline[key]/summaries[str(count)][key]
            summaries[str(count)][operation+"_speedup"] = speedup
            summaries[str(count)][operation+"_efficiency"] = speedup*min(ranks)/count
    result = dict(metadata=metadata, samples=samples, summaries=summaries,
                  residual_norm_relative_spread=(max(norms)-min(norms))/max(norms),
                  interpretation="CPU/MPI assembly of a deterministic synthetic state on the GPU partition; not CUDA acceleration or end-to-end IDA scaling. Derivative timings measure the cached elastic mass-matrix path. Startup is rank-zero time; residual/Jacobian samples use maximum rank time.")
    (output/"results.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(summaries, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--parameters", type=Path, default=Path("build/adavn_full/assembly.prm"))
    parser.add_argument("--output-dir", type=Path, default=Path("build/adavn_full/performance"))
    parser.add_argument("--launcher", default="mpiexec")
    parser.add_argument("--launcher-arg", action="append", default=[])
    parser.add_argument("--ranks", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()
    if min(args.ranks) < 1 or args.repeats < 1 or args.iterations < 1:
        parser.error("Rank counts, repeats and iterations must be positive")
    benchmark(args.executable, args.parameters, args.output_dir, args.launcher,
              args.launcher_arg, args.ranks, args.repeats, args.iterations)
