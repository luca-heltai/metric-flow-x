#!/usr/bin/env python3
"""Compare archived builds on one host, retaining raw data in a local run folder.

Each build directory contains benchmark_assembly, libmetric_flow_x.so and
revision.txt. Run without competing builds/benchmarks. Numerical comparisons
are exact by default; explicit --atol/--rtol use |a-b| <= atol + rtol*max(|a|,|b|).
Only results.json and its figure need to be published; logs/snapshots are local.
Matplotlib is required for plotting, never by the solver or its tests.
"""
import argparse
import hashlib
from itertools import zip_longest
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare_snapshots(before, after, rtol=0.0, atol=0.0):
    """Compare matching indexed hexadecimal snapshots, including empty ranks."""
    result = {"snapshot": before.name, "sha256": digest(before),
              "after_sha256": digest(after), "bytes": before.stat().st_size,
              "after_bytes": after.stat().st_size, "entries": 0,
              "changed_entries": 0, "violations": 0, "max_abs_error": 0.0,
              "max_relative_error": 0.0, "max_scaled_error": 0.0}
    exact = result["sha256"] == result["after_sha256"]
    with before.open() as left, after.open() as right:
        for number, (a, b) in enumerate(zip_longest(left, right), 1):
            if a is None or b is None:
                raise RuntimeError(f"Snapshot lengths differ: {before}:{number}")
            a, b = a.split(), b.split()
            if len(a) not in (2, 3) or len(a) != len(b):
                raise RuntimeError(f"Invalid snapshot row: {before}:{number}")
            try:
                indices_a = tuple(int(i) for i in a[:-1])
                indices_b = tuple(int(i) for i in b[:-1])
                x, y = float.fromhex(a[-1]), float.fromhex(b[-1])
            except ValueError as error:
                raise RuntimeError(f"Invalid snapshot value: {before}:{number}") from error
            if indices_a != indices_b or any(i < 0 for i in indices_a):
                raise RuntimeError(f"Snapshot indices differ: {before}:{number}")
            if not (math.isfinite(x) and math.isfinite(y)):
                raise RuntimeError(f"Nonfinite snapshot value: {before}:{number}")
            difference = abs(x - y)
            scale = max(abs(x), abs(y))
            tolerance = atol + rtol * scale
            if not (math.isfinite(difference) and math.isfinite(tolerance)):
                raise RuntimeError(f"Comparison overflow: {before}:{number}")
            changed = x.hex() != y.hex()
            result["entries"] += 1
            result["changed_entries"] += changed
            result["violations"] += difference > tolerance
            result["max_abs_error"] = max(result["max_abs_error"], difference)
            result["max_relative_error"] = max(
                result["max_relative_error"], difference / scale if scale else 0.0)
            if tolerance and result["max_scaled_error"] is not None:
                scaled_error = difference / tolerance
                result["max_scaled_error"] = (max(result["max_scaled_error"], scaled_error)
                                               if math.isfinite(scaled_error) else None)
            elif not tolerance and difference:
                # A zero tolerance has no finite normalized error ratio.
                result["max_scaled_error"] = None
    result["exact"] = exact
    result["accepted"] = exact if rtol == atol == 0 else result["violations"] == 0
    return result


def summarize(records):
    """Compact paired results; no fixed workloads, rank counts or run counts."""
    rows = []
    for case in sorted({r["case"] for r in records}):
        selected = [r for r in records if r["case"] == case]
        pairs = {run: {r["variant"]: r for r in selected if r["run"] == run}
                 for run in sorted({r["run"] for r in selected})}
        if any(set(pair) != {"before", "after"} for pair in pairs.values()):
            raise RuntimeError(f"Incomplete process pair for {case}")
        ranks = selected[0]["mpi_ranks"]
        workload = case.removesuffix(f"_mpi{ranks}")
        for metric in ["residual", "jacobian", "derivative"]:
            row = dict(case=case, workload=workload, ranks=ranks, metric=metric)
            for variant in ["before", "after"]:
                values = [1000 * pair[variant][metric + "_median_s"] for pair in pairs.values()]
                if any(not math.isfinite(v) or v < 0 for v in values):
                    raise RuntimeError(f"Invalid timing for {case}/{metric}")
                row.update({variant + "_median_ms": statistics.median(values),
                            variant + "_min_ms": min(values), variant + "_max_ms": max(values)})
            ratios = [pair["before"][metric + "_median_s"] / pair["after"][metric + "_median_s"]
                      for pair in pairs.values() if pair["after"][metric + "_median_s"] > 0]
            row["median_paired_speedup"] = statistics.median(ratios) if len(ratios) == len(pairs) else None
            rows.append(row)
    singles = {(r["workload"], r["metric"]): r for r in rows if r["ranks"] == 1}
    for row in rows:
        single = singles.get((row["workload"], row["metric"]))
        if single:
            for variant in ["before", "after"]:
                duration = row[variant + "_median_ms"]
                speedup = single[variant + "_median_ms"] / duration if duration else None
                row[variant + "_strong_scaling_speedup"] = speedup
                row[variant + "_parallel_efficiency"] = speedup / row["ranks"] if speedup is not None else None
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--ranks", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--rtol", type=float, default=0.0)
    parser.add_argument("--atol", type=float, default=0.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.runs, args.repetitions, *args.ranks) < 1:
        parser.error("Runs, repetitions, and ranks must be positive")
    if any(not math.isfinite(t) or t < 0 for t in [args.rtol, args.atol]):
        parser.error("Numerical tolerances must be finite and nonnegative")
    if len(set(args.ranks)) != len(args.ranks) or len({p.stem for p in args.inputs}) != len(args.inputs):
        parser.error("Rank counts and input stems must be unique")
    args.output.mkdir(parents=True, exist_ok=False)
    builds = {"before": args.before.resolve(), "after": args.after.resolve()}
    metadata = {
        "host": platform.node(), "platform": platform.platform(),
        "cpu_info": Path("/proc/cpuinfo").read_text().split("\n\n")[0],
        "runs": args.runs, "repetitions": args.repetitions,
        "builds": {name: {"path": str(path),
                         "revision": (path / "revision.txt").read_text().strip(),
                         "executable_sha256": digest(path / "benchmark_assembly"),
                         "library_sha256": digest(path / "libmetric_flow_x.so")}
                   for name, path in builds.items()},
        "inputs": {str(p.resolve()): digest(p) for p in args.inputs},
        "thread_environment": {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"},
        "numerical_tolerances": {"rtol": args.rtol, "atol": args.atol},
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    records, comparisons = [], []
    for parameters in args.inputs:
        for ranks in args.ranks:
            case = f"{parameters.stem}_mpi{ranks}"
            for run in range(args.runs):
                order = ["before", "after"] if run % 2 == 0 else ["after", "before"]
                for variant in order:
                    build = builds[variant]
                    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
                    env["LD_LIBRARY_PATH"] = str(build) + ":" + env.get("LD_LIBRARY_PATH", "")
                    command = ["mpirun", "--bind-to", "core", "--map-by", "core",
                               "-n", str(ranks), str(build / "benchmark_assembly"),
                               str(parameters.resolve()), str(args.repetitions)]
                    if run == 0:
                        command.append(str((args.output / f"{case}_{variant}").resolve()))
                    result = subprocess.run(command, env=env, cwd=args.output.resolve(), text=True,
                                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                    log = args.output / f"{case}_{variant}_{run}.log"
                    log.write_text(result.stdout)
                    if result.returncode:
                        raise RuntimeError(f"Benchmark failed; see {log}")
                    lines = [line[len("BENCHMARK "):] for line in result.stdout.splitlines()
                             if line.startswith("BENCHMARK ")]
                    if len(lines) != 1:
                        raise RuntimeError(f"Expected one BENCHMARK record; see {log}")
                    record = dict(json.loads(lines[0]), case=case, variant=variant,
                                  run=run, command=command)
                    records.append(record)
                    with (args.output / "timings.jsonl").open("a") as stream:
                        stream.write(json.dumps(record) + "\n")
                    print(case, run, variant, record["residual_median_s"], flush=True)
            snapshots = sorted(args.output.glob(f"{case}_before.state*"))
            after_snapshots = list(args.output.glob(f"{case}_after.state*"))
            if len(snapshots) != 9 * ranks or len(after_snapshots) != 9 * ranks:
                raise RuntimeError(f"Expected {9 * ranks} snapshots per variant for {case}")
            for before in snapshots:
                after = before.with_name(before.name.replace("_before.", "_after.", 1))
                comparison = dict(case=case, **compare_snapshots(before, after, args.rtol, args.atol))
                comparisons.append(comparison)
                (args.output / "equivalence.json").write_text(json.dumps(comparisons, indent=2) + "\n")
                if not comparison["accepted"]:
                    raise RuntimeError(f"Numerical result outside tolerance: {before}; see equivalence.json")
    rows = summarize(records)
    compact_metadata = {key: metadata[key] for key in ["host", "runs", "repetitions"]}
    compact_metadata.update({
        "baseline": metadata["builds"]["before"]["revision"],
        "candidate": metadata["builds"]["after"]["revision"], "ranks": args.ranks,
        "input_sha256": {Path(p).name: h for p, h in metadata["inputs"].items()},
        "numerical_validation": {"status": "exact" if all(c["exact"] for c in comparisons) else "within_tolerance",
                                 "snapshot_pairs": len(comparisons), "rtol": args.rtol, "atol": args.atol,
                                 "max_abs_error": max(c["max_abs_error"] for c in comparisons),
                                 "max_relative_error": max(c["max_relative_error"] for c in comparisons),
                                 "max_scaled_error": max(c["max_scaled_error"] for c in comparisons),
                                 "changed_entries": sum(c["changed_entries"] for c in comparisons)},
    })
    (args.output / "results.json").write_text(json.dumps({"metadata": compact_metadata, "rows": rows}, indent=2) + "\n")
    # Delete large snapshots only after all gates pass; preserve hashes/errors.
    for snapshot in args.output.glob("*.state*"):
        snapshot.unlink()
    subprocess.run([sys.executable, str(Path(__file__).with_name("plot_assembly_results.py")),
                    str(args.output / "results.json"), "--output", str(args.output / "comparison.svg")], check=True)


if __name__ == "__main__":
    main()
