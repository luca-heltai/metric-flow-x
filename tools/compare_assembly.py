#!/usr/bin/env python3
"""Compare two archived Release builds, numerical snapshots, and timings.

Each directory contains benchmark_assembly, its shared library, and revision.txt.
Run on the same host, without competing benchmark/build jobs. Matplotlib is only
required for this offline reporting tool, never by the solver or its tests.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--ranks", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.runs, args.repetitions, *args.ranks) < 1:
        parser.error("Runs, repetitions, and ranks must be positive")
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
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    records = []
    comparisons = []
    for parameters in args.inputs:
        for ranks in args.ranks:
            case = f"{parameters.stem}_mpi{ranks}"
            for run in range(args.runs):
                # Alternate process order to reduce monotonic host drift.
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
            # Exact hexadecimal snapshots include all owned vector entries and
            # all matrix entries, including explicit zeros. No norm-only gate.
            expected = 9 * ranks
            snapshots = sorted(args.output.glob(f"{case}_before.state*"))
            if len(snapshots) != expected:
                raise RuntimeError(f"Expected {expected} snapshot files for {case}")
            for before in snapshots:
                after = before.with_name(before.name.replace("_before.", "_after.", 1))
                if digest(before) != digest(after):
                    raise RuntimeError(f"Numerical result changed: {before} vs {after}")
                comparisons.append({"case": case, "snapshot": before.name,
                                    "sha256": digest(before), "bytes": before.stat().st_size})
    (args.output / "equivalence.json").write_text(json.dumps(comparisons, indent=2) + "\n")
    # Snapshots can be large; preserve their hashes and sizes after comparison.
    for snapshot in args.output.glob("*.state*"):
        snapshot.unlink()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    metrics = ["residual", "jacobian", "derivative"]
    cases = sorted({r["case"] for r in records})
    fig, axes = plt.subplots(1, 3, figsize=(15, max(3, len(cases) * .55 + 1)))
    summary = []
    for ax, metric in zip(axes, metrics):
        for offset, variant, color in [(-.18, "before", "#666666"), (.18, "after", "#0072B2")]:
            values = [[1000 * r[f"{metric}_median_s"] for r in records
                       if r["case"] == case and r["variant"] == variant] for case in cases]
            centers = [statistics.median(v) for v in values]
            errors = [[c - min(v) for c, v in zip(centers, values)],
                      [max(v) - c for c, v in zip(centers, values)]]
            ax.barh([i + offset for i in range(len(cases))], centers, height=.34,
                    xerr=errors, label=variant, color=color, capsize=3)
            summary.extend({"case": case, "metric": metric, "variant": variant,
                            "median_ms": c, "min_ms": min(v), "max_ms": max(v)}
                           for case, c, v in zip(cases, centers, values))
        ax.set_yticks(range(len(cases)), cases)
        ax.set_xlabel("ms / assembly; median of process medians (range)")
        ax.set_title(metric + " assembly")
        ax.legend()
    fig.tight_layout()
    fig.savefig(args.output / "comparison.svg")
    fig.savefig(args.output / "comparison.png", dpi=150)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
