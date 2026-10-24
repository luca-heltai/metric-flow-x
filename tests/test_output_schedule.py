#!/usr/bin/env python3
"""Check file cadence after long IDA callback grids and at the final time."""
import argparse
import csv
from fractions import Fraction
import math
import os
from pathlib import Path
import subprocess
import tempfile


def validate(executable, source_dir, destination):
    destination.mkdir(parents=True, exist_ok=True)
    template = (source_dir / "parameters/constant.prm.in").read_text().replace("@SOURCE_DIR@", str(source_dir))
    cases = ((0., 10., .01, .05), (0., .4, .03, .07), (.17, .37, .03, .05))
    environment = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    for number, (initial, final, period, cadence) in enumerate(cases):
        folder = destination / str(number)
        folder.mkdir(exist_ok=True)
        source = template.replace("= 4\n", "= 0\n")
        source = source.replace("= 1.1\n", f"= {final}\n")
        source = source.replace("set Initial time                      = 0\n",
                                f"set Initial time                      = {initial}\n")
        source = source.replace("= 0.005\n", f"= {period}\n")
        extra = (f"  set Output directory = {folder / 'output'}\n"
                 f"  set Output time interval = {cadence}\n"
                 "  set Initial condition policy = native\n")
        source = source.replace("  set Output filename", extra + "  set Output filename")
        (folder / "input.prm").write_text(source)
        result = subprocess.run([str(executable), "input.prm"], cwd=folder,
                                env=environment, capture_output=True, text=True, timeout=90)
        (folder / "run.log").write_text(result.stdout + result.stderr)
        assert result.returncode == 0, result.stdout + result.stderr
        with (folder / "output/mass_conservation.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        start, stop, step, interval = map(lambda x: Fraction(str(x)),
                                        (initial, final, period, cadence))
        duration = stop - start
        last_step = math.ceil(duration / step)
        selected = {math.ceil(k * interval / step)
                    for k in range(math.floor(duration / interval) + 1)}
        selected.add(last_step)
        expected = [float(start + min(n * step, duration)) for n in sorted(selected)]
        actual = [float(row["t"]) for row in rows]
        assert len(actual) == len(expected), (number, len(actual), len(expected))
        callback_times = [initial]
        for _ in range(last_step):
            callback_times.append(callback_times[-1] + period)
        # Scheduling must not relabel the physical interpolation timestamps.
        assert all(t == callback_times[n]
                   for t, n in zip(actual[:-1], sorted(selected)[:-1]))
        assert actual[-1] == final
        # Physical callback timestamps retain repeated-addition roundoff. The
        # bound covers that sum, while a callback-period delay remains visible.
        tolerance = 4 * math.ulp(max(1., abs(final))) * (last_step + 1)
        assert max(abs(t - target) for t, target in zip(actual, expected)) <= tolerance, (number, actual, expected)
        assert all(t < u for t, u in zip(actual, actual[1:])), actual
        assert all(math.isfinite(float(value)) for row in rows for value in row.values())
        with (folder / "output/ida_step_history.csv").open() as stream:
            history = list(csv.DictReader(stream))
        assert len(history) == len(rows)
        assert all(abs(float(item["output_time"]) - t) <= tolerance
                   for item, t in zip(history, actual))
        if number == 0:
            assert len([t for t in expected if 9. <= t <= 10.]) == 21
    print("Long-run cadence, nonmultiple intervals, nonzero start and unique final output: PASSED")


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
