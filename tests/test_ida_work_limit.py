#!/usr/bin/env python3
"""A native IDA work-limit failure must exit before emitting an invalid state."""
import argparse
import csv
import os
from pathlib import Path
import subprocess
import tempfile


def validate(executable, source_dir, destination):
    destination.mkdir(parents=True, exist_ok=True)
    template = (source_dir / "parameters/constant.prm.in").read_text().replace("@SOURCE_DIR@", str(source_dir))
    template = template.replace("= 4\n", "= 0\n").replace("= 1.1\n", "= 0.01\n")
    environment = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    for limit in (1, 10000):
        folder = destination / str(limit)
        folder.mkdir(exist_ok=True)
        output = folder / "output"
        extra = (f"  set Output directory = {output}\n"
                 f"  set Maximum IDA internal steps per output = {limit}\n"
                 "  set Initial condition policy = native\n")
        (folder / "input.prm").write_text(template.replace("  set Output filename", extra + "  set Output filename"))
        result = subprocess.run([str(executable), "input.prm"], cwd=folder,
                                env=environment, capture_output=True, text=True, timeout=60)
        (folder / "run.log").write_text(result.stdout + result.stderr)
        with (output / "mass_conservation.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        if limit == 1:
            assert result.returncode != 0
            assert "SUNDIALS IDA reported error -" in result.stderr
            assert "IDASolve" in result.stderr and "mxstep" in result.stderr
            assert len(rows) == 1 and float(rows[0]["t"]) == 0.
            assert not (output / "initialization_statistics.json").exists()
            assert not (output / "ida_statistics.csv").exists()
        else:
            assert result.returncode == 0, result.stdout + result.stderr
            assert len(rows) == 3 and float(rows[-1]["t"]) == .01
            assert (output / "initialization_statistics.json").exists()
            assert (output / "ida_statistics.csv").exists()
    print("IDA work exhaustion exits without invalid output; unchanged controls succeed with sufficient work: PASSED")


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
