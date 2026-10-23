#!/usr/bin/env python3
"""Check the CTest build lock with four nested Ninja writers; compile nothing."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time


def record_writer(build):
    events = build / "events"
    events.mkdir(exist_ok=True)
    start = time.monotonic_ns()
    with (build / "shared_output.txt").open("a") as output:
        output.write(f"start {os.getpid()}\n")
        output.flush()
        time.sleep(0.5)
        output.write(f"end {os.getpid()}\n")
    end = time.monotonic_ns()
    (events / f"{os.getpid()}.json").write_text(
        json.dumps({"start": start, "end": end}))


def command(arguments):
    result = subprocess.run(arguments, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(result.stdout + result.stderr)
    return result.stdout


def check(root, directory, cmake, ctest, ninja):
    for mode, enabled in [("unlocked", "OFF"), ("locked", "ON")]:
        build = directory / mode
        command([cmake, "-S", str(root), "-B", str(build), "-G", "Ninja",
                 f"-DCMAKE_MAKE_PROGRAM={ninja}", f"-DAPPLY_LOCK={enabled}",
                 f"-DPython3_EXECUTABLE={sys.executable}",
                 f"-DPROJECT_TEST_CMAKE={root.parents[2] / 'tests/CMakeLists.txt'}"])
        command([ctest, "--test-dir", str(build), "-j4", "--output-on-failure"])
        listing = json.loads(command(
            [ctest, "--test-dir", str(build), "--show-only=json-v1"]))
        if len(listing["tests"]) != 4:
            raise RuntimeError("Expected four nested-Ninja tests")
        for test in listing["tests"]:
            locks = [p["value"] for p in test["properties"]
                     if p["name"] == "RESOURCE_LOCK"]
            expected = [["metric_flow_x_test_build"]] if enabled == "ON" else []
            if locks != expected:
                raise RuntimeError(f"Unexpected lock properties: {locks}")
        records = [json.loads(path.read_text())
                   for path in (build / "events").glob("*.json")]
        if len(records) != 4:
            raise RuntimeError(f"Expected four writer intervals, got {len(records)}")
        timeline = sorted([(record["start"], 1) for record in records] +
                          [(record["end"], -1) for record in records])
        active = maximum = 0
        for _, delta in timeline:
            active += delta
            maximum = max(maximum, active)
        expected = 1 if enabled == "ON" else 4
        if maximum != expected:
            raise RuntimeError(f"{mode}: expected {expected} concurrent writers, got {maximum}")
        print(f"{mode}: 4/4 passed; maximum concurrent shared-output writers: {maximum}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cmake", default=shutil.which("cmake"))
    parser.add_argument("--ctest", default=shutil.which("ctest"))
    parser.add_argument("--ninja", default=shutil.which("ninja") or shutil.which("ninja-build"))
    args = parser.parse_args()
    if not all((args.cmake, args.ctest, args.ninja)):
        parser.error("CMake, CTest and Ninja must be on PATH or provided explicitly")
    root = Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory(prefix="metric-flow-x-test-build-lock-") as directory:
        check(root, Path(directory), args.cmake, args.ctest, args.ninja)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--worker":
        record_writer(Path(sys.argv[2]))
    else:
        main()
