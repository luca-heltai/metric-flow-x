#!/usr/bin/env python3
"""Collect completed new runs into portable SI sample tables, without comparisons."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import sys


SOURCE_COLUMNS = ("time_s", "P_dynpcm2", "Q_cm3ps", "A_cm2", "U_cmps")
OUTPUT_COLUMNS = ("time_s", "pressure_pa", "flow_m3_s", "area_m2", "velocity_m_s")
FACTORS = (1.0, 0.1, 1e-6, 1e-4, 1e-2)
DIAGNOSTICS = ("mass_conservation.csv", "ida_statistics.csv", "scaling_timings.csv")


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def collect(root: Path, output: Path, vessels: list[int] | None,
            start: float, end: float) -> dict:
    receipts = sorted(root.glob("*/run.json"))
    completed = [(path, json.loads(path.read_text())) for path in receipts]
    completed = [(path, receipt) for path, receipt in completed if receipt["status"] == "completed"]
    if not completed:
        raise ValueError("No completed run receipts found")
    output.mkdir(parents=True, exist_ok=False)
    (output / "diagnostics").mkdir()
    provenance = {
        "sample_location": "Solver vessel probes near the arc-length midpoint; no junction traces inferred.",
        "source_columns": SOURCE_COLUMNS, "output_columns": OUTPUT_COLUMNS,
        "conversion_factors": FACTORS,
        "sampling": "Original solver sample times; no interpolation or resampling.",
        "time_window_s": [start if math.isfinite(start) else None,
                          end if math.isfinite(end) else None],
        "vessels": vessels, "sources": [], "missing_diagnostics": [],
        "validation": "Collected new outputs only; no archived-data or reference-curve comparison.",
    }
    rows = 0
    with (output / "samples.csv").open("w", newline="") as samples, \
            (output / "runs.csv").open("w", newline="") as runs:
        writer = csv.writer(samples)
        writer.writerow(("case_id", "vessel_id", *OUTPUT_COLUMNS))
        summary = csv.writer(runs)
        summary.writerow(("case_id", "mpi", "final_time_override", "workflow_revision",
                          "executable_revision", "receipt_sha256"))
        for receipt_path, receipt in completed:
            case_id = receipt["case"]["id"]
            if not re.fullmatch(r"[a-z0-9_\-]+", case_id):
                raise ValueError(f"Invalid case ID in {receipt_path}")
            provenance["sources"].append({"path": str(receipt_path), "sha256": digest(receipt_path)})
            summary.writerow((case_id, receipt["mpi"], receipt.get("final_time_override"),
                              receipt.get("workflow_revision"), receipt.get("executable_revision"),
                              digest(receipt_path)))
            results = receipt_path.parent / "results"
            probes = sorted(results.glob("HDG_IDA_Vessel_*.csv"))
            if not probes:
                raise ValueError(f"No vessel probe CSVs in completed case {case_id}")
            found = set()
            for path in probes:
                vessel = int(path.stem.removeprefix("HDG_IDA_Vessel_"))
                if vessels is not None and vessel not in vessels:
                    continue
                found.add(vessel)
                provenance["sources"].append({"path": str(path), "sha256": digest(path)})
                with path.open(newline="") as source:
                    reader = csv.DictReader(source)
                    if not set(SOURCE_COLUMNS).issubset(reader.fieldnames or []):
                        raise ValueError(f"Missing explicit probe-unit columns in {path}")
                    for row in reader:
                        values = [float(row[key]) * factor for key, factor in zip(SOURCE_COLUMNS, FACTORS)]
                        if not all(math.isfinite(value) for value in values):
                            raise ValueError(f"Non-finite sample in {path}")
                        if start <= values[0] <= end:
                            writer.writerow((case_id, vessel, *values))
                            rows += 1
            if vessels is not None and set(vessels) - found:
                raise ValueError(f"Missing requested vessels {sorted(set(vessels) - found)} in {case_id}")
            for name in DIAGNOSTICS:
                path = results / name
                if path.is_file():
                    shutil.copyfile(path, output / "diagnostics" / f"{case_id}_{name}")
                    provenance["sources"].append({"path": str(path), "sha256": digest(path)})
                else:
                    provenance["missing_diagnostics"].append({"case_id": case_id, "file": name})
    provenance["sample_rows"] = rows
    provenance["completed_runs"] = len(completed)
    (output / "collection.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return provenance


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--output", type=Path, required=True, help="New directory for collected tables")
    parser.add_argument("--vessels", nargs="+", type=int, help="Optional explicit vessel IDs")
    parser.add_argument("--start", type=float, default=-math.inf, help="First sample time in seconds, inclusive")
    parser.add_argument("--end", type=float, default=math.inf, help="Last sample time in seconds, inclusive")
    args = parser.parse_args(argv)
    if math.isnan(args.start) or math.isnan(args.end) or args.start > args.end:
        parser.error("Invalid sample time window")
    try:
        result = collect(args.run_root.resolve(), args.output.resolve(),
                         args.vessels, args.start, args.end)
    except (OSError, ValueError, KeyError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    print(f"Collected {result['sample_rows']} samples from {result['completed_runs']} completed runs")
    if result["missing_diagnostics"]:
        print(f"{len(result['missing_diagnostics'])} diagnostic files absent; see collection.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
