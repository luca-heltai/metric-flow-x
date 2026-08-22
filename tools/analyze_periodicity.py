#!/usr/bin/env python3
"""Compare the final two complete periods in explicitly structured trace CSV data.

The input schema is deliberately narrow: one ``time`` column followed by one or
more columns whose names start with ``trace_``.  A period is supplied explicitly
by the caller (it is not estimated from the data).  The final timestamp anchors
the two windows ``[t_end - 2P, t_end - P]`` and ``[t_end - P, t_end]``.  Values
are linearly interpolated on the union of observed phase locations before the
normalized L2 difference is evaluated.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from bisect import bisect_right
from pathlib import Path
from typing import Sequence


class PeriodicityError(ValueError):
    """Raised when a trace file cannot support the requested comparison."""


def _read_csv(path: str | Path) -> tuple[list[float], dict[str, list[float]]]:
    path = Path(path)
    try:
        with path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames is None:
                raise PeriodicityError(f"{path}: CSV has no header")
            fields = [field.strip() for field in reader.fieldnames]
            if "time" not in fields:
                raise PeriodicityError(f"{path}: required 'time' column is missing")
            traces = [field for field in fields if field.startswith("trace_")]
            if not traces:
                raise PeriodicityError(
                    f"{path}: no trace_* columns found; name each value column 'trace_<name>'"
                )
            unexpected = [field for field in fields if field != "time" and field not in traces]
            if unexpected:
                raise PeriodicityError(
                    f"{path}: unsupported columns {unexpected}; use only 'time' and 'trace_*' columns"
                )
            times: list[float] = []
            values = {name: [] for name in traces}
            for row_number, row in enumerate(reader, start=2):
                try:
                    time = float(row["time"])
                    row_values = {name: float(row[name]) for name in traces}
                except (TypeError, ValueError) as exc:
                    raise PeriodicityError(f"{path}: non-numeric value on row {row_number}") from exc
                if not math.isfinite(time) or any(not math.isfinite(value) for value in row_values.values()):
                    raise PeriodicityError(f"{path}: non-finite value on row {row_number}")
                times.append(time)
                for name, value in row_values.items():
                    values[name].append(value)
    except OSError as exc:
        raise PeriodicityError(f"{path}: cannot read CSV: {exc}") from exc

    if len(times) < 4:
        raise PeriodicityError(f"{path}: at least four samples are required for two complete periods")
    if any(later <= earlier for earlier, later in zip(times, times[1:])):
        raise PeriodicityError(f"{path}: time values must be strictly increasing")
    return times, values


def _interpolate(times: Sequence[float], values: Sequence[float], target: float) -> float:
    if target < times[0] or target > times[-1]:
        raise PeriodicityError(f"cannot interpolate outside trace range at t={target:g}")
    index = bisect_right(times, target)
    if index == 0:
        return values[0]
    if index == len(times):
        return values[-1]
    left = index - 1
    if times[left] == target:
        return values[left]
    fraction = (target - times[left]) / (times[index] - times[left])
    return values[left] + fraction * (values[index] - values[left])


def _phases(times: Sequence[float], start: float, period: float) -> list[float]:
    """Return deterministic phase nodes observed in either complete window."""
    end = start + 2.0 * period
    nodes = {0.0, period}
    for timestamp in times:
        if start <= timestamp <= end:
            phase = timestamp - start
            if phase <= period:
                nodes.add(phase)
            else:
                nodes.add(phase - period)
    return sorted(nodes)


def _sample_count(times: Sequence[float], start: float, end: float) -> int:
    return sum(start <= timestamp <= end for timestamp in times)


def analyze(path: str | Path, period: float) -> dict[str, object]:
    """Return normalized L2 differences for the final two complete periods.

    ``period`` is required because this tool intentionally does not infer a
    frequency from potentially transient or noisy solver output.
    """
    if not math.isfinite(period) or period <= 0:
        raise PeriodicityError("period must be a finite positive number")
    times, traces = _read_csv(path)
    end = times[-1]
    first_start, middle, second_end = end - 2.0 * period, end - period, end
    if times[0] > first_start:
        raise PeriodicityError(
            f"insufficient data for two complete periods: need samples from t={first_start:g} to t={end:g}"
        )
    if _sample_count(times, first_start, middle) < 2 or _sample_count(times, middle, second_end) < 2:
        raise PeriodicityError(
            "insufficient data for two complete periods: each cycle needs at least two samples"
        )
    phases = _phases(times, first_start, period)
    per_trace: dict[str, float] = {}
    numerator_total = 0.0
    denominator_total = 0.0
    for name, values in traces.items():
        reference = [_interpolate(times, values, first_start + phase) for phase in phases]
        candidate = [_interpolate(times, values, middle + phase) for phase in phases]
        numerator = math.sqrt(sum((a - b) ** 2 for a, b in zip(reference, candidate)))
        denominator = math.sqrt(sum(a * a for a in reference))
        if denominator == 0.0:
            if numerator != 0.0:
                raise PeriodicityError(f"trace {name!r} has zero reference norm; normalized L2 is undefined")
            difference = 0.0
        else:
            difference = numerator / denominator
        per_trace[name] = difference
        numerator_total += numerator * numerator
        denominator_total += denominator * denominator
    if denominator_total == 0.0:
        normalized = 0.0
    else:
        normalized = math.sqrt(numerator_total / denominator_total)
    return {
        "period": period,
        "windows": {
            "reference": [first_start, middle],
            "candidate": [middle, second_end],
        },
        "phase_samples": len(phases),
        "traces": per_trace,
        "normalized_l2": normalized,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help="CSV with time and trace_* columns")
    parser.add_argument("--period", type=float, required=True, help="documented cycle period")
    parser.add_argument("--json", type=Path, help="write the result JSON to this path")
    args = parser.parse_args(argv)
    try:
        result = analyze(args.csv, args.period)
    except PeriodicityError as exc:
        print(f"analyze_periodicity.py: {exc}", file=sys.stderr)
        return 2
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.json:
        args.json.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
