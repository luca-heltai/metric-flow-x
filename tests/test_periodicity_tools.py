"""Deterministic focused tests for the WP-10 periodicity diagnostic."""
from __future__ import annotations

import csv
import math
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from analyze_periodicity import PeriodicityError, analyze  # noqa: E402


def _write_trace(path: Path, times: list[float], values: list[float], *, name: str = "trace_flow") -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["time", name])
        writer.writerows(zip(times, values))


def test_identical_final_periods_have_zero_normalized_l2(tmp_path: Path) -> None:
    # Two identical cycles with an intentionally nonuniform sampling grid.
    times = [0.0, 0.25, 0.75, 1.0, 1.25, 1.75, 2.0]
    values = [0.0, 1.0, -1.0, 0.0, 1.0, -1.0, 0.0]
    path = tmp_path / "trace.csv"
    _write_trace(path, times, values)
    result = analyze(path, period=1.0)
    assert result["normalized_l2"] == pytest.approx(0.0)
    assert result["windows"] == {"reference": [0.0, 1.0], "candidate": [1.0, 2.0]}


def test_multiple_traces_report_deterministic_difference(tmp_path: Path) -> None:
    path = tmp_path / "traces.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["time", "trace_a", "trace_b"])
        writer.writerows([[0, 0, 0], [0.5, 0, 0], [1, 1, 2], [1.5, 1, 2], [2, 2, 4], [2.5, 2, 4], [3, 3, 6]])
    result = analyze(path, period=1.0)
    expected = math.sqrt(1.0 / 2.0)
    assert result["traces"]["trace_a"] == pytest.approx(expected)
    assert result["traces"]["trace_b"] == pytest.approx(expected)


def test_insufficient_data_and_ambiguous_columns_fail_clearly(tmp_path: Path) -> None:
    short = tmp_path / "short.csv"
    _write_trace(short, [0, 0.5, 1.0], [1, 1, 1])
    with pytest.raises(PeriodicityError, match="at least four samples"):
        analyze(short, period=1.0)

    malformed = tmp_path / "malformed.csv"
    malformed.write_text("time,value\n0,1\n1,1\n2,1\n3,1\n")
    with pytest.raises(PeriodicityError, match="trace_.*columns"):
        analyze(malformed, period=1.0)
