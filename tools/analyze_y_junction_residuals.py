#!/usr/bin/env python3
"""Compute scaled Y-junction residual metrics from explicitly named raw columns.

Input is CSV (one sample per row).  This tool intentionally does not guess which
solver output columns are residuals: absent columns are an error, not a reason to
invent zero values or substitute state variables.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Iterable

DEFAULT_MASS = ("mass_residual",)
DEFAULT_HEAD = ("head_residual_0", "head_residual_1")
DEFAULT_COMPATIBILITY = (
    "compatibility_residual_0",
    "compatibility_residual_1",
    "compatibility_residual_2",
)


def _names(value: str) -> tuple[str, ...]:
    names = tuple(item.strip() for item in value.split(",") if item.strip())
    if not names:
        raise ValueError("column list must contain at least one explicitly named column")
    return names


def _read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    try:
        with path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames is None:
                raise ValueError("CSV input has no header; explicit raw column names are required")
            rows = list(reader)
            return list(reader.fieldnames), rows
    except OSError as exc:
        raise ValueError(f"cannot read input {path}: {exc}") from exc


def _metric(values: Iterable[float], scale: float) -> dict[str, float | int]:
    scaled = [abs(value) * scale for value in values]
    if not scaled:
        return {"max_abs": 0.0, "rms": 0.0, "samples": 0}
    return {
        "max_abs": max(scaled),
        "rms": math.sqrt(sum(value * value for value in scaled) / len(scaled)),
        "samples": len(scaled),
    }


def analyze(
    path: str | Path,
    *,
    mass_columns: tuple[str, ...] = DEFAULT_MASS,
    head_columns: tuple[str, ...] = DEFAULT_HEAD,
    compatibility_columns: tuple[str, ...] = DEFAULT_COMPATIBILITY,
    mass_scale: float = 1.0,
    head_scale: float = 1.0,
    compatibility_scale: float = 1.0,
) -> dict[str, object]:
    input_path = Path(path)
    columns, rows = _read_rows(input_path)
    required = mass_columns + head_columns + compatibility_columns
    missing = [name for name in required if name not in columns]
    if missing:
        raise ValueError(
            "missing required raw residual columns: " + ", ".join(missing) +
            "; refusing to fabricate residual values (export these solver columns explicitly)"
        )
    for name, scale in (("mass", mass_scale), ("head", head_scale), ("compatibility", compatibility_scale)):
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError(f"{name} scale must be finite and positive, got {scale!r}")

    def values(names: tuple[str, ...]) -> list[float]:
        result: list[float] = []
        for row_number, row in enumerate(rows, start=2):
            for name in names:
                raw = row[name].strip()
                try:
                    value = float(raw)
                except ValueError as exc:
                    raise ValueError(f"row {row_number}, raw column {name!r} is not numeric: {raw!r}") from exc
                if not math.isfinite(value):
                    raise ValueError(f"row {row_number}, raw column {name!r} is not finite: {raw!r}")
                result.append(value)
        return result

    return {
        "schema": "blood-flow.y-junction.residual-metrics.v1",
        "input": input_path.name,
        "raw_columns": {
            "mass": list(mass_columns),
            "head": list(head_columns),
            "compatibility": list(compatibility_columns),
        },
        "scales": {"mass": mass_scale, "head": head_scale, "compatibility": compatibility_scale},
        "metrics": {
            "mass": _metric(values(mass_columns), mass_scale),
            "head": _metric(values(head_columns), head_scale),
            "compatibility": _metric(values(compatibility_columns), compatibility_scale),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help="CSV output containing named raw residual columns")
    parser.add_argument("--mass-columns", default=','.join(DEFAULT_MASS))
    parser.add_argument("--head-columns", default=','.join(DEFAULT_HEAD))
    parser.add_argument("--compatibility-columns", default=','.join(DEFAULT_COMPATIBILITY))
    parser.add_argument("--mass-scale", type=float, default=1.0)
    parser.add_argument("--head-scale", type=float, default=1.0)
    parser.add_argument("--compatibility-scale", type=float, default=1.0)
    args = parser.parse_args(argv)
    try:
        result = analyze(
            args.csv,
            mass_columns=_names(args.mass_columns),
            head_columns=_names(args.head_columns),
            compatibility_columns=_names(args.compatibility_columns),
            mass_scale=args.mass_scale,
            head_scale=args.head_scale,
            compatibility_scale=args.compatibility_scale,
        )
    except ValueError as exc:
        print(f"analyze_y_junction_residuals.py: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
