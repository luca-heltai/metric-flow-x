#!/usr/bin/env python3
"""Parse the immutable convergence report and render tables.

The parser retains the displayed error/rate strings verbatim.  This prevents
rounding or hand-copying from silently changing the reference evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Dict, List

DEGREES = re.compile(r"polynomial degree\s*=\s*(\d+)", re.IGNORECASE)
ROW = re.compile(
    r"^\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|"
    r"\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|"
    r"\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*$"
)
COLUMNS = ["Cycle", "DoFs", "A L2", "A L2 rate", "A H1", "A H1 rate", "U L2", "U L2 rate", "U H1", "U H1 rate"]


def parse_report(text: str) -> List[Dict]:
    """Parse all degree sections from *text* in one pass."""
    tables: List[Dict] = []
    current = None
    for line in text.splitlines():
        match = DEGREES.search(line)
        if match:
            current = {"polynomial_degree": int(match.group(1)), "columns": COLUMNS[:], "rows": []}
            tables.append(current)
            continue
        if current is None:
            continue
        row = ROW.match(line.strip())
        if row:
            values = [value.strip() for value in row.groups()]
            current["rows"].append(
                {"Cycle": int(values[0]), "DoFs": int(values[1]), **dict(zip(COLUMNS[2:], values[2:]))}
            )
    if not tables or any(not table["rows"] for table in tables):
        raise ValueError("reference report did not contain complete degree tables")
    return tables


def canonical_document(text: str, source_name: str = "convergence.txt") -> Dict:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return {
        "schema": "blood-flow-convergence-v1",
        "provenance": {
            "source": source_name,
            "sha256": digest,
            "note": "Values are parsed once from the immutable TXT reference; displayed error/rate strings are retained verbatim.",
        },
        "tables": parse_report(text),
    }


def markdown(document: Dict) -> str:
    lines = []
    for table in document["tables"]:
        lines += [f"#### Polynomial degree {table['polynomial_degree']}", "", "| " + " | ".join(table["columns"]) + " |", "|" + "|".join("---" for _ in table["columns"]) + "|"]
        for row in table["rows"]:
            lines.append("| " + " | ".join(str(row[column]) for column in table["columns"]) + " |")
        lines.append("")
    return "\n".join(lines)


def tex(document: Dict) -> str:
    lines = []
    for table in document["tables"]:
        lines += [f"% Polynomial degree {table['polynomial_degree']}", r"\begin{tabular}{rrrrrrrrrr}", " & ".join(table["columns"]) + r" \\", r"\hline"]
        for row in table["rows"]:
            lines.append(" & ".join(str(row[column]) for column in table["columns"]) + r" \\")
        lines += [r"\end{tabular}", ""]
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--json", type=Path, help="write canonical JSON")
    parser.add_argument("--format", choices=["markdown", "tex"], help="render a table")
    parser.add_argument("--output", type=Path, help="table output path (default stdout)")
    args = parser.parse_args(argv)
    text = args.input.read_text(encoding="utf-8")
    document = canonical_document(text, str(args.input))
    if args.json:
        args.json.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    if args.format:
        rendered = markdown(document) if args.format == "markdown" else tex(document)
        if args.output:
            args.output.write_text(rendered, encoding="utf-8")
        else:
            print(rendered)
    elif not args.json:
        print(json.dumps(document, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
