#!/usr/bin/env python3
"""Print a deterministic, read-only license-notice audit.

This tool reports what recognizable notices say. It deliberately does not
assign licenses to files without notices and never edits or normalizes files.
An unresolved mixed-license tree is reported with exit status 1.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Iterable


SPDX_RE = re.compile(r"SPDX-License-Identifier:\s*([^\s*]+)")

# The audit is intentionally limited to project source/metadata locations. It
# does not inspect build products, virtual environments, caches, or subagent
# transcripts, whose third-party notices would obscure this project's audit.
SOURCE_DIRS = ("apps", "include", "source", "tests", "scripts")
TOP_LEVEL_FILES = ("LICENSE.md", "README.md", "CITATION.cff")
EXCLUDED_SUFFIXES = {".output"}


def candidate_paths(root: Path) -> Iterable[Path]:
    paths = [root / name for name in TOP_LEVEL_FILES]
    for directory in SOURCE_DIRS:
        base = root / directory
        if base.exists():
            paths.extend(
                path
                for path in base.rglob("*")
                if path.is_file() and path.suffix not in EXCLUDED_SUFFIXES
            )
    return sorted(path for path in paths if path.is_file())


def notice_for(path: Path, text: str) -> tuple[list[str], list[str]]:
    """Return sorted license identifiers and provenance labels for *text*."""

    licenses = set(SPDX_RE.findall(text))
    lowered = text.lower()
    normalized = re.sub(r"[^a-z0-9]+", " ", lowered)

    if "mit license" in lowered or "licensed under the mit" in lowered:
        licenses.add("MIT")
    if "apache-2.0" in lowered and "llvm-exception" in lowered:
        licenses.add("Apache-2.0 WITH LLVM-exception")
    if "gnu lesser general public license" in normalized:
        if "version 3 0" in normalized:
            licenses.add("LGPL-3.0-or-later")
        elif "version 2 1" in normalized:
            licenses.add("LGPL-2.1-or-later")
    if "lgpl-3.0-or-later" in lowered:
        licenses.add("LGPL-3.0-or-later")
    if "lgpl-2.1-or-later" in lowered:
        licenses.add("LGPL-2.1-or-later")

    provenance = set()
    if "fsi-suite" in lowered or "parsedtools" in lowered:
        provenance.add("FSI-suite/ParsedTools")
    if "deal.ii" in lowered and path.name not in {"README.md", "CITATION.cff"}:
        provenance.add("deal.II")
    if path.name == "LICENSE.md":
        provenance.add("root project notice")
    elif path.name in {"README.md", "CITATION.cff"}:
        provenance.add("project metadata")
    if not provenance and licenses:
        provenance.add("project or unclassified notice")
    return sorted(licenses), sorted(provenance)


def audit(root: Path) -> dict:
    files = []
    for path in candidate_paths(root):
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        licenses, provenance = notice_for(path, text)
        if licenses:
            files.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "licenses": licenses,
                    "provenance": provenance,
                }
            )

    root_notice = next(
        (entry for entry in files if entry["path"] == "LICENSE.md"),
        {"licenses": []},
    )
    root_licenses = root_notice["licenses"]
    conflicts = []
    for entry in files:
        non_root = [license_id for license_id in entry["licenses"] if license_id not in root_licenses]
        if non_root and entry["path"] != "LICENSE.md":
            conflicts.append(
                {
                    "path": entry["path"],
                    "reason": "file notice differs from root notice",
                    "licenses": entry["licenses"],
                }
            )
        if len(entry["licenses"]) > 1:
            conflicts.append(
                {
                    "path": entry["path"],
                    "reason": "multiple recognized license families in one file",
                    "licenses": entry["licenses"],
                }
            )

    # A file may trigger both checks; retain a stable unique conflict list.
    conflicts = sorted(
        {
            (item["path"], item["reason"], tuple(item["licenses"])): item
            for item in conflicts
        }.values(),
        key=lambda item: (item["path"], item["reason"], item["licenses"]),
    )
    return {
        "root": {"path": "LICENSE.md", "licenses": root_licenses},
        "files": files,
        "conflicts": conflicts,
        "status": "blocked" if conflicts else "clear",
        "proposed_message": "WP-14: record licensing conflict pending approval.",
    }


def print_text(report: dict) -> None:
    print("Licensing audit (read-only)")
    print("Root notice: {}".format(", ".join(report["root"]["licenses"]) or "unrecognized"))
    print("File notices:")
    for entry in report["files"]:
        provenance = ", ".join(entry["provenance"]) or "unclassified"
        print(
            "- {}: {} [{}]".format(
                entry["path"], ", ".join(entry["licenses"]), provenance
            )
        )
    print("Conflicts:")
    if report["conflicts"]:
        for conflict in report["conflicts"]:
            print(
                "- {}: {}; {}".format(
                    conflict["path"],
                    conflict["reason"],
                    ", ".join(conflict["licenses"]),
                )
            )
    else:
        print("- none")
    print("Status: {}".format(report["status"]))
    print("Proposed message: {}".format(report["proposed_message"]))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json", action="store_true", help="emit stable machine-readable JSON"
    )
    args = parser.parse_args(argv)
    report = audit(Path(__file__).resolve().parents[1])
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print_text(report)
    return 1 if report["conflicts"] else 0


if __name__ == "__main__":
    sys.exit(main())
