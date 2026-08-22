#!/usr/bin/env python3
"""Check the repository's canonical bibliography and tracked citations.

This is deliberately a small source-backed checker, not a bibliographic
metadata resolver. It validates keys and citation references and checks only
DOIs that are already present in the tracked bibliography inputs.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


KEY_RE = re.compile(r"^[A-Za-z][A-Za-z0-9:_+.-]*$")
ENTRY_RE = re.compile(r"@([A-Za-z][A-Za-z0-9_-]*)\s*\{")
CITATION_RE = re.compile(r"\\(?:no)?cite[A-Za-z*]*\s*\{([^{}]*)\}")
MYST_CITATION_RE = re.compile(r"\{cite[A-Za-z-]*\}`([^`]+)`")
PANDOC_CITATION_RE = re.compile(r"\[@([^\]]+)\]")
DOI_RE = re.compile(r"\bdoi\s*=\s*[\{\"]\s*([^\}\"\s,]+)", re.IGNORECASE)

# This value is copied verbatim from the MR3345205 entry in both tracked
# source bibliographies. Do not add DOI values without source-backed metadata.
CORE_DOI_ALLOWLIST = {"MR3345205": "10.1016/j.jcp.2015.04.009"}
SOURCE_BIBS = ("doc/references.bib", "latex/metric_flow.bib")


@dataclass
class BibEntry:
    key: str
    body: str
    start_line: int


def line_number(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def parse_bib(text: str, source: str) -> tuple[list[BibEntry], list[str]]:
    """Parse entry boundaries and keys without attempting full BibTeX parsing."""
    errors: list[str] = []
    entries: list[BibEntry] = []
    positions = list(ENTRY_RE.finditer(text))
    for index, match in enumerate(positions):
        opening = match.end() - 1
        depth = 0
        in_quote = False
        escaped = False
        closing = None
        for position in range(opening, len(text)):
            char = text[position]
            if escaped:
                escaped = False
                continue
            if char == "\\":
                escaped = True
                continue
            if char == '"':
                in_quote = not in_quote
            if in_quote:
                continue
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    closing = position
                    break
        if closing is None:
            errors.append(
                f"{source}:{line_number(text, match.start())}: unclosed bibliography entry"
            )
            continue

        body = text[match.end() : closing]
        comma = body.find(",")
        if comma < 0:
            errors.append(
                f"{source}:{line_number(text, match.start())}: entry has no key separator"
            )
            continue
        key = body[:comma].strip()
        if not key:
            errors.append(
                f"{source}:{line_number(text, match.start())}: empty bibliography key"
            )
            continue
        entries.append(BibEntry(key, body[comma + 1 :], line_number(text, match.start())))

        # If another @entry marker occurs before this entry's closing brace,
        # the source is malformed (the next marker will still be parsed).
        if index + 1 < len(positions) and positions[index + 1].start() < closing:
            errors.append(
                f"{source}:{line_number(text, positions[index + 1].start())}: nested bibliography entry"
            )
    return entries, errors


def control_character_errors(text: str, source: str) -> list[str]:
    errors = []
    for offset, char in enumerate(text):
        codepoint = ord(char)
        is_control = (codepoint < 32 and char not in "\n\r\t") or 0x7F <= codepoint <= 0x9F
        if is_control:
            errors.append(
                f"{source}:{line_number(text, offset)}: control character U+{codepoint:04X}"
            )
    return errors


def read_tracked_sources(root: Path) -> list[Path]:
    try:
        output = subprocess.check_output(
            [
                "git",
                "-C",
                str(root),
                "ls-files",
                "-z",
                "--",
                "*.tex",
                "*.md",
                "*.markdown",
            ],
            text=False,
        )
    except (OSError, subprocess.CalledProcessError):
        # This fallback keeps the checker useful in a source archive without a
        # .git directory; normal repository runs always use tracked files.
        return sorted(
            path
            for pattern in ("*.tex", "*.md", "*.markdown")
            for path in root.rglob(pattern)
            if ".git" not in path.parts
        )
    return [root / item.decode("utf-8") for item in output.split(b"\0") if item]


def split_citation_keys(value: str) -> Iterable[str]:
    for item in re.split(r"[,;]", value):
        key = item.strip()
        if key.startswith("@"):
            key = key[1:].strip()
        if key and key != "*":
            yield key


def citations_from_file(text: str) -> tuple[list[str], list[str]]:
    keys: list[str] = []
    malformed: list[str] = []
    for match in CITATION_RE.finditer(text):
        for key in split_citation_keys(match.group(1)):
            if KEY_RE.fullmatch(key):
                keys.append(key)
            else:
                malformed.append(key)
    for pattern in (MYST_CITATION_RE, PANDOC_CITATION_RE):
        for match in pattern.finditer(text):
            for key in split_citation_keys(match.group(1)):
                if KEY_RE.fullmatch(key):
                    keys.append(key)
                else:
                    malformed.append(key)
    return keys, malformed


def source_dois(root: Path) -> dict[str, set[str]]:
    values: dict[str, set[str]] = {}
    for relative in SOURCE_BIBS:
        path = root / relative
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        entries, _ = parse_bib(text, relative)
        for entry in entries:
            match = DOI_RE.search(entry.body)
            if match:
                values.setdefault(entry.key, set()).add(match.group(1))
    return values


def check(root: Path, bib_path: Path) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    notes: list[str] = []
    try:
        bib_text = bib_path.read_text(encoding="utf-8")
    except OSError as exc:
        return [f"{bib_path}: cannot read canonical bibliography: {exc}"], notes

    errors.extend(control_character_errors(bib_text, str(bib_path)))
    entries, parse_errors = parse_bib(bib_text, str(bib_path))
    errors.extend(parse_errors)

    keys: dict[str, BibEntry] = {}
    for entry in entries:
        if not KEY_RE.fullmatch(entry.key):
            errors.append(
                f"{bib_path}:{entry.start_line}: malformed bibliography key {entry.key!r}"
            )
        if entry.key in keys:
            errors.append(
                f"{bib_path}:{entry.start_line}: duplicate bibliography key {entry.key!r}"
            )
        keys.setdefault(entry.key, entry)

    cited: dict[str, list[str]] = {}
    for path in read_tracked_sources(root):
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            errors.append(f"{path}: cannot read tracked source: {exc}")
            continue
        errors.extend(control_character_errors(text, str(path)))
        found, malformed = citations_from_file(text)
        for key in found:
            cited.setdefault(key, []).append(str(path))
        for key in malformed:
            errors.append(f"{path}: malformed citation key {key!r}")

    for key, paths in sorted(cited.items()):
        if key not in keys:
            errors.append(
                f"missing citation key {key!r} (referenced by {', '.join(sorted(set(paths)))})"
            )

    backed_dois = source_dois(root)
    for key, expected in CORE_DOI_ALLOWLIST.items():
        source_values = backed_dois.get(key, set())
        if expected not in source_values:
            errors.append(
                f"DOI allowlist entry {key!r} is not present verbatim in source-backed metadata"
            )
            continue
        entry = keys.get(key)
        if entry is None:
            errors.append(f"DOI allowlist key {key!r} is missing from canonical bibliography")
            continue
        actual = DOI_RE.search(entry.body)
        if actual is None:
            errors.append(f"DOI allowlist key {key!r} has no DOI in canonical bibliography")
        elif actual.group(1) != expected:
            errors.append(
                f"DOI mismatch for {key!r}: expected {expected!r}, found {actual.group(1)!r}"
            )
        else:
            notes.append(f"checked source-backed DOI {key}={expected}")

    return errors, notes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="repository root (default: inferred from this script)",
    )
    parser.add_argument(
        "--bib",
        type=Path,
        default=None,
        help="canonical bibliography path (default: ROOT/bibliography/references.bib)",
    )
    args = parser.parse_args(argv)
    root = args.root.resolve()
    bib_path = (args.bib or root / "bibliography" / "references.bib").resolve()
    errors, notes = check(root, bib_path)
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        print(f"bibliography check failed: {len(errors)} error(s)", file=sys.stderr)
        return 1
    print(f"bibliography check passed: {bib_path}")
    for note in notes:
        print(f"  {note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
