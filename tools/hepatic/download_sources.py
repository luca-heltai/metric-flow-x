#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Download checked upstream hepatic datasets without committing medical data."""

import argparse
import hashlib
import json
import shutil
import stat
import subprocess
import zipfile
from pathlib import Path, PurePosixPath

DEFAULT_ROOT = Path(__file__).resolve().parents[2] / "datasets" / "hepatic"


def file_digest(path, algorithm="sha256"):
    digest = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_archive(path, source):
    """Nextcloud ZIP timestamps vary: pin its immutable member bytes instead."""
    if "checksum" in source:
        algorithm, expected = source["checksum"].split(":", 1)
        if file_digest(path, algorithm) != expected:
            raise ValueError(f"Checksum mismatch for {path}")
    with zipfile.ZipFile(path) as archive:
        members = [i for i in archive.infolist() if not i.is_dir()]
        if len({i.filename for i in members}) != len(members):
            raise ValueError("Duplicate ZIP member")
        if "member_sha256" in source:
            expected = source["member_sha256"]
            if {i.filename for i in members} != set(expected):
                raise ValueError("Unexpected archive contents")
            for item in members:
                with archive.open(item) as stream:
                    actual = hashlib.file_digest(stream, "sha256").hexdigest()
                if actual != expected[item.filename]:
                    raise ValueError(f"Checksum mismatch for {item.filename}")


def extract_zip(path, destination):
    """Extract regular files only, retaining provider names and license notices."""
    destination = destination.resolve()
    with zipfile.ZipFile(path) as archive:
        for item in archive.infolist():
            parts = PurePosixPath(item.filename).parts
            if not parts or "__MACOSX" in parts or parts[-1].startswith("._"):
                continue
            target = (destination / item.filename).resolve()
            if (
                PurePosixPath(item.filename).is_absolute()
                or ".." in parts
                or not target.is_relative_to(destination)
                or stat.S_ISLNK(item.external_attr >> 16)
            ):
                raise ValueError(f"Unsafe ZIP member: {item.filename}")
            if item.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(item) as src, target.open("wb") as dst:
                    shutil.copyfileobj(src, dst)


def download(name, root, extract=False):
    catalog = json.loads(Path(__file__).with_name("sources.json").read_text())
    source = catalog[name]
    directory = root / name
    directory.mkdir(parents=True, exist_ok=True)
    archive = directory / "source.zip"
    if not archive.exists():
        temporary = archive.with_suffix(".part")
        try:
            subprocess.run(
                [
                    "curl",
                    "--fail",
                    "--location",
                    "--retry",
                    "3",
                    "--output",
                    str(temporary),
                    source["url"],
                ],
                check=True,
            )
            verify_archive(temporary, source)
            temporary.replace(archive)
        finally:
            temporary.unlink(missing_ok=True)
    else:
        verify_archive(archive, source)
    if extract:
        extract_zip(archive, directory)
        for relative in source.get("nested_archives", []):
            nested = directory / relative
            extract_zip(nested, nested.parent)
    provenance = dict(
        source,
        downloaded_archive_sha256=file_digest(archive),
        extraction_requested=extract,
    )
    (directory / "download.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(directory)


def main():
    catalog = json.loads(Path(__file__).with_name("sources.json").read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=sorted(catalog))
    parser.add_argument("--destination", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--extract", action="store_true")
    args = parser.parse_args()
    try:
        download(args.dataset, args.destination, args.extract)
    except (
        ValueError,
        OSError,
        zipfile.BadZipFile,
        subprocess.CalledProcessError,
    ) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
