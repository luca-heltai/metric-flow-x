#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Fetch checksum-pinned ADAVN data; atlas access is optional and restricted."""
import argparse
import hashlib
import json
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from recover_geometry import DEFAULT_DATA_DIR

SOURCES = Path(__file__).with_name("sources.json")


def download(url, target, expected_sha256):
    """Reuse verified files and reject changed bytes before writing a file."""
    if target.exists():
        data = target.read_bytes()
    else:
        for attempt in range(3):
            try:
                with urllib.request.urlopen(url, timeout=30) as response:
                    data = response.read()
                break
            except (urllib.error.URLError, TimeoutError):
                if attempt == 2:
                    raise
    actual = hashlib.sha256(data).hexdigest()
    if actual != expected_sha256:
        raise ValueError(
            f"SHA-256 mismatch for {target}: expected {expected_sha256}, got {actual}"
        )
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    return {"path": str(target), "url": url, "sha256": actual}


def fetch(root=DEFAULT_DATA_DIR, include_atlas=False, atlas_permission=False):
    if include_atlas and not atlas_permission:
        raise ValueError(
            "Atlas modification is restricted by HeMoLab terms. Use --atlas-permission only if you have the required written permission; otherwise use the CSV-only default."
        )
    source = json.loads(SOURCES.read_text())
    item = source["csv"]
    receipts = [download(item["url"], root / "source" / item["path"], item["sha256"])]
    if include_atlas:
        atlas = source["atlas"]

        def fetch_surface(item):
            name, checksum = item
            filename = "SW_" + name + ".vtk"
            return download(
                atlas["base_url"] + "MESHES/arterial/" + filename,
                root / "source/surfaces" / filename,
                checksum,
            )

        with ThreadPoolExecutor(max_workers=6) as pool:
            for i, receipt in enumerate(
                pool.map(fetch_surface, atlas["files"].items()), 1
            ):
                receipts.append(receipt)
                if i % 200 == 0:
                    print(f'Verified {i}/{len(atlas["files"])} surfaces', flush=True)
    root.mkdir(parents=True, exist_ok=True)
    (root / "download_receipt.json").write_text(json.dumps(receipts, indent=2) + "\n")
    print(f"Verified {len(receipts)} source files in {root}")
    return receipts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--include-atlas",
        action="store_true",
        help="Also download the separately licensed surface atlas",
    )
    parser.add_argument(
        "--atlas-permission",
        action="store_true",
        help="Confirm that you hold the written permission required for your intended atlas use; this does not grant permission",
    )
    args = parser.parse_args()
    fetch(args.data_dir, args.include_atlas, args.atlas_permission)
