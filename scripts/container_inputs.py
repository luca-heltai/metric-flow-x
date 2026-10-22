#!/usr/bin/env python3
"""Hash tracked files that affect the installed image or its runtime checks."""

import hashlib
from pathlib import Path
import subprocess


INPUT_PATHS = (
    "Dockerfile", ".dockerignore", "CMakeLists.txt", "LICENSE.md",
    "apps", "include", "source", "cmake", "third_party", "tests",
    "tools/package_smoke", "scripts/run_container.sh",
    "scripts/container_inputs.py", ".github/workflows/container.yml",
)


def fingerprint(root):
    root = Path(root)
    files = subprocess.check_output(
        ["git", "ls-files", "-z", "--", *INPUT_PATHS], cwd=root
    ).split(b"\0")
    digest = hashlib.sha256()
    for name in sorted(filter(None, files)):
        path = root / name.decode("utf-8")
        content = path.read_bytes()
        digest.update(len(name).to_bytes(8, "big"))
        digest.update(name)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


if __name__ == "__main__":
    print(fingerprint(Path(__file__).resolve().parents[1]))
