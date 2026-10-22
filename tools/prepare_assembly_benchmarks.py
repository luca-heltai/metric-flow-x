#!/usr/bin/env python3
"""Prepare explicit, hashed input files for the assembly benchmark."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def assign(text, field, value):
    result, count = re.subn(r"(?m)^(\s*set\s+" + re.escape(field) + r"\s*=).*$",
                            lambda match: match.group(1) + " " + str(value), text)
    if count != 1:
        raise ValueError(f"Expected one parameter entry for {field}, found {count}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adan56-mesh", type=Path, required=True)
    parser.add_argument("--network10k-mesh", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    destination = args.output_dir.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    templates = {
        "adan56": (ROOT / "tutorials/04_adan56/parameters.prm.in", args.adan56_mesh.resolve()),
        "network10k": (ROOT / "parameters/networks_by_svv/network_10k.prm", args.network10k_mesh.resolve()),
    }
    manifest = {"source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "inputs": {}}
    for name, (template, mesh) in templates.items():
        text = template.read_text().replace("@SOURCE_DIR@", str(ROOT))
        text = text.replace("subsection BloodFlowSystem<", "subsection MetricFlowSystem<")
        text = text.replace("subsection Blood Flow Parameters", "subsection Metric Flow Parameters")
        text = assign(text, "Vtk file path for mesh input", mesh)
        if name == "network10k":
            text = assign(text, "Finite element degree", 1)
            text = assign(text, "Number of global refinement", 0)
            lines = text.splitlines(keepends=True)
            inlets = [i for i, line in enumerate(lines) if line.strip().startswith("set Inflow function")]
            text = "".join(line for i, line in enumerate(lines) if i not in inlets[:-1])
        output = destination / f"{name}.prm"
        output.write_text(text)
        manifest["inputs"][name] = {"parameter_file": str(output), "parameter_sha256": digest(output),
                                     "mesh_file": str(mesh), "mesh_sha256": digest(mesh)}
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(destination / "manifest.json")


if __name__ == "__main__":
    main()
