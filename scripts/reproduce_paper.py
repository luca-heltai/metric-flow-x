#!/usr/bin/env python3
"""Prepare or run the canonical current-code paper experiments (Python stdlib)."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "reproducibility/manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def replace_setting(text: str, key: str, value: str) -> str:
    pattern = rf"(?m)^(\s*set {re.escape(key)}\s*=)[^\n]*$"
    result, count = re.subn(pattern, lambda match: match[1] + " " + value, text)
    if count != 1:
        raise ValueError(f"Expected one setting {key!r}; found {count}")
    return result


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def verify_executable_revision(required: str, actual: str | None, root: Path = ROOT) -> None:
    """Check the explicitly attested build revision against the MMS prerequisite."""
    if actual is None or re.fullmatch(r"[0-9a-fA-F]{40}", actual) is None:
        raise ValueError("MMS execution requires --executable-revision with the full 40-character "
                         "source commit recorded when the executable was built")
    if actual.lower() == required.lower():
        return  # Exact required revision also works without a complete Git history.
    result = subprocess.run(
        ["git", "-C", str(root), "merge-base", "--is-ancestor", required, actual],
        capture_output=True, text=True, check=False,
    )
    if result.returncode:
        raise ValueError(f"Cannot establish that executable revision {actual} includes the MMS "
                         f"prerequisite {required}. Use a supported build and fetch its commit "
                         "history into this checkout (including shallow/release checkouts), "
                         "or attest the exact prerequisite revision if that is what was built.")


def run_case(case: dict, manifest: dict, args: argparse.Namespace) -> int:
    base = case["base"]
    mesh_info = manifest["meshes"][base]
    mesh = args.dataset_root / mesh_info["path"]
    if not mesh.is_file():
        raise ValueError(f"Missing {mesh}; run scripts/download_datasets.sh {args.dataset_root}")
    if sha256(mesh) != mesh_info["sha256"]:
        raise ValueError(f"Dataset checksum mismatch: {mesh}; download the pinned inputs again")
    template = ROOT / f"reproducibility/parameters/{base}.prm.in"
    output = args.output_root / case["id"]
    if output.exists():
        raise ValueError(f"Run directory already exists: {output}; choose a fresh --output-root")
    text = template.read_text()
    settings = dict(case["settings"])
    if args.final_time is not None:
        settings["Final time"] = str(args.final_time)
    for key, value in settings.items():
        text = replace_setting(text, key, value)
    text = text.replace("@DATASET_ROOT@", str(args.dataset_root))
    text = text.replace("@OUTPUT_DIR@", str(output / "results"))
    if re.search(r"@[A-Z_]+@", text):
        raise ValueError(f"Unresolved template token in {template}")
    mpi = args.mpi if args.mpi is not None else case["mpi"]
    command = []
    executable = None
    if not args.prepare_only:
        if args.runtime == "native":
            found = shutil.which(args.executable)
            if found is None:
                raise ValueError(f"Executable not found or not executable: {args.executable}")
            executable = Path(found).resolve()
            if mpi > 1:
                launcher = shutil.which(args.mpi_launcher)
                if launcher is None:
                    raise ValueError(f"MPI launcher not found: {args.mpi_launcher}")
                command = [launcher, "-np", str(mpi)]
            command += [str(executable), str(output / "parameters.prm")]
        else:
            for path in (mesh, output):
                if not path.is_relative_to(args.container_workdir):
                    raise ValueError(f"{path} is outside container workdir {args.container_workdir}; "
                                     "choose an explicit --container-workdir containing both datasets and outputs")
            launcher = ROOT / "scripts/run_container.sh"
            if not launcher.is_file():
                raise ValueError("scripts/run_container.sh is missing; use a checkout that includes the container launcher")
            command = ["bash", str(launcher), "--runtime", args.runtime,
                       "--image", args.image, "--ranks", str(mpi),
                       "--workdir", str(args.container_workdir), str(output / "parameters.prm")]
    git = shutil.which("git")
    revision = dirty = None
    if git:
        revision = subprocess.run(
            [git, "-C", str(ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=False,
        )
        dirty = subprocess.run(
            [git, "-C", str(ROOT), "status", "--porcelain"],
            capture_output=True, text=True, check=False,
        )
    output.mkdir(parents=True)
    parameter_file = output / "parameters.prm"
    parameter_file.write_text(text)
    receipt = {
        "case": case, "status": "prepared", "created_utc": timestamp(),
        "workflow_revision": (revision.stdout.strip() or None) if revision else None,
        "workflow_dirty": bool(dirty.stdout.strip()) if dirty and dirty.returncode == 0 else None,
        "executable_revision": args.executable_revision,
        "executable_revision_source": "user-supplied build provenance" if args.executable_revision else None,
        "manifest_sha256": sha256(MANIFEST), "template_sha256": sha256(template),
        "parameters_sha256": sha256(parameter_file),
        "mesh": {"path": str(mesh), "sha256": mesh_info["sha256"]},
        "settings": settings, "mpi": mpi, "command": command,
        "runtime": args.runtime,
        "container_workdir": str(args.container_workdir) if args.runtime != "native" else None,
        "image_reference": args.image,
        "image_digest": args.image.split("@", 1)[1] if args.image and "@" in args.image else None,
        "final_time_override": args.final_time,
        "execution_blocker": manifest["provenance"][base].get("execution_blocker"),
        "executable_sha256": sha256(executable) if executable else None,
        "platform": platform.platform(),
        "thread_environment": {key: os.environ.get(key) for key in
                               ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
        "validation": "No comparison with archived paper results is performed.",
    }
    receipt_file = output / "run.json"

    def save() -> None:
        receipt_file.write_text(json.dumps(receipt, indent=2) + "\n")

    save()
    print(f"Prepared {case['id']}: {parameter_file}", flush=True)
    if args.prepare_only:
        return 0
    receipt.update(status="running", started_utc=timestamp())
    save()
    try:
        with (output / "solver.log").open("w") as log:
            result = subprocess.run(command, cwd=output, stdout=log, stderr=subprocess.STDOUT)
        receipt.update(returncode=result.returncode,
                       status="completed" if result.returncode == 0 else "failed")
    except (OSError, KeyboardInterrupt) as error:
        receipt.update(status="failed", error=str(error), returncode=1)
    receipt["finished_utc"] = timestamp()
    save()
    print(f"{case['id']}: {receipt['status']}; log: {output / 'solver.log'}", flush=True)
    return receipt["returncode"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", nargs="?", help="Experiment ID shown by --list")
    parser.add_argument("--list", action="store_true", help="List cases without downloading or running")
    parser.add_argument("--suite", help="mms, benchmarks, mesh, time, applications, scaling, or all")
    parser.add_argument("--dataset-root", type=Path, default=ROOT)
    parser.add_argument("--output-root", type=Path, default=ROOT / "paper-runs")
    parser.add_argument("--executable", default="blood_flow")
    parser.add_argument("--executable-revision", help="Known source revision of a reused executable (user supplied)")
    parser.add_argument("--runtime", choices=("native", "docker", "podman", "apptainer"), default="native")
    parser.add_argument("--image", help="Explicit container tag or digest; tags are recorded without claiming a resolved digest")
    parser.add_argument("--container-workdir", type=Path, default=ROOT,
                        help="Shared container mount root containing datasets and outputs (default: checkout)")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--mpi", type=int, help="Override the manifest MPI count")
    parser.add_argument("--mpi-launcher", default="mpirun", help="MPI launcher executable")
    parser.add_argument("--final-time", type=float, help="Override final time for a separately recorded smoke run")
    args = parser.parse_args(argv)
    manifest = json.loads(MANIFEST.read_text())
    cases = manifest["runs"]
    if args.list:
        for case in cases:
            candidate = " [prepare only]" if manifest["provenance"][case["base"]].get("execution_blocker") else ""
            print(f"{case['id']:24s} {case['suite']:12s} MPI={case['mpi']:2d}  {case['paper']}{candidate}")
        return 0
    if bool(args.case) == bool(args.suite):
        parser.error("Choose one case ID or --suite")
    if args.mpi is not None and args.mpi < 1:
        parser.error("--mpi must be positive")
    if args.final_time is not None and (not math.isfinite(args.final_time) or args.final_time <= 0):
        parser.error("--final-time must be finite and positive")
    if args.runtime != "native" and not args.prepare_only and not args.image:
        parser.error("Container execution requires an explicit --image tag or digest")
    selected = [case for case in cases if case["id"] == args.case or
                (args.suite and args.suite in ("all", case["suite"]))]
    if not selected:
        parser.error("Unknown case or suite; use --list")
    args.dataset_root = args.dataset_root.resolve()
    args.output_root = args.output_root.resolve()
    args.container_workdir = args.container_workdir.resolve()
    if any(character in str(args.dataset_root) + str(args.output_root) for character in "\n\r#"):
        parser.error("Paths cannot contain newlines or # (deal.II parameter syntax)")
    try:
        prerequisites = {
            manifest["provenance"][case["base"]].get("minimum_executable_revision")
            for case in selected
        } - {None}
        if not args.prepare_only:
            for required in prerequisites:
                verify_executable_revision(required, args.executable_revision)
        # Refuse an overlapping suite before launching any expensive experiment.
        for case in selected:
            blocker = manifest["provenance"][case["base"]].get("execution_blocker")
            if blocker and not args.prepare_only:
                raise ValueError(f"{case['id']}: {blocker} Use --prepare-only to inspect candidate inputs.")
            if (args.output_root / case["id"]).exists():
                raise ValueError(f"Run directory already exists: {args.output_root / case['id']}")
        for case in selected:
            status = run_case(case, manifest, args)
            if status:
                return status
    except (OSError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
