#!/usr/bin/env python3
"""Check which changes require the nightly image to be rebuilt."""

import importlib.util
from pathlib import Path
import subprocess
import tempfile
import unittest

spec = importlib.util.spec_from_file_location(
    "container_inputs", Path(__file__).resolve().parents[1] / "scripts/container_inputs.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ContainerInputs(unittest.TestCase):
    def test_runtime_changes_and_documentation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(["git", "init", "-q", directory], check=True)
            for name in ("Dockerfile", "source/solver.cc", "doc/math.md",
                         "reproducibility/parameters/mms.prm.in"):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("initial\n")
            subprocess.run(["git", "add", "."], cwd=root, check=True)
            original = module.fingerprint(root)
            (root / "doc/math.md").write_text("new documentation\n")
            (root / "reproducibility/parameters/mms.prm.in").write_text("new input\n")
            self.assertEqual(original, module.fingerprint(root))
            (root / "source/solver.cc").write_text("new implementation\n")
            self.assertNotEqual(original, module.fingerprint(root))
            changed = module.fingerprint(root)
            (root / "source/another.cc").write_text("new file\n")
            subprocess.run(["git", "add", "."], cwd=root, check=True)
            added = module.fingerprint(root)
            self.assertNotEqual(changed, added)
            (root / "source/solver.cc").unlink()
            subprocess.run(["git", "add", "-u"], cwd=root, check=True)
            self.assertNotEqual(added, module.fingerprint(root))


if __name__ == "__main__":
    unittest.main()
