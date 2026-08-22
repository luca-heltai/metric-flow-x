import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "validate_vtk_network.py"
spec = importlib.util.spec_from_file_location("validate_vtk_network", TOOL)
assert spec and spec.loader
validator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validator)


class ValidateVtkNetworkTests(unittest.TestCase):
    def test_repository_networks_are_valid(self):
        for relative in (
            "notebooks/37_vessel_network.vtk",
            "notebooks/bifurcation_network.vtk",
            "parameters/aortic.vtk",
            "parameters/single_vessel.vtk",
        ):
            result = validator.validate(ROOT / relative)
            self.assertTrue(result["valid"], result)

    def test_invalid_cell_type_has_json_failure_and_nonzero_exit(self):
        source = (ROOT / "notebooks" / "bifurcation_network.vtk").read_text()
        invalid = source.replace("CELL_TYPES 3\n3\n3\n3", "CELL_TYPES 3\n3\n5\n3")
        with tempfile.NamedTemporaryFile("w", suffix=".vtk") as handle:
            handle.write(invalid)
            handle.flush()
            result = subprocess.run(
                [sys.executable, str(TOOL), handle.name],
                check=False,
                capture_output=True,
                text=True,
            )
        self.assertNotEqual(result.returncode, 0)
        payload = json.loads(result.stdout)
        self.assertFalse(payload["valid"])
        self.assertTrue(any("VTK_LINE" in error for error in payload["errors"]))


if __name__ == "__main__":
    unittest.main()
