import importlib.util
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CONVERTER = ROOT / "notebooks/networks_by_svv/convert_to_standard_vtk.py"
spec = importlib.util.spec_from_file_location("vtk_validator", ROOT / "tools/validate_vtk_network.py")
validator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validator)


class SvvConversionTests(unittest.TestCase):
    def convert(self, folder, units, length_scale=1.0, time_scale=1.0, options=()):
        data = np.zeros((3, 31))
        data[:, 0:3] = [(0, 0, 0), (1, 0, 0), (1, 0, 0)]
        data[:, 3:6] = [(1, 0, 0), (2, 1, 0), (2, -1, 0)]
        data[:, 18] = [0, 1, 1]
        data[:, 19] = [1, 2, 3]
        data[:, 20] = [1, math.sqrt(2), math.sqrt(2)]
        data[:, 21] = [0.2, 0.1, 0.1]
        data[:, 22] = [2, 1, 1]
        data[:, :6] *= length_scale
        data[:, 20:22] *= length_scale
        data[:, 22] *= length_scale**3 / time_scale
        source = folder / "tree.npz"
        output = folder / "network.vtk"
        np.savez(source, data=data, parameters=np.array([{"unit_system": units}], dtype=object))
        result = subprocess.run([sys.executable, str(CONVERTER), str(source), str(output), *options],
                                capture_output=True, text=True)
        return result, validator.parse_vtk(output) if result.returncode == 0 else None

    def test_cgs_geometry_and_outlets_are_si(self):
        with tempfile.TemporaryDirectory() as folder:
            result, mesh = self.convert(Path(folder), {"length": "cm", "time": "s"})
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(mesh.points[2], (0.02, 0.01, 0.0))
            self.assertAlmostEqual(mesh.cell_arrays["L"][0], 0.01)
            self.assertAlmostEqual(mesh.cell_arrays["r_d"][0], 0.002)
            self.assertAlmostEqual(mesh.cell_arrays["a_d"][0] / (math.pi * 0.002**2), 1, places=6)
            self.assertAlmostEqual(mesh.point_arrays["R2"][2] / 1.25e11, 1)
            self.assertEqual(mesh.point_arrays["boundary_id"], [0, 300655, 1, 2])
            self.assertEqual(mesh.point_array_types["boundary_id"], "unsigned_int")

    def test_si_and_cgs_representations_agree_including_rcr(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            (base / "cgs").mkdir()
            (base / "si").mkdir()
            cg_result, cgs = self.convert(base / "cgs", {"length": "cm", "time": "s"}, options=("--rcr",))
            si_result, si = self.convert(base / "si", {"length": "m", "time": "s"}, 0.01, options=("--rcr",))
            self.assertEqual(cg_result.returncode, 0, cg_result.stderr)
            self.assertEqual(si_result.returncode, 0, si_result.stderr)
            self.assertEqual(cgs.points, si.points)
            self.assertEqual(cgs.cell_arrays, si.cell_arrays)
            self.assertEqual(cgs.point_arrays, si.point_arrays)
            self.assertAlmostEqual(cgs.point_arrays["C"][2] / (0.283 / 1.25e11), 1)
            self.assertAlmostEqual((cgs.point_arrays["R1"][2] + cgs.point_arrays["R2"][2]) / 1.25e11, 1)

    def test_time_units_affect_flow(self):
        with tempfile.TemporaryDirectory() as folder:
            result, mesh = self.convert(Path(folder), {"length": "mm", "time": "ms"})
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertAlmostEqual(mesh.point_arrays["R2"][2] / 1.25e11, 1)

    def test_missing_and_unknown_units_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            for units in ({}, {"length": "furlong", "time": "s"}):
                result, _ = self.convert(Path(folder), units)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("source units are never inferred", result.stderr)

    def test_invalid_si_model_constants_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            for options in (("--young-modulus", "0"),
                            ("--perfusion-pressure", "nan"),
                            ("--diastolic-pressure", "inf"),
                            ("--outlet-pressure", "nan")):
                result, mesh = self.convert(Path(folder), {"length": "cm", "time": "s"}, options=options)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("must be finite", result.stderr)
                self.assertIsNone(mesh)

    def test_outlet_pressure_is_si(self):
        with tempfile.TemporaryDirectory() as folder:
            result, mesh = self.convert(Path(folder), {"length": "cm", "time": "s"},
                                        options=("--outlet-pressure", "432.6"))
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(mesh.point_arrays["P_out"], [432.6] * 4)

    def test_explicit_overrides(self):
        with tempfile.TemporaryDirectory() as folder:
            result, mesh = self.convert(Path(folder), {}, options=("--length-unit", "cm", "--time-unit", "s"))
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertAlmostEqual(mesh.cell_arrays["r_d"][0], 0.002)


if __name__ == "__main__":
    unittest.main()
