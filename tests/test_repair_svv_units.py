import math
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from repair_legacy_svv_units import repair
from validate_vtk_network import parse_vtk


class RepairSvvUnitsTests(unittest.TestCase):
    def test_geometry_physics_and_rc_time_constant(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "network.vtk"
            path.write_bytes((ROOT / "parameters/single_vessel.vtk").read_bytes())
            old = parse_vtk(path)
            self.assertTrue(repair(path))
            new = parse_vtk(path)
            self.assertEqual(new.cells, old.cells)
            self.assertEqual(new.cell_arrays["vessel_id"], old.cell_arrays["vessel_id"])
            self.assertEqual(new.cell_arrays["E"], old.cell_arrays["E"])
            self.assertEqual(new.cell_arrays["p_d"], old.cell_arrays["p_d"])
            self.assertAlmostEqual(new.points[1][1], old.points[1][1] * 0.01)
            self.assertAlmostEqual(new.cell_arrays["r_d"][0] / old.cell_arrays["r_d"][0], 0.01)
            self.assertAlmostEqual(new.cell_arrays["a_d"][0] / old.cell_arrays["a_d"][0], 1e-4)
            self.assertAlmostEqual(new.point_arrays["R2"][1] / old.point_arrays["R2"][1], 1e6)
            self.assertAlmostEqual(new.point_arrays["C"][1] / old.point_arrays["C"][1], 1e-6)
            self.assertAlmostEqual(new.point_arrays["R2"][1] * new.point_arrays["C"][1],
                                   old.point_arrays["R2"][1] * old.point_arrays["C"][1])
            self.assertGreater(new.cell_arrays["h_wall"][0], 0)
            self.assertLess(new.cell_arrays["h_wall"][0], new.cell_arrays["r_d"][0])
            corrected = path.read_bytes()
            self.assertFalse(repair(path))
            self.assertEqual(path.read_bytes(), corrected)

    def test_interior_ids_fit_uint32(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "network.vtk"
            text = (ROOT / "notebooks/bifurcation_network.vtk").read_text()
            mesh = parse_vtk(ROOT / "notebooks/bifurcation_network.vtk")
            path.write_text(text)
            repair(path)
            corrected = parse_vtk(path)
            degree = [0] * len(mesh.points)
            for first, second in mesh.cells:
                degree[first] += 1
                degree[second] += 1
            for old, new, count in zip(mesh.point_arrays["boundary_id"],
                                       corrected.point_arrays["boundary_id"], degree):
                self.assertGreaterEqual(new, 0)
                self.assertLess(new, 2**32 - 1)
                self.assertEqual(new, old if count == 1 else 300655)


if __name__ == "__main__":
    unittest.main()
