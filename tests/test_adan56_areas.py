import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from validate_vtk_network import parse_vtk


class Adan56AreaTests(unittest.TestCase):
    def test_reference_areas_match_radii(self):
        for name in ("notebooks/56_adnr.vtk", "notebooks/56_adnr_new.vtk",
                     "tutorials/04_adan56/network.vtk"):
            mesh = parse_vtk(ROOT / name)
            for vessel, area, radius in zip(mesh.cell_arrays["vessel_id"],
                                            mesh.cell_arrays["a_d"],
                                            mesh.cell_arrays["r_d"]):
                with self.subTest(mesh=name, vessel=int(vessel)):
                    expected = math.pi * radius**2
                    self.assertLess(abs(area / expected - 1), 0.02)

    def test_tutorial_remains_an_exact_copy(self):
        self.assertEqual((ROOT / "notebooks/56_adnr_new.vtk").read_bytes(),
                         (ROOT / "tutorials/04_adan56/network.vtk").read_bytes())


if __name__ == "__main__":
    unittest.main()
