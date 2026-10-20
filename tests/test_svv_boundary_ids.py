import importlib.util
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("validator", ROOT / "tools/validate_vtk_network.py")
validator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validator)


class SvvBoundaryIdsTests(unittest.TestCase):
    def validate_with_ids(self, ids):
        lines = (ROOT / "notebooks/bifurcation_network.vtk").read_text().splitlines(keepends=True)
        start = next(i for i, line in enumerate(lines) if line.startswith("SCALARS boundary_id ")) + 2
        end = next((i for i in range(start, len(lines)) if lines[i].startswith("SCALARS ")), len(lines))
        lines[start:end] = [f"{value}\n" for value in ids]
        with tempfile.NamedTemporaryFile("w", suffix=".vtk") as output:
            output.write("".join(lines))
            output.flush()
            return validator.validate(Path(output.name))

    def boundary_ids(self):
        mesh = validator.parse_vtk(ROOT / "notebooks/bifurcation_network.vtk")
        degree = [0] * len(mesh.points)
        for first, second in mesh.cells:
            degree[first] += 1
            degree[second] += 1
        return list(mesh.point_arrays["boundary_id"]), degree

    def test_graph_degree_distinguishes_interior_marker_and_outlet_255(self):
        ids, degree = self.boundary_ids()
        for point, count in enumerate(degree):
            if count > 1:
                ids[point] = 300655
            elif ids[point] == 1:
                ids[point] = 255
        result = self.validate_with_ids(ids)
        self.assertTrue(result["valid"], result)
        self.assertIn(255, result["terminal_boundary_ids"])
        self.assertNotIn(300655, result["terminal_boundary_ids"])

    def test_out_of_range_interior_marker_is_rejected(self):
        ids, degree = self.boundary_ids()
        ids[degree.index(3)] = 25555555555
        result = self.validate_with_ids(ids)
        self.assertFalse(result["valid"])
        self.assertTrue(any("32-bit" in error for error in result["errors"]))


if __name__ == "__main__":
    unittest.main()
