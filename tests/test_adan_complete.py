# SPDX-License-Identifier: MIT
"""Offline ADAN reconstruction tests with independent topology/wall checks."""
import contextlib
import hashlib
import io
import math
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools/adan"))
import build_vtk
import check_conversion
import download_sources
import recover_geometry


def fixture(root):
    source = root / "source"
    source.mkdir()
    # One bifurcation; one outlet is the first CSV node, and one is purely
    # resistive. A venous row must be excluded from the arterial network.
    rows = [
        [
            "A_aortic_arch_C_0",
            0,
            1,
            1,
            0.2,
            0.18,
            1,
            1,
            -1,
            "-",
            "-",
            "-",
            "-",
            "-",
            "",
        ],
        ["A_branch_L_1", 1, 2, 2, 0.1, 0.08, 1, 2, -1, 1e5, 1e-6, 20, 2, "-", ""],
        ["A_branch_R_2", 3, 1, 3, 0.02, 0.03, 1, 3, -1, 2e5, 0, 21, 3, "-", ""],
        ["V_vein_3", 4, 5, 1, 0.2, 0.2, 0, 1, -1, "-", "-", "-", "-", "-", ""],
    ]
    import csv

    with (source / "data_sheet_1.csv").open("w") as stream:
        writer = csv.writer(stream, delimiter=";")
        writer.writerow(["# fixture"] + [""] * 14)
        writer.writerows(rows)


class AdanCompleteTests(unittest.TestCase):
    def test_csv_only_end_to_end(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(
            io.StringIO()
        ):
            root = Path(folder)
            fixture(root)
            recover_geometry.schematic(root)
            receipt = build_vtk.build(root / "adan_complete.vtk", 1040.0, 666.0, root)
            checks = check_conversion.check(root)
            self.assertEqual(receipt["source_arterial_segments"], 3)
            self.assertEqual(receipt["vtk_cells"], 6)
            self.assertEqual(receipt["terminals"], 2)
            self.assertIsNone(receipt["surface_source"])
            self.assertTrue(checks["source_connectivity_and_radii_preserved"])
            self.assertLess(checks["max_relative_source_length_error"], 1e-12)
            self.assertEqual(receipt["geometry_mode"], "schematic")
            self.assertFalse((root / "source/surfaces").exists())
            self.assertEqual(receipt["source_data_license"], "CC-BY-4.0")

    def test_radius_group_boundaries(self):
        self.assertEqual(build_vtk.wall(0.0018)["adavn_E_c"], 2e7)
        self.assertEqual(build_vtk.wall(0.00180001)["adavn_E_c"], 5e6)
        self.assertEqual(build_vtk.wall(0.0007)["adavn_E_c"], 2e7)
        self.assertEqual(build_vtk.wall(0.00069999)["adavn_E_c"], 0.0)

    def test_download_checksum_rejection(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "data"
            path.write_bytes(b"changed bytes")
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                download_sources.download(
                    "https://example.invalid",
                    path,
                    hashlib.sha256(b"original").hexdigest(),
                )
            self.assertEqual(path.read_bytes(), b"changed bytes")

    def test_atlas_requires_explicit_permission(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, "written permission"):
                download_sources.fetch(Path(folder), include_atlas=True)

    def test_invalid_length_fails_before_geometry(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            fixture(root)
            path = root / "source/data_sheet_1.csv"
            path.write_text(path.read_text().replace(";1;0.2;0.18;", ";-1;0.2;0.18;"))
            with self.assertRaisesRegex(ValueError, "finite and positive"):
                recover_geometry.schematic(root)

    def test_two_line_embedding_preserves_length(self):
        a, b = (0.0, 0.0, 0.0), (0.3, 0.4, 0.0)
        p = build_vtk.midpoint(a, b, 1.0, (0.15, 0.2, 1.0))
        self.assertAlmostEqual(math.dist(a, p), 0.5, places=14)
        self.assertAlmostEqual(math.dist(p, b), 0.5, places=14)


if __name__ == "__main__":
    unittest.main()
