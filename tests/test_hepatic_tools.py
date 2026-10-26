# SPDX-License-Identifier: MIT
"""Check image coordinates, topology decisions, VTK physics and downloads offline."""

import argparse
import hashlib
import importlib.util
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

import nibabel as nib
import numpy as np
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "tools/hepatic" / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


builder = load("build_vtk")
checker = load("check_conversion")
downloader = load("download_sources")


class HepaticToolsTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def tube(self):
        # Square cross section: central distance to background voxel centres is
        # exactly 3 mm, independent of axial spacing and translation.
        mask = np.zeros((11, 11, 25), dtype=np.uint8)
        mask[3:8, 3:8, 2:23] = 1
        affine = np.diag([-1.0, 1.0, 2.0, 1.0])
        affine[:3, 3] = [30, -20, 100]
        image = nib.Nifti1Image(mask, affine)
        image.header.set_xyzt_units("mm")
        path = self.root / "tube.nii.gz"
        nib.save(image, path)
        return path

    def arguments(self, mask):
        return argparse.Namespace(
            mask=mask,
            output=self.root / "tree.vtk",
            label=1,
            network_kind="artery",
            length_unit=None,
            component=None,
            break_cycles=False,
            root_point=None,
            root_largest_radius=True,
            young_modulus=4e5,
            wall_ratio=0.1,
            reference_pressure=1e4,
            terminal_resistance=1e10,
            outlet_pressure=100,
            source_url="synthetic test",
            source_license="MIT",
        )

    def test_anisotropic_tube_si_export_and_independent_check(self):
        args = self.arguments(self.tube())
        builder.build(args)
        result = checker.check(args.output)
        self.assertTrue(result["valid"])
        self.assertEqual(result["outlets"], 1)
        mesh = builder.validate(args.output)
        self.assertTrue(mesh["valid"])
        report = json.loads(args.output.with_suffix(".provenance.json").read_text())
        self.assertAlmostEqual(max(report["point_radii_m"]), 0.003)
        p = report["voxel_to_ras_m"]
        self.assertAlmostEqual(p[0][0], -0.001)
        self.assertAlmostEqual(p[2][2], 0.002)
        self.assertAlmostEqual(p[2][3], 0.1)
        self.assertAlmostEqual(result["length_m"], (result["points"] - 1) * 0.002)

    def test_label_selection_and_unknown_units(self):
        path = self.tube()
        with self.assertRaisesRegex(ValueError, "absent"):
            builder.load_mask(path, 2)
        image = nib.load(path)
        image.header.set_xyzt_units("unknown")
        nib.save(image, path)
        with self.assertRaisesRegex(ValueError, "length unit"):
            builder.load_mask(path, 1)
        self.assertTrue(builder.load_mask(path, 1, "mm")[0].any())

    def test_sheared_affine_rejected(self):
        path = self.tube()
        image = nib.load(path)
        affine = image.affine.copy()
        affine[0, 1] = 0.2
        nib.save(
            nib.Nifti1Image(np.asanyarray(image.dataobj), affine, image.header), path
        )
        with self.assertRaisesRegex(ValueError, "sheared"):
            builder.load_mask(path, 1)

    def test_disconnected_components_require_explicit_selection(self):
        mask = np.zeros((20, 20, 20), dtype=bool)
        mask[2:5, 2:5, 2:15] = True
        mask[12:15, 12:15, 2:10] = True
        with self.assertRaisesRegex(ValueError, "components"):
            builder.extract_graph(mask, np.eye(4))
        _, _, _, report = builder.extract_graph(mask, np.eye(4), "largest")
        self.assertEqual(report["discarded_mask_voxels"], 72)
        self.assertEqual(report["selected_component_rank"], 0)

    def test_cycle_is_not_silently_removed(self):
        mask = np.zeros((20, 20, 5), dtype=bool)
        mask[3:17, 3, 2] = mask[3:17, 16, 2] = True
        mask[3, 3:17, 2] = mask[16, 3:17, 2] = True
        with self.assertRaisesRegex(ValueError, "cycle"):
            builder.extract_graph(mask, np.eye(4))
        points, _, edges, report = builder.extract_graph(
            mask, np.eye(4), break_cycles=True
        )
        self.assertEqual(len(edges), len(points) - 1)
        self.assertTrue(report["cycle_edges_removed"])

    def test_bifurcation_root_and_orientation(self):
        points = np.array([[0, 0, 0], [1, 0, 0], [2, 1, 0], [2, -1, 0]])
        root, outlets, edges = builder.orient_tree(
            points, np.ones(4), [(1, 0), (2, 1), (3, 1)], [0, 0, 0]
        )
        self.assertEqual((root, outlets, edges), (0, [2, 3], [(0, 1), (1, 2), (1, 3)]))

    def test_reserved_boundary_id_is_skipped(self):
        # Deliberately large synthetic junction: exercise all terminal IDs.
        points = np.vstack(([0, 0, -1], [0, 0, 0], [[i + 1, 1, 0] for i in range(260)]))
        edges = [(0, 1)] + [(1, i) for i in range(2, 262)]
        path = self.root / "many.vtk"
        terminals = builder.write_network(
            path,
            points,
            np.ones(262),
            edges,
            0,
            list(range(2, 262)),
            vars(self.arguments(self.tube())),
        )
        ids = [t["boundary_id"] for t in terminals]
        self.assertNotIn(255, ids)
        self.assertEqual(len(set(ids)), 260)
        self.assertEqual(ids[-1], 261)

    def test_corrupt_vtk_detected(self):
        args = self.arguments(self.tube())
        builder.build(args)
        args.output.write_text(args.output.read_text().replace("400000", "800000"))
        with self.assertRaisesRegex(ValueError, "checksum"):
            checker.check(args.output)

    def test_invalid_physics_rejected_before_writing(self):
        args = self.arguments(self.tube())
        args.terminal_resistance = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            builder.build(args)
        self.assertFalse(args.output.exists())

    def test_dicom_sorted_by_position_and_lps_to_ras(self):
        folder = self.root / "dicom"
        folder.mkdir()
        uid = pydicom.uid.generate_uid()
        for index, z in enumerate((14, 10, 12)):
            meta = FileMetaDataset()
            meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
            data = FileDataset(
                str(folder / f"slice{index}"), {}, file_meta=meta, preamble=b"\0" * 128
            )
            data.SeriesInstanceUID = uid
            data.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
            data.ImagePositionPatient = [30, 40, z]
            data.PixelSpacing = [0.5, 0.75]
            data.Rows, data.Columns = 5, 6
            data.SamplesPerPixel = 1
            data.PhotometricInterpretation = "MONOCHROME2"
            data.BitsAllocated = data.BitsStored = 8
            data.HighBit, data.PixelRepresentation = 7, 0
            pixels = np.zeros((5, 6), dtype=np.uint8)
            pixels[2, 3] = 255
            data.PixelData = pixels.tobytes()
            data.save_as(data.filename, enforce_file_format=False)
        mask, affine, _ = builder.load_mask(folder, 255)
        self.assertEqual(mask.shape, (5, 6, 3))
        np.testing.assert_allclose(affine[:3, 3], [-0.03, -0.04, 0.01])
        np.testing.assert_allclose(
            affine[:3, :3], [[0, -0.00075, 0], [-0.0005, 0, 0], [0, 0, 0.002]]
        )

    def test_archive_integrity_and_safe_extraction(self):
        path = self.root / "source.zip"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("nested/mask.txt", b"original")
        checksum = hashlib.sha256(b"original").hexdigest()
        source = {"member_sha256": {"nested/mask.txt": checksum}}
        downloader.verify_archive(path, source)
        downloader.extract_zip(path, self.root / "out")
        self.assertEqual((self.root / "out/nested/mask.txt").read_bytes(), b"original")
        source["member_sha256"]["nested/mask.txt"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "Checksum"):
            downloader.verify_archive(path, source)
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("../escape.txt", b"bad")
        with self.assertRaisesRegex(ValueError, "Unsafe"):
            downloader.extract_zip(path, self.root / "out")
        self.assertFalse((self.root / "escape.txt").exists())


if __name__ == "__main__":
    unittest.main()
