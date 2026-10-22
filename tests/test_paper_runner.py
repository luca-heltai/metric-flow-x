"""Exercise CLI preparation, data integrity and run receipts without a solver."""

import json
import importlib.util
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/reproduce_paper.py"
SPEC = importlib.util.spec_from_file_location("paper_runner", SCRIPT)
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


class PaperRunnerTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="paper inputs ")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.mesh = self.root / "data/notebooks/paper_inputs/bifurcation_network.vtk"
        self.mesh.parent.mkdir(parents=True)
        shutil.copyfile(ROOT / "notebooks/bifurcation_network.vtk", self.mesh)

    def invoke(self, *args):
        return subprocess.run(
            [sys.executable, str(SCRIPT), *args,
             "--dataset-root", str(self.root / "data"),
             "--output-root", str(self.root / "runs")],
            cwd=self.root, capture_output=True, text=True,
        )

    def test_list_and_matrix_cover_paper_sweeps(self):
        result = self.invoke("--list")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(result.stdout.splitlines()), 43)
        for case in ("mms-p1-n4", "mms-p3-n64", "time37-r2-bdf5", "network100k-np12"):
            self.assertIn(case, result.stdout)

    def test_prepare_portable_input_and_refuse_reuse(self):
        result = self.invoke("bifurcation", "--prepare-only")
        self.assertEqual(result.returncode, 0, result.stderr)
        run = self.root / "runs/bifurcation"
        params = (run / "parameters.prm").read_text()
        self.assertIn(str(self.mesh), params)
        self.assertNotIn("@DATASET_ROOT@", params)
        self.assertIn("subsection MetricFlowSystem<1, 3>", params)
        self.assertEqual(json.loads((run / "run.json").read_text())["status"], "prepared")
        repeat = self.invoke("bifurcation", "--prepare-only")
        self.assertNotEqual(repeat.returncode, 0)
        self.assertEqual((run / "parameters.prm").read_text(), params)

    def test_tampered_mesh_is_rejected_before_run_directory(self):
        self.mesh.write_text("wrong network")
        result = self.invoke("bifurcation", "--prepare-only")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("checksum mismatch", result.stderr)
        self.assertFalse((self.root / "runs").exists())

    def test_failure_receipt_retains_override_and_actual_exit(self):
        executable = self.root / "failed_solver"
        executable.write_text("#!/bin/sh\necho 'fixture solver failure'\nexit 7\n")
        executable.chmod(0o755)
        result = self.invoke("bifurcation", "--executable", str(executable),
                             "--final-time", "0.001")
        self.assertEqual(result.returncode, 7, result.stderr)
        run = self.root / "runs/bifurcation"
        receipt = json.loads((run / "run.json").read_text())
        self.assertEqual(receipt["status"], "failed")
        self.assertEqual(receipt["returncode"], 7)
        self.assertEqual(receipt["final_time_override"], 0.001)
        self.assertIn("fixture solver failure", (run / "solver.log").read_text())
        self.assertRegex((run / "parameters.prm").read_text(), r"Final time\s*= 0\.001")

    def test_invalid_overrides_do_not_prepare_input(self):
        for args in (("--mpi", "0"), ("--final-time", "nan")):
            result = self.invoke("bifurcation", "--prepare-only", *args)
            self.assertNotEqual(result.returncode, 0)
        self.assertFalse((self.root / "runs").exists())

    def test_mms_and_all_suite_refuse_unimplemented_exact_boundary_data(self):
        for selection in (("mms-p1-n4",), ("--suite", "all")):
            result = self.invoke(*selection)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("MMS execution is blocked", result.stderr)
        self.assertFalse((self.root / "runs").exists())

    def test_native_mms_revision_attestation_checks_real_ancestry(self):
        checkout = self.root / "source"
        subprocess.run(["git", "init", "-q", str(checkout)], check=True)
        revisions = []
        for name in ("old implementation", "MMS support", "descendant"):
            subprocess.run(["git", "-C", str(checkout), "-c", "user.name=Test",
                            "-c", "user.email=test@example.invalid", "commit",
                            "--allow-empty", "-qm", name], check=True)
            revisions.append(subprocess.check_output(
                ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip())
        old, required, descendant = revisions
        RUNNER.verify_executable_revision(required, descendant, checkout)
        RUNNER.verify_executable_revision(required, required, self.root / "no git history")
        for unsupported in (None, "arbitrary string", old, "a" * 40):
            with self.assertRaises(ValueError):
                RUNNER.verify_executable_revision(required, unsupported, checkout)


if __name__ == "__main__":
    unittest.main()
