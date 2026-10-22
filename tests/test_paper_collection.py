"""Check unit conversion and incomplete-run handling against explicit samples."""

import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/collect_paper_outputs.py"


class PaperCollectionTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.run = self.root / "runs/bifurcation"
        (self.run / "results").mkdir(parents=True)
        (self.run / "run.json").write_text(json.dumps({
            "status": "completed", "case": {"id": "bifurcation"}, "mpi": 1,
        }))
        (self.run / "results/HDG_IDA_Vessel_0.csv").write_text(
            "time_s,P_dynpcm2,Q_cm3ps,A_cm2,U_cmps\n"
            "0,100,2,3,4\n1,200,4,6,8\n")

    def invoke(self, *options):
        return subprocess.run([sys.executable, str(SCRIPT), str(self.root / "runs"),
                               "--output", str(self.root / "collected"), *options],
                              capture_output=True, text=True)

    def test_explicit_unit_conversion_and_sample_window(self):
        result = self.invoke("--start", "1", "--end", "1", "--vessels", "0")
        self.assertEqual(result.returncode, 0, result.stderr)
        with (self.root / "collected/samples.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(float(row["time_s"]), 1)
        self.assertAlmostEqual(float(row["pressure_pa"]), 20)
        self.assertAlmostEqual(float(row["flow_m3_s"]), 4e-6)
        self.assertAlmostEqual(float(row["area_m2"]), 6e-4)
        self.assertAlmostEqual(float(row["velocity_m_s"]), 0.08)
        provenance = json.loads((self.root / "collected/collection.json").read_text())
        self.assertEqual(len(provenance["missing_diagnostics"]), 3)
        self.assertEqual(provenance["sample_rows"], 1)

    def test_failed_runs_are_not_collected(self):
        receipt = json.loads((self.run / "run.json").read_text())
        receipt["status"] = "failed"
        (self.run / "run.json").write_text(json.dumps(receipt))
        result = self.invoke()
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse((self.root / "collected").exists())

    def test_missing_requested_vessel_is_an_error(self):
        result = self.invoke("--vessels", "9")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Missing requested vessels", result.stderr)

    def test_mms_errors_retain_reported_log_precision(self):
        (self.run / "run.json").write_text(json.dumps({
            "status": "completed", "case": {"id": "mms-p2-n8", "base": "mms"},
            "mpi": 1, "settings": {"Finite element degree": "2"},
        }))
        (self.run / "solver.log").write_text(
            "=== Errors t=1.000e-01 cycle 1 ===\n"
            " A  L2=1.234e-06 rate=0.000e+00\n"
            " A  H1=2.345e-05 rate=0.000e+00\n"
            " U  L2=3.456e-04 rate=0.000e+00\n"
            " U  H1=4.567e-03 rate=0.000e+00\n")
        result = self.invoke()
        self.assertEqual(result.returncode, 0, result.stderr)
        with (self.root / "collected/mms_errors.csv").open() as stream:
            row = next(csv.DictReader(stream))
        self.assertEqual(row["p"], "2")
        self.assertEqual(row["cells"], "8")
        self.assertEqual(row["area_l2"], "1.234e-06")
        self.assertEqual(row["velocity_h1"], "4.567e-03")


if __name__ == "__main__":
    unittest.main()
