"""Regression checks for numerical snapshot gates and paired timing summaries."""
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('comparison', ROOT/'tools/compare_assembly.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

class Snapshots(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.before, self.after = [Path(self.temp.name)/p for p in ['before','after']]
    def tearDown(self): self.temp.cleanup()
    def compare(self, a, b, **kw):
        self.before.write_text(a); self.after.write_text(b)
        return m.compare_snapshots(self.before,self.after,**kw)
    def test_exact(self):
        r=self.compare('0 0x1p+0\n','0 0x1p+0\n')
        self.assertTrue(r['accepted']);self.assertEqual(r['entries'],1);self.assertEqual(r['max_abs_error'],0)
    def test_next_float_rejected_by_default(self):
        r=self.compare('0 0x1p+0\n','0 '+math.nextafter(1.,2.).hex()+'\n')
        self.assertFalse(r['accepted']);self.assertEqual(r['violations'],1);self.assertEqual(r['changed_entries'],1)
        json.dumps(r,allow_nan=False)
    def test_relative_rounding_and_threshold(self):
        a='3 4 '+(1e7).hex()+'\n'; b='3 4 '+math.nextafter(1e7,math.inf).hex()+'\n'
        r=self.compare(a,b,rtol=64*math.ulp(1.))
        self.assertTrue(r['accepted']);self.assertGreater(r['max_abs_error'],0);self.assertLess(r['max_scaled_error'],1)
        self.assertFalse(self.compare(a,b,rtol=1e-18)['accepted'])
    def test_absolute_near_zero(self):
        self.assertTrue(self.compare('0 0x0p+0\n','0 '+(1e-15).hex()+'\n',atol=1e-14)['accepted'])
        self.assertFalse(self.compare('0 0x0p+0\n','0 '+(1e-15).hex()+'\n',rtol=1e-14)['accepted'])
    def test_structure_cannot_be_tolerated(self):
        with self.assertRaisesRegex(RuntimeError,'indices differ'):
            self.compare('0 1 0x1p+0\n','0 2 0x1p+0\n',atol=100)
    def test_nonfinite_even_same_hash(self):
        for value in ['nan','inf','-inf']:
            with self.assertRaisesRegex(RuntimeError,'Nonfinite'):
                self.compare('0 '+value+'\n','0 '+value+'\n',atol=100)
    def test_empty_rank(self):
        self.assertTrue(self.compare('','')['accepted'])
    def test_signed_zero(self):
        r=self.compare('0 0x0p+0\n','0 -0x0p+0\n')
        self.assertFalse(r['accepted']);self.assertEqual(r['changed_entries'],1)
        self.assertTrue(self.compare('0 0x0p+0\n','0 -0x0p+0\n',rtol=1e-14)['accepted'])
    def test_missing_row(self):
        with self.assertRaisesRegex(RuntimeError,'lengths differ'):
            self.compare('0 0x1p+0\n','')
    def test_malformed_or_negative_index(self):
        for a,b in [('0\n','0\n'),('-1 0x1p+0\n','-1 0x1p+0\n'),('x 0x1p+0\n','x 0x1p+0\n')]:
            with self.assertRaises(RuntimeError):self.compare(a,b)
    def test_error_overflow_rejected(self):
        with self.assertRaisesRegex(RuntimeError,'overflow'):
            self.compare('0 '+(1e308).hex()+'\n','0 '+(-1e308).hex()+'\n',atol=1)
    def test_scaled_error_underflow_serializes(self):
        r=self.compare('0 '+(1e-320).hex()+'\n1 0x1p+0\n','0 '+(2e-320).hex()+'\n1 0x1p+0\n',rtol=1e-14)
        self.assertFalse(r['accepted']);json.dumps(r,allow_nan=False)
    def test_arbitrary_paired_summary(self):
        records=[]
        for ranks in [1,3]:
            for run in [0,1,2,3]:
                for variant, factor in [('before',2),('after',1)]:
                    r=dict(case=f'custom_mpi{ranks}',mpi_ranks=ranks,run=run,variant=variant)
                    r.update({metric+'_median_s':factor/ranks*(run+1) for metric in ['residual','jacobian','derivative']})
                    records.append(r)
        rows=m.summarize(records)
        self.assertEqual(len(rows),6)
        for row in rows:
            self.assertEqual(row['median_paired_speedup'],2)
            self.assertAlmostEqual(row['after_parallel_efficiency'],1)

    def assert_fast_matches_parser(self, data, **tolerances):
        self.before.write_bytes(data)
        self.after.write_bytes(data)
        expected_error = None
        with mock.patch.object(m, "canonical_snapshot_entries", return_value=None):
            try:
                expected = m.compare_snapshots(self.before, self.after, **tolerances)
            except Exception as error:
                expected_error = (type(error), str(error))
        if expected_error:
            with self.assertRaises(expected_error[0]) as error:
                m.compare_snapshots(self.before, self.after, **tolerances)
            self.assertEqual(str(error.exception), expected_error[1])
        else:
            self.assertEqual(m.compare_snapshots(self.before, self.after, **tolerances), expected)

    def test_fast_chunk_boundaries(self):
        data = (b"0 0x0p+0\n18446744073709551615 23 -0x1.fffffffffffffp+999\n"
                b"19 0x0.0000000000001p-999\n22 38 0x1.23456789abcdep-32\n") * 9
        self.before.write_bytes(data)
        for size in [1, 2, 3, 7, 17, 31, 64, 1024 * 1024]:
            self.assertEqual(m.canonical_snapshot_entries(self.before, size), 36)
        self.assert_fast_matches_parser(data)
        self.assert_fast_matches_parser(data, rtol=64*math.ulp(1.), atol=64*math.ulp(1.))

    def test_fast_nonfinite_identical_files(self):
        for token in [b"nan", b"NaN", b"inf", b"-inf", b"0x1p+1024", b"0x1p+9999"]:
            self.assert_fast_matches_parser(b"0 " + token + b"\n")

    def test_fast_large_exponent_fallback(self):
        for token in [b"0x1p+1023", b"0x1p-1022", b"0x1p-9999"]:
            data = b"0 " + token + b"\n"
            self.before.write_bytes(data)
            self.assertIsNone(m.canonical_snapshot_entries(self.before))
            self.assert_fast_matches_parser(data)

    def test_fast_large_index_fallback(self):
        data = b"123456789012345678901234 1 0x1p+0\n"
        self.before.write_bytes(data)
        self.assertIsNone(m.canonical_snapshot_entries(self.before))
        self.assert_fast_matches_parser(data)

    def test_fast_empty(self):
        self.assert_fast_matches_parser(b"")

    def test_fast_no_final_newline(self):
        data = b"1 2 0x1p+0"
        self.before.write_bytes(data)
        self.assertIsNone(m.canonical_snapshot_entries(self.before))
        self.assert_fast_matches_parser(data)

    def test_fast_malformed_rows(self):
        for data in [b"\n", b"0 1 2 0x1p+0\n", b"x 0x1p+0\n", b"-1 0x1p+0\n", b"0 bad\n"]:
            self.assert_fast_matches_parser(data)

    def test_fast_noncanonical_finite_fallback(self):
        for token in [b"+0x1p+0", b"0X1.ABCP+0", b"0x2p+0", b"0x1.123456789abcdep+0", b"0x1p0"]:
            self.assert_fast_matches_parser(b"0 " + token + b"\n")
        self.assert_fast_matches_parser(b"0\t0x1p+0\n")

    def test_fast_noncanonical_suffix(self):
        for data in [b"0 0x1p+0\ninvalidsuffix", b"bad\n1 0x1p+0\n", b"badprefix1 0x1p+0\n"]:
            self.assert_fast_matches_parser(data)

    def test_fast_unusual_tolerance_fallback(self):
        self.assert_fast_matches_parser(b"0 0x1p+999\n", rtol=1e308)
        self.assert_fast_matches_parser(b"0 0x1p+0\n", atol=-1.)
        self.assert_fast_matches_parser(b"0 0x1p+0\n", atol=float("nan"))

    def test_fast_hash_mismatch_uses_tolerant_parser(self):
        for tolerance in [0., 1e-14]:
            value = self.compare("0 0x1p+0\n", "0 0x1.0000000000001p+0\n", rtol=tolerance)
            self.assertEqual(value["accepted"], tolerance != 0.)


    def test_cli_mapping_default_and_explicit(self):
        root = Path(self.temp.name)
        builds = {name: root / (name + "_build") for name in ["before", "after"]}
        for name, build in builds.items():
            build.mkdir()
            for file in ["benchmark_assembly", "libmetric_flow_x.so", "revision.txt"]:
                (build / file).write_text(name)
        parameters = root / "workload.prm"
        parameters.write_text("test parameters")
        for mapping in [None, "pe-list=8,9:ordered"]:
            output = root / ("default" if mapping is None else "mapped")
            commands = []
            expected_mapping = mapping or "core"
            def run(command, **kwargs):
                if command[0] == "mpirun":
                    commands.append(command)
                    self.assertEqual(command[command.index("--map-by") + 1], expected_mapping)
                    self.assertEqual(command[command.index("--bind-to") + 1], "core")
                    prefix = Path(command[-1])
                    for state in range(3):
                        for metric in ["residual", "state_jacobian", "derivative_jacobian"]:
                            indices = "0 " if metric == "residual" else "0 0 "
                            Path(str(prefix) + f".state{state}.rank0.{metric}").write_text(indices + "0x1p+0\n")
                    before = command[command.index("-n") + 2] == str(builds["before"] / "benchmark_assembly")
                    record = {"mpi_ranks": 1}
                    record.update({metric + "_median_s": 2. if before else 1.
                                   for metric in ["residual", "jacobian", "derivative"]})
                    return m.subprocess.CompletedProcess(command, 0, stdout="BENCHMARK " + json.dumps(record))
                return m.subprocess.CompletedProcess(command, 0)
            argv = ["compare_assembly.py", "--before", str(builds["before"]),
                    "--after", str(builds["after"]), "--inputs", str(parameters),
                    "--ranks", "1", "--runs", "1", "--repetitions", "1", "--output", str(output)]
            if mapping:
                argv += ["--map-by", mapping]
            with mock.patch.object(m.sys, "argv", argv), mock.patch.object(m.sys, "stdout"), \
                 mock.patch.object(m.platform, "platform", return_value="test-platform"), \
                 mock.patch.object(m.subprocess, "run", side_effect=run):
                m.main()
            self.assertEqual(len(commands), 2)
            full = json.loads((output / "metadata.json").read_text())
            compact = json.loads((output / "results.json").read_text())["metadata"]
            for metadata in [full, compact]:
                self.assertEqual(metadata["map_by"], expected_mapping)
                self.assertEqual(metadata["bind_to"], "core")


if __name__=='__main__':unittest.main(verbosity=2)
