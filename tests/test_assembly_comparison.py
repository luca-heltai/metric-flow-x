"""Regression checks for numerical snapshot gates and paired timing summaries."""
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest

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

if __name__=='__main__':unittest.main(verbosity=2)
