#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Offline checks for complete-cycle acceptance and time/pressure units."""
import contextlib
import csv
import io
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools/adan'))
from analyze_cycles import summarize
from analyze_full import MMHG_PA


class CardiacCycles(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.initial = dict(tolerance_ramp_enabled=True,
                            ramp_last_relative_factor=1., ramp_last_absolute_factor=1.)
        self.write_initial()
        (self.root/'ida_statistics.csv').write_text('n_steps\n1000\n')
        self.write_probes()

    def write_initial(self):
        (self.root/'initialization_statistics.json').write_text(json.dumps(self.initial))

    def write_probes(self, count=401):
        with (self.root/'probes.csv').open('w') as stream:
            writer = csv.writer(stream)
            writer.writerow(['time_s', 'vessel_id', 'pressure_Pa', 'flow_m3_s', 'area_m2', 'velocity_m_s'])
            for tick in range(count):
                for vid in (1630, 1631):
                    p = (90+20*math.sin(2*math.pi*tick/40))*MMHG_PA
                    writer.writerow([tick*.02, vid, p, 96.36e-6, 1e-4, .9636])

    def summary(self):
        with contextlib.redirect_stdout(io.StringIO()):
            return summarize(self.root)

    def test_periodic_waveform_and_units(self):
        report = self.summary()
        self.assertEqual(len(report['final_tolerance_cycles']), 4)
        for cycle in report['final_tolerance_cycles']:
            self.assertAlmostEqual(cycle['MAP_mmHg'], 90.)
            self.assertAlmostEqual(cycle['mean_aortic_flow_ml_s'], 96.36)
            self.assertAlmostEqual(cycle['systolic_pressure_mmHg'], 110.)
            self.assertAlmostEqual(cycle['diastolic_pressure_mmHg'], 70.)
        self.assertLess(report['cycle10_vs9_max_absolute_difference']['1630']['pressure_Pa'], 1e-9)

    def test_partial_run_rejected(self):
        self.write_probes(400)
        with self.assertRaisesRegex(ValueError, 'complete ten cycles'):
            self.summary()

    def test_final_tolerances_required(self):
        self.initial['ramp_last_absolute_factor'] = 2.
        self.write_initial()
        with self.assertRaisesRegex(ValueError, 'final tolerances'):
            self.summary()


if __name__ == '__main__':
    unittest.main()
