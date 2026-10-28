"""Check manuscript column compatibility and reject invalid/mixed timing data."""
import csv
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
SPEC = importlib.util.spec_from_file_location('finalizer', ROOT / 'scripts/finalize_paper_output.py')
FINAL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FINAL)


class FinalizeTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.runs = self.root / 'campaign'
        self.paper = self.root / 'paper'
        self.profile = self.paper / 'scaling/gpu04'
        self.profile.mkdir(parents=True)
        (self.profile / 'machine.tex').write_text('hardware description\n')
        self.paths = []
        for network in ('network50k', 'network100k'):
            for mpi in (1, 2):
                folder = self.runs / 'scaling' / f'{network}-np{mpi}'
                results = folder / 'results'
                results.mkdir(parents=True)
                params = folder / 'parameters.prm'
                params.write_text('set Write solution output = false\nset Print current statistics = false\nset Final time = 1\n')
                with (results / 'scaling_timings.csv').open('w', newline='') as stream:
                    writer = csv.writer(stream)
                    writer.writerow(('n_mpi_processes', 'section', 'wall_time'))
                    # Independent known fixture: each column has a distinct value.
                    for section, value in zip(FINAL.SECTIONS, (10, 20, 30, 40)):
                        writer.writerow((mpi, section, value / mpi))
                (results / 'ida_statistics.csv').write_text('wall_time,n_steps\n100,42\n')
                (results / 'initialization_statistics.json').write_text(
                    '{"ida_setup_and_ic_wall_s": 10}')
                receipt = dict(case=dict(id=folder.name, base=network, suite='scaling', mpi=mpi),
                               status='completed', returncode=0, mpi=mpi, machine='gpu04',
                               hardware=dict(hostname='gpu04'), executable_revision='a' * 40,
                               executable_sha256='b' * 64, manifest_sha256='c' * 64,
                               parameters_sha256=FINAL.sha256(params), final_time_override=None,
                               settings={'Write solution output': 'false', 'Print current statistics': 'false'},
                               thread_environment={key: '1' for key in ('OMP_NUM_THREADS',
                                   'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'BLIS_NUM_THREADS',
                                   'DEAL_II_NUM_THREADS')}, process_wall_seconds=150)
                path = folder / 'run.json'
                path.write_text(json.dumps(receipt))
                self.paths.append(path)

    def export(self):
        return FINAL.finalize(self.runs, self.paper, 'gpu04', [2, 1])

    def change(self, **updates):
        path = self.paths[0]
        receipt = json.loads(path.read_text())
        receipt.update(updates)
        path.write_text(json.dumps(receipt))

    def assert_rejected(self, message):
        with self.assertRaisesRegex(ValueError, message):
            self.export()
        self.assertEqual([p.name for p in self.profile.iterdir()], ['machine.tex'])

    def test_submission_root_and_exact_manuscript_columns(self):
        report = self.export()
        rows = FINAL.read_table(self.profile / 'plot_data/scaling_50k.csv')
        self.assertEqual(list(rows[0]), ['n_mpi_processes', *FINAL.SECTIONS])
        self.assertEqual([row['n_mpi_processes'] for row in rows], ['1', '2'])
        self.assertEqual([float(rows[1][key]) for key in FINAL.SECTIONS], [5, 10, 15, 20])
        self.assertEqual(report['runs'][0]['integration_wall_seconds'], 90)
        parallel = next(run for run in report['runs'] if run['base'] == 'network50k' and run['mpi'] == 2)
        self.assertEqual(parallel['solve_speedup'], 1)
        self.assertEqual(parallel['solve_parallel_efficiency'], .5)
        self.assertEqual(parallel['section_speedups']['assemble_residual'], 2)
        self.assertEqual(parallel['section_parallel_efficiencies']['assemble_residual'], 1)
        self.assertFalse(report['manuscript_profile_complete'])
        self.assertEqual(len(report['raw_runs']), 4)
        self.assertTrue((self.profile / '50k_vessels/runs/network50k-np1/run.json').is_file())
        with self.assertRaisesRegex(ValueError, 'already contains results'):
            self.export()

    def test_direct_runner_root(self):
        report = FINAL.finalize(self.runs / 'scaling', self.paper, 'gpu04', [1, 2])
        self.assertTrue(report['scaling_tables_ready'])

    def test_failed_or_prepared_run(self):
        self.change(status='prepared')
        self.assert_rejected('Incomplete')

    def test_smoke_is_rejected(self):
        self.change(final_time_override=.001)
        self.assert_rejected('Smoke')

    def test_mixed_build(self):
        self.change(executable_sha256='d' * 64)
        self.assert_rejected('one executable')

    def test_mixed_host(self):
        self.change(hardware={'hostname': 'gpu03'})
        self.assert_rejected('one executable')

    def test_wrong_machine(self):
        self.change(machine='lnx')
        self.assert_rejected('Machine profile mismatch')

    def test_missing_rank(self):
        self.paths[0].unlink()
        self.assert_rejected('missing MPI counts')

    def test_modified_parameters(self):
        self.paths[0].with_name('parameters.prm').write_text('changed')
        self.assert_rejected('Parameters changed')

    def test_nonfinite_timing(self):
        path = self.paths[0].parent / 'results/scaling_timings.csv'
        path.write_text(path.read_text().replace('10.0', 'nan'))
        self.assert_rejected('Invalid')

    def test_duplicate_timing(self):
        path = self.paths[0].parent / 'results/scaling_timings.csv'
        path.write_text(path.read_text() + '1,assemble_residual,2\n')
        self.assert_rejected('Duplicate timing')

    def test_unbounded_threads(self):
        self.change(thread_environment={})
        self.assert_rejected('explicitly 1')

    def test_solution_output_cannot_be_hidden_in_receipt(self):
        path = self.paths[0].parent / 'parameters.prm'
        path.write_text(path.read_text().replace('output = false', 'output = true'))
        self.change(parameters_sha256=FINAL.sha256(path))
        self.assert_rejected('must be false')

    def test_different_numerical_settings_are_rejected(self):
        path = self.paths[0].parent / 'parameters.prm'
        path.write_text(path.read_text() + 'set Finite element degree = 2\n')
        self.change(parameters_sha256=FINAL.sha256(path))
        self.assert_rejected('input parameters differ')

    def test_partial_timing_sections(self):
        path = self.paths[0].parent / 'results/scaling_timings.csv'
        path.write_text('\n'.join(path.read_text().splitlines()[:-1]) + '\n')
        self.assert_rejected('Missing scaling sections')

    def test_historical_profile_is_protected(self):
        with self.assertRaisesRegex(ValueError, 'historical'):
            FINAL.finalize(self.runs, self.paper, 'raksha', [1, 2])


if __name__ == '__main__':
    unittest.main()
