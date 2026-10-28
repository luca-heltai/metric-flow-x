"""Check resource isolation, dependencies, and partial submission receipts."""

import importlib.util
import ast
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import re
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
SPEC = importlib.util.spec_from_file_location('gpu_submission', ROOT / 'scripts/submit_paper_gpu.py')
GPU = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GPU)


class PaperGpuSubmissionTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='paper gpu ')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.env = self.root / 'environment.sh'
        self.env.write_text('')
        self.options = ['--output-root', str(self.root / 'runs'), '--env', str(self.env)]
        source = patch.object(GPU, 'capture_source', return_value={'revision': 'a' * 40, 'files': {}})
        source.start()
        self.addCleanup(source.stop)

    def test_dry_run_and_generated_scripts(self):
        with patch.object(GPU.subprocess, 'run') as submit:
            self.assertEqual(GPU.main([*self.options, '--dry-run']), 0)
        submit.assert_not_called()
        root = self.root / 'runs'
        scaling = (root / 'jobs/scaling.sh').read_text()
        self.assertEqual(scaling.count('--output-root'), 14)
        for base in ('network50k', 'network100k'):
            for ranks in (1, 2, 4, 8, 16, 32, 64):
                self.assertIn(f'{base}-np{ranks} ', scaling)
        self.assertIn('--prtemca ras', (root / 'jobs/mpirun-local.sh').read_text())
        self.assertIn('--executable-revision "$revision"', scaling)
        self.assertIn('--build-provenance', scaling)
        self.assertEqual(len(json.loads((root / 'output-cases.json').read_text())), 35)
        for script in (root / 'jobs').glob('*.sh'):
            subprocess.run(['bash', '-n', str(script)], check=True)
            for block in re.findall(r"<<'(PY\w+)'\n(.*?)\n\1", script.read_text(), re.S):
                ast.parse(block[1])
        self.assertIn('finalize_paper_output.py', (root / 'jobs/collect.sh').read_text())

    def test_smoke_does_not_export_paper_measurements(self):
        self.assertEqual(GPU.main([*self.options, '--dry-run', '--smoke']), 0)
        root = self.root / 'runs'
        self.assertEqual((root / 'jobs/scaling.sh').read_text().count('--final-time 0.001'), 14)
        self.assertNotIn('finalize_paper_output.py', (root / 'jobs/collect.sh').read_text())

    def test_scaling_only_smaller_node(self):
        with patch.object(GPU.subprocess, 'run') as submit:
            self.assertEqual(GPU.main([*self.options, '--dry-run', '--scaling-only',
                                      '--node', 'gpu04', '--partition', 'compute',
                                      '--ranks', '1', '2', '4']), 0)
        submit.assert_not_called()
        root = self.root / 'runs'
        record = json.loads((root / 'submission.json').read_text())
        self.assertEqual(list(record['jobs']), ['build', 'scaling', 'collect'])
        self.assertEqual(record['partition'], 'compute')
        self.assertEqual((root / 'jobs/scaling.sh').read_text().count('--output-root'), 6)
        self.assertNotIn('collect_paper_outputs.py', (root / 'jobs/collect.sh').read_text())

    def test_submissions_chain_and_retain_ids_on_failure(self):
        manifest = json.loads(GPU.MANIFEST.read_text())
        manifest['meshes'] = {}  # No downloads needed by this submission test.
        fixture = self.root / 'manifest.json'
        fixture.write_text(json.dumps(manifest))
        calls = []

        def sbatch(command, **kwargs):
            calls.append(command)
            if len(calls) == 4:
                raise subprocess.CalledProcessError(1, command)
            return subprocess.CompletedProcess(command, 0, f'{100 + len(calls)};cluster\n', '')

        with patch.object(GPU, 'MANIFEST', fixture), patch.object(GPU.subprocess, 'run', side_effect=sbatch):
            self.assertEqual(GPU.main(self.options), 1)
        self.assertIn('--exclusive', calls[1])
        self.assertIn('--nodelist=gpu04', calls[1])
        self.assertIn('--ntasks=64', calls[1])
        self.assertIn('--dependency=afterok:101', calls[1])
        self.assertIn('--array=0-34', calls[2])
        self.assertIn('--ntasks=4', calls[2])
        self.assertIn('--dependency=afterok:102', calls[2])
        self.assertIn('--dependency=afterok:103', calls[3])
        receipt = json.loads((self.root / 'runs/submission.json').read_text())
        self.assertEqual(receipt['jobs'], {'build': '101', 'scaling': '102', 'outputs': '103'})


if __name__ == '__main__':
    unittest.main()
