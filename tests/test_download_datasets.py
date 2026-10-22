"""Exercise the real downloader with tiny local release-asset fixtures."""

import hashlib
import io
import os
from pathlib import Path
import subprocess
import tarfile
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/download_datasets.sh"
ASSETS = (
    ("metric_flow_x_datasets_si_v1.tar.gz", "notebooks/binary_tree_255.vtk",
     "ee7c831a626109639c32edeea5085e5cb8a6cd15cd6dce31cbd75bf173787c1a"),
    ("cerebral_datasets_v2.tar.gz", "notebooks/cerebral_data/standard_cco.vtk",
     "3f94081c6a256e2d341a45c51889e9090056273664cc9503cb22a2ffa98c5ee1"),
    ("paper_inputs_v1.tar.gz", "notebooks/paper_inputs/liver_vascular_tree.vtk",
     "452d1df2db7d62dd0c20fed3237c516d9b45cfe61553e472f53db4e97212f30c"),
)


class DownloadDatasetsTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.repo = self.root / "repository with spaces"
        (self.repo / "scripts").mkdir(parents=True)
        source = SCRIPT.read_text()
        for asset, member, checksum in ASSETS:
            archive = self.root / asset
            with tarfile.open(archive, "w:gz") as tar:
                data = b"# vtk DataFile Version 3.0\nfixture\n"
                info = tarfile.TarInfo(member)
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
            self.assertEqual(source.count(checksum), 1)
            source = source.replace(checksum, hashlib.sha256(archive.read_bytes()).hexdigest())
        self.script = self.repo / "scripts/download_datasets.sh"
        self.script.write_text(source)
        fakebin = self.root / "bin"
        fakebin.mkdir()
        curl = fakebin / "curl"
        curl.write_text("""#!/usr/bin/env python3
import os, pathlib, shutil, sys
args = sys.argv[1:]
url = args[-1]
assert url.startswith('https://github.com/luca-heltai/metric-flow-x/releases/download/datasets-final-version-v1/')
assert '--fail' in args and '--location' in args and '--retry' in args
if os.environ.get('FAIL_DOWNLOAD'):
    sys.exit(22)
shutil.copyfile(pathlib.Path(os.environ['FIXTURES']) / url.rsplit('/', 1)[1],
                args[args.index('--output') + 1])
""")
        curl.chmod(0o755)
        self.env = dict(os.environ, PATH=str(fakebin) + os.pathsep + os.environ["PATH"],
                        FIXTURES=str(self.root))

    def run_script(self, *args):
        return subprocess.run(["bash", str(self.script), *map(str, args)],
                              env=self.env, cwd=self.root, capture_output=True, text=True)

    def assert_extracted(self, destination):
        for _, member, _ in ASSETS:
            self.assertEqual((destination / member).read_bytes(),
                             b"# vtk DataFile Version 3.0\nfixture\n")

    def test_default_destination_is_repository_root(self):
        result = self.run_script()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assert_extracted(self.repo)

    def test_custom_destination_with_spaces(self):
        destination = self.root / "custom destination"
        result = self.run_script(destination)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assert_extracted(destination)
        self.assertFalse((self.repo / "notebooks").exists())

    def test_corrupt_network_archive_is_not_extracted(self):
        (self.root / ASSETS[0][0]).write_bytes(b"corrupt")
        result = self.run_script()
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse((self.repo / "notebooks").exists())

    def test_corrupt_cerebral_archive_is_not_extracted(self):
        (self.root / ASSETS[1][0]).write_bytes(b"corrupt")
        result = self.run_script()
        self.assertNotEqual(result.returncode, 0)
        self.assertTrue((self.repo / ASSETS[0][1]).exists())
        self.assertFalse((self.repo / ASSETS[1][1]).exists())

    def test_http_failure_stops_before_extraction(self):
        self.env["FAIL_DOWNLOAD"] = "1"
        result = self.run_script()
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse((self.repo / "notebooks").exists())

    def test_corrupt_paper_archive_is_not_extracted(self):
        (self.root / ASSETS[2][0]).write_bytes(b"corrupt")
        result = self.run_script()
        self.assertNotEqual(result.returncode, 0)
        self.assertTrue((self.repo / ASSETS[1][1]).exists())
        self.assertFalse((self.repo / ASSETS[2][1]).exists())


if __name__ == "__main__":
    unittest.main()
