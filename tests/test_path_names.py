"""Keep project-owned paths and published dataset paths consistently named."""
import json
import re
from pathlib import Path
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[1]
STANDARD_FILENAMES = {
    "CMakeLists.txt", ".clang-format", "README.md", "LICENSE.md", "CITATION.cff", "Doxyfile", "Dockerfile"
}


class PathNamesTest(unittest.TestCase):
    def assert_standard_path(self, path):
        for part in Path(path).parts:
            if part in STANDARD_FILENAMES:
                continue
            # deal.II discovers MPI test variants through this required suffix.
            if Path(path).parts[0] == "tests" and re.fullmatch(
                    r"test_[a-z0-9_]+\.mpirun=[1-9][0-9]*\.output", part):
                continue
            self.assertEqual(part, part.lower(), path)
            self.assertRegex(part, r"^[a-z0-9_.]+$", path)

    def test_tracked_file_and_directory_names(self):
        paths = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0")
        for path in filter(None, paths):
            with self.subTest(path=path):
                self.assert_standard_path(path)

    def test_published_dataset_names(self):
        manifest = json.loads((ROOT / "datasets/datasets_si_v1.json").read_text())
        for key in ("archive", "cerebral_archive"):
            self.assert_standard_path(manifest[key]["name"])
        for item in manifest["files"] + manifest["cerebral_files"]:
            with self.subTest(path=item["path"]):
                self.assert_standard_path(item["path"])
        for item in manifest["solver_meshes"]:
            self.assert_standard_path(item["path"])
            self.assert_standard_path(item["source"])


if __name__ == "__main__":
    unittest.main()
