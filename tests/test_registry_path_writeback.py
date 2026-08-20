"""登記進 registry 的路徑，必須是 load_registry 解得回原處的那一個。

registry 的契約是「相對路徑以 registry 檔所在目錄為基準」，但寫入端手上的是
相對於 cwd 的路徑（configs 的 `artifact_root: artifacts` 就是相對的）。兩者
不一致時會疊成 `artifacts/artifacts/…`——而 load_registry 只做形式檢查、不碰
檔案，所以寫入後的複驗照樣過，要到選用時才炸。
"""

import json
import os
import pathlib
import tempfile
import unittest

from system.registry import load_registry, path_for_registry


class PathForRegistryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = pathlib.Path(self.tmp.name).resolve()
        self.registry = self.root / "artifacts" / "registry.json"
        self.registry.parent.mkdir()

    def test_an_artifact_under_the_registry_is_stored_relative(self):
        target = self.registry.parent / "ties_only" / "abc123" / "prepare" / "merged_model"
        self.assertEqual(
            path_for_registry(self.registry, target),
            os.path.join("ties_only", "abc123", "prepare", "merged_model"))

    def test_a_cwd_relative_path_does_not_double_up(self):
        """`--artifact-root artifacts` 走過來的樣子：相對 cwd，不是相對 registry。"""
        cwd = os.getcwd()
        os.chdir(self.root)
        self.addCleanup(os.chdir, cwd)

        stored = path_for_registry(
            "artifacts/registry.json",
            os.path.join("artifacts", "pico_ta", "abc123", "prepare",
                         "merged_model"))

        self.assertEqual(
            stored,
            os.path.join("pico_ta", "abc123", "prepare", "merged_model"))
        self.assertNotIn("artifacts" + os.sep + "artifacts", stored)

    def test_an_artifact_outside_the_registry_is_stored_absolute(self):
        target = self.root / "elsewhere" / "ta" / "abc123" / "prepare"
        stored = path_for_registry(self.registry, target)

        self.assertTrue(os.path.isabs(stored))
        self.assertEqual(pathlib.Path(stored), target)


class RoundTripTests(unittest.TestCase):
    """寫進去再讀回來，必須指回原本那個目錄。"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = pathlib.Path(self.tmp.name).resolve()
        self.artifact_root = self.root / "artifacts"
        self.registry = self.artifact_root / "registry.json"
        self.target = (self.artifact_root / "ties_only" / "abc123"
                       / "prepare" / "merged_model")
        self.target.mkdir(parents=True)

    def _write(self, artifact_dir):
        self.registry.write_text(json.dumps({
            "schema_version": 1,
            "entries": [{
                "id": "ties_only", "method": "artifact", "source": "prepared",
                "artifact_dir": path_for_registry(self.registry, artifact_dir),
                "description": "",
            }],
        }), encoding="utf-8")

    def test_a_cwd_relative_artifact_root_round_trips(self):
        cwd = os.getcwd()
        os.chdir(self.root)
        self.addCleanup(os.chdir, cwd)

        self._write(os.path.join("artifacts", "ties_only", "abc123",
                                 "prepare", "merged_model"))
        entry = load_registry(self.registry).get("ties_only")

        self.assertEqual(entry.artifact_dir, self.target)

    def test_an_absolute_artifact_root_round_trips(self):
        self._write(self.target)
        entry = load_registry(self.registry).get("ties_only")

        self.assertEqual(entry.artifact_dir, self.target)

    def test_the_registry_stays_valid_after_moving_the_whole_root(self):
        """registry 與 artifact 一起搬走仍然解得開——這是存相對路徑的理由。"""
        self._write(self.target)
        moved = self.root / "moved"
        os.rename(self.artifact_root, moved)

        entry = load_registry(moved / "registry.json").get("ties_only")

        self.assertEqual(
            entry.artifact_dir,
            moved / "ties_only" / "abc123" / "prepare" / "merged_model")


if __name__ == "__main__":
    unittest.main()
