"""adapter pool 的下載端：checksum 不符不得落地，相符者不重下。"""

import hashlib
import importlib.util
import json
import pathlib
import tempfile
import unittest

_SCRIPTS = pathlib.Path(__file__).resolve().parent.parent / "scripts"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, _SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fetch_adapter_pool = _load("fetch_adapter_pool")


def _spec(payload: bytes) -> dict:
    return {"size": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


class UpToDateTests(unittest.TestCase):
    """已在本機且相符的檔案要跳過——重跑 setup 不該重下 2.6 GiB。"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = pathlib.Path(self.tmp.name) / "adapter_model.safetensors"

    def test_matching_content_is_skipped(self):
        self.path.write_bytes(b"weights")
        self.assertTrue(
            fetch_adapter_pool.up_to_date(str(self.path), _spec(b"weights")))

    def test_missing_file_is_not_up_to_date(self):
        self.assertFalse(
            fetch_adapter_pool.up_to_date(str(self.path), _spec(b"weights")))

    def test_same_size_different_content_is_not_up_to_date(self):
        """大小相同但內容不同——只比大小會漏掉，所以必須算 sha256。"""
        self.path.write_bytes(b"WEIGHTS")
        self.assertFalse(
            fetch_adapter_pool.up_to_date(str(self.path), _spec(b"weights")))

    def test_truncated_download_is_not_up_to_date(self):
        self.path.write_bytes(b"weig")
        self.assertFalse(
            fetch_adapter_pool.up_to_date(str(self.path), _spec(b"weights")))


class FetchOneTests(unittest.TestCase):
    """下載途中損壞的檔案不得留在 adapter/ 裡被當成完整的 adapter。"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = pathlib.Path(self.tmp.name)
        self.adapter_dir = self.root / "adapter"
        self.remote = self.root / "remote"
        self.remote.mkdir()

        self.contents = {
            "adapter_config.json": b'{"r": 8}',
            "adapter_model.safetensors": b"weights",
        }
        self.entry = {
            "name": "task0", "path": "task0",
            "files": {name: _spec(payload)
                      for name, payload in self.contents.items()},
        }

    def _install_downloader(self, payloads):
        """把 hf_hub_download 換成從本地假 repo 取檔，測試不碰網路。"""
        def download(repo_id, filename, revision=None, token=None):
            target = self.remote / filename.replace("/", "_")
            target.write_bytes(payloads[filename.rsplit("/", 1)[-1]])
            return str(target)

        import huggingface_hub

        original = huggingface_hub.hf_hub_download
        huggingface_hub.hf_hub_download = download
        self.addCleanup(setattr, huggingface_hub, "hf_hub_download", original)

    def test_a_clean_download_lands_both_files(self):
        self._install_downloader(self.contents)
        count = fetch_adapter_pool.fetch_one(
            "org/repo", self.entry, str(self.adapter_dir),
            revision=None, token=None)

        self.assertEqual(count, 2)
        landed = self.adapter_dir / "task0"
        self.assertEqual(
            (landed / "adapter_model.safetensors").read_bytes(), b"weights")
        self.assertEqual(json.loads(
            (landed / "adapter_config.json").read_text())["r"], 8)

    def test_a_corrupted_file_is_rejected_and_nothing_is_left_behind(self):
        corrupted = dict(self.contents)
        corrupted["adapter_model.safetensors"] = b"corrupt"  # 同長度、不同內容
        self._install_downloader(corrupted)

        with self.assertRaises(fetch_adapter_pool.AdapterPoolError) as caught:
            fetch_adapter_pool.fetch_one(
                "org/repo", self.entry, str(self.adapter_dir),
                revision=None, token=None)

        self.assertIn("sha256", str(caught.exception))
        landed = self.adapter_dir / "task0"
        self.assertFalse((landed / "adapter_model.safetensors").exists())
        self.assertFalse((landed / "adapter_model.safetensors.tmp").exists())

    def test_a_second_run_downloads_nothing(self):
        self._install_downloader(self.contents)
        fetch_adapter_pool.fetch_one(
            "org/repo", self.entry, str(self.adapter_dir),
            revision=None, token=None)
        again = fetch_adapter_pool.fetch_one(
            "org/repo", self.entry, str(self.adapter_dir),
            revision=None, token=None)

        self.assertEqual(again, 0)


class RemoteManifestTests(unittest.TestCase):
    """manifest 是唯一真相：沒有它就不是一個可用的 pool repo。"""

    def test_an_empty_adapter_list_is_refused(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = pathlib.Path(tmp.name) / "pool150_manifest.json"
        path.write_text(json.dumps({"schema_version": 1, "adapters": []}))

        import huggingface_hub

        original = huggingface_hub.hf_hub_download
        huggingface_hub.hf_hub_download = (
            lambda *a, **k: str(path))
        self.addCleanup(setattr, huggingface_hub, "hf_hub_download", original)

        with self.assertRaises(SystemExit) as caught:
            fetch_adapter_pool.remote_manifest("org/repo", None, None)
        self.assertIn("adapters", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
