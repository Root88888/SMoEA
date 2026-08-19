import json
import tempfile
import unittest
from pathlib import Path

from system.registry import RegistryError, load_registry


def write_registry(root: Path, entries, *, schema_version=1) -> Path:
    path = root / "registry.json"
    path.write_text(
        json.dumps({"schema_version": schema_version, "entries": entries}),
        encoding="utf-8",
    )
    return path


class RegistryTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def test_relative_paths_resolve_against_the_registry_directory(self):
        path = write_registry(self.root, [
            {"id": "ties", "method": "artifact",
             "artifact_dir": "ties_only/run/prepare/merged_model"},
        ])

        entry = load_registry(path).get("ties")

        self.assertEqual(
            entry.artifact_dir,
            (self.root / "ties_only/run/prepare/merged_model").resolve(),
        )

    def test_absolute_paths_are_taken_as_declared(self):
        path = write_registry(self.root, [
            {"id": "ties", "method": "artifact",
             "artifact_dir": "/mnt/runs/ties/prepare/merged_model"},
        ])

        entry = load_registry(path).get("ties")

        self.assertEqual(
            entry.artifact_dir, Path("/mnt/runs/ties/prepare/merged_model"))

    def test_parent_traversal_is_refused(self):
        path = write_registry(self.root, [
            {"id": "escape", "method": "artifact",
             "artifact_dir": "../../etc/merged_model"},
        ])

        with self.assertRaisesRegex(RegistryError, r"\.\."):
            load_registry(path)

    def test_duplicate_ids_are_refused(self):
        path = write_registry(self.root, [
            {"id": "same", "method": "base"},
            {"id": "same", "method": "base"},
        ])

        with self.assertRaisesRegex(RegistryError, "重複"):
            load_registry(path)

    def test_each_method_must_declare_the_paths_it_needs(self):
        cases = [
            ({"id": "a", "method": "artifact"}, "artifact_dir"),
            ({"id": "b", "method": "arrow"}, "adapter_manifest"),
            ({"id": "c", "method": "taskwise_k16_arrow"}, "router_dir"),
        ]
        for entry, missing in cases:
            with self.subTest(method=entry["method"]):
                path = write_registry(self.root, [entry])
                with self.assertRaisesRegex(RegistryError, missing):
                    load_registry(path)

    def test_base_needs_no_external_files(self):
        path = write_registry(self.root, [{"id": "base", "method": "base"}])

        entry = load_registry(path).get("base")

        self.assertIsNone(entry.artifact_dir)
        self.assertEqual(
            entry.as_system_overrides()["rejection_method"], "base")

    def test_an_unknown_method_is_refused(self):
        path = write_registry(self.root, [{"id": "x", "method": "lorahub"}])

        with self.assertRaisesRegex(RegistryError, "method"):
            load_registry(path)

    def test_an_unknown_id_reports_what_is_available(self):
        path = write_registry(self.root, [{"id": "base", "method": "base"}])

        with self.assertRaisesRegex(RegistryError, "base"):
            load_registry(path).get("missing")

    def test_an_unsupported_schema_version_is_refused(self):
        path = write_registry(self.root, [{"id": "base", "method": "base"}],
                              schema_version=2)

        with self.assertRaisesRegex(RegistryError, "schema"):
            load_registry(path)

    def test_an_empty_registry_is_refused(self):
        path = write_registry(self.root, [])

        with self.assertRaisesRegex(RegistryError, "至少"):
            load_registry(path)

    def test_entries_become_engine_configuration(self):
        path = write_registry(self.root, [
            {"id": "arrow", "method": "arrow",
             "adapter_manifest": "/data/pool150/manifest.json",
             "adapter_root": "/data/pool150",
             "description": "Direct Arrow, 150 experts"},
        ])

        entry = load_registry(path).get("arrow")
        overrides = entry.as_system_overrides()

        self.assertEqual(overrides["rejection_method"], "arrow")
        self.assertEqual(
            overrides["rejection_adapter_manifest"], "/data/pool150/manifest.json")
        self.assertIsNone(overrides["rejection_artifact_dir"])
        self.assertIn("Direct Arrow", entry.summary())


if __name__ == "__main__":
    unittest.main()
