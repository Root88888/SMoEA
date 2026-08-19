import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from system.inference import InferenceEngine, RejectionSelection
from system.merged_model import MergedModelError


class FakeTuner:
    def __init__(self, events):
        self.events = events

    def enable_adapter_layers(self):
        self.events.append("peft:enable")

    def disable_adapter_layers(self):
        self.events.append("peft:disable")


class FakePeftModel:
    def __init__(self, events):
        self.events = events
        self.peft_config = {"task1": object()}
        self.base_model = FakeTuner(events)

    def set_adapter(self, name):
        self.events.append(f"peft:set:{name}")


class FakeDenseController:
    def __init__(self, events):
        self.events = events

    def enable(self):
        self.events.append("dense:enable")

    def disable(self):
        self.events.append("dense:disable")


class FakeArrowController:
    def __init__(self, events):
        self.events = events

    def attach(self, model):
        self.events.append("arrow:attach")

    def enable(self):
        self.events.append("arrow:enable")

    def disable(self):
        self.events.append("arrow:disable")


class InferenceSwitchingTests(unittest.TestCase):
    @patch("system.inference.validate_inference_config")
    @patch("system.inference.validate_base_model_config")
    @patch("system.inference.load_merged_model_artifact")
    def test_the_configured_artifact_directory_is_loaded_and_validated(
        self, load_artifact, validate_base, validate_inference
    ):
        load_artifact.return_value = SimpleNamespace(base_model_revision="local")
        cfg = {
            "system": {
                "base_model": "base",
                "adapter_dir": "adapter",
                "rejection_method": "artifact",
                "rejection_artifact_dir": "/data/runs/ties_only/RUN/prepare/merged_model",
            }
        }

        engine = InferenceEngine(cfg)

        self.assertEqual(engine._selection.method, "artifact")
        load_artifact.assert_called_once_with(
            "/data/runs/ties_only/RUN/prepare/merged_model",
            expected_base_model="base",
        )
        validate_base.assert_called_once()
        validate_inference.assert_called_once()

    def test_selected_artifact_revision_is_used_for_base_loading(self):
        cfg = {
            "system": {
                "base_model": "base",
                "adapter_dir": "adapter",
                "rejection_method": "base",
            }
        }
        engine = InferenceEngine(cfg)
        engine._selection = RejectionSelection(
            id="ties", method="artifact",
            merged_artifact=SimpleNamespace(base_model_revision="exact-hub-commit"),
        )

        self.assertEqual(
            engine._base_revision_kwargs(),
            {"revision": "exact-hub-commit"},
        )

    def test_artifact_method_requires_an_explicit_artifact_directory(self):
        cfg = {
            "system": {
                "base_model": "base",
                "adapter_dir": "adapter",
                "rejection_method": "artifact",
                "rejection_artifact_dir": None,
            }
        }

        with self.assertRaisesRegex(MergedModelError, "未設定"):
            InferenceEngine(cfg)

    def test_an_unknown_rejection_method_is_refused_before_serving(self):
        cfg = {
            "system": {
                "base_model": "base",
                "adapter_dir": "adapter",
                "rejection_method": "lorahub",
            }
        }

        with self.assertRaisesRegex(MergedModelError, "rejection_method"):
            InferenceEngine(cfg)

    def test_task_to_dense_merged_to_task_keeps_updates_mutually_exclusive(self):
        cfg = {
            "system": {
                "base_model": "base",
                "adapter_dir": "adapter",
                "rejection_method": "base",
            }
        }
        events = []
        engine = InferenceEngine(cfg)
        engine.model = FakePeftModel(events)
        engine.tokenizer = object()
        engine._loaded_adapters = {"task1": "task1"}
        engine._dense_controller = FakeDenseController(events)
        engine._selection = RejectionSelection(
            id="ties", method="artifact",
            merged_artifact=SimpleNamespace(
                format="dense_delta_v1",
                condition_id="ties_only",
                run_id="run-123",
            ),
        )

        engine.ensure_adapter("task1")
        engine.ensure_rejection()
        engine.ensure_adapter("task1")

        self.assertEqual(
            events,
            [
                "dense:disable",
                "peft:enable",
                "peft:set:task1",
                "peft:disable",
                "dense:enable",
                "dense:disable",
                "peft:enable",
                "peft:set:task1",
            ],
        )
        self.assertEqual(engine._active, "task1")

    def test_base_rejection_disables_task_and_dense_updates(self):
        cfg = {
            "system": {
                "base_model": "base",
                "adapter_dir": "adapter",
                "rejection_method": "base",
            }
        }
        events = []
        engine = InferenceEngine(cfg)
        engine.model = FakePeftModel(events)
        engine.tokenizer = object()
        engine._dense_controller = FakeDenseController(events)

        info = engine.ensure_rejection()

        self.assertEqual(info["condition_id"], "base")
        self.assertEqual(events, ["dense:disable", "peft:disable"])
        self.assertIsNone(engine._active)

    def test_arrow_rejection_disables_other_updates_before_enabling_arrow(self):
        cfg = {
            "system": {
                "base_model": "base",
                "adapter_dir": "adapter",
                "rejection_method": "base",
            }
        }
        events = []
        engine = InferenceEngine(cfg)
        engine.model = FakePeftModel(events)
        engine.tokenizer = object()
        engine._dense_controller = FakeDenseController(events)
        engine._selection = RejectionSelection(
            id="arrow", method="arrow",
            arrow_artifact=SimpleNamespace(
                condition_id="arrow", run_id="run-1", adapter_paths=("a", "b")),
        )
        engine._arrow_controller = FakeArrowController(events)

        info = engine.ensure_rejection()

        self.assertEqual(info["condition_id"], "arrow")
        self.assertEqual(
            events,
            ["dense:disable", "peft:disable", "arrow:attach", "arrow:enable"],
        )


class RejectionSelectionTests(unittest.TestCase):
    """ADR-0001：依 registry 在執行期換 artifact。"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def _registry(self, entries):
        path = self.root / "registry.json"
        path.write_text(json.dumps({"schema_version": 1, "entries": entries}),
                        encoding="utf-8")
        return str(path)

    def _cfg(self, registry, **extra):
        return {"system": {"base_model": "base", "adapter_dir": "adapter",
                           "artifact_registry": registry, **extra}}

    @patch("system.inference.validate_inference_config")
    @patch("system.inference.validate_base_model_config")
    @patch("system.inference.load_merged_model_artifact")
    def test_selecting_an_entry_validates_it_and_becomes_current(
        self, load_artifact, _validate_base, _validate_inference
    ):
        load_artifact.return_value = SimpleNamespace(
            base_model_revision="local", format="dense_delta_v1",
            condition_id="ties_only", run_id="run-1")
        registry = self._registry([
            {"id": "base", "method": "base"},
            {"id": "ties", "method": "artifact",
             "artifact_dir": "/runs/ties/prepare/merged_model"},
        ])
        engine = InferenceEngine(self._cfg(registry))

        self.assertEqual(engine.current_rejection()["condition_id"], "base")
        current = engine.select_rejection("ties")

        self.assertEqual(current["id"], "ties")
        self.assertEqual(current["condition_id"], "ties_only")
        self.assertEqual(engine._selection.method, "artifact")

    @patch("system.inference.load_merged_model_artifact")
    def test_a_failed_selection_leaves_the_current_one_untouched(
        self, load_artifact
    ):
        load_artifact.side_effect = MergedModelError("checksum mismatch")
        registry = self._registry([
            {"id": "base", "method": "base"},
            {"id": "broken", "method": "artifact",
             "artifact_dir": "/runs/broken/prepare/merged_model"},
        ])
        engine = InferenceEngine(self._cfg(registry))
        before = engine.current_rejection()

        with self.assertRaisesRegex(MergedModelError, "checksum"):
            engine.select_rejection("broken")

        self.assertEqual(engine.current_rejection(), before)
        self.assertEqual(engine._selection.method, "base")

    def test_an_unknown_id_lists_what_the_registry_declares(self):
        registry = self._registry([{"id": "base", "method": "base"}])
        engine = InferenceEngine(self._cfg(registry))

        with self.assertRaisesRegex(MergedModelError, "base"):
            engine.select_rejection("nope")

    def test_selection_by_id_needs_a_registry(self):
        engine = InferenceEngine(self._cfg(None))

        with self.assertRaisesRegex(MergedModelError, "artifact_registry"):
            engine.select_rejection("ties")
        self.assertEqual(engine.available_rejections(), ())

    def test_the_registry_is_listed_without_touching_weights(self):
        registry = self._registry([
            {"id": "base", "method": "base"},
            {"id": "arrow", "method": "arrow",
             "adapter_manifest": "/data/pool150/manifest.json"},
        ])
        engine = InferenceEngine(self._cfg(registry))

        # Arrow 的資產不存在，但列出可選項目不該去讀它。
        self.assertEqual([e.id for e in engine.available_rejections()],
                         ["base", "arrow"])


if __name__ == "__main__":
    unittest.main()
