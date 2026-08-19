import unittest
from types import SimpleNamespace
from unittest.mock import patch

from system.inference import InferenceEngine
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
    def test_legacy_merged_model_dir_still_selects_artifact(
        self, load_artifact, _validate_base, _validate_inference
    ):
        load_artifact.return_value = SimpleNamespace(base_model_revision="local")
        cfg = {
            "system": {
                "base_model": "base",
                "adapter_dir": "adapter",
                "merged_model_dir": "/legacy/merged_model",
            }
        }

        engine = InferenceEngine(cfg)

        self.assertEqual(engine._rejection_method, "artifact")
        load_artifact.assert_called_once_with(
            "/legacy/merged_model", expected_base_model="base"
        )

    def test_selected_artifact_revision_is_used_for_base_loading(self):
        cfg = {
            "system": {
                "base_model": "base",
                "adapter_dir": "adapter",
                "merged_model_dir": None,
            }
        }
        engine = InferenceEngine(cfg)
        engine._merged_artifact = SimpleNamespace(
            base_model_revision="exact-hub-commit"
        )

        self.assertEqual(
            engine._base_revision_kwargs(),
            {"revision": "exact-hub-commit"},
        )

    def test_production_requires_an_explicit_merged_model_directory(self):
        cfg = {
            "system": {
                "base_model": "base",
                "adapter_dir": "adapter",
                "merged_model_dir": None,
                "merged_model_required": True,
            }
        }

        with self.assertRaisesRegex(MergedModelError, "未設定"):
            InferenceEngine(cfg)

    def test_task_to_dense_merged_to_task_keeps_updates_mutually_exclusive(self):
        cfg = {
            "system": {
                "base_model": "base",
                "adapter_dir": "adapter",
                "merged_model_dir": None,
            }
        }
        events = []
        engine = InferenceEngine(cfg)
        engine.model = FakePeftModel(events)
        engine.tokenizer = object()
        engine._loaded_adapters = {"task1": "task1"}
        engine._dense_controller = FakeDenseController(events)
        engine._merged_artifact = SimpleNamespace(
            format="dense_delta_v1",
            condition_id="adamerging_pp",
            run_id="run-123",
        )

        engine.ensure_adapter("task1")
        engine.ensure_merged()
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
        engine._arrow_artifact = SimpleNamespace(
            condition_id="arrow", run_id="run-1", adapter_paths=("a", "b")
        )
        engine._arrow_controller = FakeArrowController(events)
        engine._rejection_method = "arrow"

        info = engine.ensure_rejection()

        self.assertEqual(info["condition_id"], "arrow")
        self.assertEqual(
            events,
            ["dense:disable", "peft:disable", "arrow:attach", "arrow:enable"],
        )


if __name__ == "__main__":
    unittest.main()
