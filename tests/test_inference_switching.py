import unittest
from types import SimpleNamespace

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


class InferenceSwitchingTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
