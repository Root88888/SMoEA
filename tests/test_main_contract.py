import unittest
from unittest.mock import Mock, patch

from main import (
    external_task_key,
    generation_prompt,
    handle_rejection_command,
    read_interactive_request,
    run_interactive,
)
from system.merged_model import MergedModelError


class MainContractTests(unittest.TestCase):
    @patch(
        "builtins.input",
        side_effect=[":paste", "Task instructions", "Input text", ":send"],
    )
    def test_interactive_paste_mode_sends_one_multiline_request(self, _input):
        self.assertEqual(
            read_interactive_request(),
            "Task instructions\nInput text",
        )

    def test_generation_uses_the_answer_free_full_prompt(self):
        record = {
            "instance_id": "sample-1",
            "input": "short router input",
            "full_prompt": "Task instructions\nInput\nAnswer:",
            "output": "expected answer",
        }

        self.assertEqual(
            generation_prompt(record, "input"),
            "Task instructions\nInput\nAnswer:",
        )

    def test_generation_rejects_a_training_completion(self):
        record = {
            "instance_id": "sample-1",
            "input": "short router input",
            "full_prompt": "Task instructions\nAnswer: expected answer",
            "output": "expected answer",
        }

        with self.assertRaisesRegex(ValueError, "full_prompt.*答案"):
            generation_prompt(record, "input")

    def test_internal_9149_is_reported_as_original_ood_task149(self):
        self.assertEqual(external_task_key(9149), "task149")
        self.assertEqual(external_task_key(149), "task149")

    @patch("main.InferenceEngine")
    @patch("main.make_verifier")
    @patch("builtins.input", return_value="exit")
    def test_no_preload_does_not_load_the_verifier_before_a_query_needs_it(
        self, _input, make_verifier, _engine
    ):
        cfg = {
            "system": {"verifier_mode": "resident_4bit"},
            "verifier": {"model_path": "unused"},
        }
        router = Mock()
        router.units = [[0]]
        router.load_descriptions.return_value = {
            "0": {"description": "example"}
        }

        run_interactive(cfg, router, preload=False)

        make_verifier.assert_not_called()


class FakeEntry:
    def __init__(self, entry_id):
        self.id = entry_id

    def summary(self):
        return f"{self.id}  artifact  /runs/{self.id}"


class FakeSelectableEngine:
    def __init__(self, *, fails=False):
        self.current = {"id": "base", "condition_id": "base", "run_id": None}
        self.selected = []
        self._fails = fails

    def available_rejections(self):
        return (FakeEntry("base"), FakeEntry("ties"))

    def current_rejection(self):
        return self.current

    def select_rejection(self, entry_id):
        self.selected.append(entry_id)
        if self._fails:
            raise MergedModelError("weight checksum mismatch")
        self.current = {"id": entry_id, "condition_id": "ties_only",
                        "run_id": "run-1"}
        return self.current


class RejectionCommandTests(unittest.TestCase):
    def test_the_command_is_recognised_with_its_argument(self):
        with patch("builtins.input", side_effect=[":rejection use ties"]):
            self.assertEqual(read_interactive_request(), (":rejection", "use ties"))

    def test_bare_command_reports_the_current_selection(self):
        with patch("builtins.input", side_effect=[":rejection"]):
            self.assertEqual(read_interactive_request(), (":rejection", ""))

    def test_use_switches_the_engine(self):
        engine = FakeSelectableEngine()

        handle_rejection_command(engine, "use ties")

        self.assertEqual(engine.selected, ["ties"])
        self.assertEqual(engine.current_rejection()["id"], "ties")

    def test_a_failed_switch_keeps_the_current_selection(self):
        engine = FakeSelectableEngine(fails=True)

        handle_rejection_command(engine, "use broken")

        self.assertEqual(engine.current_rejection()["id"], "base")

    def test_listing_does_not_switch_anything(self):
        engine = FakeSelectableEngine()

        handle_rejection_command(engine, "list")

        self.assertEqual(engine.selected, [])


if __name__ == "__main__":
    unittest.main()
