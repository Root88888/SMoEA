"""遠端新增版本時，fetch 不得靜悄悄換掉正在用的權重。"""

import importlib.util
import pathlib
import unittest

_SPEC = importlib.util.spec_from_file_location(
    "fetch_artifact",
    pathlib.Path(__file__).resolve().parent.parent / "scripts" / "fetch_artifact.py",
)
fetch_artifact = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(fetch_artifact)


class RunSelectionTests(unittest.TestCase):
    def test_a_single_remote_run_is_used_without_asking(self):
        self.assertEqual(fetch_artifact.choose_run_id(["abc123"]), "abc123")

    def test_multiple_remote_runs_require_an_explicit_choice(self):
        """上傳新權重後遠端會有兩個版本——此時必須由人決定用哪一個。"""
        with self.assertRaises(SystemExit) as caught:
            fetch_artifact.choose_run_id(["old111", "new222"])

        message = str(caught.exception)
        self.assertIn("old111", message)
        self.assertIn("new222", message)
        self.assertIn("--run-id", message)

    def test_an_explicit_choice_is_honoured(self):
        self.assertEqual(
            fetch_artifact.choose_run_id(["old111", "new222"], "new222"), "new222")

    def test_an_unknown_choice_lists_what_exists(self):
        with self.assertRaises(SystemExit) as caught:
            fetch_artifact.choose_run_id(["old111", "new222"], "typo")

        self.assertIn("old111", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
