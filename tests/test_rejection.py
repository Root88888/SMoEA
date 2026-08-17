import unittest

from system.rejection import handle_rejection


class FakeEngine:
    def __init__(self):
        self.calls = []

    def ensure_merged(self):
        self.calls.append("merged")
        return {"condition_id": "adamerging_pp", "run_id": "run-123"}

    def generate(self, prompts):
        self.calls.append(("generate", prompts))
        return ["merged answer"]


class RejectionTests(unittest.TestCase):
    def test_rejection_uses_the_selected_merged_model(self):
        engine = FakeEngine()

        output = handle_rejection("complete prompt", engine)

        self.assertEqual(output, "merged answer")
        self.assertEqual(
            engine.calls,
            ["merged", ("generate", ["complete prompt"])],
        )


if __name__ == "__main__":
    unittest.main()
