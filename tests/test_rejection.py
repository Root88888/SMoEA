import unittest

from system.rejection import run_rejection


class FakeEngine:
    def __init__(self):
        self.calls = []

    def ensure_rejection(self):
        self.calls.append("rejection")
        return {"method": "artifact", "condition_id": "adamerging_pp",
                "run_id": "run-123", "format": "dense_delta_v1"}

    def generate(self, prompts):
        self.calls.append(("generate", list(prompts)))
        return [f"answer:{p}" for p in prompts]


class RejectionTests(unittest.TestCase):
    def test_rejection_uses_the_selected_method_and_reports_its_identity(self):
        engine = FakeEngine()

        outputs, identity = run_rejection(engine, ["complete prompt"])

        self.assertEqual(outputs, ["answer:complete prompt"])
        self.assertEqual(identity["condition_id"], "adamerging_pp")
        self.assertEqual(identity["run_id"], "run-123")
        self.assertEqual(
            engine.calls,
            ["rejection", ("generate", ["complete prompt"])],
        )

    def test_batching_preserves_prompt_order(self):
        engine = FakeEngine()
        prompts = [f"p{i}" for i in range(5)]

        outputs, _ = run_rejection(engine, prompts, batch_size=2)

        self.assertEqual(outputs, [f"answer:p{i}" for i in range(5)])
        self.assertEqual(
            [call[1] for call in engine.calls if call[0] == "generate"],
            [["p0", "p1"], ["p2", "p3"], ["p4"]],
        )

    def test_the_rejection_method_is_activated_once_per_run(self):
        engine = FakeEngine()

        run_rejection(engine, ["a", "b", "c"], batch_size=1)

        self.assertEqual(engine.calls.count("rejection"), 1)


if __name__ == "__main__":
    unittest.main()
