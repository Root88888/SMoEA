import json
import tempfile
import unittest
from pathlib import Path

from scripts.eval_outputs_llm_judge import build_items, summarize


class EvalOutputsJudgeTest(unittest.TestCase):
    def test_current_rejection_metadata_is_preserved_and_evaluated(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            test_dir = root / "test_data"
            ood_dir = root / "ood_test_data"
            test_dir.mkdir()
            ood_dir.mkdir()
            (test_dir / "task0_test.json").write_text(json.dumps({
                "instances": [{
                    "instance_id": "task0-1",
                    "full_prompt": "Classify this request",
                    "output": "yes",
                }],
            }), encoding="utf-8")
            (ood_dir / "task149_test.json").write_text(json.dumps({
                "instances": [{
                    "instance_id": "ood-149-1",
                    "full_prompt": "Answer this OOD request",
                    "output": "reference",
                }],
            }), encoding="utf-8")
            batch = root / "batch.jsonl"
            rows = [
                {
                    "source_task": "task0",
                    "instance_id": "task0-1",
                    "routed_to": "task0",
                    "output": "yes",
                },
                {
                    "source_task": "task149",
                    "internal_task_id": "task9149",
                    "instance_id": "ood-149-1",
                    "routed_to": None,
                    "model_source": "rejection",
                    "rejection_method": "arrow",
                    "rejection_condition_id": "direct-arrow",
                    "rejection_run_id": "run-1",
                    "output": "prediction",
                },
            ]
            batch.write_text("".join(json.dumps(r) + "\n" for r in rows),
                             encoding="utf-8")

            items, skipped = build_items(
                batch, test_dir, None, None, ood_dir)

        self.assertEqual(len(items), 2)
        self.assertEqual(skipped, [])
        rejected = next(x for x in items if x["path"] != "routed")
        self.assertEqual(rejected["path"], "rejected_arrow")
        self.assertEqual(rejected["rejection_condition_id"], "direct-arrow")
        self.assertEqual(rejected["target"], "reference")

        judged = [
            {**item, "judge": {"score": 5, "is_correct": True}, "error": None}
            for item in items
        ]
        summary = summarize(judged, skipped)
        self.assertIn("rejected_arrow", summary["per_path"])


if __name__ == "__main__":
    unittest.main()
