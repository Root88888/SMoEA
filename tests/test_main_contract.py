import unittest

from main import external_task_key, generation_prompt


class MainContractTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
