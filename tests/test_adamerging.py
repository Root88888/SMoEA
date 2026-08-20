"""AdaMerging++ 中不需要 GPU 就能驗的部分：資料切分、抽樣、編碼、防外洩。"""

import json
import tempfile
import unittest
from pathlib import Path

from system.adamerging import (
    SEALED_ADAMERGING,
    build_source_split,
    length_batches,
    sample_calibration,
    truncate_head_tail,
)


def write_task(root: Path, task: str, count: int, *, leaky=False):
    instances = []
    for index in range(count):
        target = f"answer-{index}"
        prompt = f"instruction for {task}\ninput {index}\n"
        # 訓練樣本的 full_prompt 尾端應該精確等於 output
        instances.append({
            "instance_id": f"{task}-{index}",
            "input": f"input {index}",
            "full_prompt": prompt if leaky else prompt + target,
            "output": target,
        })
    (root / f"{task}_train.json").write_text(
        json.dumps({"task_key": task, "instances": instances}), encoding="utf-8")


class SourceSplitTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        for task in ("task0", "task1", "task2"):
            write_task(self.root, task, 20)

    def _split(self, **overrides):
        options = dict(calibration_per_task=5, validation_per_task=3, seed=42,
                       file_pattern="{task}_train.json",
                       source_prompts_include_target=True)
        options.update(overrides)
        return build_source_split(
            self.root, ["task0", "task1", "task2"], **options)

    def test_calibration_and_validation_never_overlap(self):
        calibration, validation, _ = self._split()

        calibration_ids = {item["instance_id"] for item in calibration}
        validation_ids = {item["instance_id"] for item in validation}
        self.assertEqual(len(calibration), 15)
        self.assertEqual(len(validation), 9)
        self.assertEqual(calibration_ids & validation_ids, set())

    def test_the_target_is_stripped_from_the_calibration_prompt(self):
        """目標函數只看 prompt，答案必須先移除，否則等於偷看標準答案。"""
        calibration, _validation, _ = self._split()

        for item in calibration:
            self.assertNotIn(item["target"], item["prompt"])

    def test_a_prompt_that_does_not_end_with_its_target_is_refused(self):
        write_task(self.root, "task0", 20, leaky=True)

        with self.assertRaisesRegex(ValueError, "尾端"):
            self._split()

    def test_the_split_is_reproducible_from_its_seed(self):
        first = self._split()[2]["selection_sha256"]
        second = self._split()[2]["selection_sha256"]
        third = self._split(seed=7)[2]["selection_sha256"]

        self.assertEqual(first, second)
        self.assertNotEqual(first, third)

    def test_a_task_with_too_few_samples_is_refused(self):
        write_task(self.root, "task0", 4)

        with self.assertRaisesRegex(ValueError, "不足"):
            self._split()


class SamplingTests(unittest.TestCase):
    def setUp(self):
        self.records = [
            {"task": f"task{t}", "instance_id": f"task{t}-{i}",
             "input_ids": list(range(i + 1))}
            for t in range(4) for i in range(10)
        ]

    def test_sampling_is_reproducible_and_balanced(self):
        first, tasks = sample_calibration(
            self.records, tasks_per_iteration=2, examples_per_task=3,
            seed=42, iteration=1)
        second, _ = sample_calibration(
            self.records, tasks_per_iteration=2, examples_per_task=3,
            seed=42, iteration=1)

        self.assertEqual(len(first), 6)
        self.assertEqual(len(tasks), 2)
        self.assertEqual([item["instance_id"] for item in first],
                         [item["instance_id"] for item in second])

    def test_different_iterations_draw_different_tasks_over_a_cycle(self):
        seen = set()
        for iteration in range(1, 3):
            _selected, tasks = sample_calibration(
                self.records, tasks_per_iteration=2, examples_per_task=3,
                seed=42, iteration=iteration)
            seen.update(tasks)

        self.assertGreater(len(seen), 2, "整個週期應該涵蓋超過一批任務")

    def test_tasks_per_iteration_must_divide_the_task_count(self):
        with self.assertRaisesRegex(ValueError, "整除"):
            sample_calibration(self.records, tasks_per_iteration=3,
                               examples_per_task=1, seed=42, iteration=1)

    def test_batches_are_ordered_by_length_deterministically(self):
        batches = list(length_batches(self.records, 5))

        flat = [item for batch in batches for item in batch]
        lengths = [len(item["input_ids"]) for item in flat]
        self.assertEqual(lengths, sorted(lengths))


class TruncationTests(unittest.TestCase):
    def test_short_sequences_are_left_alone(self):
        self.assertEqual(truncate_head_tail([1, 2, 3], 5), ([1, 2, 3], False))

    def test_long_sequences_keep_both_ends(self):
        kept, truncated = truncate_head_tail(list(range(10)), 4)

        self.assertTrue(truncated)
        self.assertEqual(kept, [0, 1, 8, 9])

    def test_a_non_positive_budget_is_refused(self):
        with self.assertRaises(ValueError):
            truncate_head_tail([1, 2, 3], 0)


class SealedValueTests(unittest.TestCase):
    PRODUCER = {
        "lambda_init": 0.3, "lr": 0.001, "max_iterations": 500,
        "micro_batch_size": 1, "max_length": 8192, "gradient_clip_norm": 1.0,
        "tasks_per_iteration": 150, "examples_per_selected_task": 32,
        "calibration_per_task": 50, "validation_per_task": 32,
        "coefficient_min": 0.0, "coefficient_max": 1.0,
        "coefficient_parameterization": "projected_nonnegative",
        "optimization_model": "unsloth/Meta-Llama-3.1-8B-bnb-4bit",
    }

    def test_sealed_values_match_the_producer(self):
        for key, value in self.PRODUCER.items():
            with self.subTest(key=key):
                self.assertEqual(SEALED_ADAMERGING[key], value)

    def test_ties_preprocessing_uses_the_documented_density(self):
        self.assertEqual(SEALED_ADAMERGING["ties"]["density"], 0.2)
        self.assertEqual(SEALED_ADAMERGING["ties"]["tile_rows"], 16)

    def test_the_objective_never_sees_target_labels(self):
        """封板設定的目標函數是 prompt 上的 token 熵，不含標準答案。"""
        self.assertTrue(SEALED_ADAMERGING["source_prompts_include_target"],
                        "訓練樣本含答案，所以才需要在切分時移除")


if __name__ == "__main__":
    unittest.main()
