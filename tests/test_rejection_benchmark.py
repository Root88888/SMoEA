import json
import tempfile
import unittest
from pathlib import Path

from system.benchmark import (
    BBH_DATASETS,
    MMLU_PRO_DATASETS,
    NI_DATASETS,
    run_rejection_benchmark,
)


def write_benchmark(root: Path) -> None:
    prompts = root / "prompts"
    prompts.mkdir(parents=True)
    (prompts / "ood_ni_tasks.json").write_text(
        json.dumps({name: {"description": name} for name in NI_DATASETS}),
        encoding="utf-8",
    )
    ni = root / "dataset/natural_instructions/data/selected_10_tasks/test_data"
    ni.mkdir(parents=True)
    for name in NI_DATASETS:
        target = "Valid" if name == "task149" else "answer"
        (ni / f"{name}_test.json").write_text(
            json.dumps(
                {
                    "definition": name,
                    "instances": [
                        {
                            "instance_id": f"ni::{name}::0",
                            "input": f"input-{name}",
                            "full_prompt": f"FULL NI PROMPT {name}\nAnswer:",
                            "output": target,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
    ood = root / "ood/data"
    ood.mkdir(parents=True)
    for filename, benchmark, datasets, target in (
        ("bbh_test.json", "bbh", BBH_DATASETS, "Yes"),
        ("mmlu_pro_test.json", "mmlu_pro", MMLU_PRO_DATASETS, "A"),
    ):
        (ood / filename).write_text(
            json.dumps(
                {
                    "instances": [
                        {
                            "instance_id": f"{benchmark}::{name}::0",
                            "full_prompt": f"FULL {benchmark} PROMPT {name}\nAnswer:",
                            "output": target,
                        }
                        for name in datasets
                    ]
                }
            ),
            encoding="utf-8",
        )


class FakeEngine:
    def __init__(self):
        self.prompts = []
        self.generation_overrides = []

    def ensure_rejection(self):
        return {
            "method": "base",
            "condition_id": "base",
            "run_id": None,
            "format": "base_model",
        }

    def generate(self, prompts, generation_overrides=None):
        self.prompts.extend(prompts)
        self.generation_overrides.append(generation_overrides)
        answers = []
        for prompt in prompts:
            if "FULL NI" in prompt:
                answers.append("Valid")
            elif "FULL bbh" in prompt:
                answers.append("The answer is Yes.")
            else:
                answers.append("Answer: A")
        return answers


class RejectionBenchmarkTests(unittest.TestCase):
    def test_smoke_uses_full_prompts_and_same_rejection_runtime(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_benchmark(root / "benchmark")
            engine = FakeEngine()

            report = run_rejection_benchmark(
                engine,
                benchmark_root=root / "benchmark",
                output_dir=root / "results",
                batch_size=2,
                smoke=True,
            )

            saved = json.loads(
                (root / "results/metrics.json").read_text(encoding="utf-8")
            )

        self.assertEqual(report["n"], 3)
        self.assertEqual(report["metrics"]["classification"]["micro_accuracy"], 1.0)
        self.assertEqual(saved["rejection"]["condition_id"], "base")
        self.assertEqual(len(engine.prompts), 3)
        self.assertTrue(all(prompt.startswith("FULL ") for prompt in engine.prompts))
        self.assertTrue(
            all(
                policy["max_input_tokens"] == 8192
                and policy["max_new_tokens"] == 1024
                and policy["reject_prompt_truncation"]
                for policy in engine.generation_overrides
            )
        )


if __name__ == "__main__":
    unittest.main()
