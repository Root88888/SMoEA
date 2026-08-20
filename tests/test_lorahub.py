"""LoRAHub 中不需要 GPU 就能驗的部分：候選選取、權重合成、設定守衛、身分指紋。"""

import json
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file

from system.lorahub import (
    SEALED_LORAHUB,
    LoRAHubError,
    build_causal_lm_batch,
    compose_state_dict,
    examples_fingerprint,
    paper_l1_regularization,
    select_candidate_adapters,
    validate_paper_settings,
)


class CandidateSelectionTests(unittest.TestCase):
    def setUp(self):
        self.names = [f"task{i}" for i in range(50)]

    def test_selection_is_reproducible_from_its_seed(self):
        first = select_candidate_adapters(self.names, 20, 42)
        second = select_candidate_adapters(self.names, 20, 42)
        other = select_candidate_adapters(self.names, 20, 43)

        self.assertEqual(first, second)
        self.assertNotEqual(first, other)
        self.assertEqual(len(set(first)), 20, "不得重複選到同一個 adapter")

    def test_asking_for_more_than_the_pool_holds_is_refused(self):
        with self.assertRaisesRegex(LoRAHubError, "只有"):
            select_candidate_adapters(self.names, 100, 42)


class CompositionTests(unittest.TestCase):
    def setUp(self):
        self.cache = {
            "a": {"w": torch.tensor([1.0, 2.0])},
            "b": {"w": torch.tensor([10.0, 20.0])},
        }

    def test_composition_is_the_weighted_sum(self):
        merged = compose_state_dict([0.5, 2.0], ["a", "b"], self.cache)

        torch.testing.assert_close(merged["w"], torch.tensor([20.5, 41.0]))

    def test_the_source_cache_is_not_mutated(self):
        compose_state_dict([0.5, 2.0], ["a", "b"], self.cache)

        torch.testing.assert_close(self.cache["a"]["w"], torch.tensor([1.0, 2.0]))
        torch.testing.assert_close(self.cache["b"]["w"], torch.tensor([10.0, 20.0]))

    def test_mismatched_lengths_are_refused(self):
        with self.assertRaises(LoRAHubError):
            compose_state_dict([1.0], ["a", "b"], self.cache)

    def test_a_state_dict_with_different_keys_is_refused(self):
        self.cache["b"] = {"other": torch.tensor([1.0, 1.0])}

        with self.assertRaisesRegex(LoRAHubError, "鍵"):
            compose_state_dict([1.0, 1.0], ["a", "b"], self.cache)


class PaperSettingsTests(unittest.TestCase):
    def test_the_sealed_settings_match_the_paper(self):
        policy = validate_paper_settings(dict(SEALED_LORAHUB))

        self.assertEqual(policy["candidate_count"], 20)
        self.assertEqual(policy["generations"], 40)
        self.assertEqual(policy["population_size"], 12)
        self.assertEqual(policy["objective_evaluations_per_run"], 480)

    def test_a_different_dimension_is_refused(self):
        settings = dict(SEALED_LORAHUB, candidate_count=10)

        with self.assertRaisesRegex(LoRAHubError, "N=20"):
            validate_paper_settings(settings)

    def test_a_population_that_does_not_match_the_dimension_is_refused(self):
        """族群數由維度決定；改了就不是同一個搜尋。"""
        settings = dict(SEALED_LORAHUB, population_size=8)

        with self.assertRaisesRegex(LoRAHubError, "族群數"):
            validate_paper_settings(settings)

    def test_a_generation_count_that_contradicts_the_evaluation_budget_is_refused(self):
        settings = dict(SEALED_LORAHUB, generations=40, population_size=12,
                        expected_objective_evaluations_per_run=100)

        with self.assertRaisesRegex(LoRAHubError, "評估次數"):
            validate_paper_settings(settings)

    def test_the_l1_term_is_the_sum_of_absolute_weights(self):
        self.assertAlmostEqual(
            paper_l1_regularization([1.0, -2.0, 0.5], 0.05), 0.175)


class FingerprintTests(unittest.TestCase):
    """權重只對它擬合的那組樣本有意義，指紋是身分的一部分。"""

    EXAMPLES = [{"instance_id": "a", "prompt": "p1", "output": "o1"},
                {"instance_id": "b", "prompt": "p2", "output": "o2"}]

    def test_the_same_examples_give_the_same_fingerprint(self):
        self.assertEqual(examples_fingerprint(self.EXAMPLES),
                         examples_fingerprint(list(self.EXAMPLES)))

    def test_changing_one_example_changes_the_fingerprint(self):
        altered = [dict(self.EXAMPLES[0]), dict(self.EXAMPLES[1], output="o3")]

        self.assertNotEqual(examples_fingerprint(self.EXAMPLES),
                            examples_fingerprint(altered))


class FakeTokenizer:
    eos_token_id = 2
    pad_token_id = 0

    def __call__(self, text, add_special_tokens=True, truncation=False):
        return {"input_ids": [ord(character) % 50 + 3 for character in text]}


class BatchTests(unittest.TestCase):
    def test_the_prompt_is_masked_and_the_target_is_kept(self):
        examples = [{"prompt": "abc", "output": "de", "instance_id": "x"}]

        batch, truncations = build_causal_lm_batch(
            FakeTokenizer(), examples, 100, torch.device("cpu"))

        labels = batch["labels"][0].tolist()
        self.assertEqual(labels[:3], [-100, -100, -100], "prompt 必須被遮罩")
        self.assertNotIn(-100, labels[3:], "目標與 eos 必須參與計分")
        self.assertFalse(truncations[0]["truncated"])

    def test_an_overlong_prompt_is_cut_from_the_left_and_flagged(self):
        examples = [{"prompt": "a" * 50, "output": "de", "instance_id": "x"}]

        batch, truncations = build_causal_lm_batch(
            FakeTokenizer(), examples, 10, torch.device("cpu"))

        self.assertTrue(truncations[0]["truncated"])
        self.assertEqual(batch["input_ids"].shape[1], 10)
        # 目標仍在，只有 prompt 被砍
        self.assertEqual(int((batch["labels"][0] != -100).sum()), 3)


class PeftArtifactTests(unittest.TestCase):
    """LoRAHub 的產物是 PEFT adapter，要能被 SMoEA 的契約打包與讀回。"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def test_a_written_peft_adapter_loads_back_through_the_validator(self):
        from system.merged_model import (
            load_merged_model_artifact,
            write_peft_adapter_artifact,
        )

        source = self.root / "adapter"
        source.mkdir()
        tensors = {}
        for layer in range(2):
            stem = f"base_model.model.model.layers.{layer}.mlp.down_proj"
            tensors[f"{stem}.lora_A.weight"] = torch.randn(2, 6, dtype=torch.bfloat16)
            tensors[f"{stem}.lora_B.weight"] = torch.randn(8, 2, dtype=torch.bfloat16)
        save_file(tensors, str(source / "adapter_model.safetensors"))
        (source / "adapter_config.json").write_text("{}", encoding="utf-8")

        artifact = write_peft_adapter_artifact(
            self.root / "merged_model", source=source, condition_id="lorahub",
            run_id="run-1", base_model_name="base", base_model_revision="local",
            base_model_config_sha256="a" * 64, torch_dtype="bfloat16")

        reloaded = load_merged_model_artifact(self.root / "merged_model")
        self.assertEqual(reloaded.format, "peft_adapter_v1")
        self.assertEqual(reloaded.condition_id, "lorahub")
        self.assertEqual(len(reloaded.modules), 4)
        self.assertEqual(artifact.manifest["expected_module_count"], 2)

    def test_an_unpaired_lora_tensor_is_refused(self):
        from system.merged_model import write_peft_adapter_artifact

        source = self.root / "adapter"
        source.mkdir()
        save_file({"m.lora_A.weight": torch.randn(2, 6, dtype=torch.bfloat16)},
                  str(source / "adapter_model.safetensors"))
        (source / "adapter_config.json").write_text("{}", encoding="utf-8")

        with self.assertRaisesRegex(Exception, "成對"):
            write_peft_adapter_artifact(
                self.root / "merged_model", source=source, condition_id="lorahub",
                run_id="r", base_model_name="base", base_model_revision="local",
                base_model_config_sha256="a" * 64, torch_dtype="bfloat16")


if __name__ == "__main__":
    unittest.main()
