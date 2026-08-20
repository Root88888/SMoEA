"""線上 merge（ADR-0002）：pool 載入、三種方法、以及 writer/validator 的往返一致。"""

import json
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file

from system.adapter_pool import (
    AdapterPoolError,
    load_adapter_pool,
    manifest_from_adapter_dir,
)
from system.merged_model import (
    load_merged_model_artifact,
    write_dense_delta_artifact,
)
from system.merging import CANONICAL_SEED, SEALED, merge_pool

RANK, OUT_FEATURES, IN_FEATURES, LAYERS = 2, 8, 6, 2
DIGEST = "a" * 64


def lora_key(layer, kind):
    return (f"base_model.model.model.layers.{layer}.mlp.down_proj"
            f".lora_{kind}.weight")


class SyntheticPool:
    """一個微型但結構真實的 adapter 池（CPU、無需 GPU）。"""

    def __init__(self, root: Path, count=3, seed=0):
        self.root = root
        self.count = count
        generator = torch.Generator().manual_seed(seed)
        entries = []
        for index in range(count):
            directory = root / f"task{index}"
            directory.mkdir(parents=True, exist_ok=True)
            tensors = {}
            for layer in range(LAYERS):
                tensors[lora_key(layer, "A")] = torch.randn(
                    RANK, IN_FEATURES, generator=generator)
                tensors[lora_key(layer, "B")] = torch.randn(
                    OUT_FEATURES, RANK, generator=generator)
            save_file(tensors, str(directory / "adapter_model.safetensors"))
            (directory / "adapter_config.json").write_text(json.dumps({
                "base_model_name_or_path": "base", "r": RANK, "lora_alpha": 4,
                "target_modules": ["down_proj"], "bias": "none",
            }), encoding="utf-8")
            entries.append({"name": f"task{index}", "path": str(directory)})
        self.manifest = root / "manifest.json"
        self.manifest.write_text(json.dumps(
            {"schema_version": 1, "pool_id": "synthetic", "count": count,
             "adapters": entries}), encoding="utf-8")

    def load(self):
        return load_adapter_pool(self.manifest, expected_adapters=self.count)


class MergingTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self.pool = SyntheticPool(self.root / "pool").load()

    def test_task_arithmetic_averages_rather_than_sums(self):
        """封板的 ta 是 reduction=mean：係數為 lambda / 任務數，不是 lambda。

        直接用 lambda 會讓輸出放大 150 倍——producer 在呼叫端做這個換算，
        搬過來時漏掉就會靜默地產生完全不同的權重。
        """
        out = self.root / "ta.safetensors"
        merge_pool("ta", self.pool, out, output_dtype=torch.float32)

        from safetensors.torch import load_file
        produced = load_file(str(out))
        for layer in range(LAYERS):
            total = sum(
                (weights[lora_key(layer, "B")] @ weights[lora_key(layer, "A")]
                 for weights in self.pool.weights),
                torch.zeros(OUT_FEATURES, IN_FEATURES),
            ) * self.pool.scaling
            expected = total * (SEALED["ta"]["lambda"] / len(self.pool.names))
            torch.testing.assert_close(
                produced[f"model.layers.{layer}.mlp.down_proj.delta_weight"],
                expected, rtol=1e-5, atol=1e-5)

    def test_every_method_covers_each_layer_once(self):
        # lora_lego 的 output_rank 是 16，需要至少 16 個 MSU（任務數 × rank）。
        big = SyntheticPool(self.root / "big", count=10, seed=5).load()
        for method in SEALED:
            with self.subTest(method=method):
                out = self.root / f"{method}.safetensors"
                report = merge_pool(method, big, out)
                self.assertEqual(report["module_count"], LAYERS)
                self.assertEqual(report["status"], "complete")

    def test_lego_refuses_a_pool_with_too_few_msus(self):
        """output_rank 超過 MSU 總數時必須明講，而不是產出無意義的分群。"""
        with self.assertRaisesRegex(ValueError, "output_rank"):
            merge_pool("lora_lego", self.pool, self.root / "x.safetensors")

    def test_pico_and_lego_are_deterministic(self):
        big = SyntheticPool(self.root / "det", count=10, seed=6).load()
        for method in ("pico_ta", "lora_lego"):
            with self.subTest(method=method):
                first = merge_pool(method, big, self.root / f"{method}-1.st")
                second = merge_pool(method, big, self.root / f"{method}-2.st")
                self.assertEqual(first["output_sha256"], second["output_sha256"])

    def test_merging_the_same_pool_twice_is_bit_identical(self):
        first = merge_pool("dare_ties_ta", self.pool, self.root / "d1.safetensors")
        second = merge_pool("dare_ties_ta", self.pool, self.root / "d2.safetensors")

        self.assertEqual(first["output_sha256"], second["output_sha256"])

    def test_ties_keeps_only_the_agreed_signs(self):
        out = self.root / "ties.safetensors"
        report = merge_pool("ties_only", self.pool, out)

        selected = sum(item["selected_coordinates"]
                       for item in report["module_reports"])
        trimmed = sum(item["trimmed_coordinates"]
                      for item in report["module_reports"])
        self.assertGreater(trimmed, 0)
        self.assertLessEqual(selected, trimmed)

    def test_an_unknown_method_is_refused(self):
        with self.assertRaisesRegex(ValueError, "adamerging|不支援"):
            merge_pool("adamerging_pp", self.pool, self.root / "x.safetensors")


class PoolFingerprintTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def test_changing_one_adapter_changes_the_pool_fingerprint(self):
        first = SyntheticPool(self.root / "a", seed=1).load().fingerprint()
        second = SyntheticPool(self.root / "b", seed=2).load().fingerprint()

        self.assertEqual(first["adapter_ids"], second["adapter_ids"])
        self.assertNotEqual(first["adapters_sha256"], second["adapters_sha256"])

    def test_a_pool_of_the_wrong_size_is_refused(self):
        pool = SyntheticPool(self.root / "c", count=3)

        with self.assertRaisesRegex(AdapterPoolError, "150"):
            load_adapter_pool(pool.manifest)


class ArtifactRoundTripTests(unittest.TestCase):
    """writer 與 validator 必須同意同一份契約（ADR-0002 的漂移防線）。"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self.pool = SyntheticPool(self.root / "pool").load()

    def _write(self, **overrides):
        delta = self.root / "delta.safetensors"
        merge_pool("ta", self.pool, delta)
        options = dict(
            source=delta, condition_id="ta", run_id="run-1",
            base_model_name="base", base_model_revision="local",
            base_model_config_sha256=DIGEST, torch_dtype="bfloat16",
            adapter_pool=self.pool.fingerprint(),
        )
        options.update(overrides)
        return write_dense_delta_artifact(self.root / "merged_model", **options)

    def test_a_written_artifact_loads_back_through_the_validator(self):
        artifact = self._write()

        reloaded = load_merged_model_artifact(self.root / "merged_model")
        self.assertEqual(reloaded.condition_id, "ta")
        self.assertEqual(reloaded.run_id, "run-1")
        self.assertEqual(reloaded.torch_dtype, "bfloat16")
        self.assertEqual(len(reloaded.modules), LAYERS)
        self.assertEqual(artifact.manifest["expected_module_count"], LAYERS)

    def test_module_and_tensor_names_are_written_separately(self):
        self._write()

        payload = json.loads(
            (self.root / "merged_model" / "result.json").read_text(encoding="utf-8"))
        module = payload["modules"][0]
        self.assertEqual(module["tensor_name"], module["name"] + ".delta_weight")
        self.assertNotIn(".delta_weight", module["name"])

    def test_the_pool_fingerprint_is_recorded_in_the_artifact(self):
        self._write()

        payload = json.loads(
            (self.root / "merged_model" / "result.json").read_text(encoding="utf-8"))
        self.assertEqual(payload["adapter_pool"]["adapter_count"], 3)
        self.assertIn("adapters_sha256", payload["adapter_pool"])

    def test_a_dtype_that_contradicts_the_weights_is_refused(self):
        with self.assertRaisesRegex(Exception, "dtype"):
            self._write(torch_dtype="float32")

    def test_a_malformed_base_config_digest_is_refused(self):
        with self.assertRaisesRegex(Exception, "SHA-256"):
            self._write(base_model_config_sha256="short")


class ManifestFromDirectoryTests(unittest.TestCase):
    """交付時公司方可能只有 adapter/task{N}/，沒有 manifest。"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def _adapter(self, name, *, checkpoints=()):
        directory = self.root / name
        directory.mkdir(parents=True)
        targets = [directory / item for item in checkpoints] or [directory]
        for target in targets:
            target.mkdir(exist_ok=True)
            (target / "adapter_config.json").write_text("{}", encoding="utf-8")

    def test_tasks_are_ordered_by_number_not_by_name(self):
        for name in ("task0", "task2", "task10"):
            self._adapter(name)

        manifest = manifest_from_adapter_dir(self.root)

        # 字串排序會給出 task0, task10, task2——順序會改變 merge 結果。
        self.assertEqual([item["name"] for item in manifest["adapters"]],
                         ["task0", "task2", "task10"])

    def test_the_largest_checkpoint_is_chosen(self):
        self._adapter("task0", checkpoints=("checkpoint-9", "checkpoint-100"))

        manifest = manifest_from_adapter_dir(self.root)

        self.assertTrue(manifest["adapters"][0]["path"].endswith("checkpoint-100"))

    def test_an_empty_directory_is_refused(self):
        with self.assertRaisesRegex(AdapterPoolError, "task"):
            manifest_from_adapter_dir(self.root)


class SealedHyperparameterTests(unittest.TestCase):
    """封板超參數必須與 producer 的 conditions.py 逐項相同。

    這些值決定產物的每一個位元。搬移時最容易漏的不是公式本身，而是
    **呼叫端的換算**——例如 ta 的 reduction=mean 要把 lambda 除以任務數。
    """

    PRODUCER_VALUES = {
        "ta": {"lambda": 1.0, "reduction": "mean"},
        "ties_only": {"lambda": 0.3, "density": 0.2, "reduction": "disjoint_mean",
                      "tile_rows": 4},
        "dare_ties_ta": {"density": 1.0, "lambda": 0.25, "sign_method": "total",
                         "rescale": True, "tile_rows": 16, "mask_block_rows": 8},
        "pico_ta": {"reduction": "mean", "eps": 1e-12},
        "lora_lego": {"output_rank": 16, "output_ref_rank": 8, "lego_seed": 0,
                      "n_init": 10, "max_iter": 300, "parameter_reweight": True,
                      "output_reweight": True, "eps": 1e-12},
    }

    def test_sealed_values_match_the_producer(self):
        for method, expected in self.PRODUCER_VALUES.items():
            with self.subTest(method=method):
                for key, value in expected.items():
                    self.assertEqual(SEALED[method][key], value,
                                     f"{method}.{key} 與封板值不符")

    def test_the_canonical_seed_is_pinned(self):
        self.assertEqual(CANONICAL_SEED, 42)

    def test_task_arithmetic_mean_divides_by_the_pool_size(self):
        """換算漏掉時輸出會放大 len(pool) 倍——這裡直接盯住比例。"""
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        small = SyntheticPool(root / "small", count=2, seed=3).load()
        large = SyntheticPool(root / "large", count=2, seed=3).load()
        # 同樣的權重、把池複製成兩倍大小
        large.names = large.names * 2
        large.weights = large.weights * 2
        large.source_files = large.source_files * 2

        from safetensors.torch import load_file
        merge_pool("ta", small, root / "s.safetensors", output_dtype=torch.float32)
        merge_pool("ta", large, root / "l.safetensors", output_dtype=torch.float32)
        key = "model.layers.0.mlp.down_proj.delta_weight"
        first = load_file(str(root / "s.safetensors"))[key]
        second = load_file(str(root / "l.safetensors"))[key]

        # 池變兩倍、內容相同 → mean 之後應該不變（sum 的話會變兩倍）
        torch.testing.assert_close(first, second, rtol=1e-5, atol=1e-5)


class ManifestMigrationTests(unittest.TestCase):
    """補寫舊 manifest 時的 dtype 命名必須與 torch 一致。"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def test_dtype_is_written_in_torch_naming_not_safetensors(self):
        """safetensors 叫 BF16、torch 叫 bfloat16——混用會讓載入時比對失敗。"""
        import subprocess
        import sys as _sys

        pool = SyntheticPool(self.root / "pool").load()
        directory = self.root / "merged_model"
        delta = self.root / "delta.safetensors"
        merge_pool("ta", pool, delta)
        write_dense_delta_artifact(
            directory, source=delta, condition_id="ta", run_id="r",
            base_model_name="base", base_model_revision="local",
            base_model_config_sha256=DIGEST, torch_dtype="bfloat16")

        # 還原成舊格式：拿掉 inference、把 tensor_name 併回 name
        payload = json.loads((directory / "result.json").read_text(encoding="utf-8"))
        payload.pop("inference")
        payload["modules"] = [
            {"name": m["tensor_name"], "shape": m["shape"], "dtype": m["dtype"]}
            for m in payload["modules"]
        ]
        (directory / "result.json").write_text(
            json.dumps(payload), encoding="utf-8")

        subprocess.run(
            [_sys.executable, "scripts/migrate_artifact_manifest.py", str(directory)],
            check=True, capture_output=True)

        migrated = json.loads((directory / "result.json").read_text(encoding="utf-8"))
        self.assertEqual(migrated["inference"]["torch_dtype"], "bfloat16")
        self.assertNotEqual(migrated["inference"]["torch_dtype"], "bf16")
        # 補完後必須能通過正式驗證器
        artifact = load_merged_model_artifact(directory)
        self.assertEqual(artifact.torch_dtype, artifact.modules[0].dtype)


if __name__ == "__main__":
    unittest.main()
