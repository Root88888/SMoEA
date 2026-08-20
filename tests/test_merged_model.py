import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from system.merged_model import (
    MergedModelError,
    load_merged_model_artifact,
    validate_base_model_config,
    validate_inference_config,
)


class MergedModelArtifactTests(unittest.TestCase):
    def _write_peft_artifact(self, root):
        config = root / "adapter_config.json"
        model = root / "adapter_model.safetensors"
        config.write_text('{"peft_type":"LORA"}', encoding="utf-8")
        model.write_bytes(b"tiny-peft-adapter")
        weights = []
        for path in (config, model):
            weights.append(
                {
                    "path": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
        payload = {
            "schema_version": 1,
            "format": "peft_adapter_v1",
            "condition_id": "task_arithmetic",
            "run_id": "run-peft",
            "base_model": {
                "name": "base",
                "revision": "revision",
                "config_sha256": "c" * 64,
            },
            "inference": {"torch_dtype": "bfloat16", "quantization": "none"},
            "weights": weights,
            "expected_module_count": 1,
            "expected_tensor_count": 2,
            "modules": [
                {
                    "name": "model.layers.0.mlp.down_proj",
                    "tensor_name": "base_model.model.model.layers.0.mlp.down_proj.lora_A.weight",
                    "role": "lora_A",
                    "shape": [1, 3],
                    "dtype": "float32",
                },
                {
                    "name": "model.layers.0.mlp.down_proj",
                    "tensor_name": "base_model.model.model.layers.0.mlp.down_proj.lora_B.weight",
                    "role": "lora_B",
                    "shape": [2, 1],
                    "dtype": "float32",
                },
            ],
        }
        (root / "result.json").write_text(json.dumps(payload), encoding="utf-8")
        return payload

    def _write_dense_artifact(self, root, *, weight_path="dense_delta.safetensors"):
        weight = root / "dense_delta.safetensors"
        weight.write_bytes(b"tiny-dense-delta")
        digest = hashlib.sha256(weight.read_bytes()).hexdigest()
        payload = {
            "schema_version": 1,
            "format": "dense_delta_v1",
            "condition_id": "adamerging_pp",
            "run_id": "run-123",
            "base_model": {
                "name": "unsloth/Meta-Llama-3.1-8B",
                "revision": "revision-abc",
                "config_sha256": "c" * 64,
            },
            "inference": {
                "torch_dtype": "bfloat16",
                "quantization": "none",
            },
            "producer": {
                "repository": "moea-trainer-delivery",
                "commit": "deadbeef",
            },
            "adapter_pool": {
                "id": "pool150_in_domain",
                "adapter_ids": ["task0", "task1"],
            },
            "weights": [
                {
                    "path": weight_path,
                    "bytes": weight.stat().st_size,
                    "sha256": digest,
                }
            ],
            "expected_module_count": 1,
            "modules": [
                {
                    "name": "model.layers.0.mlp.down_proj",
                    "tensor_name": "model.layers.0.mlp.down_proj.delta_weight",
                    "shape": [2, 3],
                    "dtype": "bfloat16",
                }
            ],
        }
        (root / "result.json").write_text(json.dumps(payload), encoding="utf-8")
        return payload

    def test_validates_the_exact_base_model_config_content(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "base"
            model.mkdir()
            (model / "config.json").write_text(
                '{\n  "model_type": "llama",\n  "hidden_size": 2\n}\n',
                encoding="utf-8",
            )
            artifact = SimpleNamespace(
                base_model_name=str(model),
                base_model_revision="local",
                base_model_config_sha256=(
                    "e01aa4af77230cb8dcc6b014b17227a17ccb64d62642e9997f169f785c2d72bb"
                ),
                torch_dtype="bfloat16",
                quantization="none",
            )

            validate_base_model_config(artifact)
            validate_inference_config(
                artifact,
                {"dtype": "bfloat16", "load_in_4bit": False},
            )

            artifact.base_model_config_sha256 = "0" * 64
            with self.assertRaisesRegex(MergedModelError, "does not match"):
                validate_base_model_config(artifact)

    def test_loads_a_valid_dense_delta_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_dense_artifact(root)
            weight = root / "dense_delta.safetensors"

            artifact = load_merged_model_artifact(
                root,
                expected_base_model="unsloth/Meta-Llama-3.1-8B",
            )

        self.assertEqual(artifact.format, "dense_delta_v1")
        self.assertEqual(artifact.condition_id, "adamerging_pp")
        self.assertEqual(artifact.run_id, "run-123")
        self.assertEqual(artifact.weight_files, (weight,))
        self.assertEqual(artifact.modules[0].name, "model.layers.0.mlp.down_proj")
        self.assertEqual(
            artifact.modules[0].tensor_name,
            "model.layers.0.mlp.down_proj.delta_weight",
        )
        self.assertEqual(artifact.modules[0].shape, (2, 3))
        self.assertEqual(artifact.torch_dtype, "bfloat16")
        self.assertEqual(artifact.quantization, "none")

    def test_rejects_a_weight_path_that_escapes_the_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_dense_artifact(root, weight_path="../dense_delta.safetensors")

            with self.assertRaisesRegex(MergedModelError, "unsafe weight path"):
                load_merged_model_artifact(root)

    def test_rejects_a_weight_checksum_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = self._write_dense_artifact(root)
            payload["weights"][0]["sha256"] = "0" * 64
            (root / "result.json").write_text(
                json.dumps(payload), encoding="utf-8"
            )

            with self.assertRaisesRegex(MergedModelError, "checksum mismatch"):
                load_merged_model_artifact(root)

    def test_rejects_a_different_configured_base_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_dense_artifact(root)

            with self.assertRaisesRegex(MergedModelError, "requires base"):
                load_merged_model_artifact(
                    root,
                    expected_base_model="a-different-base",
                )

    def test_rejects_a_runtime_precision_that_changes_the_selected_model(self):
        artifact = SimpleNamespace(
            torch_dtype="bfloat16",
            quantization="none",
        )

        with self.assertRaisesRegex(MergedModelError, "bfloat16"):
            validate_inference_config(
                artifact,
                {"dtype": "float16", "load_in_4bit": False},
            )

    def test_loads_a_standard_peft_adapter_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_peft_artifact(root)

            artifact = load_merged_model_artifact(
                root,
                expected_base_model="base",
            )

        self.assertEqual(artifact.format, "peft_adapter_v1")
        self.assertEqual(
            [path.name for path in artifact.weight_files],
            ["adapter_config.json", "adapter_model.safetensors"],
        )
        self.assertEqual(len(artifact.modules), 2)


if __name__ == "__main__":
    unittest.main()
