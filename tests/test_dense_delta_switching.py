import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file

from system.merged_model import DenseDeltaController, load_merged_model_artifact


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        block = torch.nn.Module()
        block.mlp = torch.nn.Module()
        block.mlp.down_proj = torch.nn.Linear(2, 2, bias=False)
        block.mlp.down_proj.weight.data.copy_(torch.eye(2))
        self.model.layers = torch.nn.ModuleList([block])

    def forward(self, inputs):
        return self.model.layers[0].mlp.down_proj(inputs)


class DenseDeltaSwitchingTests(unittest.TestCase):
    def test_dense_delta_can_be_enabled_and_disabled_without_changing_base_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            weight = root / "dense_delta.safetensors"
            save_file(
                {"model.layers.0.mlp.down_proj.delta_weight": torch.ones(2, 2)},
                weight,
            )
            digest = hashlib.sha256(weight.read_bytes()).hexdigest()
            (root / "result.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "format": "dense_delta_v1",
                        "condition_id": "adamerging_pp",
                        "run_id": "run-123",
                        "base_model": {
                            "name": "base",
                            "revision": "revision",
                            "config_sha256": "c" * 64,
                        },
                        "inference": {
                            "torch_dtype": "float32",
                            "quantization": "none",
                        },
                        "producer": {"repository": "delivery", "commit": "abc"},
                        "adapter_pool": {"id": "pool", "adapter_ids": []},
                        "weights": [
                            {
                                "path": weight.name,
                                "bytes": weight.stat().st_size,
                                "sha256": digest,
                            }
                        ],
                        "expected_module_count": 1,
                        "modules": [
                            {
                                "name": "model.layers.0.mlp.down_proj",
                                "tensor_name": (
                                    "model.layers.0.mlp.down_proj.delta_weight"
                                ),
                                "shape": [2, 2],
                                "dtype": "float32",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            artifact = load_merged_model_artifact(root, expected_base_model="base")
            model = TinyModel()
            original = model.model.layers[0].mlp.down_proj.weight.detach().clone()
            controller = DenseDeltaController.attach(model, artifact)
            inputs = torch.tensor([[1.0, 2.0]])

            base_output = model.model.layers[0].mlp.down_proj(inputs)
            controller.enable()
            merged_output = model.model.layers[0].mlp.down_proj(inputs)
            controller.disable()
            restored_output = model.model.layers[0].mlp.down_proj(inputs)

            from peft import LoraConfig, get_peft_model

            peft_model = get_peft_model(
                model,
                LoraConfig(
                    r=1,
                    lora_alpha=1,
                    target_modules=["down_proj"],
                    bias="none",
                ),
            )
            peft_model.base_model.disable_adapter_layers()
            controller.enable()
            merged_after_peft_injection = peft_model(inputs)

        self.assertEqual(base_output.tolist(), [[1.0, 2.0]])
        self.assertEqual(merged_output.tolist(), [[4.0, 5.0]])
        self.assertEqual(restored_output.tolist(), [[1.0, 2.0]])
        self.assertEqual(merged_after_peft_injection.tolist(), [[4.0, 5.0]])
        self.assertTrue(
            torch.equal(model.model.layers[0].mlp.down_proj.weight, original)
        )


if __name__ == "__main__":
    unittest.main()
