import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file

from system.arrow_runtime import (
    ArrowController,
    ArrowRuntimeError,
    load_arrow_runtime_artifact,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_adapter(root: Path, name: str, a, b) -> Path:
    directory = root / name
    directory.mkdir(parents=True)
    (directory / "adapter_config.json").write_text(
        json.dumps({"r": 1, "lora_alpha": 1}), encoding="utf-8"
    )
    save_file(
        {
            "base_model.model.model.layers.0.mlp.down_proj.lora_A.weight": (
                torch.tensor([a], dtype=torch.float32)
            ),
            "base_model.model.model.layers.0.mlp.down_proj.lora_B.weight": (
                torch.tensor([[b]], dtype=torch.float32)
            ),
        },
        directory / "adapter_model.safetensors",
    )
    return directory


def write_direct_arrow(root: Path):
    adapters = root / "adapters"
    first = write_adapter(adapters, "task0", [1.0, 0.0], 2.0)
    second = write_adapter(adapters, "task1", [0.0, 1.0], 3.0)
    manifest = root / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "count": 2,
                "adapters": [
                    {"name": "task0", "path": str(first)},
                    {"name": "task1", "path": str(second)},
                ],
            }
        ),
        encoding="utf-8",
    )
    prepared = root / "prepare"
    prepared.mkdir()
    prototype = prepared / "prototypes.safetensors"
    save_file(
        {"layer.0.mlp.down_proj": torch.eye(2, dtype=torch.float32)},
        prototype,
    )
    (prepared / "method.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "method": "direct_arrow",
                "run_id": "run-1",
                "adapter_ids": ["task0", "task1"],
                "target_module": "down_proj",
                "top_k": 1,
                "prototype_artifact": {
                    "sha256": sha256(prototype),
                    "bytes": prototype.stat().st_size,
                    "layers": [0],
                },
            }
        ),
        encoding="utf-8",
    )
    return manifest, prepared


def write_taskwise_arrow(root: Path) -> Path:
    prepared = root / "prepare"
    cluster_root = prepared / "cluster_adapters"
    entries = []
    for index in range(16):
        adapter = write_adapter(
            cluster_root, f"cluster_{index}", [1.0, 0.0], 1.0
        )
        config = adapter / "adapter_config.json"
        weights = adapter / "adapter_model.safetensors"
        entries.append(
            {
                "name": f"cluster_{index}",
                "path": f"cluster_adapters/cluster_{index}",
                "artifacts": {
                    "adapter_config.json": {
                        "sha256": sha256(config), "bytes": config.stat().st_size
                    },
                    "adapter_model.safetensors": {
                        "sha256": sha256(weights), "bytes": weights.stat().st_size
                    },
                },
            }
        )
    cluster_manifest = prepared / "cluster_manifest.json"
    cluster_manifest.write_text(
        json.dumps({"count": 16, "adapters": entries}), encoding="utf-8"
    )
    prototype = prepared / "prototypes.safetensors"
    save_file(
        {"layer.0.mlp.down_proj": torch.ones(16, 2, dtype=torch.float32)},
        prototype,
    )
    (prepared / "method.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "method": "taskwise_k16_arrow",
                "run_id": "taskwise-run",
                "representative_count": 16,
                "target_module": "down_proj",
                "top_k": 1,
                "cluster_manifest_artifact": {
                    "sha256": sha256(cluster_manifest),
                    "bytes": cluster_manifest.stat().st_size,
                },
                "prototype_artifact": {
                    "sha256": sha256(prototype),
                    "bytes": prototype.stat().st_size,
                    "layers": [0],
                },
            }
        ),
        encoding="utf-8",
    )
    return prepared


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        block = torch.nn.Module()
        block.mlp = torch.nn.Module()
        block.mlp.down_proj = torch.nn.Linear(2, 1, bias=False)
        torch.nn.init.zeros_(block.mlp.down_proj.weight)
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([block])


class ArrowRuntimeTests(unittest.TestCase):
    def test_prepared_direct_arrow_routes_each_token_to_top1_expert(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest, prepared = write_direct_arrow(Path(directory))
            artifact = load_arrow_runtime_artifact(
                "arrow",
                router_dir=prepared,
                adapter_manifest=manifest,
            )
            model = FakeModel()
            controller = ArrowController(artifact)
            controller.attach(model)
            controller.enable()

            output = model.model.layers[0].mlp.down_proj(
                torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
            )

        self.assertTrue(torch.equal(output, torch.tensor([[[2.0], [3.0]]])))
        self.assertEqual(artifact.adapter_names, ("task0", "task1"))
        self.assertEqual(artifact.run_id, "run-1")

    def test_controller_can_be_disabled_without_changing_base_output(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest, prepared = write_direct_arrow(Path(directory))
            artifact = load_arrow_runtime_artifact(
                "arrow", router_dir=prepared, adapter_manifest=manifest
            )
            model = FakeModel()
            controller = ArrowController(artifact)
            controller.attach(model)
            controller.disable()

            output = model.model.layers[0].mlp.down_proj(
                torch.tensor([[[1.0, 0.0]]])
            )

        self.assertTrue(torch.equal(output, torch.zeros(1, 1, 1)))

    def test_direct_arrow_can_compute_prototypes_from_raw_adapters(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest, _prepared = write_direct_arrow(Path(directory))
            artifact = load_arrow_runtime_artifact(
                "arrow", router_dir=None, adapter_manifest=manifest
            )
            model = FakeModel()
            controller = ArrowController(artifact)
            controller.attach(model)
            controller.enable()

            output = model.model.layers[0].mlp.down_proj(
                torch.tensor([[[0.0, 1.0]]])
            )

        self.assertTrue(torch.equal(output, torch.tensor([[[3.0]]])))
        self.assertEqual(artifact.run_id, "runtime-computed")

    def test_rejects_prototype_checksum_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest, prepared = write_direct_arrow(Path(directory))
            prototype = prepared / "prototypes.safetensors"
            prototype.write_bytes(prototype.read_bytes() + b"corrupt")

            with self.assertRaisesRegex(ArrowRuntimeError, "mismatch"):
                load_arrow_runtime_artifact(
                    "arrow", router_dir=prepared, adapter_manifest=manifest
                )

    def test_loads_all_taskwise_k16_representatives_from_prepared_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            prepared = write_taskwise_arrow(Path(directory))

            artifact = load_arrow_runtime_artifact(
                "taskwise_k16_arrow",
                router_dir=prepared,
                adapter_manifest=None,
            )

        self.assertEqual(artifact.condition_id, "taskwise_k16_arrow")
        self.assertEqual(artifact.run_id, "taskwise-run")
        self.assertEqual(len(artifact.adapter_paths), 16)
        self.assertEqual(artifact.adapter_names[0], "cluster_0")
        self.assertEqual(artifact.adapter_names[-1], "cluster_15")


if __name__ == "__main__":
    unittest.main()
