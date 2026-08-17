#!/usr/bin/env python3
"""GPU smoke for a real selected dense artifact and two task adapters."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from system.inference import InferenceEngine  # noqa: E402
from system.merged_model import load_merged_model_artifact  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--task-a", default="task0")
    parser.add_argument("--task-b", default="task1")
    parser.add_argument("--prompt", default="Q: What is 2 + 2?\nA:")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def last_token_logits(engine, prompt):
    import torch

    tokens = engine.tokenizer(prompt, return_tensors="pt")
    tokens = {key: value.to(engine.model.device) for key, value in tokens.items()}
    with torch.no_grad():
        return engine.model(**tokens).logits[:, -1, :].float().cpu()


def main():
    import torch
    import torch.nn.functional as functional
    from safetensors.torch import load_file

    args = parse_args()
    artifact = load_merged_model_artifact(args.artifact)
    cfg = {
        "system": {
            "base_model": artifact.base_model_name,
            "adapter_dir": args.adapter_dir,
            "merged_model_dir": args.artifact,
            "merged_model_required": True,
            "load_in_4bit": False,
            "dtype": artifact.torch_dtype,
            "max_input_tokens": 512,
            "max_new_tokens": 16,
        }
    }
    engine = InferenceEngine(cfg)
    torch.cuda.reset_peak_memory_stats()
    engine.load_base()

    # Check one real module against the explicit base(x) + delta(x) equation.
    spec = artifact.modules[0]
    module = dict(engine.model.named_modules())[spec.name]
    delta = load_file(str(artifact.weight_files[0]), device="cpu")[spec.name]
    inputs = torch.randn(
        1,
        2,
        module.in_features,
        device=module.weight.device,
        dtype=module.weight.dtype,
    )
    engine._dense_controller.disable()
    with torch.no_grad():
        base_output = module(inputs)
        expected = base_output + functional.linear(
            inputs,
            delta.to(device=module.weight.device, dtype=module.weight.dtype),
        ).to(base_output.dtype)
        engine._dense_controller.enable()
        actual = module(inputs)
        layer_max_abs_error = float((actual.float() - expected.float()).abs().max())
    if layer_max_abs_error != 0.0:
        raise AssertionError(
            f"dense layer equation mismatch: max_abs_error={layer_max_abs_error}"
        )

    engine.ensure_adapter(args.task_a)
    task_a_first = last_token_logits(engine, args.prompt)
    merged_info = engine.ensure_merged()
    merged_logits = last_token_logits(engine, args.prompt)
    merged_text = engine.generate([args.prompt])[0]
    engine.ensure_adapter(args.task_b)
    task_b_logits = last_token_logits(engine, args.prompt)
    engine.ensure_adapter(args.task_a)
    task_a_second = last_token_logits(engine, args.prompt)

    task_a_restore_error = float(
        (task_a_first - task_a_second).abs().max()
    )
    task_a_vs_merged = float((task_a_first - merged_logits).abs().max())
    task_b_vs_merged = float((task_b_logits - merged_logits).abs().max())
    if task_a_restore_error != 0.0:
        raise AssertionError(
            f"task adapter was not restored exactly: {task_a_restore_error}"
        )
    if task_a_vs_merged == 0.0 or task_b_vs_merged == 0.0:
        raise AssertionError("task and merged modes produced identical logits")

    result = {
        "status": "complete",
        "artifact": str(Path(args.artifact).resolve()),
        "condition_id": merged_info["condition_id"],
        "run_id": merged_info["run_id"],
        "format": merged_info["format"],
        "base_model": artifact.base_model_name,
        "torch_dtype": artifact.torch_dtype,
        "task_sequence": [args.task_a, "merged_model", args.task_b, args.task_a],
        "layer_max_abs_error": layer_max_abs_error,
        "task_a_restore_max_abs_error": task_a_restore_error,
        "task_a_vs_merged_max_abs_difference": task_a_vs_merged,
        "task_b_vs_merged_max_abs_difference": task_b_vs_merged,
        "merged_generation": merged_text,
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "gpu": torch.cuda.get_device_name(0),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
