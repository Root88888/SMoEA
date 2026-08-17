"""Validate a selected ``merged_model/`` directory without producer imports."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class MergedModelError(RuntimeError):
    """The selected serving artifact is missing, unsafe or incompatible."""


@dataclass(frozen=True)
class ModuleSpec:
    name: str
    tensor_name: str
    shape: tuple[int, ...]
    dtype: str


@dataclass(frozen=True)
class MergedModelArtifact:
    directory: Path
    format: str
    condition_id: str
    run_id: str
    base_model_name: str
    base_model_revision: str
    base_model_config_sha256: str
    torch_dtype: str
    quantization: str
    weight_files: tuple[Path, ...]
    modules: tuple[ModuleSpec, ...]
    manifest: dict[str, Any]


class DenseDeltaController:
    """Non-destructive forward hooks for one validated exact dense update."""

    def __init__(self) -> None:
        self.enabled = False
        self._handles: list[Any] = []
        self._deltas: list[Any] = []

    @classmethod
    def attach(
        cls, model: Any, artifact: MergedModelArtifact
    ) -> "DenseDeltaController":
        if artifact.format != "dense_delta_v1":
            raise MergedModelError(
                f"DenseDeltaController cannot load {artifact.format!r}"
            )
        if len(artifact.weight_files) != 1:
            raise MergedModelError("dense delta requires exactly one weight file")

        import torch
        import torch.nn.functional as functional
        from safetensors.torch import load_file

        tensors = load_file(str(artifact.weight_files[0]), device="cpu")
        expected_names = {spec.tensor_name for spec in artifact.modules}
        if set(tensors) != expected_names:
            raise MergedModelError(
                "dense delta tensor names differ from result.json module inventory"
            )
        modules = dict(model.named_modules())
        controller = cls()
        for spec in artifact.modules:
            module = modules.get(spec.name)
            if not isinstance(module, torch.nn.Linear):
                found = type(module).__name__ if module is not None else "missing"
                raise MergedModelError(
                    f"dense target {spec.name!r} must be Linear; found {found}"
                )
            delta = tensors[spec.tensor_name]
            if tuple(delta.shape) != spec.shape:
                raise MergedModelError(
                    f"dense tensor shape mismatch for {spec.name}: "
                    f"manifest={spec.shape}, file={tuple(delta.shape)}"
                )
            if tuple(module.weight.shape) != spec.shape:
                raise MergedModelError(
                    f"base module shape mismatch for {spec.name}: "
                    f"artifact={spec.shape}, base={tuple(module.weight.shape)}"
                )
            file_dtype = str(delta.dtype).removeprefix("torch.")
            if file_dtype != spec.dtype:
                raise MergedModelError(
                    f"dense tensor dtype mismatch for {spec.name}: "
                    f"manifest={spec.dtype}, file={file_dtype}"
                )
            delta = delta.to(device=module.weight.device, dtype=module.weight.dtype)
            controller._deltas.append(delta)

            def add_delta(_module, inputs, output, *, update=delta, owner=controller):
                if not owner.enabled:
                    return output
                delta_output = functional.linear(inputs[0], update)
                return output + delta_output.to(output.dtype)

            controller._handles.append(module.register_forward_hook(add_delta))
        return controller

    def enable(self) -> None:
        self.enabled = True

    def disable(self) -> None:
        self.enabled = False

    def close(self) -> None:
        self.disable()
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._deltas.clear()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_base_model_config(artifact: MergedModelArtifact) -> None:
    """Verify that serving resolves the exact config recorded by the producer."""

    from transformers.utils.hub import cached_file

    revision = artifact.base_model_revision
    if revision in {"local", "unresolved"}:
        revision = None
    try:
        resolved = cached_file(
            artifact.base_model_name,
            "config.json",
            revision=revision,
        )
    except Exception as exc:
        raise MergedModelError(
            f"cannot resolve configured base model: {exc}"
        ) from exc
    if resolved is None:
        raise MergedModelError(
            f"cannot resolve config for base model {artifact.base_model_name!r}"
        )


def validate_inference_config(
    artifact: MergedModelArtifact, system_config: dict[str, Any]
) -> None:
    """Reject runtime settings that would silently change the selected model."""

    configured_quantization = (
        "bitsandbytes_4bit" if system_config.get("load_in_4bit", False) else "none"
    )
    if configured_quantization != artifact.quantization:
        raise MergedModelError(
            "merged model requires quantization "
            f"{artifact.quantization!r}; configured {configured_quantization!r}"
        )
    configured_dtype = system_config.get("dtype")
    if configured_dtype != artifact.torch_dtype:
        raise MergedModelError(
            "merged model requires torch dtype "
            f"{artifact.torch_dtype!r}; configured {configured_dtype!r}"
        )
    try:
        config = json.loads(Path(resolved).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MergedModelError(f"cannot read base model config {resolved}: {exc}") from exc
    canonical = json.dumps(
        config, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    digest = hashlib.sha256(canonical).hexdigest()
    if digest != artifact.base_model_config_sha256:
        raise MergedModelError(
            "configured base model config does not match merged model artifact: "
            f"expected {artifact.base_model_config_sha256}, found {digest}"
        )


def _text(payload: dict[str, Any], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value:
        raise MergedModelError(f"result.json field {field!r} must be non-empty text")
    return value


def _safe_file(root: Path, relative: Any) -> Path:
    if not isinstance(relative, str) or not relative:
        raise MergedModelError("weight path must be non-empty relative text")
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise MergedModelError(f"unsafe weight path: {relative!r}")
    resolved_root = root.resolve()
    resolved = (root / candidate).resolve()
    if not resolved.is_relative_to(resolved_root):
        raise MergedModelError(f"weight path escapes artifact directory: {relative!r}")
    if not resolved.is_file():
        raise MergedModelError(f"weight file is missing: {resolved}")
    return resolved


def load_merged_model_artifact(
    directory: str | Path,
    *,
    expected_base_model: str | None = None,
) -> MergedModelArtifact:
    """Read and fully validate the explicitly selected artifact directory."""

    root = Path(directory)
    manifest_path = root / "result.json"
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MergedModelError(f"cannot read merged model manifest {manifest_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise MergedModelError("result.json must contain one JSON object")
    if payload.get("schema_version") != 1:
        raise MergedModelError(
            f"unsupported merged model schema: {payload.get('schema_version')!r}"
        )
    format_name = _text(payload, "format")
    if format_name not in {"dense_delta_v1", "peft_adapter_v1"}:
        raise MergedModelError(f"unsupported merged model format: {format_name!r}")

    base = payload.get("base_model")
    if not isinstance(base, dict):
        raise MergedModelError("result.json field 'base_model' must be an object")
    base_name = _text(base, "name")
    revision = _text(base, "revision")
    config_digest = _text(base, "config_sha256")
    if len(config_digest) != 64:
        raise MergedModelError("base model config SHA-256 must have 64 hex digits")
    try:
        int(config_digest, 16)
    except ValueError as exc:
        raise MergedModelError(
            "base model config SHA-256 must have 64 hex digits"
        ) from exc
    if expected_base_model is not None and base_name != expected_base_model:
        raise MergedModelError(
            f"merged model requires base {base_name!r}; configured {expected_base_model!r}"
        )

    inference = payload.get("inference")
    if not isinstance(inference, dict):
        raise MergedModelError("result.json field 'inference' must be an object")
    torch_dtype = _text(inference, "torch_dtype")
    quantization = _text(inference, "quantization")

    raw_weights = payload.get("weights")
    if not isinstance(raw_weights, list) or not raw_weights:
        raise MergedModelError("result.json must list at least one weight file")
    weight_files = []
    for item in raw_weights:
        if not isinstance(item, dict):
            raise MergedModelError("each weight entry must be an object")
        path = _safe_file(root, item.get("path"))
        expected_bytes = item.get("bytes")
        if not isinstance(expected_bytes, int) or expected_bytes < 0:
            raise MergedModelError(f"invalid byte size for weight file {path.name}")
        if path.stat().st_size != expected_bytes:
            raise MergedModelError(
                f"weight size mismatch for {path}: "
                f"expected {expected_bytes}, found {path.stat().st_size}"
            )
        expected_digest = item.get("sha256")
        if not isinstance(expected_digest, str) or _sha256(path) != expected_digest:
            raise MergedModelError(f"weight checksum mismatch for {path}")
        weight_files.append(path)

    raw_modules = payload.get("modules", [])
    if not isinstance(raw_modules, list):
        raise MergedModelError("result.json field 'modules' must be a list")
    modules = []
    for item in raw_modules:
        if not isinstance(item, dict):
            raise MergedModelError("each module entry must be an object")
        shape = item.get("shape")
        if not isinstance(shape, list) or not all(
            isinstance(value, int) and value >= 0 for value in shape
        ):
            raise MergedModelError("module shape must be a list of non-negative integers")
        modules.append(
            ModuleSpec(
                name=_text(item, "name"),
                tensor_name=_text(item, "tensor_name"),
                shape=tuple(shape),
                dtype=_text(item, "dtype"),
            )
        )
    if len({module.name for module in modules}) != len(modules):
        raise MergedModelError("merged model module names must be unique")
    if len({module.tensor_name for module in modules}) != len(modules):
        raise MergedModelError("merged model tensor names must be unique")
    if format_name == "dense_delta_v1":
        expected_count = payload.get("expected_module_count")
        if not isinstance(expected_count, int) or expected_count <= 0:
            raise MergedModelError("dense delta requires a positive module count")
        if len(modules) != expected_count:
            raise MergedModelError(
                f"dense delta lists {len(modules)} modules; expected {expected_count}"
            )

    return MergedModelArtifact(
        directory=root.resolve(),
        format=format_name,
        condition_id=_text(payload, "condition_id"),
        run_id=_text(payload, "run_id"),
        base_model_name=base_name,
        base_model_revision=revision,
        base_model_config_sha256=config_digest,
        torch_dtype=torch_dtype,
        quantization=quantization,
        weight_files=tuple(weight_files),
        modules=tuple(modules),
        manifest=payload,
    )
