"""Runtime-only Arrow routing for the SMoEA rejection path.

This module consumes either the ``prepare/`` directory written by
``moea-repro prepare arrow|taskwise_k16_arrow`` or a raw ordered adapter
manifest for Direct Arrow.  It intentionally contains no benchmark or
clustering implementation: serving only loads validated assets and routes
tokens over the retained LoRA experts.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from system.merged_model import MergedModelError


class ArrowRuntimeError(MergedModelError):
    """Arrow serving assets are missing, unsafe, or incompatible."""


_LAYER_RE = re.compile(
    r"^base_model\.model\.model\.layers\.(\d+)\.mlp\.([a-z_]+)\.lora_([AB])\.weight$"
)


@dataclass(frozen=True)
class ArrowRuntimeArtifact:
    directory: Path | None
    condition_id: str
    run_id: str
    base_model_name: str | None
    target_module: str
    top_k: int
    adapter_names: tuple[str, ...]
    adapter_paths: tuple[Path, ...]
    prototype_file: Path | None
    prototype_sha256: str | None
    prototype_layers: tuple[int, ...]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArrowRuntimeError(f"cannot read Arrow metadata {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ArrowRuntimeError(f"Arrow metadata must be a JSON object: {path}")
    return payload


def _safe_relative(root: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ArrowRuntimeError(f"{label} must be a non-empty relative path")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ArrowRuntimeError(f"unsafe {label}: {value!r}")
    resolved_root = root.resolve()
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(resolved_root):
        raise ArrowRuntimeError(f"{label} escapes its root: {value!r}")
    return resolved


def _validate_file(path: Path, metadata: Any, label: str) -> None:
    if not path.is_file():
        raise ArrowRuntimeError(f"missing {label}: {path}")
    if not isinstance(metadata, dict):
        raise ArrowRuntimeError(f"missing checksum metadata for {label}: {path}")
    expected_size = metadata.get("bytes")
    if isinstance(expected_size, int) and path.stat().st_size != expected_size:
        raise ArrowRuntimeError(f"size mismatch for {label}: {path}")
    expected_digest = metadata.get("sha256")
    if not isinstance(expected_digest, str) or _sha256(path) != expected_digest:
        raise ArrowRuntimeError(f"checksum mismatch for {label}: {path}")


def _resolve_manifest_paths(
    manifest_path: Path,
    *,
    adapter_root: str | Path | None,
    expected_names: list[str] | None = None,
) -> tuple[tuple[str, ...], tuple[Path, ...]]:
    payload = _read_json(manifest_path)
    entries = payload.get("adapters")
    if not isinstance(entries, list) or not entries:
        raise ArrowRuntimeError(f"adapter manifest has no adapters: {manifest_path}")
    root = Path(adapter_root).resolve() if adapter_root else manifest_path.parent.resolve()
    names: list[str] = []
    paths: list[Path] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict) or not entry.get("name") or not entry.get("path"):
            raise ArrowRuntimeError(f"invalid adapter manifest entry {index}")
        name = str(entry["name"])
        path = Path(str(entry["path"]))
        if not path.is_absolute():
            path = root / path
        path = path.resolve()
        for filename in ("adapter_config.json", "adapter_model.safetensors"):
            if not (path / filename).is_file():
                raise ArrowRuntimeError(f"missing Arrow adapter file: {path / filename}")
        names.append(name)
        paths.append(path)
    if len(set(names)) != len(names):
        raise ArrowRuntimeError("Arrow adapter manifest contains duplicate names")
    if len(set(paths)) != len(paths):
        raise ArrowRuntimeError("Arrow adapter manifest contains duplicate paths")
    if expected_names is not None and names != expected_names:
        raise ArrowRuntimeError(
            "Arrow adapter order differs from the prepared prototype artifact"
        )
    return tuple(names), tuple(paths)


def load_arrow_runtime_artifact(
    condition_id: str,
    *,
    router_dir: str | Path | None,
    adapter_manifest: str | Path | None,
    adapter_root: str | Path | None = None,
    expected_base_model: str | None = None,
) -> ArrowRuntimeArtifact:
    """Resolve Direct Arrow or Taskwise-K16 assets without producer imports."""

    if condition_id not in {"arrow", "taskwise_k16_arrow"}:
        raise ArrowRuntimeError(f"unsupported Arrow condition: {condition_id!r}")

    # Direct Arrow can derive its small prototypes once at process startup.
    if router_dir is None:
        if condition_id != "arrow":
            raise ArrowRuntimeError(
                "taskwise_k16_arrow requires its prepared router directory"
            )
        if adapter_manifest is None:
            raise ArrowRuntimeError("arrow requires an adapter manifest")
        names, paths = _resolve_manifest_paths(
            Path(adapter_manifest), adapter_root=adapter_root
        )
        return ArrowRuntimeArtifact(
            directory=None,
            condition_id="arrow",
            run_id="runtime-computed",
            base_model_name=expected_base_model,
            target_module="down_proj",
            top_k=1,
            adapter_names=names,
            adapter_paths=paths,
            prototype_file=None,
            prototype_sha256=None,
            prototype_layers=(),
        )

    prepared = Path(router_dir).resolve()
    marker_path = prepared / "method.json"
    marker = _read_json(marker_path)
    expected_method = (
        "direct_arrow" if condition_id == "arrow" else "taskwise_k16_arrow"
    )
    if marker.get("status") != "complete" or marker.get("method") != expected_method:
        raise ArrowRuntimeError(
            f"{marker_path} is not a complete {condition_id} preparation"
        )
    if marker.get("target_module") != "down_proj" or marker.get("top_k") != 1:
        raise ArrowRuntimeError("serving supports the locked down_proj/top-1 Arrow methods")

    prototype_meta = marker.get("prototype_artifact")
    if not isinstance(prototype_meta, dict):
        raise ArrowRuntimeError("Arrow marker has no prototype artifact")
    prototype_file = prepared / "prototypes.safetensors"
    _validate_file(prototype_file, prototype_meta, "Arrow prototypes")
    raw_layers = prototype_meta.get("layers")
    if not isinstance(raw_layers, list) or not raw_layers:
        raise ArrowRuntimeError("Arrow prototype layer inventory is empty")
    layers = tuple(int(layer) for layer in raw_layers)

    if condition_id == "arrow":
        if adapter_manifest is None:
            raise ArrowRuntimeError(
                "prepared Direct Arrow still requires the ordered adapter manifest"
            )
        expected_names = marker.get("adapter_ids")
        if not isinstance(expected_names, list) or not expected_names:
            raise ArrowRuntimeError("Direct Arrow marker has no adapter order")
        names, paths = _resolve_manifest_paths(
            Path(adapter_manifest),
            adapter_root=adapter_root,
            expected_names=[str(name) for name in expected_names],
        )
    else:
        manifest_file = prepared / "cluster_manifest.json"
        _validate_file(
            manifest_file,
            marker.get("cluster_manifest_artifact"),
            "Taskwise K16 cluster manifest",
        )
        cluster_manifest = _read_json(manifest_file)
        entries = cluster_manifest.get("adapters")
        expected_count = int(marker.get("representative_count", 0))
        if (
            not isinstance(entries, list)
            or expected_count != 16
            or len(entries) != expected_count
            or cluster_manifest.get("count") != expected_count
        ):
            raise ArrowRuntimeError("Taskwise K16 representative inventory is incomplete")
        names_list: list[str] = []
        paths_list: list[Path] = []
        for index, entry in enumerate(entries):
            if not isinstance(entry, dict):
                raise ArrowRuntimeError("invalid Taskwise K16 manifest entry")
            expected_name = f"cluster_{index}"
            if entry.get("name") != expected_name:
                raise ArrowRuntimeError("Taskwise K16 representative order is invalid")
            directory = _safe_relative(prepared, entry.get("path"), "representative path")
            artifacts = entry.get("artifacts")
            if not isinstance(artifacts, dict):
                raise ArrowRuntimeError("representative checksum metadata is missing")
            for filename in ("adapter_config.json", "adapter_model.safetensors"):
                _validate_file(
                    directory / filename,
                    artifacts.get(filename),
                    f"{expected_name}/{filename}",
                )
            names_list.append(expected_name)
            paths_list.append(directory)
        names, paths = tuple(names_list), tuple(paths_list)

    return ArrowRuntimeArtifact(
        directory=prepared,
        condition_id=condition_id,
        run_id=str(marker.get("run_id", "unresolved")),
        base_model_name=expected_base_model,
        target_module="down_proj",
        top_k=1,
        adapter_names=names,
        adapter_paths=paths,
        prototype_file=prototype_file,
        prototype_sha256=str(prototype_meta["sha256"]),
        prototype_layers=layers,
    )


def _read_adapter_scale(
    path: Path, expected_base_model: str | None
) -> tuple[int, int]:
    config = _read_json(path / "adapter_config.json")
    try:
        rank = int(config["r"])
        alpha = int(config["lora_alpha"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ArrowRuntimeError(f"invalid LoRA rank/alpha in {path}") from exc
    if rank <= 0:
        raise ArrowRuntimeError(f"invalid LoRA rank in {path}")
    adapter_base = config.get("base_model_name_or_path")
    if (
        expected_base_model is not None
        and adapter_base is not None
        and str(adapter_base) != expected_base_model
    ):
        raise ArrowRuntimeError(
            f"Arrow adapter {path} requires base {adapter_base!r}; "
            f"configured {expected_base_model!r}"
        )
    targets = config.get("target_modules")
    if targets is not None and "down_proj" not in targets:
        raise ArrowRuntimeError(f"Arrow adapter does not target down_proj: {path}")
    return alpha, rank


def _parse_lora_stacks(artifact: ArrowRuntimeArtifact):
    import torch
    from safetensors.torch import load_file

    per_adapter: list[dict[int, dict[str, Any]]] = []
    layer_sets: list[set[int]] = []
    expected_scale = _read_adapter_scale(
        artifact.adapter_paths[0], artifact.base_model_name
    )
    for path in artifact.adapter_paths:
        if _read_adapter_scale(path, artifact.base_model_name) != expected_scale:
            raise ArrowRuntimeError("Arrow adapters disagree on LoRA rank/alpha")
        state = load_file(str(path / "adapter_model.safetensors"), device="cpu")
        layers: dict[int, dict[str, Any]] = {}
        for key, tensor in state.items():
            match = _LAYER_RE.match(key)
            if match is None or match.group(2) != artifact.target_module:
                continue
            layer = int(match.group(1))
            layers.setdefault(layer, {})[match.group(3)] = tensor.float()
        if not layers or any(set(pair) != {"A", "B"} for pair in layers.values()):
            raise ArrowRuntimeError(f"adapter has incomplete down_proj LoRA weights: {path}")
        per_adapter.append(layers)
        layer_sets.append(set(layers))
    if any(layers != layer_sets[0] for layers in layer_sets):
        raise ArrowRuntimeError("Arrow adapters disagree on LoRA layer inventory")

    layer_ids = sorted(layer_sets[0])
    a_stacks: dict[int, Any] = {}
    b_stacks: dict[int, Any] = {}
    for layer in layer_ids:
        a_values = [item[layer]["A"] for item in per_adapter]
        b_values = [item[layer]["B"] for item in per_adapter]
        a_shape, b_shape = tuple(a_values[0].shape), tuple(b_values[0].shape)
        if any(tuple(value.shape) != a_shape for value in a_values) or any(
            tuple(value.shape) != b_shape for value in b_values
        ):
            raise ArrowRuntimeError(f"Arrow adapter shape mismatch at layer {layer}")
        a_stacks[layer] = torch.stack(a_values)
        b_stacks[layer] = torch.stack(b_values)
    alpha, rank = expected_scale
    return a_stacks, b_stacks, layer_ids, alpha / rank


def _first_right_singular_vector(a, b):
    import torch

    q_a, r_a = torch.linalg.qr(a.float().T, mode="reduced")
    _q_b, r_b = torch.linalg.qr(b.float(), mode="reduced")
    _u, _s, vh = torch.linalg.svd(r_b @ r_a.T, full_matrices=False)
    vector = q_a @ vh[0]
    return vector / (vector.norm() + 1e-12)


def _compute_prototypes(a_stacks, b_stacks):
    import torch

    return {
        layer: torch.stack(
            [
                _first_right_singular_vector(a_stacks[layer][index], b_stacks[layer][index])
                for index in range(a_stacks[layer].shape[0])
            ]
        ).float()
        for layer in sorted(a_stacks)
    }


def _load_prototypes(artifact: ArrowRuntimeArtifact, expected_layers: list[int]):
    from safetensors.torch import load_file

    if artifact.prototype_file is None:
        return None
    tensors = load_file(str(artifact.prototype_file), device="cpu")
    expected_keys = {
        f"layer.{layer}.mlp.{artifact.target_module}" for layer in expected_layers
    }
    if set(tensors) != expected_keys:
        raise ArrowRuntimeError("Arrow prototype tensors differ from adapter layers")
    result = {}
    for layer in expected_layers:
        tensor = tensors[f"layer.{layer}.mlp.{artifact.target_module}"]
        if tensor.ndim != 2 or tensor.shape[0] != len(artifact.adapter_paths):
            raise ArrowRuntimeError(f"invalid Arrow prototype shape at layer {layer}")
        result[layer] = tensor.float()
    return result


def _target_modules(model: Any, layer_ids: list[int], target_module: str):
    modules = dict(model.named_modules())
    resolved = {}
    for layer in layer_ids:
        suffix = f".layers.{layer}.mlp.{target_module}"
        matches = [
            (name, module)
            for name, module in modules.items()
            if name == suffix[1:] or name.endswith(suffix)
        ]
        if not matches:
            raise ArrowRuntimeError(f"base model is missing Arrow layer {layer}")
        # Prefer the shortest exact wrapper path and not its nested base_layer.
        name, module = min(matches, key=lambda item: len(item[0]))
        base = getattr(module, "base_layer", module)
        weight = getattr(base, "weight", None)
        if weight is None or getattr(weight, "ndim", 0) != 2:
            raise ArrowRuntimeError(f"Arrow target is not linear: {name}")
        resolved[layer] = (module, weight)
    return resolved


class ArrowController:
    """Enable/disable token-level Arrow deltas on the engine's current model."""

    def __init__(self, artifact: ArrowRuntimeArtifact) -> None:
        self.artifact = artifact
        self.enabled = False
        self._cpu_stacks = None
        self._handles: list[Any] = []
        self._module_ids: tuple[int, ...] = ()
        self._device_tensors: list[Any] = []

    def _load_cpu(self):
        if self._cpu_stacks is not None:
            return self._cpu_stacks
        a_stacks, b_stacks, layers, scaling = _parse_lora_stacks(self.artifact)
        prototypes = _load_prototypes(self.artifact, layers)
        if prototypes is None:
            print("[system] 計算 Direct Arrow routing prototypes…", flush=True)
            prototypes = _compute_prototypes(a_stacks, b_stacks)
        self._cpu_stacks = (a_stacks, b_stacks, prototypes, layers, scaling)
        return self._cpu_stacks

    def attach(self, model: Any) -> None:
        import torch

        a_stacks, b_stacks, prototypes, layers, scaling = self._load_cpu()
        targets = _target_modules(model, layers, self.artifact.target_module)
        module_ids = tuple(id(targets[layer][0]) for layer in layers)
        if module_ids == self._module_ids and self._handles:
            return
        self._remove_hooks()
        for layer in layers:
            module, weight = targets[layer]
            a_stack = a_stacks[layer].to(device=weight.device, dtype=weight.dtype)
            b_stack = b_stacks[layer].to(device=weight.device, dtype=weight.dtype)
            prototype = prototypes[layer].to(device=weight.device, dtype=weight.dtype)
            if (
                a_stack.shape[0] != b_stack.shape[0]
                or a_stack.shape[0] != prototype.shape[0]
                or a_stack.shape[2] != weight.shape[1]
                or b_stack.shape[1] != weight.shape[0]
            ):
                raise ArrowRuntimeError(f"Arrow/base dimension mismatch at layer {layer}")
            self._device_tensors.extend((a_stack, b_stack, prototype))

            def add_arrow(
                _module,
                inputs,
                output,
                *,
                a=a_stack,
                b=b_stack,
                p=prototype,
                scale=scaling,
                owner=self,
            ):
                if not owner.enabled:
                    return output
                x = inputs[0]
                flat = x.reshape(-1, x.shape[-1])
                selected = torch.einsum("md,nd->mn", flat, p).abs().argmax(dim=-1)
                chosen_a = a.index_select(0, selected)
                chosen_b = b.index_select(0, selected)
                low_rank = torch.bmm(chosen_a, flat.unsqueeze(-1))
                delta = torch.bmm(chosen_b, low_rank).squeeze(-1)
                delta = delta.reshape(*x.shape[:-1], b.shape[1])
                return output + float(scale) * delta.to(output.dtype)

            self._handles.append(module.register_forward_hook(add_arrow))
        self._module_ids = module_ids

    def enable(self) -> None:
        self.enabled = True

    def disable(self) -> None:
        self.enabled = False

    def _remove_hooks(self) -> None:
        self.disable()
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._module_ids = ()
        self._device_tensors.clear()

    def close(self) -> None:
        self._remove_hooks()
        self._cpu_stacks = None
