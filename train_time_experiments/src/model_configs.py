"""Resolve supported model IDs, dimensions, aliases, and experiment layers.

Keeping model metadata here ensures the 1B and 3B training paths select valid
probe, activation, and LoRA layers whether models are local or on Hugging Face.
"""

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


TRAIN_TIME_DIR = Path(__file__).resolve().parent.parent

LLAMA32_1B = "meta-llama/Llama-3.2-1B-Instruct"
LLAMA32_3B = "meta-llama/Llama-3.2-3B-Instruct"
LLAMA3_8B = "meta-llama/Meta-Llama-3-8B-Instruct"
LLAMA3_8B_BASE = "meta-llama/Meta-Llama-3-8B"


@dataclass(frozen=True)
class ModelSpec:
    repo_id: str
    alias: str
    hidden_size: int
    num_hidden_layers: int
    default_probe_layers: tuple[int, ...]
    default_activation_layers: tuple[int, ...]
    local_dir_names: tuple[str, ...] = ()
    has_eleuther_sae: bool = False


@dataclass(frozen=True)
class ResolvedModelConfig:
    requested_name: str
    model_name: str
    repo_id: str
    alias: str
    hidden_size: int
    num_hidden_layers: int
    default_probe_layers: tuple[int, ...]
    default_activation_layers: tuple[int, ...]
    has_eleuther_sae: bool
    is_local: bool


MODEL_SPECS: dict[str, ModelSpec] = {
    LLAMA32_1B: ModelSpec(
        repo_id=LLAMA32_1B,
        alias="llama32_1b",
        hidden_size=2048,
        num_hidden_layers=16,
        default_probe_layers=(2, 4, 6, 8, 10, 12),
        default_activation_layers=(0, 2, 4, 6, 8, 10, 12, 14, 16),
        local_dir_names=("Llama-3.2-1B-Instruct",),
    ),
    LLAMA32_3B: ModelSpec(
        repo_id=LLAMA32_3B,
        alias="llama32_3b",
        hidden_size=3072,
        num_hidden_layers=28,
        default_probe_layers=(4, 8, 12, 16, 20, 24),
        default_activation_layers=(0, 4, 8, 12, 16, 20, 24, 28),
        local_dir_names=("Llama-3.2-3B-Instruct",),
    ),
    LLAMA3_8B: ModelSpec(
        repo_id=LLAMA3_8B,
        alias="llama3_8b",
        hidden_size=4096,
        num_hidden_layers=32,
        default_probe_layers=(4, 8, 12, 16, 20, 24),
        default_activation_layers=(0, 4, 8, 12, 16, 20, 24, 28, 32),
        local_dir_names=("Meta-Llama-3-8B-Instruct",),
        has_eleuther_sae=True,
    ),
    LLAMA3_8B_BASE: ModelSpec(
        repo_id=LLAMA3_8B_BASE,
        alias="llama3_8b_base",
        hidden_size=4096,
        num_hidden_layers=32,
        default_probe_layers=(4, 8, 12, 16, 20, 24),
        default_activation_layers=(0, 4, 8, 12, 16, 20, 24, 28, 32),
        local_dir_names=("Meta-Llama-3-8B",),
        has_eleuther_sae=True,
    ),
}

ALIASES = {
    "llama3.2-1b": LLAMA32_1B,
    "llama32-1b": LLAMA32_1B,
    "llama32_1b": LLAMA32_1B,
    "1b": LLAMA32_1B,
    "llama3.2-3b": LLAMA32_3B,
    "llama32-3b": LLAMA32_3B,
    "llama32_3b": LLAMA32_3B,
    "3b": LLAMA32_3B,
    "llama3-8b": LLAMA3_8B,
    "llama3_8b": LLAMA3_8B,
    "8b": LLAMA3_8B,
    "llama3-8b-base": LLAMA3_8B_BASE,
    "llama3_8b_base": LLAMA3_8B_BASE,
}

UNSUPPORTED_MODEL_MESSAGES = {
    "meta-llama/llama-3.2-2b-instruct": (
        "meta-llama/Llama-3.2-2B-Instruct is not a valid Meta Llama 3.2 "
        "instruct repo. Use meta-llama/Llama-3.2-1B-Instruct or "
        "meta-llama/Llama-3.2-3B-Instruct."
    ),
    "llama3.2-2b": (
        "Llama 3.2 2B Instruct is not available from meta-llama; use 1B or 3B."
    ),
    "llama32-2b": (
        "Llama 3.2 2B Instruct is not available from meta-llama; use 1B or 3B."
    ),
    "llama32_2b": (
        "Llama 3.2 2B Instruct is not available from meta-llama; use 1B or 3B."
    ),
    "2b": "Llama 3.2 2B Instruct is not available from meta-llama; use 1B or 3B.",
}


def sanitize_model_alias(value: str) -> str:
    alias = re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()
    return alias or "model"


def _candidate_local_paths(model_name: str) -> list[Path]:
    path = Path(os.path.expanduser(model_name))
    candidates = [path]
    if not path.is_absolute():
        candidates.append(TRAIN_TIME_DIR / model_name)
    return candidates


def _find_local_path(model_name: str) -> Path | None:
    for candidate in _candidate_local_paths(model_name):
        if candidate.is_dir():
            return candidate.resolve()
    return None


def _local_path_for_spec(spec: ModelSpec) -> Path | None:
    for dirname in spec.local_dir_names:
        candidate = TRAIN_TIME_DIR / dirname
        if candidate.is_dir():
            return candidate.resolve()
    return None


def _read_local_config(model_path: Path) -> dict:
    config_path = model_path / "config.json"
    if not config_path.exists():
        raise ValueError(f"Local model directory has no config.json: {model_path}")
    with config_path.open("r") as f:
        return json.load(f)


def _spec_for_local_path(model_path: Path) -> ModelSpec | None:
    path_name = model_path.name.lower()
    for spec in MODEL_SPECS.values():
        names = [spec.repo_id.split("/")[-1], *spec.local_dir_names]
        if path_name in {name.lower() for name in names}:
            return spec

    config = _read_local_config(model_path)
    hidden_size = config.get("hidden_size")
    num_hidden_layers = config.get("num_hidden_layers")
    for spec in MODEL_SPECS.values():
        if (
            spec.hidden_size == hidden_size
            and spec.num_hidden_layers == num_hidden_layers
        ):
            return spec
    return None


def _default_probe_layers(num_hidden_layers: int) -> tuple[int, ...]:
    if num_hidden_layers <= 16:
        return tuple(range(2, max(2, num_hidden_layers - 2), 2))
    return tuple(layer for layer in range(4, num_hidden_layers, 4))


def _default_activation_layers(num_hidden_layers: int) -> tuple[int, ...]:
    step = 2 if num_hidden_layers <= 16 else 4
    return tuple(range(0, num_hidden_layers + 1, step))


def _resolve_repo_id(model_name: str) -> str:
    normalized = model_name.strip()
    key = normalized.lower()
    if key in UNSUPPORTED_MODEL_MESSAGES:
        raise ValueError(UNSUPPORTED_MODEL_MESSAGES[key])
    return ALIASES.get(key, normalized)


def resolve_model_config(model_name: str | None = None) -> ResolvedModelConfig:
    requested_name = model_name or LLAMA32_1B
    local_path = _find_local_path(requested_name)
    if local_path is not None:
        spec = _spec_for_local_path(local_path)
        if spec is not None:
            return ResolvedModelConfig(
                requested_name=requested_name,
                model_name=str(local_path),
                repo_id=spec.repo_id,
                alias=spec.alias,
                hidden_size=spec.hidden_size,
                num_hidden_layers=spec.num_hidden_layers,
                default_probe_layers=spec.default_probe_layers,
                default_activation_layers=spec.default_activation_layers,
                has_eleuther_sae=spec.has_eleuther_sae,
                is_local=True,
            )

        config = _read_local_config(local_path)
        hidden_size = int(config["hidden_size"])
        num_hidden_layers = int(config["num_hidden_layers"])
        alias = sanitize_model_alias(local_path.name)
        return ResolvedModelConfig(
            requested_name=requested_name,
            model_name=str(local_path),
            repo_id=str(local_path),
            alias=alias,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            default_probe_layers=_default_probe_layers(num_hidden_layers),
            default_activation_layers=_default_activation_layers(num_hidden_layers),
            has_eleuther_sae=False,
            is_local=True,
        )

    repo_id = _resolve_repo_id(requested_name)
    if repo_id not in MODEL_SPECS:
        raise ValueError(
            f"Unsupported model '{requested_name}'. Supported train-time models are: "
            + ", ".join(MODEL_SPECS.keys())
        )

    spec = MODEL_SPECS[repo_id]
    local_spec_path = _local_path_for_spec(spec)
    model_name = str(local_spec_path) if local_spec_path is not None else spec.repo_id
    return ResolvedModelConfig(
        requested_name=requested_name,
        model_name=model_name,
        repo_id=spec.repo_id,
        alias=spec.alias,
        hidden_size=spec.hidden_size,
        num_hidden_layers=spec.num_hidden_layers,
        default_probe_layers=spec.default_probe_layers,
        default_activation_layers=spec.default_activation_layers,
        has_eleuther_sae=spec.has_eleuther_sae,
        is_local=local_spec_path is not None,
    )


def parse_layer_list(value, default: Iterable[int] | None = None) -> list[int]:
    if value is None:
        if default is None:
            return []
        return list(default)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.lower() in {"", "none", "default"}:
            return [] if default is None else list(default)
        return [int(part.strip()) for part in stripped.split(",") if part.strip()]
    return [int(layer) for layer in value]


def validate_probe_layers(
    layers: Iterable[int], model_config: ResolvedModelConfig
) -> list[int]:
    layers = list(layers)
    invalid = [
        layer
        for layer in layers
        if layer < 0 or layer >= model_config.num_hidden_layers
    ]
    if invalid:
        raise ValueError(
            f"Invalid probe/block layers for {model_config.alias}: {invalid}. "
            f"Valid range is 0..{model_config.num_hidden_layers - 1}."
        )
    return layers


def validate_hidden_state_layers(
    layers: Iterable[int], model_config: ResolvedModelConfig
) -> list[int]:
    layers = list(layers)
    invalid = [
        layer
        for layer in layers
        if layer < 0 or layer > model_config.num_hidden_layers
    ]
    if invalid:
        raise ValueError(
            f"Invalid hidden-state layers for {model_config.alias}: {invalid}. "
            f"Valid range is 0..{model_config.num_hidden_layers}."
        )
    return layers


def resolve_probe_layers(value, model_config: ResolvedModelConfig) -> list[int]:
    return validate_probe_layers(
        parse_layer_list(value, model_config.default_probe_layers), model_config
    )


def resolve_activation_layers(value, model_config: ResolvedModelConfig) -> list[int]:
    return validate_hidden_state_layers(
        parse_layer_list(value, model_config.default_activation_layers), model_config
    )


def resolve_lora_layers(
    value, model_config: ResolvedModelConfig, default_layers: Iterable[int] | None = None
) -> list[int]:
    if isinstance(value, str) and value.strip().lower() == "all":
        return list(range(model_config.num_hidden_layers))
    if value is None:
        if default_layers is None:
            return list(range(model_config.num_hidden_layers))
        return validate_probe_layers(default_layers, model_config)
    return validate_probe_layers(parse_layer_list(value), model_config)
