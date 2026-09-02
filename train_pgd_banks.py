#!/usr/bin/env python3
"""Train and store reusable PGD attack banks for the 1B and 3B checkpoints.

The training path mirrors the primary per-batch PGD implementation in
``activation_analysis4.py``:

* the adapted model is attacked at its input embeddings;
* harmful and benign examples are trained separately;
* probe-targeted and behavior-only objectives get independent attacks;
* prompt, target, and probe masks use the repository's generation masking; and
* every stored attack is checked against a portable replay hook.

Unlike the earlier checkpoint format, the on-disk representation is not a pickle.
Metadata and loss histories are JSON, while variable-length attack tensors are
stored with safetensors.  ``load_pgd_banks`` reconstructs the same nested bank
interface used by the analysis experiments.

This file is included because the public result grids depend on fixed,
hash-addressed attacks. Keeping attack construction here—separate from the
experiment runners—makes bank provenance explicit and prevents each ablation
from silently optimizing a different input perturbation.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
import random
import shutil
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict, load_dataset
from peft import LoraConfig, PeftModel
from safetensors.torch import load_file as load_safetensors
from safetensors.torch import save_file as save_safetensors
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parent
# Resolve paths relative to this file so commands behave identically from an
# interactive checkout, a batch job, or the eventual public repository layout.
TRAIN_TIME_DIR = REPO_ROOT / "obfuscated-activations" / "train_time_experiments"
if not TRAIN_TIME_DIR.is_dir():
    # In the GPU repository, train_time_experiments is already at repo root.
    TRAIN_TIME_DIR = REPO_ROOT / "train_time_experiments"
ARTIFACT_ROOT = REPO_ROOT / "probe_weights_comp_only2"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "pgd_banks"

if str(TRAIN_TIME_DIR) not in sys.path:
    sys.path.insert(0, str(TRAIN_TIME_DIR))

os.environ.setdefault("OAT_LIGHTWEIGHT_IMPORTS", "1")

from src.attacks import add_hooks, clear_hooks, train_attack
from src.chat_formatting import format_dataset_chat
from src.probe_training import load_probe_state_dicts
from src.token_ranges import get_token_ranges
from src.utils import get_valid_token_mask


SCHEMA_NAME = "activation-analysis-pgd-banks"
SCHEMA_VERSION = 1
DATASET_NAME = "Mechanistic-Anomaly-Detection/llama3-jailbreaks"
HARMFUL_SPLIT = "circuit_breakers_test"
BENIGN_SPLIT = "benign_instructions_test"
REQUIRED_SPLITS = (HARMFUL_SPLIT, BENIGN_SPLIT)
MODEL_ORDER = ("1B", "3B")
POPULATION_ORDER = ("harmful", "benign")
ATTACK_SPECS = {
    "probe_targeted": {"probe_loss_coef": 1.0},
    "behavior_only": {"probe_loss_coef": 0.0},
}
ATTACK_ORDER = tuple(ATTACK_SPECS)
PRIMARY_PGD_BATCH_SIZE = 2
DEFAULT_EPSILON = 10.0
DEFAULT_N_PAIRS = 20
DEFAULT_PGD_ITERATIONS = 32
DEFAULT_SEED = 42
REPLAY_LOGIT_TOLERANCE = 3e-3

MODEL_SPECS = {
    "1B": {
        "label": "Llama-3.2-1B",
        "stem": "llama32-1b-generation-linear",
        "local_model_dir": TRAIN_TIME_DIR / "Llama-3.2-1B-Instruct",
        "repo_id": "meta-llama/Llama-3.2-1B-Instruct",
        "hidden_size": 2048,
        "num_hidden_layers": 16,
    },
    "3B": {
        "label": "Llama-3.2-3B",
        "stem": "llama32-3b-generation-linear",
        "local_model_dir": TRAIN_TIME_DIR / "Llama-3.2-3B-Instruct",
        "repo_id": "meta-llama/Llama-3.2-3B-Instruct",
        "hidden_size": 3072,
        "num_hidden_layers": 28,
    },
}

LORA_TARGET_MODULES = {
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
}

LOSS_COLUMNS = (
    "population",
    "attack_kind",
    "batch_index",
    "step",
    "toward_loss",
    "probe_loss",
    "total_loss",
    "iterations",
)


@dataclass(frozen=True)
class PGDAttackRecord:
    """One replayable per-example, per-position prompt attack."""

    model_key: str
    population: str
    attack_kind: str
    pair_id: int
    dataset_split: str
    input_token_hash: str
    prompt_positions: torch.Tensor
    prompt_deltas: torch.Tensor
    sequence_length: int
    epsilon: float
    learning_rate: float
    iterations: int
    final_toward_loss: float
    final_probe_loss: float
    final_total_loss: float
    attack_norm_mean: float
    attack_norm_max: float
    fingerprint: str


@dataclass
class PGDAttackBank:
    """A collection of attacks trained together with one objective and seed."""

    model_key: str
    model_label: str
    population: str
    attack_kind: str
    dataset_split: str
    start: int
    n_examples: int
    batch_size: int
    iterations: int
    epsilon: float
    learning_rate: float
    seed: int
    records: dict[int, PGDAttackRecord]
    losses: pd.DataFrame
    fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        digest = hashlib.sha256()
        for pair_id in sorted(self.records):
            digest.update(self.records[pair_id].fingerprint.encode())
        self.fingerprint = digest.hexdigest()[:16]

    def record(self, pair_id: int) -> PGDAttackRecord:
        pair_id = int(pair_id)
        if pair_id not in self.records:
            raise KeyError(f"PGD bank has no record for pair_id={pair_id}")
        return self.records[pair_id]


class FixedBatchPromptAdversary(nn.Module):
    """Replay a fixed per-row, per-position PGD tensor at prompt tokens only."""

    def __init__(self, applied_deltas: torch.Tensor, attack_mask: torch.Tensor):
        super().__init__()
        self.register_buffer("applied_deltas", applied_deltas.detach().float())
        self.attack_mask = attack_mask.detach().bool()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.applied_deltas.ndim == 2:
            deltas = self.applied_deltas.to(device=x.device, dtype=x.dtype).unsqueeze(1)
            deltas = deltas.expand(-1, x.shape[1], -1)
        elif self.applied_deltas.ndim == 3:
            if self.applied_deltas.shape != x.shape:
                raise ValueError(
                    f"Applied PGD tensor {self.applied_deltas.shape} does not match "
                    f"{x.shape}"
                )
            deltas = self.applied_deltas.to(device=x.device, dtype=x.dtype)
        else:
            raise ValueError("Applied PGD tensor must be rank two or three")

        mask = self.attack_mask.to(x.device)
        if mask.shape != x.shape[:2]:
            raise ValueError(f"Attack mask {mask.shape} does not match {x.shape[:2]}")
        return torch.where(mask.unsqueeze(-1), x + deltas, x)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return value


def _json_safe(value: Any) -> Any:
    """Convert values to strict JSON, representing non-finite floats as null."""

    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        value = value.item()
    if torch.is_tensor(value):
        if value.numel() != 1:
            raise TypeError("Only scalar tensors can be represented directly in JSON")
        value = value.detach().cpu().item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(_json_safe(value), handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(_json_safe(row), sort_keys=True, allow_nan=False) + "\n"
            )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TypeError(f"{path}:{line_number} is not a JSON object")
            rows.append(row)
    return rows


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parameter_label(value: float) -> str:
    return format(float(value), ".15g")


def build_bundle_name(
    epsilon: float,
    n_pairs: int,
    pgd_iterations: int,
    seed: int,
) -> str:
    """Build a deterministic portable-bank directory name."""

    return (
        f"ep{_parameter_label(epsilon)}_n{int(n_pairs)}_"
        f"iter{int(pgd_iterations)}_seed{int(seed)}"
    )


def attack_seed_from_base(seed: int) -> int:
    """Match activation_analysis4's Experiment 7 attack-seed derivation."""

    return int(seed) + 7000


def bank_seed(attack_seed: int, population: str, attack_kind: str) -> int:
    """Derive a stable seed for each population and attack objective."""

    if population not in POPULATION_ORDER:
        raise ValueError(f"Unknown population {population!r}")
    if attack_kind not in ATTACK_ORDER:
        raise ValueError(f"Unknown attack kind {attack_kind!r}")
    return (
        int(attack_seed)
        + 100 * int(population == "benign")
        + ATTACK_ORDER.index(attack_kind)
    )


def _validate_parameters(
    epsilon: float,
    n_pairs: int,
    pgd_iterations: int,
    seed: int,
) -> None:
    if not math.isfinite(float(epsilon)) or float(epsilon) <= 0:
        raise ValueError("epsilon must be a finite positive number")
    if int(n_pairs) <= 0:
        raise ValueError("n_pairs must be positive")
    if int(pgd_iterations) <= 0:
        raise ValueError("pgd_iterations must be positive")
    if isinstance(seed, bool):
        raise TypeError("seed must be an integer")
    int(seed)


def _resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device {device} was requested but CUDA is unavailable")
        if device.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is unavailable")
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _set_global_seed(seed: int) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _empty_cache(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.empty_cache()


def _package_versions() -> dict[str, str | None]:
    packages = ("torch", "numpy", "pandas", "datasets", "peft", "safetensors", "transformers")
    versions: dict[str, str | None] = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _usable_model_path(spec: Mapping[str, Any], info: Mapping[str, Any]) -> str:
    candidates = [Path(spec["local_model_dir"])]
    if info.get("model_name"):
        candidates.append(Path(info["model_name"]))
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return str(info.get("model_repo_id") or spec["repo_id"])


def get_model_config(model_key: str) -> dict[str, Any]:
    """Load and validate adapter, probe, and base-model metadata for one size."""

    if model_key not in MODEL_SPECS:
        raise ValueError(f"Unknown model_key {model_key!r}; expected one of {MODEL_ORDER}")

    spec = dict(MODEL_SPECS[model_key])
    stem = spec["stem"]
    info_path = ARTIFACT_ROOT / f"{stem}_info.json"
    probes_path = ARTIFACT_ROOT / f"{stem}_probes_state_dict.pt"
    adapter_dir = ARTIFACT_ROOT / f"{stem}_model"
    adapter_config_path = adapter_dir / "adapter_config.json"
    adapter_weights_path = adapter_dir / "adapter_model.safetensors"
    required = (info_path, probes_path, adapter_config_path, adapter_weights_path)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required artifacts:\n" + "\n".join(missing))

    info = _read_json(info_path)
    adapter_config = _read_json(adapter_config_path)
    target_modules = set(adapter_config["target_modules"])
    if target_modules != LORA_TARGET_MODULES:
        raise ValueError(
            f"{spec['label']} target modules differ from the training manifest: "
            f"{sorted(target_modules)}"
        )

    probe_layers = [int(layer) for layer in info["layers"]]
    lora_layers = [int(layer) for layer in info["lora_layers"]]
    configured_layers = [
        int(layer)
        for layer in adapter_config.get("layers_to_transform", lora_layers)
    ]
    if configured_layers != lora_layers:
        raise ValueError(f"{spec['label']} manifest and adapter layer lists differ")
    if not probe_layers or not lora_layers:
        raise ValueError(f"{spec['label']} has empty probe or LoRA layer metadata")
    if max(lora_layers) >= int(spec["num_hidden_layers"]):
        raise ValueError(f"{spec['label']} adapted layer exceeds model depth")

    optimization = info.get("optimization", {})
    return {
        **spec,
        "model_key": model_key,
        "info_path": info_path,
        "probes_path": probes_path,
        "adapter_dir": adapter_dir,
        "adapter_config_path": adapter_config_path,
        "adapter_weights_path": adapter_weights_path,
        "model_path": _usable_model_path(spec, info),
        "model_repo_id": info.get("model_repo_id", spec["repo_id"]),
        "probe_layers": probe_layers,
        "lora_layers": lora_layers,
        "max_length": int(info.get("max_length", info.get("base_max_length", 512))),
        "manifest_epsilon": float(optimization.get("epsilon", DEFAULT_EPSILON)),
        "attack_learning_rate": float(optimization.get("adversary_lr", 1e-3)),
        "manifest_pgd_iterations": int(
            info.get("pgd_iterations", DEFAULT_PGD_ITERATIONS)
        ),
        "artifact_format_version": info.get("artifact_format_version"),
        "dataset_fingerprints": info.get("dataset_fingerprints", {}),
        "model_config_sha256": info.get("model_config_sha256"),
        "recorded_artifacts": info.get("artifacts", {}),
        "info_sha256": _sha256_file(info_path),
    }


def _load_tokenizer(config: Mapping[str, Any]):
    tokenizer = AutoTokenizer.from_pretrained(config["model_path"])
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _load_probes(
    config: Mapping[str, Any],
    *,
    device: torch.device,
    model_dtype: torch.dtype,
) -> dict[int, nn.Module]:
    probe_dtype = model_dtype if device.type in {"cuda", "mps"} else torch.float32
    probes = load_probe_state_dicts(
        config["probes_path"],
        map_location="cpu",
        device=device,
        dtype=probe_dtype,
    )
    probes = {int(layer): probe for layer, probe in probes.items()}
    if set(probes) != set(config["probe_layers"]):
        raise ValueError(f"{config['label']} probe layers do not match the manifest")
    for layer, probe in probes.items():
        width = int(probe.linear.weight.shape[-1])
        if width != int(config["hidden_size"]):
            raise ValueError(
                f"Probe at layer {layer} has width {width}, "
                f"expected {config['hidden_size']}"
            )
        probe.eval()
        probe.requires_grad_(False)
    return probes


def _compatible_lora_config(adapter_dir: Path) -> LoraConfig:
    raw_config = _read_json(adapter_dir / "adapter_config.json")
    supported = set(inspect.signature(LoraConfig).parameters)
    compatible = {key: value for key, value in raw_config.items() if key in supported}
    dropped = sorted(set(raw_config) - supported)
    if dropped:
        print(f"PEFT compatibility: ignoring unsupported config fields: {dropped}")
    return LoraConfig(**compatible)


def _load_adapted_model(
    config: Mapping[str, Any],
    *,
    device: torch.device,
    model_dtype: torch.dtype,
):
    base = AutoModelForCausalLM.from_pretrained(
        config["model_path"],
        torch_dtype=model_dtype,
        low_cpu_mem_usage=True,
    ).to(device)
    base.eval()
    base.requires_grad_(False)
    model = PeftModel.from_pretrained(
        base,
        config["adapter_dir"],
        config=_compatible_lora_config(config["adapter_dir"]),
    )
    model.eval()
    model.requires_grad_(False)
    model.config.use_cache = False
    if int(model.config.hidden_size) != int(config["hidden_size"]):
        raise ValueError(
            f"Loaded {config['label']} hidden size is {model.config.hidden_size}, "
            f"expected {config['hidden_size']}"
        )
    return model


def _model_layers_module(model: nn.Module) -> str:
    return (
        "base_model.model.model.layers"
        if hasattr(model, "peft_config")
        else "model.layers"
    )


def _cached_arrow_path(split: str) -> Path | None:
    cache_root = Path(
        os.environ.get(
            "HF_DATASETS_CACHE",
            Path.home() / ".cache" / "huggingface" / "datasets",
        )
    )
    dataset_root = cache_root / "Mechanistic-Anomaly-Detection___llama3-jailbreaks"
    matches = list(dataset_root.glob(f"**/llama3-jailbreaks-{split}.arrow"))
    return max(matches, key=lambda path: path.stat().st_mtime) if matches else None


def _load_dataset():
    cached = {split: _cached_arrow_path(split) for split in REQUIRED_SPLITS}
    if all(cached.values()):
        dataset = DatasetDict(
            {
                split: Dataset.from_file(str(path))
                for split, path in cached.items()
                if path is not None
            }
        )
    else:
        dataset = load_dataset(DATASET_NAME)
    for split in REQUIRED_SPLITS:
        if split not in dataset:
            raise KeyError(f"Required dataset split {split!r} is unavailable")
        if not {"prompt", "completion"}.issubset(dataset[split].column_names):
            raise ValueError(f"Split {split!r} lacks prompt/completion columns")
    return dataset


def _make_examples(
    dataset,
    split: str,
    tokenizer,
    *,
    start: int,
    n_examples: int,
) -> list[dict[str, Any]]:
    stop = int(start) + int(n_examples)
    if start < 0 or stop > len(dataset[split]):
        raise IndexError(
            f"Requested rows [{start}:{stop}] from split {split!r} "
            f"of length {len(dataset[split])}"
        )
    rows = dataset[split].select(range(start, stop))
    return [
        {
            "pair_id": pair_id,
            "text": format_dataset_chat(
                tokenizer,
                row["prompt"],
                row["completion"],
            ),
        }
        for pair_id, row in enumerate(rows, start=start)
    ]


def _build_batch(
    examples: Sequence[Mapping[str, Any]],
    tokenizer,
    config: Mapping[str, Any],
    *,
    device: torch.device,
) -> dict[str, Any]:
    encoded = tokenizer(
        [example["text"] for example in examples],
        padding=True,
        truncation=True,
        max_length=int(config["max_length"]),
        add_special_tokens=False,
        return_tensors="pt",
    )
    input_ids = encoded.input_ids.to(device)
    attention_mask = encoded.attention_mask.to(device)
    token_ranges = get_token_ranges("generation", tokenizer)
    prompt_mask = get_valid_token_mask(
        input_ids,
        token_ranges["only_choose_prompt_tokens_between"],
    ).to(device).bool()
    target_mask = get_valid_token_mask(
        input_ids,
        token_ranges["only_return_on_tokens_between"],
    ).to(device).bool()
    probe_mask = get_valid_token_mask(
        input_ids,
        token_ranges["only_probe_tokens_between"],
    ).to(device).bool()

    valid = attention_mask.bool()
    masks = {"prompt": prompt_mask, "target": target_mask, "probe": probe_mask}
    for name, mask in masks.items():
        if mask.shape != input_ids.shape or not torch.all(mask.sum(dim=1) > 0):
            raise ValueError(
                f"Invalid {name} mask; increase max_length or inspect chat formatting"
            )
        if torch.any(mask & ~valid):
            raise ValueError(f"{name} mask includes padding tokens")
    if torch.any(prompt_mask & target_mask):
        raise ValueError("Prompt and target masks overlap")

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "prompt_mask": prompt_mask,
        "target_mask": target_mask,
        "probe_mask": probe_mask,
    }


def _iter_batches(
    examples: Sequence[Mapping[str, Any]],
    batch_size: int,
) -> Iterable[Sequence[Mapping[str, Any]]]:
    for start in range(0, len(examples), int(batch_size)):
        yield examples[start : start + int(batch_size)]


def _token_hash(input_ids: torch.Tensor, attention_mask: torch.Tensor) -> str:
    ids = (
        input_ids[attention_mask.bool()]
        .detach()
        .cpu()
        .to(torch.int64)
        .contiguous()
    )
    return hashlib.sha256(ids.numpy().tobytes()).hexdigest()


def _delta_fingerprint(
    pair_id: int,
    positions: torch.Tensor,
    deltas: torch.Tensor,
) -> str:
    digest = hashlib.sha256()
    digest.update(str(int(pair_id)).encode())
    digest.update(
        positions.detach().cpu().to(torch.int64).contiguous().numpy().tobytes()
    )
    digest.update(
        deltas.detach().cpu().float().contiguous().numpy().tobytes()
    )
    return digest.hexdigest()[:16]


def _as_float(value: Any, default: float = math.nan) -> float:
    if value is None:
        return float(default)
    if torch.is_tensor(value):
        value = value.detach().cpu().item()
    return float(value)


def _train_bank(
    *,
    model_key: str,
    population: str,
    attack_kind: str,
    model: nn.Module,
    tokenizer,
    probes: Mapping[int, nn.Module],
    dataset,
    config: Mapping[str, Any],
    device: torch.device,
    start: int,
    n_examples: int,
    iterations: int,
    epsilon: float,
    learning_rate: float,
    seed: int,
) -> PGDAttackBank:
    if population not in POPULATION_ORDER:
        raise ValueError(f"Unknown population {population!r}")
    if attack_kind not in ATTACK_ORDER:
        raise ValueError(f"Unknown attack kind {attack_kind!r}")

    split = HARMFUL_SPLIT if population == "harmful" else BENIGN_SPLIT
    examples = _make_examples(
        dataset,
        split,
        tokenizer,
        start=start,
        n_examples=n_examples,
    )
    records: dict[int, PGDAttackRecord] = {}
    history_rows: list[dict[str, Any]] = []
    probe_loss_coef = float(ATTACK_SPECS[attack_kind]["probe_loss_coef"])
    attack_probes = probes if probe_loss_coef else None

    try:
        for batch_index, example_batch in enumerate(
            _iter_batches(examples, PRIMARY_PGD_BATCH_SIZE)
        ):
            batch_seed = int(seed) + batch_index
            torch.manual_seed(batch_seed)
            np.random.seed(batch_seed)
            random.seed(batch_seed)
            batch = _build_batch(
                example_batch,
                tokenizer,
                config,
                device=device,
            )
            loss_history, wrappers = train_attack(
                adv_tokens=batch["input_ids"],
                prompt_mask=batch["prompt_mask"],
                target_mask=batch["target_mask"],
                model=model,
                tokenizer=tokenizer,
                model_layers_module=_model_layers_module(model),
                layer=["embedding"],
                epsilon=float(epsilon),
                learning_rate=float(learning_rate),
                pgd_iterations=int(iterations),
                probes=attack_probes,
                probe_mask=batch["probe_mask"],
                probe_loss_coef=probe_loss_coef,
                towards_loss_coef=1.0,
                return_loss_over_time=True,
                device=device,
                clip_grad=1.0,
                adversary_type="pgd",
                verbose=False,
                attention_mask=batch["attention_mask"],
            )
            if len(wrappers) != 1 or not hasattr(wrappers[0], "hook_fn"):
                raise AssertionError("Repository PGD did not return one embedding hook")
            if not loss_history:
                raise AssertionError("Repository PGD returned an empty loss history")

            adversary = wrappers[0].hook_fn
            attack = adversary.attack.detach().float().cpu()
            final_losses = dict(loss_history[-1])

            # Confirm that the portable record representation reproduces the live hook.
            with torch.inference_mode():
                live_last_logits = (
                    model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        use_cache=False,
                    )
                    .logits[:, -1]
                    .detach()
                    .float()
                    .cpu()
                )
            clear_hooks(model)
            replay_deltas = torch.zeros_like(attack)
            replay_mask_cpu = batch["prompt_mask"].detach().cpu().bool()
            replay_deltas[replay_mask_cpu] = attack[replay_mask_cpu]
            parent = _model_layers_module(model).replace(".layers", "")
            add_hooks(
                model,
                create_adversary=lambda _: FixedBatchPromptAdversary(
                    replay_deltas.to(device),
                    batch["prompt_mask"],
                ),
                adversary_locations=[(parent, "embed_tokens")],
            )
            with torch.inference_mode():
                replay_last_logits = (
                    model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        use_cache=False,
                    )
                    .logits[:, -1]
                    .detach()
                    .float()
                    .cpu()
                )
            replay_error = (
                live_last_logits - replay_last_logits
            ).abs().max().item()
            if replay_error > REPLAY_LOGIT_TOLERANCE:
                raise AssertionError(
                    "Live/replayed repository PGD mismatch: "
                    f"max logit error={replay_error:.3e}"
                )
            clear_hooks(model)

            for step, step_losses in enumerate(loss_history):
                history_rows.append(
                    {
                        "population": population,
                        "attack_kind": attack_kind,
                        "batch_index": batch_index,
                        "step": step,
                        "toward_loss": _as_float(step_losses.get("toward")),
                        "probe_loss": _as_float(step_losses.get("probe")),
                        "total_loss": _as_float(step_losses.get("total")),
                        "iterations": int(iterations),
                    }
                )

            for row_index, example in enumerate(example_batch):
                positions = torch.where(
                    batch["prompt_mask"][row_index].detach().cpu()
                )[0].to(torch.int64)
                deltas = attack[row_index, positions].clone().float().contiguous()
                norms = deltas.norm(dim=-1)
                if positions.numel() == 0:
                    raise AssertionError("PGD record has no prompt positions")
                if not torch.isfinite(deltas).all():
                    raise FloatingPointError("PGD record contains non-finite deltas")
                if norms.max().item() > float(epsilon) + 1e-4:
                    raise AssertionError("Repository PGD exceeded epsilon")

                pair_id = int(example["pair_id"])
                record = PGDAttackRecord(
                    model_key=model_key,
                    population=population,
                    attack_kind=attack_kind,
                    pair_id=pair_id,
                    dataset_split=split,
                    input_token_hash=_token_hash(
                        batch["input_ids"][row_index],
                        batch["attention_mask"][row_index],
                    ),
                    prompt_positions=positions.clone().contiguous(),
                    prompt_deltas=deltas,
                    sequence_length=int(
                        batch["attention_mask"][row_index].sum().item()
                    ),
                    epsilon=float(epsilon),
                    learning_rate=float(learning_rate),
                    iterations=int(iterations),
                    final_toward_loss=_as_float(final_losses.get("toward")),
                    final_probe_loss=_as_float(final_losses.get("probe")),
                    final_total_loss=_as_float(final_losses.get("total")),
                    attack_norm_mean=float(norms.mean().item()),
                    attack_norm_max=float(norms.max().item()),
                    fingerprint=_delta_fingerprint(pair_id, positions, deltas),
                )
                if pair_id in records:
                    raise AssertionError(f"Duplicate PGD record for pair_id={pair_id}")
                records[pair_id] = record

            clear_hooks(model)
            model.zero_grad(set_to_none=True)
            _empty_cache(device)
    finally:
        clear_hooks(model)
        model.zero_grad(set_to_none=True)
        _empty_cache(device)

    expected_pair_ids = {int(example["pair_id"]) for example in examples}
    if set(records) != expected_pair_ids:
        raise AssertionError("PGD bank record coverage is incomplete")

    bank = PGDAttackBank(
        model_key=model_key,
        model_label=str(config["label"]),
        population=population,
        attack_kind=attack_kind,
        dataset_split=split,
        start=int(start),
        n_examples=int(n_examples),
        batch_size=PRIMARY_PGD_BATCH_SIZE,
        iterations=int(iterations),
        epsilon=float(epsilon),
        learning_rate=float(learning_rate),
        seed=int(seed),
        records=records,
        losses=pd.DataFrame(history_rows, columns=LOSS_COLUMNS),
    )
    _validate_bank(
        bank,
        hidden_size=int(config["hidden_size"]),
        expected_start=start,
        expected_n_examples=n_examples,
        expected_iterations=iterations,
        expected_epsilon=epsilon,
        expected_learning_rate=learning_rate,
        expected_seed=seed,
    )
    return bank


def train_model_banks(
    model_key: str,
    *,
    dataset,
    device: torch.device,
    epsilon: float,
    n_pairs: int,
    pgd_iterations: int,
    attack_seed: int,
    start: int = 0,
) -> tuple[dict[str, dict[str, PGDAttackBank]], dict[str, Any]]:
    """Train all four independent PGD banks for one model."""

    config = get_model_config(model_key)
    model_dtype = torch.float16 if device.type in {"cuda", "mps"} else torch.float32
    tokenizer = _load_tokenizer(config)
    probes = _load_probes(config, device=device, model_dtype=model_dtype)
    model = _load_adapted_model(
        config,
        device=device,
        model_dtype=model_dtype,
    )
    banks: dict[str, dict[str, PGDAttackBank]] = {
        population: {} for population in POPULATION_ORDER
    }
    try:
        for population in POPULATION_ORDER:
            for attack_kind in ATTACK_ORDER:
                seed = bank_seed(attack_seed, population, attack_kind)
                print(
                    f"  Training {model_key}/{population}/{attack_kind} "
                    f"(seed={seed})"
                )
                banks[population][attack_kind] = _train_bank(
                    model_key=model_key,
                    population=population,
                    attack_kind=attack_kind,
                    model=model,
                    tokenizer=tokenizer,
                    probes=probes,
                    dataset=dataset,
                    config=config,
                    device=device,
                    start=start,
                    n_examples=n_pairs,
                    iterations=pgd_iterations,
                    epsilon=epsilon,
                    learning_rate=float(config["attack_learning_rate"]),
                    seed=seed,
                )
    finally:
        clear_hooks(model)
        del model
        _empty_cache(device)

    provenance = {
        "model_key": model_key,
        "model_label": config["label"],
        "model_path": config["model_path"],
        "model_repo_id": config["model_repo_id"],
        "hidden_size": int(config["hidden_size"]),
        "max_length": int(config["max_length"]),
        "probe_layers": config["probe_layers"],
        "lora_layers": config["lora_layers"],
        "artifact_format_version": config["artifact_format_version"],
        "artifact_info_sha256": config["info_sha256"],
        "model_config_sha256": config["model_config_sha256"],
        "recorded_artifacts": config["recorded_artifacts"],
        "manifest_epsilon": config["manifest_epsilon"],
        "manifest_pgd_iterations": config["manifest_pgd_iterations"],
        "attack_learning_rate": config["attack_learning_rate"],
    }
    return banks, provenance


def _validate_record(
    record: PGDAttackRecord,
    *,
    bank: PGDAttackBank,
    hidden_size: int,
) -> None:
    expected_metadata = {
        "model_key": bank.model_key,
        "population": bank.population,
        "attack_kind": bank.attack_kind,
        "dataset_split": bank.dataset_split,
        "epsilon": float(bank.epsilon),
        "learning_rate": float(bank.learning_rate),
        "iterations": int(bank.iterations),
    }
    for name, expected in expected_metadata.items():
        observed = getattr(record, name)
        if isinstance(expected, float):
            matches = math.isclose(
                float(observed),
                expected,
                rel_tol=0,
                abs_tol=1e-12,
            )
        else:
            matches = observed == expected
        if not matches:
            raise ValueError(
                f"PGD record {record.pair_id} has {name}={observed!r}, "
                f"expected {expected!r}"
            )

    if int(record.pair_id) not in bank.records:
        raise ValueError(f"PGD record {record.pair_id} is not indexed by its pair ID")
    positions = record.prompt_positions
    deltas = record.prompt_deltas
    if not torch.is_tensor(positions) or positions.ndim != 1:
        raise TypeError("prompt_positions must be a rank-one tensor")
    if positions.dtype not in {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    }:
        raise TypeError("prompt_positions must use an integer dtype")
    if not torch.is_tensor(deltas) or deltas.ndim != 2:
        raise TypeError("prompt_deltas must be a rank-two tensor")

    positions_cpu = positions.detach().cpu().to(torch.int64)
    deltas_cpu = deltas.detach().cpu().float()
    expected_shape = (positions_cpu.numel(), int(hidden_size))
    if tuple(deltas_cpu.shape) != expected_shape:
        raise ValueError(
            f"PGD record {record.pair_id} delta shape {tuple(deltas_cpu.shape)} "
            f"does not match {expected_shape}"
        )
    if positions_cpu.numel() == 0:
        raise ValueError(f"PGD record {record.pair_id} has no prompt positions")
    if int(record.sequence_length) <= 0:
        raise ValueError(f"PGD record {record.pair_id} has invalid sequence length")
    if torch.any(positions_cpu < 0) or torch.any(
        positions_cpu >= int(record.sequence_length)
    ):
        raise ValueError(f"PGD record {record.pair_id} has out-of-range positions")
    if positions_cpu.numel() > 1 and not torch.all(
        positions_cpu[1:] > positions_cpu[:-1]
    ):
        raise ValueError(f"PGD record {record.pair_id} positions are not ordered")
    if not torch.isfinite(deltas_cpu).all():
        raise FloatingPointError(f"PGD record {record.pair_id} has non-finite deltas")

    norms = deltas_cpu.norm(dim=-1)
    if norms.max().item() > float(bank.epsilon) + 1e-4:
        raise ValueError(f"PGD record {record.pair_id} exceeds epsilon")
    if not math.isclose(
        float(record.attack_norm_mean),
        float(norms.mean().item()),
        rel_tol=1e-6,
        abs_tol=1e-6,
    ):
        raise ValueError(f"PGD record {record.pair_id} has an invalid mean norm")
    if not math.isclose(
        float(record.attack_norm_max),
        float(norms.max().item()),
        rel_tol=1e-6,
        abs_tol=1e-6,
    ):
        raise ValueError(f"PGD record {record.pair_id} has an invalid max norm")
    if record.fingerprint != _delta_fingerprint(
        record.pair_id,
        positions_cpu,
        deltas_cpu,
    ):
        raise ValueError(f"PGD record {record.pair_id} fingerprint mismatch")
    if (
        not isinstance(record.input_token_hash, str)
        or len(record.input_token_hash) != 64
        or any(character not in "0123456789abcdef" for character in record.input_token_hash)
    ):
        raise ValueError(f"PGD record {record.pair_id} has an invalid token hash")


def _validate_bank(
    bank: PGDAttackBank,
    *,
    hidden_size: int,
    expected_start: int | None = None,
    expected_n_examples: int | None = None,
    expected_iterations: int | None = None,
    expected_epsilon: float | None = None,
    expected_learning_rate: float | None = None,
    expected_seed: int | None = None,
) -> None:
    if bank.population not in POPULATION_ORDER:
        raise ValueError(f"Invalid bank population {bank.population!r}")
    if bank.attack_kind not in ATTACK_ORDER:
        raise ValueError(f"Invalid bank attack kind {bank.attack_kind!r}")
    if bank.batch_size != PRIMARY_PGD_BATCH_SIZE:
        raise ValueError("PGD bank batch size is not the required value 2")

    exact_expectations = {
        "start": expected_start,
        "n_examples": expected_n_examples,
        "iterations": expected_iterations,
        "seed": expected_seed,
    }
    for name, expected in exact_expectations.items():
        if expected is not None and int(getattr(bank, name)) != int(expected):
            raise ValueError(
                f"PGD bank {name}={getattr(bank, name)!r}, expected {expected!r}"
            )
    float_expectations = {
        "epsilon": expected_epsilon,
        "learning_rate": expected_learning_rate,
    }
    for name, expected in float_expectations.items():
        if expected is not None and not math.isclose(
            float(getattr(bank, name)),
            float(expected),
            rel_tol=0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                f"PGD bank {name}={getattr(bank, name)!r}, expected {expected!r}"
            )

    expected_pair_ids = set(
        range(int(bank.start), int(bank.start) + int(bank.n_examples))
    )
    if set(bank.records) != expected_pair_ids:
        raise ValueError(
            f"PGD bank has pair IDs {sorted(bank.records)}, "
            f"expected {sorted(expected_pair_ids)}"
        )
    for pair_id in sorted(bank.records):
        record = bank.records[pair_id]
        if int(record.pair_id) != pair_id:
            raise ValueError(f"PGD record key {pair_id} does not match its pair_id")
        _validate_record(record, bank=bank, hidden_size=hidden_size)

    expected_fingerprint = hashlib.sha256(
        "".join(bank.records[pair_id].fingerprint for pair_id in sorted(bank.records)).encode()
    ).hexdigest()[:16]
    if bank.fingerprint != expected_fingerprint:
        raise ValueError("PGD bank fingerprint mismatch")

    if not isinstance(bank.losses, pd.DataFrame):
        raise TypeError("PGD bank losses must be a pandas DataFrame")
    missing_columns = set(LOSS_COLUMNS) - set(bank.losses.columns)
    if missing_columns:
        raise ValueError(f"PGD loss history is missing {sorted(missing_columns)}")
    expected_batches = math.ceil(bank.n_examples / bank.batch_size)
    expected_loss_rows = expected_batches * bank.iterations
    if len(bank.losses) != expected_loss_rows:
        raise ValueError(
            f"PGD loss history has {len(bank.losses)} rows, "
            f"expected {expected_loss_rows}"
        )
    for batch_index in range(expected_batches):
        batch_rows = bank.losses[bank.losses["batch_index"] == batch_index]
        if list(batch_rows["step"]) != list(range(bank.iterations)):
            raise ValueError(
                f"PGD loss steps are incomplete for batch {batch_index}"
            )
    if not (
        (bank.losses["population"] == bank.population).all()
        and (bank.losses["attack_kind"] == bank.attack_kind).all()
        and (bank.losses["iterations"] == bank.iterations).all()
    ):
        raise ValueError("PGD loss history metadata does not match its bank")


def _tensor_key(
    population: str,
    attack_kind: str,
    pair_id: int,
    tensor_name: str,
) -> str:
    return f"{population}.{attack_kind}.pair_{int(pair_id)}.{tensor_name}"


def _record_metadata(record: PGDAttackRecord) -> dict[str, Any]:
    # Build this explicitly: dataclasses.asdict() deep-copies tensors and would
    # briefly duplicate the largest part of a trained bank in memory.
    return {
        "model_key": record.model_key,
        "population": record.population,
        "attack_kind": record.attack_kind,
        "pair_id": record.pair_id,
        "dataset_split": record.dataset_split,
        "input_token_hash": record.input_token_hash,
        "sequence_length": record.sequence_length,
        "epsilon": record.epsilon,
        "learning_rate": record.learning_rate,
        "iterations": record.iterations,
        "final_toward_loss": record.final_toward_loss,
        "final_probe_loss": record.final_probe_loss,
        "final_total_loss": record.final_total_loss,
        "attack_norm_mean": record.attack_norm_mean,
        "attack_norm_max": record.attack_norm_max,
        "fingerprint": record.fingerprint,
        "prompt_positions_key": _tensor_key(
            record.population,
            record.attack_kind,
            record.pair_id,
            "prompt_positions",
        ),
        "prompt_deltas_key": _tensor_key(
            record.population,
            record.attack_kind,
            record.pair_id,
            "prompt_deltas",
        ),
    }


def _save_model_bundle(
    model_dir: Path,
    *,
    model_key: str,
    banks: Mapping[str, Mapping[str, PGDAttackBank]],
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    model_dir.mkdir(parents=True, exist_ok=False)
    tensors: dict[str, torch.Tensor] = {}
    loss_rows: list[dict[str, Any]] = []
    bank_metadata: list[dict[str, Any]] = []

    for population in POPULATION_ORDER:
        if set(banks.get(population, {})) != set(ATTACK_ORDER):
            raise ValueError(f"{model_key}/{population} has incomplete PGD banks")
        for attack_kind in ATTACK_ORDER:
            bank = banks[population][attack_kind]
            _validate_bank(
                bank,
                hidden_size=int(provenance["hidden_size"]),
            )
            metadata = {
                "model_key": bank.model_key,
                "model_label": bank.model_label,
                "population": bank.population,
                "attack_kind": bank.attack_kind,
                "dataset_split": bank.dataset_split,
                "start": bank.start,
                "n_examples": bank.n_examples,
                "batch_size": bank.batch_size,
                "iterations": bank.iterations,
                "epsilon": bank.epsilon,
                "learning_rate": bank.learning_rate,
                "seed": bank.seed,
                "fingerprint": bank.fingerprint,
                "records": [],
            }
            for pair_id in sorted(bank.records):
                record = bank.records[pair_id]
                record_info = _record_metadata(record)
                positions_key = record_info["prompt_positions_key"]
                deltas_key = record_info["prompt_deltas_key"]
                tensors[positions_key] = (
                    record.prompt_positions.detach().cpu().to(torch.int64).contiguous()
                )
                tensors[deltas_key] = (
                    record.prompt_deltas.detach().cpu().float().contiguous()
                )
                metadata["records"].append(record_info)
            bank_metadata.append(metadata)
            loss_rows.extend(bank.losses.to_dict(orient="records"))

    metadata_path = model_dir / "metadata.json"
    tensors_path = model_dir / "banks.safetensors"
    losses_path = model_dir / "losses.jsonl"
    _write_json_atomic(
        metadata_path,
        {
            "schema_name": SCHEMA_NAME,
            "schema_version": SCHEMA_VERSION,
            "model_key": model_key,
            "provenance": provenance,
            "banks": bank_metadata,
        },
    )
    save_safetensors(
        tensors,
        str(tensors_path),
        metadata={
            "schema_name": SCHEMA_NAME,
            "schema_version": str(SCHEMA_VERSION),
            "model_key": model_key,
        },
    )
    _write_jsonl(losses_path, loss_rows)

    files = {}
    for path in (metadata_path, tensors_path, losses_path):
        files[path.name] = {
            "size_bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
    return {
        "directory": model_key.lower(),
        "hidden_size": int(provenance["hidden_size"]),
        "files": files,
    }


def _optional_float(value: Any) -> float:
    return math.nan if value is None else float(value)


def _load_model_bundle(
    model_dir: Path,
    *,
    expected_model_key: str,
    expected_files: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[dict[str, dict[str, PGDAttackBank]], dict[str, Any]]:
    required_paths = {
        "metadata.json": model_dir / "metadata.json",
        "banks.safetensors": model_dir / "banks.safetensors",
        "losses.jsonl": model_dir / "losses.jsonl",
    }
    missing = [str(path) for path in required_paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("PGD model bundle is incomplete:\n" + "\n".join(missing))
    if expected_files is not None:
        for filename, path in required_paths.items():
            expected = expected_files.get(filename)
            if expected is None:
                raise ValueError(f"Manifest has no checksum for {path}")
            observed_size = path.stat().st_size
            if observed_size != int(expected["size_bytes"]):
                raise ValueError(f"Size mismatch for {path}")
            if _sha256_file(path) != expected["sha256"]:
                raise ValueError(f"SHA-256 mismatch for {path}")

    metadata = _read_json(required_paths["metadata.json"])
    if metadata.get("schema_name") != SCHEMA_NAME:
        raise ValueError(f"Unknown PGD schema in {required_paths['metadata.json']}")
    if int(metadata.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported PGD schema version {metadata.get('schema_version')!r}"
        )
    if metadata.get("model_key") != expected_model_key:
        raise ValueError(
            f"Model bundle contains {metadata.get('model_key')!r}, "
            f"expected {expected_model_key!r}"
        )

    provenance = metadata.get("provenance")
    if not isinstance(provenance, dict):
        raise TypeError("PGD model provenance must be a JSON object")
    hidden_size = int(provenance["hidden_size"])
    tensors = load_safetensors(str(required_paths["banks.safetensors"]), device="cpu")
    all_loss_rows = _read_jsonl(required_paths["losses.jsonl"])
    banks: dict[str, dict[str, PGDAttackBank]] = {
        population: {} for population in POPULATION_ORDER
    }
    consumed_tensor_keys: set[str] = set()

    raw_banks = metadata.get("banks")
    if not isinstance(raw_banks, list):
        raise TypeError("PGD model bank metadata must be a list")
    for bank_info in raw_banks:
        population = str(bank_info["population"])
        attack_kind = str(bank_info["attack_kind"])
        if population not in POPULATION_ORDER or attack_kind not in ATTACK_ORDER:
            raise ValueError(
                f"Unexpected bank {population!r}/{attack_kind!r} in {model_dir}"
            )
        if attack_kind in banks[population]:
            raise ValueError(f"Duplicate bank {population}/{attack_kind}")

        records: dict[int, PGDAttackRecord] = {}
        for record_info in bank_info["records"]:
            pair_id = int(record_info["pair_id"])
            positions_key = record_info["prompt_positions_key"]
            deltas_key = record_info["prompt_deltas_key"]
            if positions_key not in tensors or deltas_key not in tensors:
                raise KeyError(f"Missing tensors for PGD record {pair_id}")
            consumed_tensor_keys.update((positions_key, deltas_key))
            records[pair_id] = PGDAttackRecord(
                model_key=str(record_info["model_key"]),
                population=str(record_info["population"]),
                attack_kind=str(record_info["attack_kind"]),
                pair_id=pair_id,
                dataset_split=str(record_info["dataset_split"]),
                input_token_hash=str(record_info["input_token_hash"]),
                prompt_positions=tensors[positions_key].to(torch.int64).contiguous(),
                prompt_deltas=tensors[deltas_key].float().contiguous(),
                sequence_length=int(record_info["sequence_length"]),
                epsilon=float(record_info["epsilon"]),
                learning_rate=float(record_info["learning_rate"]),
                iterations=int(record_info["iterations"]),
                final_toward_loss=_optional_float(
                    record_info.get("final_toward_loss")
                ),
                final_probe_loss=_optional_float(record_info.get("final_probe_loss")),
                final_total_loss=_optional_float(record_info.get("final_total_loss")),
                attack_norm_mean=float(record_info["attack_norm_mean"]),
                attack_norm_max=float(record_info["attack_norm_max"]),
                fingerprint=str(record_info["fingerprint"]),
            )

        loss_rows = [
            {
                **row,
                "toward_loss": _optional_float(row.get("toward_loss")),
                "probe_loss": _optional_float(row.get("probe_loss")),
                "total_loss": _optional_float(row.get("total_loss")),
            }
            for row in all_loss_rows
            if row.get("population") == population
            and row.get("attack_kind") == attack_kind
        ]
        bank = PGDAttackBank(
            model_key=str(bank_info["model_key"]),
            model_label=str(bank_info["model_label"]),
            population=population,
            attack_kind=attack_kind,
            dataset_split=str(bank_info["dataset_split"]),
            start=int(bank_info["start"]),
            n_examples=int(bank_info["n_examples"]),
            batch_size=int(bank_info["batch_size"]),
            iterations=int(bank_info["iterations"]),
            epsilon=float(bank_info["epsilon"]),
            learning_rate=float(bank_info["learning_rate"]),
            seed=int(bank_info["seed"]),
            records=records,
            losses=pd.DataFrame(loss_rows, columns=LOSS_COLUMNS),
        )
        if bank.fingerprint != bank_info["fingerprint"]:
            raise ValueError(f"Stored bank fingerprint mismatch for {population}/{attack_kind}")
        _validate_bank(bank, hidden_size=hidden_size)
        banks[population][attack_kind] = bank

    for population in POPULATION_ORDER:
        if set(banks[population]) != set(ATTACK_ORDER):
            raise ValueError(f"Model bundle has incomplete {population} banks")
    if consumed_tensor_keys != set(tensors):
        unused = sorted(set(tensors) - consumed_tensor_keys)
        missing = sorted(consumed_tensor_keys - set(tensors))
        raise ValueError(
            f"Tensor index mismatch; unused={unused[:5]}, missing={missing[:5]}"
        )
    expected_loss_rows = sum(
        len(banks[population][attack_kind].losses)
        for population in POPULATION_ORDER
        for attack_kind in ATTACK_ORDER
    )
    if expected_loss_rows != len(all_loss_rows):
        raise ValueError("Loss history contains unindexed or duplicate rows")
    return banks, provenance


def load_pgd_banks(
    bundle_dir: str | Path,
    *,
    epsilon: float | None = None,
    n_pairs: int | None = None,
    pgd_iterations: int | None = None,
    seed: int | None = None,
) -> dict[str, dict[str, dict[str, PGDAttackBank]]]:
    """Load and fully validate a completed portable PGD bundle."""

    bundle_dir = Path(bundle_dir)
    manifest_path = bundle_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"PGD bundle manifest not found: {manifest_path}")
    manifest = _read_json(manifest_path)
    if manifest.get("schema_name") != SCHEMA_NAME:
        raise ValueError(f"Unknown PGD bundle schema in {manifest_path}")
    if int(manifest.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported PGD schema version {manifest.get('schema_version')!r}"
        )
    if manifest.get("status") != "complete":
        raise ValueError(
            f"PGD bundle is not complete (status={manifest.get('status')!r})"
        )
    parameters = manifest.get("parameters")
    if not isinstance(parameters, dict):
        raise TypeError("PGD bundle parameters must be a JSON object")

    expected_values = {
        "epsilon": epsilon,
        "n_pairs": n_pairs,
        "pgd_iterations": pgd_iterations,
        "seed": seed,
    }
    for name, expected in expected_values.items():
        if expected is None:
            continue
        observed = parameters[name]
        if name == "epsilon":
            matches = math.isclose(
                float(observed),
                float(expected),
                rel_tol=0,
                abs_tol=1e-12,
            )
        else:
            matches = int(observed) == int(expected)
        if not matches:
            raise ValueError(
                f"PGD bundle {name}={observed!r}, expected {expected!r}"
            )

    models = manifest.get("models")
    if not isinstance(models, dict) or set(models) != set(MODEL_ORDER):
        raise ValueError("PGD bundle must contain exactly the 1B and 3B models")
    banks_by_model = {}
    for model_key in MODEL_ORDER:
        model_info = models[model_key]
        model_dir = bundle_dir / model_info["directory"]
        banks, provenance = _load_model_bundle(
            model_dir,
            expected_model_key=model_key,
            expected_files=model_info["files"],
        )
        if int(provenance["hidden_size"]) != int(model_info["hidden_size"]):
            raise ValueError(f"Hidden-size mismatch in {model_key} manifest")
        for population in POPULATION_ORDER:
            for attack_kind in ATTACK_ORDER:
                bank = banks[population][attack_kind]
                _validate_bank(
                    bank,
                    hidden_size=int(provenance["hidden_size"]),
                    expected_start=int(parameters["start"]),
                    expected_n_examples=int(parameters["n_pairs"]),
                    expected_iterations=int(parameters["pgd_iterations"]),
                    expected_epsilon=float(parameters["epsilon"]),
                    expected_learning_rate=float(provenance["attack_learning_rate"]),
                    expected_seed=bank_seed(
                        int(parameters["attack_seed"]),
                        population,
                        attack_kind,
                    ),
                )
        banks_by_model[model_key] = banks
    return banks_by_model


def _initial_manifest(
    *,
    epsilon: float,
    n_pairs: int,
    pgd_iterations: int,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    return {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "status": "running",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "completed_at": None,
        "parameters": {
            "epsilon": float(epsilon),
            "n_pairs": int(n_pairs),
            "pgd_iterations": int(pgd_iterations),
            "seed": int(seed),
            "attack_seed": attack_seed_from_base(seed),
            "start": 0,
            "pgd_batch_size": PRIMARY_PGD_BATCH_SIZE,
        },
        "dataset": {
            "name": DATASET_NAME,
            "harmful_split": HARMFUL_SPLIT,
            "benign_split": BENIGN_SPLIT,
        },
        "runtime": {
            "device": str(device),
            "model_dtype": (
                "float16" if device.type in {"cuda", "mps"} else "float32"
            ),
            "python": sys.version.split()[0],
            "packages": _package_versions(),
        },
        "models": {},
    }


def _commit_model_directory(
    temporary_dir: Path,
    final_dir: Path,
) -> None:
    if final_dir.exists():
        raise FileExistsError(f"Model output already exists: {final_dir}")
    os.replace(temporary_dir, final_dir)


def generate_pgd_banks(
    *,
    epsilon: float = DEFAULT_EPSILON,
    n_pairs: int = DEFAULT_N_PAIRS,
    pgd_iterations: int = DEFAULT_PGD_ITERATIONS,
    seed: int = DEFAULT_SEED,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    device: str = "auto",
    force: bool = False,
) -> Path:
    """Train both model families, store them, and validate the saved bundle."""

    _validate_parameters(epsilon, n_pairs, pgd_iterations, seed)
    resolved_device = _resolve_device(device)
    output_dir = Path(output_dir)
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle_dir = output_dir / build_bundle_name(
        epsilon,
        n_pairs,
        pgd_iterations,
        seed,
    )

    if bundle_dir.exists():
        if not force:
            try:
                load_pgd_banks(
                    bundle_dir,
                    epsilon=epsilon,
                    n_pairs=n_pairs,
                    pgd_iterations=pgd_iterations,
                    seed=seed,
                )
            except Exception as error:
                raise RuntimeError(
                    f"Existing PGD output is incomplete or invalid: {bundle_dir}. "
                    "Inspect it or rerun with --force to replace this exact bundle."
                ) from error
            print(f"Reusing validated PGD banks: {bundle_dir}")
            return bundle_dir
        shutil.rmtree(bundle_dir)

    bundle_dir.mkdir(parents=False)
    manifest_path = bundle_dir / "manifest.json"
    manifest = _initial_manifest(
        epsilon=epsilon,
        n_pairs=n_pairs,
        pgd_iterations=pgd_iterations,
        seed=seed,
        device=resolved_device,
    )
    _write_json_atomic(manifest_path, manifest)
    attack_seed = int(manifest["parameters"]["attack_seed"])

    print(f"Device: {resolved_device}")
    print(f"Output: {bundle_dir}")
    print(
        f"Parameters: epsilon={epsilon:g}, n_pairs={n_pairs}, "
        f"pgd_iterations={pgd_iterations}, seed={seed}, attack_seed={attack_seed}"
    )

    try:
        dataset = _load_dataset()
        manifest["dataset"]["split_fingerprints"] = {
            split: getattr(dataset[split], "_fingerprint", None)
            for split in REQUIRED_SPLITS
        }
        _write_json_atomic(manifest_path, manifest)

        for model_key in MODEL_ORDER:
            print(f"Training PGD banks for {model_key}")
            _set_global_seed(attack_seed)
            banks, provenance = train_model_banks(
                model_key,
                dataset=dataset,
                device=resolved_device,
                epsilon=float(epsilon),
                n_pairs=int(n_pairs),
                pgd_iterations=int(pgd_iterations),
                attack_seed=attack_seed,
            )
            temporary_model_dir = bundle_dir / f".{model_key.lower()}.tmp-{os.getpid()}"
            if temporary_model_dir.exists():
                shutil.rmtree(temporary_model_dir)
            try:
                model_manifest = _save_model_bundle(
                    temporary_model_dir,
                    model_key=model_key,
                    banks=banks,
                    provenance=provenance,
                )
                # Release the in-memory tensors before reading the saved copy back.
                del banks
                _empty_cache(resolved_device)
                reloaded_banks, _ = _load_model_bundle(
                    temporary_model_dir,
                    expected_model_key=model_key,
                    expected_files=model_manifest["files"],
                )
                del reloaded_banks
                _commit_model_directory(
                    temporary_model_dir,
                    bundle_dir / model_manifest["directory"],
                )
            finally:
                if temporary_model_dir.exists():
                    shutil.rmtree(temporary_model_dir)

            manifest["models"][model_key] = model_manifest
            _write_json_atomic(manifest_path, manifest)
            _empty_cache(resolved_device)
            print(f"Saved and validated {model_key} banks")

        manifest["status"] = "complete"
        manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
        _write_json_atomic(manifest_path, manifest)
        load_pgd_banks(
            bundle_dir,
            epsilon=epsilon,
            n_pairs=n_pairs,
            pgd_iterations=pgd_iterations,
            seed=seed,
        )
        print(f"Saved reusable PGD banks: {bundle_dir}")
        return bundle_dir
    except Exception as error:
        manifest["status"] = "failed"
        manifest["failed_at"] = datetime.now(timezone.utc).isoformat()
        manifest["error"] = {
            "type": type(error).__name__,
            "message": str(error),
        }
        _write_json_atomic(manifest_path, manifest)
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse fixed-PGD optimization, device, and output options."""

    parser = argparse.ArgumentParser(
        description=(
            "Train portable PGD attack banks for the adapted Llama-3.2 1B and "
            "3B models."
        )
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=DEFAULT_EPSILON,
        help=f"Per-position PGD L2 bound (default: {DEFAULT_EPSILON:g}).",
    )
    parser.add_argument(
        "--n-pairs",
        type=int,
        default=DEFAULT_N_PAIRS,
        help=f"Number of harmful and benign examples (default: {DEFAULT_N_PAIRS}).",
    )
    parser.add_argument(
        "--pgd-iterations",
        "--iterations",
        dest="pgd_iterations",
        type=int,
        default=DEFAULT_PGD_ITERATIONS,
        help=(
            "Number of PGD optimization steps "
            f"(default: {DEFAULT_PGD_ITERATIONS})."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Base seed used to derive bank seeds (default: {DEFAULT_SEED}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Parent output directory (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device such as auto, cuda, cuda:0, mps, or cpu (default: auto).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace the exact parameter-matched output bundle if it exists.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Generate a validated portable bank bundle for the requested cell."""

    args = parse_args(argv)
    generate_pgd_banks(
        epsilon=args.epsilon,
        n_pairs=args.n_pairs,
        pgd_iterations=args.pgd_iterations,
        seed=args.seed,
        output_dir=args.output_dir,
        device=args.device,
        force=args.force,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
