"""Train an OAT-style Llama adapter together with activation probes.

This entry point formats the jailbreak dataset, trains linear or nonlinear
activation probes with an optional LoRA adapter, evaluates held-out probe AUC,
and saves portable weights plus provenance metadata.  It is included to show
how the published Hugging Face checkpoints were produced.  The released runs
co-trained LoRA during probe warmup, so they are OAT-style experiments rather
than exact reproductions of the source paper's training schedule.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
import random
import shutil
from typing import Literal

import numpy as np

# Third-party library imports
import torch
from datasets import load_dataset
from src.chat_formatting import (
    LLAMA32_CHAT_TEMPLATE_DATE,
    chat_template_token_overhead,
    format_dataset_chat,
)
from src.model_loading import load_model_encoder
from src.model_configs import (
    LLAMA32_1B,
    resolve_lora_layers,
    resolve_model_config,
    resolve_probe_layers,
)
from src.oat_artifacts import sha256_artifact
from src.probe_archs import LinearProbe, NonlinearProbe
from src.probe_training import (
    DEFAULT_LORA_TARGET_MODULES,
    evaluate_probe_auc,
    save_probe_state_dicts,
    save_probes,
    train_online_probe,
)
from src.utils import convert_to_serializable


DEFAULT_OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "probe_weights_comp_only"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train probes with configurable masking and probe types"
    )
    parser.add_argument(
        "--masking-type",
        type=str,
        choices=["instruction", "generation"],
        default="generation",
        help="Type of masking to use (instruction or generation)",
    )
    parser.add_argument(
        "--probe-type",
        type=str,
        choices=["linear", "nonlinear"],
        default="linear",
        help="Type of probe to use (linear or nonlinear)",
    )
    parser.add_argument(
        "--no-lora-probes",
        action="store_true",
        help="Disable LoRA probes (enabled by default)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=LLAMA32_1B,
        help="HF repo ID, supported alias, or local model directory",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="Mechanistic-Anomaly-Detection/llama3-jailbreaks",
        help="Jailbreak dataset to use",
    )
    parser.add_argument(
        "--layers",
        type=str,
        default=None,
        help="Comma-separated probe/block layers. Defaults are model-specific.",
    )
    parser.add_argument(
        "--lora-layers",
        type=str,
        default=None,
        help='Comma-separated LoRA block layers, or "all". Defaults to 0..max(probe layers).',
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--n-grad-accum", type=int, default=8)
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--pgd-iterations", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--disable-softprompt-eval",
        action="store_true",
        help="Skip the expensive soft-prompt benchmark during OAT training.",
    )
    parser.add_argument(
        "--run-softprompt-eval-every",
        type=int,
        default=256,
        help="Run soft-prompt benchmark every N adversarial steps. Use 0 to disable.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help='Torch device, e.g. "cuda", "mps", "cpu", or "auto".',
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for artifacts (defaults beside this training script).",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional artifact basename. Defaults to the legacy model/mask/probe name.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacement of an existing run with the same artifact basename.",
    )
    return parser.parse_args()


def create_linear_probe():
    return LinearProbe(encoder.model.config.hidden_size)


def create_nonlinear_probe():
    return NonlinearProbe(encoder.model.config.hidden_size, 64)


def get_probe_creator(probe_type: Literal["linear", "nonlinear"]):
    if probe_type == "linear":
        return lambda: LinearProbe(encoder.model.config.hidden_size)
    else:
        return lambda: NonlinearProbe(encoder.model.config.hidden_size, 64)


def _single_token_id(tokenizer, text):
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) != 1:
        raise ValueError(f"Expected {text!r} to tokenize to one token, got {ids}")
    return ids[0]


def sample_examples_from_datasets(
    datasets,
    proportions,
    tokenizer,
    total_examples=1000,
    only_prompts=False,
    seed=42,
):
    # This function samples examples from multiple datasets, ensuring that the final list has the desired proportions
    # of examples from each dataset. The final list is shuffled.

    # Ensure the proportions sum to 1
    if len(datasets) != len(proportions):
        raise ValueError("Number of datasets must match number of proportions")

    if abs(sum(proportions) - 1) > 1e-6:
        raise ValueError("Proportions must sum to 1")

    examples = []
    rng = np.random.RandomState(seed)
    for dataset, proportion in zip(datasets, proportions):
        n_samples = int(total_examples * proportion)

        if not len(dataset):
            raise ValueError("Cannot sample from an empty dataset")
        sampled_indices = rng.choice(len(dataset), size=n_samples, replace=True)
        sampled = dataset.select(sampled_indices)

        if only_prompts:
            examples.extend(
                [format_dataset_chat(tokenizer, item["prompt"]) for item in sampled]
            )
        else:
            examples.extend(
                [
                    format_dataset_chat(
                        tokenizer, item["prompt"], item["completion"]
                    )
                    for item in sampled
                ]
            )

    # Shuffle the final list to mix examples from different datasets
    random.Random(seed).shuffle(examples)

    return examples


def split_dataset(dataset, train_ratio=0.9, val_ratio=0.1, test_ratio=0.0):
    # Function to split dataset into train, validation, and test sets
    assert (
        abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    ), "Ratios must sum to 1"

    total_size = len(dataset)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    test_size = total_size - train_size - val_size

    train_set = dataset[:train_size]
    val_set = dataset[train_size : train_size + val_size]
    test_set = dataset[train_size + val_size :]

    return train_set, val_set, test_set


def _sha256_strings(strings):
    digest = hashlib.sha256()
    for value in strings:
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _model_config_sha256(config):
    canonical_config = config.to_dict()
    canonical_config.pop("_name_or_path", None)
    canonical_config.pop("transformers_version", None)
    return hashlib.sha256(
        json.dumps(canonical_config, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _special_token_id(tokenizer, token_name, fallback=None):
    token_id = tokenizer.convert_tokens_to_ids(token_name)
    if token_id is None or token_id == tokenizer.unk_token_id:
        if fallback is None:
            raise ValueError(f"Tokenizer has no token ID for {token_name}")
        return fallback
    return token_id


def _make_sequence_end_matcher(sequence):
    sequence = tuple(int(token_id) for token_id in sequence)
    if not sequence:
        raise ValueError("Cannot create a matcher for an empty token sequence")

    def matches_sequence_end(seq_idx, token, tokens):
        start = seq_idx - len(sequence) + 1
        if start < 0 or token != sequence[-1]:
            return False
        return tuple(int(value) for value in tokens[start : seq_idx + 1]) == sequence

    return matches_sequence_end


def _make_token_after_sequence(sequence):
    sequence = tuple(int(token_id) for token_id in sequence)

    def is_token_after_sequence(seq_idx, token, tokens):
        start = seq_idx - len(sequence)
        if start < 0:
            return False
        return tuple(int(value) for value in tokens[start:seq_idx]) == sequence

    return is_token_after_sequence


def get_token_ranges(masking_type, tokenizer):
    start_header_token = _special_token_id(tokenizer, "<|start_header_id|>")
    end_header_token = _special_token_id(tokenizer, "<|end_header_id|>")
    eot_token = _special_token_id(tokenizer, "<|eot_id|>", tokenizer.eos_token_id)
    header_break = tokenizer.encode("\n\n", add_special_tokens=False)
    if not header_break:
        raise ValueError("Tokenizer produced no tokens for the role-header separator")

    def role_header(role):
        role_tokens = tokenizer.encode(role, add_special_tokens=False)
        if not role_tokens:
            raise ValueError(f"Tokenizer produced no tokens for role {role!r}")
        return [start_header_token, *role_tokens, end_header_token]

    assistant_header = role_header("assistant")
    assistant_header_break = [*assistant_header, *header_break]
    user_header_break = [*role_header("user"), *header_break]
    is_after_assistant_header = _make_sequence_end_matcher(assistant_header)
    is_after_assistant_header_break = _make_sequence_end_matcher(
        assistant_header_break
    )
    is_after_user_header_break = _make_sequence_end_matcher(user_header_break)
    is_first_assistant_content_token = _make_token_after_sequence(
        assistant_header_break
    )

    if masking_type == "generation":
        return {
            "only_return_on_tokens_between": [
                is_after_assistant_header_break,
                eot_token,
            ],
            "only_choose_prompt_tokens_between": [
                is_after_user_header_break,
                eot_token,
            ],
            "only_probe_tokens_between": [
                is_after_assistant_header_break,
                eot_token,
            ],
        }

    elif masking_type == "instruction":
        return {
            "only_return_on_tokens_between": [
                is_after_assistant_header_break,
                eot_token,
            ],
            "only_choose_prompt_tokens_between": [
                is_after_user_header_break,
                eot_token,
            ],
            "only_probe_tokens_between": [
                is_after_assistant_header,
                is_first_assistant_content_token,
            ],
        }
    else:
        raise ValueError(f"Unknown masking_type: {masking_type}")


def main():
    global encoder

    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    probes_folder = os.path.abspath(args.output_dir)
    os.makedirs(probes_folder, exist_ok=True)
    model_config = resolve_model_config(args.model_name)
    layers = resolve_probe_layers(args.layers, model_config)
    default_lora_layers = list(range(0, max(layers) + 1))
    lora_layers = resolve_lora_layers(
        args.lora_layers, model_config, default_layers=default_lora_layers
    )
    masking_type = args.masking_type
    use_lora_probes = not args.no_lora_probes
    adapter_name = "lora" if use_lora_probes else "probe"
    default_name = f"{model_config.alias}_{adapter_name}_oat_{masking_type}_{'nonlinear' if args.probe_type == 'nonlinear' else 'linear'}"
    name = args.run_name or default_name
    planned_artifacts = [
        os.path.join(probes_folder, f"{name}_probes.pt"),
        os.path.join(probes_folder, f"{name}_probes_state_dict.pt"),
        os.path.join(probes_folder, f"{name}_info.json"),
    ]
    if use_lora_probes:
        planned_artifacts.append(os.path.join(probes_folder, f"{name}_model"))
    existing_artifacts = [path for path in planned_artifacts if os.path.exists(path)]
    if existing_artifacts and not args.overwrite:
        formatted_paths = "\n  ".join(existing_artifacts)
        raise FileExistsError(
            "Refusing to mix or overwrite an existing OAT run. Use --run-name for "
            f"a distinct experiment or --overwrite intentionally:\n  {formatted_paths}"
        )
    info_path = os.path.join(probes_folder, f"{name}_info.json")
    if args.overwrite and os.path.exists(info_path):
        # Metadata is the completion marker; remove it before replacing run parts.
        os.remove(info_path)
    device = normalize_torch_device(args.device)
    n_grad_accum = max(1, min(args.n_grad_accum, args.n_steps))
    while n_grad_accum > 1 and args.n_steps % n_grad_accum != 0:
        n_grad_accum -= 1
    print(f"Using torch device: {device}")
    print(f"Using model: {model_config.model_name}")
    print(f"Using probe layers: {layers}")
    if use_lora_probes:
        print(f"Using LoRA layers: {lora_layers}")
    print(f"Using grad accumulation: {n_grad_accum}")
    run_softprompt_eval_every = (
        0 if args.disable_softprompt_eval else args.run_softprompt_eval_every
    )
    if run_softprompt_eval_every <= 0:
        print("Soft-prompt benchmark: disabled")
    else:
        print(f"Soft-prompt benchmark: every {run_softprompt_eval_every} adversarial steps")

    # Load model and dataset
    encoder = load_model_encoder(model_config.model_name, device=device)
    if encoder.model.config.model_type != "llama":
        raise ValueError(
            "This OAT data/masking pipeline currently supports Llama instruct models only"
        )
    required_control_tokens = (
        "<|start_header_id|>",
        "<|end_header_id|>",
        "<|eot_id|>",
    )
    for control_token in required_control_tokens:
        _special_token_id(encoder.tokenizer, control_token)
    template_overhead = chat_template_token_overhead(encoder.tokenizer)
    base_max_length = 512
    max_length = base_max_length + template_overhead
    print(
        f"Using max length {max_length} ({base_max_length} legacy tokens + "
        f"{template_overhead} chat-template tokens)"
    )
    jailbreaks_dataset = load_dataset(args.dataset_name)

    forget_examples_train = sample_examples_from_datasets(
        [jailbreaks_dataset["circuit_breakers_train"]],
        [1.0],
        encoder.tokenizer,
        seed=args.seed,
    )

    retain_examples_train = sample_examples_from_datasets(
        [jailbreaks_dataset["xstest"], jailbreaks_dataset["benign_instructions_train"]],
        [0.15, 0.85],
        encoder.tokenizer,
        seed=args.seed,
    )

    # Also get examples with just the prompts
    forget_examples_train_prompts = sample_examples_from_datasets(
        [jailbreaks_dataset["circuit_breakers_train"]],
        [1.0],
        encoder.tokenizer,
        only_prompts=True,
        seed=args.seed,
    )

    # Train/test split
    forget_examples_train, forget_examples_val, _ = split_dataset(forget_examples_train)
    retain_examples_train, retain_examples_val, _ = split_dataset(retain_examples_train)
    _, forget_examples_val_prompts, _ = split_dataset(forget_examples_train_prompts)

    ranges = get_token_ranges(masking_type, encoder.tokenizer)
    only_return_on_tokens_between = ranges["only_return_on_tokens_between"]
    only_choose_prompt_tokens_between = ranges["only_choose_prompt_tokens_between"]
    only_probe_tokens_between = ranges["only_probe_tokens_between"]

    probes, lora_model, info = train_online_probe(
        encoder=encoder,
        positive_examples=forget_examples_train,
        negative_examples=retain_examples_train,
        create_probe_fn=get_probe_creator(args.probe_type),
        layers=layers,
        max_length=max_length,
        n_steps_per_logging=8,
        batch_size=args.batch_size,
        n_grad_accum=n_grad_accum,
        adversary_lr=1e-3,
        adapter_lr=5e-5,
        n_steps=args.n_steps,
        run_softprompt_eval_every=run_softprompt_eval_every,
        pgd_iterations=args.pgd_iterations,
        kl_penalty=10,
        device=device,
        lora_layers=lora_layers,
        only_return_on_tokens_between=only_return_on_tokens_between,
        only_choose_prompt_tokens_between=only_choose_prompt_tokens_between,
        only_probe_tokens_between=only_probe_tokens_between,
        adversarial_training=True,
        use_lora_adapter=use_lora_probes,
        freeze_probes_during_adversarial_training=use_lora_probes,
        softprompt_evals_data={
            "test_negative_examples": retain_examples_val,
            "test_positive_examples": forget_examples_val,
            "test_positive_prompts": forget_examples_val_prompts,
            "only_return_on_tokens_between": only_return_on_tokens_between,
            "only_choose_prompt_tokens_between": only_choose_prompt_tokens_between,
            "only_probe_tokens_between": only_probe_tokens_between,
            "max_length": max_length,
        },
    )
    info.update(
        {
            "artifact_format_version": 2,
            "model_name": model_config.model_name,
            "model_alias": model_config.alias,
            "model_repo_id": model_config.repo_id,
            "layers": layers,
            "lora_layers": lora_layers if use_lora_probes else None,
            "run_softprompt_eval_every": run_softprompt_eval_every,
            "masking_type": masking_type,
            "probe_type": args.probe_type,
            "dataset_name": args.dataset_name,
            "batch_size": args.batch_size,
            "n_grad_accum": n_grad_accum,
            "n_steps": args.n_steps,
            "pgd_iterations": args.pgd_iterations,
            "optimization": {
                "probe_lr": 1e-3,
                "adapter_lr": 5e-5,
                "adversary_lr": 1e-3,
                "epsilon": 10.0,
                "kl_penalty": 10.0,
                "kl_objective": "normalized_kl_adapted_to_base",
                "start_adv_training_at_step": 1024,
                "freeze_probes_during_adversarial_training": use_lora_probes,
                "freeze_lora_during_warmup": False,
            },
            "lora": (
                {
                    "r": 64,
                    "alpha": 128,
                    "dropout": 0.0,
                    "bias": "none",
                }
                if use_lora_probes
                else None
            ),
            "base_max_length": base_max_length,
            "chat_template_token_overhead": template_overhead,
            "max_length": max_length,
            "chat_template_date": LLAMA32_CHAT_TEMPLATE_DATE,
            "seed": args.seed,
            "lora_target_modules": (
                list(DEFAULT_LORA_TARGET_MODULES) if use_lora_probes else None
            ),
            "chat_template_sha256": hashlib.sha256(
                encoder.tokenizer.chat_template.encode("utf-8")
            ).hexdigest(),
            "model_config_sha256": _model_config_sha256(encoder.model.config),
            "special_token_ids": {
                "bos": encoder.tokenizer.bos_token_id,
                "eos": encoder.tokenizer.eos_token_id,
                "pad": encoder.tokenizer.pad_token_id,
                **{
                    token: encoder.tokenizer.convert_tokens_to_ids(token)
                    for token in required_control_tokens
                },
            },
            "dataset_fingerprints": {
                split: jailbreaks_dataset[split]._fingerprint
                for split in (
                    "circuit_breakers_train",
                    "xstest",
                    "benign_instructions_train",
                )
            },
            "output_dir": probes_folder,
            "run_name": name,
            "sample_hashes": {
                "positive_train": _sha256_strings(forget_examples_train),
                "positive_validation": _sha256_strings(forget_examples_val),
                "negative_train": _sha256_strings(retain_examples_train),
                "negative_validation": _sha256_strings(retain_examples_val),
            },
            "package_versions": {
                package: importlib.metadata.version(package)
                for package in ("torch", "transformers", "datasets", "peft")
            },
        }
    )

    # Save results
    legacy_probe_path = os.path.join(probes_folder, f"{name}_probes.pt")
    safe_probe_path = os.path.join(
        probes_folder, f"{name}_probes_state_dict.pt"
    )
    adapter_path = os.path.join(probes_folder, f"{name}_model")
    save_probes(
        probes=probes,
        save_path=legacy_probe_path,
    )
    save_probe_state_dicts(
        probes=probes,
        save_path=safe_probe_path,
    )
    reloaded_probes = load_probe_state_dicts(safe_probe_path)
    if set(reloaded_probes) != set(probes):
        raise RuntimeError("Saved probe artifact failed layer validation")

    if use_lora_probes:
        if args.overwrite and os.path.isdir(adapter_path):
            shutil.rmtree(adapter_path)
        for peft_config in lora_model.peft_config.values():
            peft_config.base_model_name_or_path = model_config.repo_id
        lora_model.save_pretrained(adapter_path)
        with open(os.path.join(adapter_path, "adapter_config.json")) as config_file:
            saved_adapter_config = json.load(config_file)
        if set(saved_adapter_config["target_modules"]) != set(
            DEFAULT_LORA_TARGET_MODULES
        ):
            raise RuntimeError("Saved adapter target modules do not match the run")

    # Persist trained weights before optional evaluation so a backend-specific
    # evaluation failure cannot discard a completed optimization run.
    print("RUNNING HELD-OUT PROBE EVALUATION")
    info["heldout_probe_eval"] = evaluate_probe_auc(
        model=lora_model,
        tokenizer=encoder.tokenizer,
        probes=probes,
        positive_examples=forget_examples_val,
        negative_examples=retain_examples_val,
        only_probe_tokens_between=only_probe_tokens_between,
        max_length=max_length,
        batch_size=args.batch_size,
        device=device,
    )
    print("Held-out probe AUC:", info["heldout_probe_eval"]["auc"])

    artifact_paths = {
        "legacy_probe_pickle": legacy_probe_path,
        "probe_state_dict": safe_probe_path,
    }
    if use_lora_probes:
        artifact_paths["adapter"] = adapter_path
    info["artifacts"] = {
        name: {
            "path": os.path.relpath(path, probes_folder),
            "sha256": sha256_artifact(path),
        }
        for name, path in artifact_paths.items()
    }

    # Publish metadata last and atomically; its presence marks a complete run.
    temporary_info_path = f"{info_path}.tmp"
    with open(temporary_info_path, "w") as f:
        json.dump(info, f, indent=2, default=convert_to_serializable)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary_info_path, info_path)


if __name__ == "__main__":
    main()
