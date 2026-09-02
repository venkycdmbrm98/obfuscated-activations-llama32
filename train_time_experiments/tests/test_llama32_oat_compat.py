"""Regression tests for the Llama 3.2 OAT artifact and analysis pipeline.

The suite checks device precision, chat masks, LoRA placement, portable probe
serialization, artifact checksums, and small-model attack semantics.  Tests
that require gated local tokenizer snapshots are skipped when those snapshots
are absent from a clean public checkout.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
from datasets import Dataset
from peft import PeftModel
from safetensors.torch import load_file as load_safetensors
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM


EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

from compute_oat_probes import get_token_ranges  # noqa: E402
from src.attacks import clear_hooks, prepare_soft_prompt_inputs, train_attack  # noqa: E402
from src.chat_formatting import (  # noqa: E402
    LLAMA32_CHAT_TEMPLATE_DATE,
    append_to_last_user_message,
    chat_template_token_overhead,
    format_dataset_chat,
    parse_llama3_chat,
)
from src.probe_archs import LinearProbe, NonlinearProbe  # noqa: E402
from src.probe_training import (  # noqa: E402
    DEFAULT_LORA_TARGET_MODULES,
    evaluate_probe_auc,
    initialize_lora_adapter,
    load_probe_state_dicts,
    save_probe_state_dicts,
    train_online_probe,
)
from src.oat_artifacts import load_oat_run_manifest, sha256_artifact  # noqa: E402
from src.utils import get_valid_token_mask, torch_autocast  # noqa: E402


LEGACY_PROMPT = (
    "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
    "How can I stop a Python process?"
    "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
)
COMPLETION = "Use the process ID with the operating system's termination command."
LOCAL_TOKENIZER_SNAPSHOTS = (
    EXPERIMENT_DIR / "Llama-3.2-1B-Instruct",
    EXPERIMENT_DIR / "Llama-3.2-3B-Instruct",
)


class DevicePrecisionCompatibilityTest(unittest.TestCase):
    def test_autocast_uses_device_appropriate_default_dtype(self):
        with mock.patch("src.utils.torch.autocast") as autocast:
            torch_autocast("cuda")
            autocast.assert_called_once_with(
                device_type="cuda", dtype=torch.bfloat16
            )

        with mock.patch("src.utils.torch.autocast") as autocast:
            torch_autocast("mps")
            autocast.assert_called_once_with(
                device_type="mps", dtype=torch.float16
            )

    def test_autocast_preserves_explicit_dtype(self):
        with mock.patch("src.utils.torch.autocast") as autocast:
            torch_autocast("cuda", dtype=torch.float32)
            autocast.assert_called_once_with(
                device_type="cuda", dtype=torch.float32
            )


@unittest.skipUnless(
    all(path.is_dir() for path in LOCAL_TOKENIZER_SNAPSHOTS),
    "requires local gated Llama 3.2 tokenizer snapshots",
)
class Llama32TokenizerCompatibilityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizers = {
            size: AutoTokenizer.from_pretrained(EXPERIMENT_DIR / model_dir)
            for size, model_dir in {
                "1B": "Llama-3.2-1B-Instruct",
                "3B": "Llama-3.2-3B-Instruct",
            }.items()
        }
        for tokenizer in cls.tokenizers.values():
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "right"

    def test_model_configs_have_multi_eos_and_matching_templates(self):
        templates = []
        for model_dir in ("Llama-3.2-1B-Instruct", "Llama-3.2-3B-Instruct"):
            config = json.loads((EXPERIMENT_DIR / model_dir / "config.json").read_text())
            generation_config = json.loads(
                (EXPERIMENT_DIR / model_dir / "generation_config.json").read_text()
            )
            self.assertEqual(config["eos_token_id"], [128001, 128008, 128009])
            self.assertEqual(
                generation_config["eos_token_id"], [128001, 128008, 128009]
            )
            templates.append(self.tokenizers[model_dir.split("-")[2]].chat_template)
        self.assertEqual(templates[0], templates[1])

    def test_control_token_ids_match_both_local_snapshots(self):
        expected = {
            "<|begin_of_text|>": 128000,
            "<|start_header_id|>": 128006,
            "<|end_header_id|>": 128007,
            "<|eot_id|>": 128009,
            "user": 882,
            "assistant": 78191,
            "\n\n": 271,
        }
        for tokenizer in self.tokenizers.values():
            for text, token_id in expected.items():
                self.assertEqual(
                    tokenizer.encode(text, add_special_tokens=False), [token_id]
                )

    def test_template_rendering_and_generation_masks_for_both_models(self):
        for size, tokenizer in self.tokenizers.items():
            with self.subTest(size=size):
                rendered = format_dataset_chat(tokenizer, LEGACY_PROMPT, COMPLETION)
                parsed = parse_llama3_chat(rendered)
                self.assertEqual(
                    [message["role"] for message in parsed.messages],
                    ["system", "user", "assistant"],
                )
                self.assertIn("Cutting Knowledge Date: December 2023", rendered)
                self.assertIn(f"Today Date: {LLAMA32_CHAT_TEMPLATE_DATE}", rendered)
                self.assertTrue(rendered.endswith(f"{COMPLETION}<|eot_id|>"))

                encoded = tokenizer(
                    rendered, add_special_tokens=False, return_tensors="pt"
                )["input_ids"]
                self.assertEqual(encoded[0, 0].item(), tokenizer.bos_token_id)
                self.assertEqual(
                    int((encoded == tokenizer.bos_token_id).sum().item()), 1
                )

                default_encoded = tokenizer(rendered, return_tensors="pt")["input_ids"]
                self.assertEqual(
                    int((default_encoded == tokenizer.bos_token_id).sum().item()), 2
                )

                ranges = get_token_ranges("generation", tokenizer)
                masks = {
                    name: get_valid_token_mask(encoded, token_range)
                    for name, token_range in ranges.items()
                }
                target_text = tokenizer.decode(
                    encoded[0][masks["only_return_on_tokens_between"][0]]
                )
                prompt_text = tokenizer.decode(
                    encoded[0][masks["only_choose_prompt_tokens_between"][0]]
                )
                self.assertEqual(target_text, COMPLETION)
                self.assertEqual(prompt_text, "How can I stop a Python process?")
                self.assertTrue(
                    torch.equal(
                        masks["only_return_on_tokens_between"],
                        masks["only_probe_tokens_between"],
                    )
                )

    def test_prompt_rendering_has_one_empty_assistant_generation_turn(self):
        for tokenizer in self.tokenizers.values():
            rendered = format_dataset_chat(tokenizer, LEGACY_PROMPT)
            parsed = parse_llama3_chat(rendered)
            self.assertTrue(parsed.has_generation_prompt)
            self.assertEqual(parsed.messages[-1]["role"], "user")
            self.assertTrue(
                rendered.endswith(
                    "<|start_header_id|>assistant<|end_header_id|>\n\n"
                )
            )

    def test_already_native_prompt_is_not_given_a_second_system_block(self):
        for tokenizer in self.tokenizers.values():
            native_prompt = format_dataset_chat(tokenizer, LEGACY_PROMPT)
            rendered = format_dataset_chat(tokenizer, native_prompt, COMPLETION)
            self.assertEqual(rendered.count("Cutting Knowledge Date"), 1)
            self.assertEqual(rendered.count("Today Date"), 1)

    def test_instruction_mask_selects_only_the_role_transition(self):
        for tokenizer in self.tokenizers.values():
            rendered = format_dataset_chat(tokenizer, LEGACY_PROMPT, COMPLETION)
            encoded = tokenizer(
                rendered, add_special_tokens=False, return_tensors="pt"
            )["input_ids"]
            probe_range = get_token_ranges("instruction", tokenizer)[
                "only_probe_tokens_between"
            ]
            mask = get_valid_token_mask(encoded, probe_range)
            self.assertEqual(tokenizer.decode(encoded[0][mask[0]]), "\n\n")

    def test_attack_suffix_is_added_only_to_the_user_message(self):
        tokenizer = self.tokenizers["1B"]
        rendered = format_dataset_chat(tokenizer, LEGACY_PROMPT, COMPLETION)
        modified = append_to_last_user_message(tokenizer, rendered, " ATTACK")
        parsed = parse_llama3_chat(modified)
        user_messages = [m["content"] for m in parsed.messages if m["role"] == "user"]
        assistant_messages = [
            m["content"] for m in parsed.messages if m["role"] == "assistant"
        ]
        self.assertTrue(user_messages[-1].endswith(" ATTACK"))
        self.assertNotIn("ATTACK", assistant_messages[-1])
        self.assertEqual(modified.count("Cutting Knowledge Date"), 1)
        self.assertEqual(
            rendered.split("<|start_header_id|>user", 1)[0],
            modified.split("<|start_header_id|>user", 1)[0],
        )

    def test_template_overhead_preserves_long_prompt_masks(self):
        cache_root = (
            Path.home()
            / ".cache/huggingface/datasets/Mechanistic-Anomaly-Detection___llama3-jailbreaks"
        )
        arrow_paths = list(cache_root.glob("**/llama3-jailbreaks-benign_instructions_train.arrow"))
        if not arrow_paths:
            self.skipTest("Cached llama3-jailbreaks dataset is unavailable")
        row = Dataset.from_file(str(arrow_paths[0]))[10136]

        for tokenizer in self.tokenizers.values():
            overhead = chat_template_token_overhead(tokenizer)
            self.assertEqual(overhead, 26)
            rendered = format_dataset_chat(tokenizer, row["prompt"], row["completion"])
            encoded = tokenizer(
                rendered,
                add_special_tokens=False,
                truncation=True,
                max_length=512 + overhead,
                return_tensors="pt",
            )["input_ids"]
            masks = {
                name: get_valid_token_mask(encoded, token_range)
                for name, token_range in get_token_ranges(
                    "generation", tokenizer
                ).items()
            }
            self.assertTrue(masks["only_choose_prompt_tokens_between"].any())
            self.assertTrue(masks["only_return_on_tokens_between"].any())
            self.assertTrue(masks["only_probe_tokens_between"].any())

    def test_malformed_legacy_prompt_fails_loudly(self):
        tokenizer = self.tokenizers["1B"]
        malformed = LEGACY_PROMPT.replace("\n\n", " ", 1)
        with self.assertRaisesRegex(ValueError, "Expected"):
            format_dataset_chat(tokenizer, malformed, COMPLETION)

    def test_valid_token_mask_preserves_1d_and_2d_shapes(self):
        tokenizer = self.tokenizers["1B"]
        rendered = format_dataset_chat(tokenizer, LEGACY_PROMPT, COMPLETION)
        token_ids = tokenizer.encode(rendered, add_special_tokens=False)
        token_range = get_token_ranges("generation", tokenizer)[
            "only_probe_tokens_between"
        ]
        one_dimensional = get_valid_token_mask(torch.tensor(token_ids), token_range)
        two_dimensional = get_valid_token_mask(
            torch.tensor([token_ids]), token_range
        )
        self.assertEqual(one_dimensional.shape, (len(token_ids),))
        self.assertEqual(two_dimensional.shape, (1, len(token_ids)))


class ArtifactAndAdapterCompatibilityTest(unittest.TestCase):
    def test_format_v2_manifest_verifies_artifact_checksums(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            artifact_path = root / "probe.pt"
            artifact_path.write_bytes(b"probe-data")
            info_path = root / "run_info.json"
            info_path.write_text(
                json.dumps(
                    {
                        "artifact_format_version": 2,
                        "artifacts": {
                            "probe_state_dict": {
                                "path": artifact_path.name,
                                "sha256": sha256_artifact(artifact_path),
                            }
                        },
                    }
                )
            )
            manifest = load_oat_run_manifest(info_path, require_v2=True)
            self.assertEqual(
                manifest["resolved_artifacts"]["probe_state_dict"],
                str(artifact_path.resolve()),
            )
            artifact_path.write_bytes(b"corrupted")
            with self.assertRaisesRegex(ValueError, "Checksum mismatch"):
                load_oat_run_manifest(info_path, require_v2=True)

    def test_safe_probe_artifact_loads_with_weights_only(self):
        probes = {2: LinearProbe(8), 4: NonlinearProbe(8, 4, dropout=0.0)}
        inputs = torch.randn(2, 3, 8)
        expected = {
            layer: probe.eval()(inputs).detach() for layer, probe in probes.items()
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "probes_state_dict.pt"
            save_probe_state_dicts(probes, path)
            loaded = torch.load(path, weights_only=True)
            reconstructed = load_probe_state_dicts(path)
        self.assertEqual(set(loaded), {2, 4})
        self.assertEqual(loaded[2]["class"], "LinearProbe")
        self.assertEqual(loaded[2]["state_dict"]["linear.weight"].shape, (1, 8))
        self.assertEqual(loaded[4]["class"], "NonlinearProbe")
        for layer in probes:
            self.assertTrue(torch.equal(expected[layer], reconstructed[layer](inputs)))

    def test_default_lora_targets_include_the_complete_llama_mlp(self):
        self.assertIn("gate_proj", DEFAULT_LORA_TARGET_MODULES)
        config = LlamaConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
        )
        encoder = SimpleNamespace(model=LlamaForCausalLM(config))
        model = initialize_lora_adapter(
            encoder,
            layers=[0],
            lora_params={"r": 2, "alpha": 4},
        )
        self.assertIsInstance(model, PeftModel)
        lora_modules = {
            name.rsplit(".", 1)[-1]
            for name, module in model.named_modules()
            if hasattr(module, "lora_A")
        }
        self.assertEqual(lora_modules, set(DEFAULT_LORA_TARGET_MODULES))
        with tempfile.TemporaryDirectory() as temp_dir:
            model.save_pretrained(temp_dir)
            adapter_config = json.loads(
                (Path(temp_dir) / "adapter_config.json").read_text()
            )
            adapter_state = load_safetensors(
                Path(temp_dir) / "adapter_model.safetensors"
            )
        self.assertEqual(
            set(adapter_config["target_modules"]),
            set(DEFAULT_LORA_TARGET_MODULES),
        )
        self.assertTrue(any("gate_proj.lora_A" in key for key in adapter_state))
        self.assertTrue(any("gate_proj.lora_B" in key for key in adapter_state))


@unittest.skipUnless(
    LOCAL_TOKENIZER_SNAPSHOTS[0].is_dir(),
    "requires a local gated Llama 3.2 tokenizer snapshot",
)
class TrainingSemanticsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained(
            EXPERIMENT_DIR / "Llama-3.2-1B-Instruct"
        )
        cls.tokenizer.pad_token = cls.tokenizer.eos_token
        cls.tokenizer.padding_side = "right"

    @staticmethod
    def tiny_model():
        config = LlamaConfig(
            vocab_size=128256,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            use_cache=True,
        )
        return LlamaForCausalLM(config)

    def test_heldout_probe_prediction_stays_inside_autocast(self):
        state = {"autocast_active": False}

        @contextmanager
        def tracked_autocast(_device):
            self.assertFalse(state["autocast_active"])
            state["autocast_active"] = True
            try:
                yield
            finally:
                state["autocast_active"] = False

        class EvaluationModel(torch.nn.Module):
            def forward(self, input_ids, attention_mask, output_hidden_states):
                del attention_mask, output_hidden_states
                hidden = torch.ones(*input_ids.shape, 4, device=input_ids.device)
                return SimpleNamespace(hidden_states=(hidden, hidden))

        class EvaluationProbe(torch.nn.Module):
            def predict(self, activations):
                if not state["autocast_active"]:
                    raise AssertionError("probe prediction ran outside autocast")
                return torch.sigmoid(activations[..., 0])

        positive = format_dataset_chat(self.tokenizer, LEGACY_PROMPT, COMPLETION)
        negative = format_dataset_chat(
            self.tokenizer, LEGACY_PROMPT, "Shut it down cleanly."
        )
        ranges = get_token_ranges("generation", self.tokenizer)
        with mock.patch(
            "src.probe_training.torch_autocast", side_effect=tracked_autocast
        ):
            result = evaluate_probe_auc(
                model=EvaluationModel(),
                tokenizer=self.tokenizer,
                probes={0: EvaluationProbe()},
                positive_examples=[positive],
                negative_examples=[negative],
                only_probe_tokens_between=ranges["only_probe_tokens_between"],
                max_length=128,
                batch_size=2,
                device="cpu",
            )

        self.assertEqual(result["n_positive"], 1)
        self.assertEqual(result["n_negative"], 1)

    def test_inner_pgd_does_not_create_probe_gradients(self):
        model = self.tiny_model()
        model.requires_grad_(False)
        probe = LinearProbe(16)
        rendered = format_dataset_chat(self.tokenizer, LEGACY_PROMPT, COMPLETION)
        encoding = self.tokenizer(
            rendered, add_special_tokens=False, return_tensors="pt"
        )
        ranges = get_token_ranges("generation", self.tokenizer)
        prompt_mask = get_valid_token_mask(
            encoding.input_ids, ranges["only_choose_prompt_tokens_between"]
        )
        target_mask = get_valid_token_mask(
            encoding.input_ids, ranges["only_return_on_tokens_between"]
        )
        losses, _ = train_attack(
            adv_tokens=encoding.input_ids,
            attention_mask=encoding.attention_mask,
            prompt_mask=prompt_mask,
            target_mask=target_mask,
            model=model,
            tokenizer=self.tokenizer,
            model_layers_module="model.layers",
            layer=["embedding"],
            epsilon=1.0,
            learning_rate=1e-3,
            pgd_iterations=1,
            probes={0: probe},
            probe_mask=target_mask,
            device="cpu",
        )
        self.assertIn("probe", losses)
        self.assertTrue(all(param.grad is None for param in probe.parameters()))
        self.assertTrue(all(param.requires_grad for param in probe.parameters()))
        clear_hooks(model)

    def test_single_batch_soft_prompt_shifts_probe_and_attention_masks(self):
        model = self.tiny_model()
        model.requires_grad_(False)
        probe = LinearProbe(16)
        rendered = format_dataset_chat(self.tokenizer, LEGACY_PROMPT, COMPLETION)
        encoding = self.tokenizer(
            rendered, add_special_tokens=False, return_tensors="pt"
        )
        ranges = get_token_ranges("generation", self.tokenizer)
        prompt_mask = get_valid_token_mask(
            encoding.input_ids, ranges["only_choose_prompt_tokens_between"]
        )
        target_mask = get_valid_token_mask(
            encoding.input_ids, ranges["only_return_on_tokens_between"]
        )
        losses, _ = train_attack(
            adv_tokens=encoding.input_ids,
            attention_mask=encoding.attention_mask,
            prompt_mask=prompt_mask,
            target_mask=target_mask,
            model=model,
            tokenizer=self.tokenizer,
            model_layers_module="model.layers",
            layer=["embedding"],
            epsilon=1.0,
            learning_rate=1e-3,
            pgd_iterations=1,
            probes={0: probe},
            probe_mask=target_mask,
            adversary_type="soft_prompt",
            initial_soft_prompt_text=" x",
            device="cpu",
        )
        self.assertIn("probe", losses)
        self.assertTrue(all(param.grad is None for param in probe.parameters()))
        clear_hooks(model)

    def test_soft_prompt_attention_mask_is_inserted_and_shifted(self):
        tokens = torch.tensor([[1, 2, 3, 0]])
        prompt_mask = torch.tensor([[False, True, False, False]])
        target_mask = torch.tensor([[False, False, True, False]])
        attention_mask = torch.tensor([[True, True, True, False]])
        soft_prompt = torch.tensor([[9, 10]])
        result = prepare_soft_prompt_inputs(
            tokens,
            prompt_mask,
            target_mask,
            soft_prompt,
            attention_mask=attention_mask,
        )
        new_tokens, _, _, insert_mask, _, new_attention_mask = result
        self.assertEqual(new_tokens.tolist(), [[1, 2, 9, 10, 3, 0]])
        self.assertEqual(insert_mask.tolist(), [[False, False, True, True, False, False]])
        self.assertEqual(
            new_attention_mask.tolist(), [[True, True, True, True, True, False]]
        )

    def test_no_lora_path_completes_an_adversarial_step(self):
        model = self.tiny_model()
        encoder = SimpleNamespace(model=model, tokenizer=self.tokenizer)
        positive = [
            format_dataset_chat(self.tokenizer, LEGACY_PROMPT, COMPLETION),
            format_dataset_chat(
                self.tokenizer,
                LEGACY_PROMPT.replace("Python", "shell"),
                COMPLETION,
            ),
        ]
        negative = [
            format_dataset_chat(
                self.tokenizer,
                LEGACY_PROMPT.replace("stop", "inspect"),
                "Read the process table.",
            ),
            format_dataset_chat(
                self.tokenizer,
                LEGACY_PROMPT.replace("stop", "start"),
                "Use the interpreter.",
            ),
        ]
        ranges = get_token_ranges("generation", self.tokenizer)
        probes, returned_model, _ = train_online_probe(
            encoder=encoder,
            positive_examples=positive,
            negative_examples=negative,
            create_probe_fn=lambda: LinearProbe(16),
            layers=[0],
            adversarial_training=True,
            use_lora_adapter=False,
            freeze_probes_during_adversarial_training=False,
            only_return_on_tokens_between=ranges[
                "only_return_on_tokens_between"
            ],
            only_choose_prompt_tokens_between=ranges[
                "only_choose_prompt_tokens_between"
            ],
            only_probe_tokens_between=ranges["only_probe_tokens_between"],
            n_steps=1,
            start_adv_training_at_step=0,
            pgd_iterations=1,
            batch_size=2,
            n_grad_accum=1,
            max_length=128,
            device="cpu",
            run_softprompt_eval_every=0,
        )
        self.assertIs(returned_model, model)
        self.assertEqual(set(probes), {0})
        clear_hooks(model)

    def test_frozen_lora_warmup_does_not_backpropagate_kl(self):
        model = self.tiny_model()
        encoder = SimpleNamespace(model=model, tokenizer=self.tokenizer)
        positive = [
            format_dataset_chat(self.tokenizer, LEGACY_PROMPT, COMPLETION),
            format_dataset_chat(
                self.tokenizer,
                LEGACY_PROMPT.replace("Python", "shell"),
                COMPLETION,
            ),
        ]
        negative = [
            format_dataset_chat(
                self.tokenizer,
                LEGACY_PROMPT.replace("stop", "inspect"),
                "Read the process table.",
            ),
            format_dataset_chat(
                self.tokenizer,
                LEGACY_PROMPT.replace("stop", "start"),
                "Use the interpreter.",
            ),
        ]
        ranges = get_token_ranges("generation", self.tokenizer)
        _, returned_model, _ = train_online_probe(
            encoder=encoder,
            positive_examples=positive,
            negative_examples=negative,
            create_probe_fn=lambda: LinearProbe(16),
            layers=[0],
            lora_params={"r": 2, "alpha": 4},
            lora_layers=[0],
            adversarial_training=True,
            use_lora_adapter=True,
            freeze_lora_during_warmup=True,
            freeze_probes_during_adversarial_training=True,
            only_return_on_tokens_between=ranges[
                "only_return_on_tokens_between"
            ],
            only_choose_prompt_tokens_between=ranges[
                "only_choose_prompt_tokens_between"
            ],
            only_probe_tokens_between=ranges["only_probe_tokens_between"],
            n_steps=2,
            start_adv_training_at_step=1,
            pgd_iterations=1,
            batch_size=2,
            n_grad_accum=1,
            max_length=128,
            device="cpu",
            run_softprompt_eval_every=0,
        )
        self.assertIsInstance(returned_model, PeftModel)
        clear_hooks(returned_model)


if __name__ == "__main__":
    unittest.main()
