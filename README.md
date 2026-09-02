# Obfuscated Activations in Llama 3.2

Code for a mechanistic case study of probe-aware, OAT-style LoRA adapters on
Llama 3.2 1B and 3B. Large adapters, probe weights, PGD banks, and result tables
are hosted on Hugging Face rather than committed to Git.

## Code

- `train_time_experiments/compute_oat_probes.py`: train adapters and probes.
- `train_pgd_banks.py`: build reusable embedding-space PGD attack banks.
- `activation_analysis4.py`: write-side Experiments 1–8 and statistical replay.
- `activation_analysis4_read.py`: read-side R1–R7 interventions.

## Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
# Install the appropriate PyTorch build for your machine first.
pip install -r requirements.txt
huggingface-cli login
python download_artifacts.py
```

The base Llama checkpoints are gated; accept Meta's terms on Hugging Face before
running model experiments.

## Example

```bash
python train_pgd_banks.py --pgd-iterations 32 --seed 42
python activation_analysis4.py --experiments 1,7,8 --iterations 32 --seeds 42
python activation_analysis4_read.py --experiments R1 --pgd-iterations 32 --seeds 42
```

Use `python <script> --help` for the complete options. Machine-readable results
can be downloaded with `python download_artifacts.py --include-results`. To use
the downloaded banks, pass `--pgd-banks-root published_artifacts/pgd_banks` to
the write-side script or `--bank-root published_artifacts/pgd_banks` to the
read-side script.

## Published artifacts

- [Hugging Face collection](https://huggingface.co/collections/venky-cdmbrm14/obfuscated-activations-in-llama-32-6a986a8b16943d61264cf9d4)
- [1B adapter and probes](https://huggingface.co/venky-cdmbrm14/Llama-3.2-1B-OAT-Adapter)
- [3B adapter and probes](https://huggingface.co/venky-cdmbrm14/Llama-3.2-3B-OAT-Adapter)
- [PGD banks and result artifacts](https://huggingface.co/datasets/venky-cdmbrm14/obfuscated-activations-llama32-artifacts)

The released evidence covers one trained adapter per model size. Fixed attacks
are replayed across interventions; this is not an adaptive reoptimization test.
Only read-side R1 has a completed public result grid.
