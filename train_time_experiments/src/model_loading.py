"""Load the base model and tokenizer required by adapter/probe training.

The original research encoder module also bundled sparse-autoencoder tooling.
This release needs only a lightweight object exposing ``model`` and
``tokenizer``, so optional SAE dependencies are deliberately excluded.
"""

from dataclasses import dataclass

from .model_configs import resolve_model_config
from .utils import load_hf_model_and_tokenizer


@dataclass
class ModelEncoder:
    """The model/tokenizer interface consumed by the probe-training loop."""

    model: object
    tokenizer: object


def load_model_encoder(model_name=None, device="cuda", layer=None, *args, **kwargs):
    """Load a supported causal LM without the unused SAE compatibility layer."""

    if layer is not None:
        raise ValueError("The public OAT training path uses raw activations, not an SAE")
    resolved = resolve_model_config(model_name)
    model, tokenizer = load_hf_model_and_tokenizer(
        resolved.model_name,
        device_map=device,
    )
    return ModelEncoder(model=model, tokenizer=tokenizer)
