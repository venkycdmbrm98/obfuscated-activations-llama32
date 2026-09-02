#!/usr/bin/env python3
"""Download published adapters, probes, PGD banks, and result artifacts.

By default this script downloads only the two model adapters and portable probe
state dictionaries, arranging them in the paths expected by the experiment
runners.  ``--include-results`` additionally downloads the machine-readable
Hugging Face artifact release; large files remain ignored by Git.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download


ROOT = Path(__file__).resolve().parent
MODEL_RELEASES = {
    "1B": {
        "repo_id": "venky-cdmbrm14/Llama-3.2-1B-OAT-Adapter",
        "revision": "4c8571375a633e5150e48ed3c6d6ff360b612092",
        "stem": "llama32-1b-generation-linear",
    },
    "3B": {
        "repo_id": "venky-cdmbrm14/Llama-3.2-3B-OAT-Adapter",
        "revision": "5825b0aebb80361bbb621308974e67eae8da40f4",
        "stem": "llama32-3b-generation-linear",
    },
}
RESULT_RELEASE = {
    "repo_id": "venky-cdmbrm14/obfuscated-activations-llama32-artifacts",
    "revision": "700ac528a3a7e4982d274413b8c37c10d6890201",
}


def sha256_file(path: Path) -> str:
    """Hash one local file without loading it entirely into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tree(path: Path) -> str:
    """Match the directory hashing convention used by OAT run manifests."""

    digest = hashlib.sha256()
    for file_path in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(str(file_path.relative_to(path)).encode("utf-8"))
        with file_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def download_model(model_key: str, destination: Path) -> None:
    """Download one pinned adapter/probe release into the runner's layout."""

    release = MODEL_RELEASES[model_key]
    repo_id = release["repo_id"]
    revision = release["revision"]
    stem = release["stem"]
    required = (
        "adapter_config.json",
        "adapter_model.safetensors",
        f"{stem}_probes_state_dict.pt",
        "training_metadata.json",
    )
    downloaded = {
        name: Path(
            hf_hub_download(repo_id=repo_id, filename=name, revision=revision)
        )
        for name in required
    }

    destination.mkdir(parents=True, exist_ok=True)
    adapter_dir = destination / f"{stem}_model"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        shutil.copy2(downloaded[name], adapter_dir / name)
    probe_path = destination / f"{stem}_probes_state_dict.pt"
    shutil.copy2(downloaded[probe_path.name], probe_path)

    metadata = json.loads(downloaded["training_metadata.json"].read_text())
    # Remove machine-specific paths from the published training run and point
    # model loading at the immutable base-model repository instead.
    metadata["model_name"] = metadata["model_repo_id"]
    metadata.pop("output_dir", None)
    metadata["source_hf_repo"] = repo_id
    metadata["source_hf_revision"] = revision
    metadata["artifacts"] = {
        "probe_state_dict": {
            "path": probe_path.name,
            "sha256": sha256_file(probe_path),
        },
        "adapter": {
            "path": adapter_dir.name,
            "sha256": sha256_tree(adapter_dir),
        },
    }
    info_path = destination / f"{stem}_info.json"
    info_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(f"Downloaded {model_key} adapter and probes from {repo_id}@{revision}")


def parse_models(value: str) -> tuple[str, ...]:
    """Parse a comma-separated subset of the supported model keys."""

    models = tuple(item.strip().upper() for item in value.split(",") if item.strip())
    unknown = sorted(set(models) - set(MODEL_RELEASES))
    if not models or unknown:
        raise argparse.ArgumentTypeError(
            f"models must be a comma-separated subset of {sorted(MODEL_RELEASES)}"
        )
    return models


def parse_args() -> argparse.Namespace:
    """Parse artifact selection and destination options."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=parse_models, default=("1B", "3B"))
    parser.add_argument(
        "--model-output",
        type=Path,
        default=ROOT / "probe_weights_comp_only2",
    )
    parser.add_argument(
        "--include-results",
        action="store_true",
        help="Download the approximately 2.1 GB PGD-bank/result release.",
    )
    parser.add_argument(
        "--results-output",
        type=Path,
        default=ROOT / "published_artifacts",
    )
    return parser.parse_args()


def main() -> int:
    """Download the selected immutable Hugging Face release revisions."""

    args = parse_args()
    for model_key in args.models:
        download_model(model_key, args.model_output.resolve())
    if args.include_results:
        snapshot_download(
            repo_id=RESULT_RELEASE["repo_id"],
            repo_type="dataset",
            revision=RESULT_RELEASE["revision"],
            local_dir=args.results_output.resolve(),
        )
        print(f"Downloaded result artifacts to {args.results_output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
