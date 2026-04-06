"""GhostLM HuggingFace Hub uploader — pushes model weights and model card to HuggingFace."""

import argparse
import json
import os
import shutil
import sys
import tempfile
from dataclasses import asdict, fields
from pathlib import Path

try:
    from huggingface_hub import HfApi, login, upload_folder
except ImportError:
    print("Install huggingface_hub: pip install huggingface_hub")
    sys.exit(1)

from ghostlm.config import GhostLMConfig


def parse_args():
    """Parse command-line arguments for HuggingFace Hub upload.

    Returns:
        argparse.Namespace with all parsed upload arguments.
    """
    parser = argparse.ArgumentParser(description="Upload GhostLM to HuggingFace Hub")

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to GhostLM checkpoint (.pt file)",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="HuggingFace repo ID (e.g. joemunene/GhostLM-tiny)",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="HuggingFace API token (or set HF_TOKEN env var)",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make the repository private",
    )
    parser.add_argument(
        "--commit-message",
        type=str,
        default="Upload GhostLM weights",
        help="Commit message for the upload",
    )

    return parser.parse_args()


def prepare_upload_folder(checkpoint_path: str, config: GhostLMConfig) -> str:
    """Prepare a temporary directory with all files needed for HuggingFace upload.

    Copies the checkpoint, writes config JSON, tokenizer config, and model card
    into a structure compatible with the HuggingFace model hub.

    Args:
        checkpoint_path: Path to the GhostLM .pt checkpoint file.
        config: GhostLMConfig instance with model hyperparameters.

    Returns:
        Path to the prepared temporary directory.
    """
    temp_dir = tempfile.mkdtemp(prefix="ghostlm_upload_")

    # Copy model checkpoint
    shutil.copy2(checkpoint_path, os.path.join(temp_dir, "pytorch_model.pt"))

    # Write model config as JSON
    config_dict = asdict(config)
    with open(os.path.join(temp_dir, "config.json"), "w") as f:
        json.dump(config_dict, f, indent=2)

    # Copy model card as README if it exists
    model_card = Path("MODEL_CARD.md")
    if model_card.exists():
        shutil.copy2(model_card, os.path.join(temp_dir, "README.md"))

    # Write tokenizer config
    tokenizer_config = {
        "tokenizer_type": "tiktoken",
        "vocab_size": 50261,
        "model": "gpt2",
    }
    with open(os.path.join(temp_dir, "tokenizer_config.json"), "w") as f:
        json.dump(tokenizer_config, f, indent=2)

    return temp_dir


def main():
    """Upload GhostLM checkpoint and config to HuggingFace Hub.

    Loads the checkpoint, prepares an upload directory with all required
    files, creates the repository if needed, and pushes everything to
    the HuggingFace model hub.
    """
    args = parse_args()

    # Verify checkpoint exists
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    # Load checkpoint to extract config
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    saved_config = checkpoint["config"]
    config = GhostLMConfig(**{
        f.name: saved_config[f.name]
        for f in fields(GhostLMConfig)
        if f.name in saved_config
    })

    # Print upload summary
    print("=" * 50)
    print("GhostLM HuggingFace Hub Upload")
    print("=" * 50)
    print(f"  Repo:       {args.repo_id}")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Preset:     {config.n_layers} layers, {config.d_model} dim")
    print(f"  Params:     {sum(p.numel() for p in [torch.zeros(config.vocab_size, config.d_model)]):,} (approx)")
    print(f"  Private:    {args.private}")
    print("=" * 50)

    # Login to HuggingFace
    token = args.token or os.environ.get("HF_TOKEN")
    if token:
        login(token=token)
        print("  Logged in to HuggingFace Hub.")
    else:
        print("  Warning: No token provided. Set --token or HF_TOKEN env var.")
        print("  Attempting to use cached credentials...")

    # Create repo if it doesn't exist
    api = HfApi()
    try:
        api.repo_info(repo_id=args.repo_id, repo_type="model")
        print(f"  Repository {args.repo_id} already exists.")
    except Exception:
        api.create_repo(
            repo_id=args.repo_id,
            repo_type="model",
            private=args.private,
        )
        print(f"  Created repository {args.repo_id}.")

    # Prepare upload folder
    print("\n  Preparing upload files...")
    upload_dir = prepare_upload_folder(str(checkpoint_path), config)

    try:
        # Upload to HuggingFace Hub
        print("  Uploading to HuggingFace Hub...")
        upload_folder(
            folder_path=upload_dir,
            repo_id=args.repo_id,
            repo_type="model",
            commit_message=args.commit_message,
        )
        print(f"\nGhostLM uploaded to https://huggingface.co/{args.repo_id}")
    finally:
        # Clean up temp folder
        shutil.rmtree(upload_dir, ignore_errors=True)
        print("  Temporary files cleaned up.")


if __name__ == "__main__":
    main()
