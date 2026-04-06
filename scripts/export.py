"""GhostLM ONNX exporter — converts trained model to ONNX for deployment outside PyTorch."""

import argparse
import sys
from dataclasses import fields
from pathlib import Path

import torch

from ghostlm.config import GhostLMConfig
from ghostlm.model import GhostLM
from ghostlm.tokenizer import GhostTokenizer


def parse_args():
    """Parse command-line arguments for ONNX export.

    Returns:
        argparse.Namespace with all parsed export arguments.
    """
    parser = argparse.ArgumentParser(description="Export GhostLM to ONNX format")

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to GhostLM checkpoint (.pt file)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="exports/ghostlm.onnx",
        help="Output path for the ONNX model file",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=64,
        help="Sequence length for export",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=14,
        help="ONNX opset version",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to run export on: cpu, cuda",
    )

    return parser.parse_args()


def load_model(checkpoint_path: str, device: str) -> tuple:
    """Load GhostLM model from a checkpoint for export.

    Reconstructs the model configuration from the checkpoint metadata,
    instantiates the model, and loads the saved weights.

    Args:
        checkpoint_path: Path to the .pt checkpoint file.
        device: Target device string.

    Returns:
        Tuple of (model, config) with model in eval mode.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    saved_config = checkpoint["config"]
    config = GhostLMConfig(**{
        f.name: saved_config[f.name]
        for f in fields(GhostLMConfig)
        if f.name in saved_config
    })

    model = GhostLM(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    model = model.to(device)

    return model, config


def export_onnx(model: GhostLM, config: GhostLMConfig, output_path: str, seq_len: int, opset: int) -> None:
    """Export the GhostLM model to ONNX format.

    Creates a dummy input tensor and traces the model forward pass
    to produce an ONNX graph with dynamic batch and sequence dimensions.

    Args:
        model: GhostLM model in eval mode.
        config: GhostLMConfig with model hyperparameters.
        output_path: Destination path for the ONNX file.
        seq_len: Sequence length to use for the export trace.
        opset: ONNX opset version for the exported graph.
    """
    dummy_input = torch.zeros(1, seq_len, dtype=torch.long)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["input_ids"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "logits": {0: "batch", 1: "sequence"},
        },
        opset_version=opset,
    )

    file_size = Path(output_path).stat().st_size / (1024 * 1024)
    print(f"  Exported ONNX model: {output_path} ({file_size:.1f} MB)")


def verify_onnx(output_path: str) -> None:
    """Verify the exported ONNX model by loading and running a quick check.

    Attempts to import onnx and onnxruntime to validate the model graph
    and run a test inference pass.

    Args:
        output_path: Path to the exported ONNX file.
    """
    try:
        import onnx
        import onnxruntime as ort

        model = onnx.load(output_path)
        onnx.checker.check_model(model)

        session = ort.InferenceSession(output_path)
        dummy_input = {"input_ids": torch.zeros(1, 64, dtype=torch.long).numpy()}
        outputs = session.run(None, dummy_input)

        print(f"  ONNX verification passed — output shape: {outputs[0].shape}")
    except ImportError:
        print("  Install onnx and onnxruntime to verify: pip install onnx onnxruntime")
    except Exception as e:
        print(f"  ONNX verification failed: {e}")


def main():
    """Run the GhostLM ONNX export pipeline.

    Loads a trained checkpoint, exports the model to ONNX format with
    dynamic axes for flexible batch and sequence lengths, and optionally
    verifies the exported model.
    """
    args = parse_args()

    # Verify checkpoint exists
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    # Load model
    model, config = load_model(str(checkpoint_path), args.device)

    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Print header
    print("=" * 50)
    print("GhostLM ONNX Export")
    print("=" * 50)
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Output:     {output_path}")
    print(f"  Seq length: {args.seq_len}")
    print(f"  Opset:      {args.opset}")
    print(f"  Params:     {model.num_params():,}")
    print("=" * 50)

    # Export
    print("\nExporting to ONNX...")
    export_onnx(model, config, str(output_path), args.seq_len, args.opset)

    # Verify
    print("\nVerifying ONNX model...")
    verify_onnx(str(output_path))

    print(f"\nExport complete: {output_path}")


if __name__ == "__main__":
    main()
