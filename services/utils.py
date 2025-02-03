import logging
import torch
from pathlib import Path

def setup_logging():
    """Sets up logging for the application."""
    logging.basicConfig(level=logging.INFO)


def quantize_model(model):
    """Quantizes a model for reduced memory usage using float16 precision."""
    model.half()  # Convert model weights to float16 for better efficiency
    return model


def export_to_onnx(model, onnx_path):
    """Exports the model to ONNX format if not already present."""
    onnx_dir = onnx_path.parent
    onnx_dir.mkdir(parents=True, exist_ok=True)

    if onnx_path.exists():
        logging.info("ONNX model already exists. Skipping export.")
        return onnx_path

    dummy_input = {
        "input_ids": torch.randint(0, 1000, (1, 512), dtype=torch.long),
        "attention_mask": torch.ones(1, 512, dtype=torch.long),
        "decoder_input_ids": torch.randint(0, 1000, (1, 10), dtype=torch.long),  # Small decoder input
    }

    try:
        logging.info(f"Exporting model to {onnx_path}...")

        model.eval()

        # Ensure no deprecated `past_key_values` are used and avoid any unintended use
        with torch.no_grad():
            # Explicitly pass required inputs for ONNX export
            torch.onnx.export(
                model,
                (
                    dummy_input["input_ids"],
                    dummy_input["attention_mask"],
                    dummy_input["decoder_input_ids"],
                ),
                str(onnx_path),
                input_names=["input_ids", "attention_mask", "decoder_input_ids"],
                output_names=["logits"],
                opset_version=14,
                do_constant_folding=True,
                dynamic_axes={
                    "input_ids": {0: "batch_size", 1: "sequence_length"},
                    "attention_mask": {0: "batch_size", 1: "sequence_length"},
                    "decoder_input_ids": {0: "batch_size", 1: "decoder_length"},
                    "logits": {0: "batch_size", 1: "sequence_length"},
                },
            )

        logging.info(f"Model successfully exported to {onnx_path}")
    except Exception as e:
        logging.error(f"Error exporting model: {e}")
        raise e

    return onnx_path