import logging
import torch
import onnxruntime
from pathlib import Path

def setup_logging():
    """Sets up logging for the application."""
    logging.basicConfig(level=logging.INFO)

def quantize_model(model):
    """Quantizes a model for reduced memory usage."""
    return torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)

def export_to_onnx(model, onnx_path):
    """Exports the model to ONNX format if not already present."""
    onnx_dir = onnx_path.parent
    onnx_dir.mkdir(parents=True, exist_ok=True)
    
    if onnx_path.exists():
        logging.info("ONNX model already exists. Skipping export.")
        return onnx_path

    dummy_input = {
        "input_ids": torch.randint(0, 1000, (1, 512)),
        "attention_mask": torch.ones(1, 512),
    }

    try:
        logging.info(f"Exporting model to {onnx_path}...")

        model.eval()
        with torch.no_grad():
            outputs = model(**dummy_input)
            start_logits, end_logits = outputs.start_logits, outputs.end_logits

        torch.onnx.export(
            model, 
            (dummy_input["input_ids"], dummy_input["attention_mask"]),
            str(onnx_path),
            input_names=["input_ids", "attention_mask"],
            output_names=["start_logits", "end_logits"],
            opset_version=14,
            do_constant_folding=True,
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "sequence_length"},
                "attention_mask": {0: "batch_size", 1: "sequence_length"},
                "start_logits": {0: "batch_size", 1: "sequence_length"},
                "end_logits": {0: "batch_size", 1: "sequence_length"},
            },
        )

        logging.info(f"Model successfully exported to {onnx_path}")
    except Exception as e:
        logging.error(f"Error exporting model: {e}")
        raise e

    return onnx_path

def create_onnx_session(onnx_path):
    """Creates and returns an ONNX Runtime session."""
    options = onnxruntime.SessionOptions()
    options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL

    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX model not found at {onnx_path}. Ensure it's exported first.")

    return onnxruntime.InferenceSession(str(onnx_path), options)
