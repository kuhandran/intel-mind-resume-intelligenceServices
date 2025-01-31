import logging
import torch
import onnxruntime
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from transformers.onnx import export
from pathlib import Path

# Setup logging
log_file = './model_export.log'
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(),  # Log to console
        logging.FileHandler(log_file)  # Log to file
    ]
)

# Directories and model info
CACHE_DIR = "./cache"
MODEL_NAME = "deepset/roberta-base-squad2"
ONNX_DIR = Path("onnx")
ONNX_PATH = ONNX_DIR / "roberta-qa.onnx"


# Quantization function for model size optimization
def quantize_model(model):
    """Quantizes a model for reduced memory usage."""
    return torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)

def export_to_onnx(model, output_dir=ONNX_DIR):
    """Exports the model to ONNX format."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create dummy input for export (input_ids, attention_mask)
    dummy_input = {
        "input_ids": torch.randint(0, 1000, (1, 512)),  # Example input with size 512
        "attention_mask": torch.ones(1, 512),  # Attention mask of ones with size 512
    }

    try:
        logging.info(f"Exporting model to {ONNX_PATH}...")

        # Forward pass through the model to get start and end logits
        model.eval()
        with torch.no_grad():
            outputs = model(**dummy_input)

            # Ensure outputs are tensors and not None
            start_logits = outputs.start_logits
            end_logits = outputs.end_logits

        # Export to ONNX, passing only the relevant logits
        torch.onnx.export(
            model,
            (dummy_input["input_ids"], dummy_input["attention_mask"]),
            str(ONNX_PATH),  # Path to save the ONNX model
            input_names=["input_ids", "attention_mask"],  # Names for the input nodes
            output_names=["start_logits", "end_logits"],  # Names for the output nodes
            opset_version=14,  # ONNX opset version
            do_constant_folding=True,  # Enable constant folding for optimizations
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "sequence_length"},  # Dynamic batch and sequence length
                "attention_mask": {0: "batch_size", 1: "sequence_length"},
                "start_logits": {0: "batch_size", 1: "sequence_length"},
                "end_logits": {0: "batch_size", 1: "sequence_length"},
            },  # Allow dynamic batch and sequence lengths
        )

        logging.info(f"Model successfully exported to {ONNX_PATH}")
    except Exception as e:
        logging.error(f"Error during export to ONNX: {e}")
        raise e

    return ONNX_PATH

# Create an ONNX session for inference
def create_onnx_session(onnx_path):
    """Creates an ONNX Runtime session."""
    options = onnxruntime.SessionOptions()
    options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    logging.info(f"Loading ONNX model from {onnx_path}...")
    return onnxruntime.InferenceSession(str(onnx_path), options)


# Load and prepare the model, tokenizer, and ONNX session
def load_models():
    """Loads and prepares models."""
    logging.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)

    logging.info("Loading model...")
    model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
    model = quantize_model(model)

    # Ensure the model is in evaluation mode before export
    model.eval()

    logging.info("Exporting model to ONNX...")
    onnx_path = export_to_onnx(model)  # No need to pass tokenizer

    logging.info("Creating ONNX session...")
    onnx_session = create_onnx_session(onnx_path)

    return model, tokenizer, onnx_session