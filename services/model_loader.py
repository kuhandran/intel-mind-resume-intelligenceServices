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
ONNX_DIR = "./onnx"
MODEL_NAME = "deepset/roberta-base-squad2"

# Quantization function for model size optimization
def quantize_model(model):
    """Quantizes a model for reduced memory usage."""
    return torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)


# Export the model to ONNX format
def export_to_onnx(model, tokenizer, output_dir=ONNX_DIR):
    """Exports the model to ONNX format."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = output_dir / "roberta-qa.onnx"

    try:
        logging.info(f"Exporting model to {onnx_path}...")
        export(model, tokenizer, feature="question-answering", opset=12, output=onnx_path)
        logging.info(f"Model successfully exported to {onnx_path}")
    except Exception as e:
        logging.error(f"Error during export to ONNX: {e}")
        raise e

    return onnx_path


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

    logging.info("Exporting model to ONNX...")
    onnx_path = export_to_onnx(model, tokenizer)

    logging.info("Creating ONNX session...")
    onnx_session = create_onnx_session(onnx_path)

    return model, tokenizer, onnx_session


