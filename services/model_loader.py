import logging
import torch
import onnxruntime
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from pathlib import Path
from services.utils import quantize_model, export_to_onnx, create_onnx_session
# from transformers import EncoderDecoderCache

# Logging setup
logging.basicConfig(level=logging.INFO)

# Configuration
CACHE_DIR = Path("./cache")
MODEL_NAME = "google/flan-t5-large"  # Use generative model
ONNX_DIR = CACHE_DIR / "onnx"
ONNX_PATH = ONNX_DIR / "flan-t5.onnx"

def load_models():
    """Loads the tokenizer and ONNX session."""
    logging.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=str(CACHE_DIR))

    # Ensure ONNX directory exists
    ONNX_DIR.mkdir(parents=True, exist_ok=True)

    if not ONNX_PATH.exists():
        logging.info("ONNX model not found, exporting...")
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, cache_dir=str(CACHE_DIR))
        model = quantize_model(model)  # Quantize model for optimization
        export_to_onnx(model, ONNX_PATH)  # Export to ONNX

    logging.info("Creating ONNX session...")
    onnx_session = create_onnx_session(ONNX_PATH)

    return tokenizer, onnx_session

# Load once and reuse
tokenizer, onnx_session = load_models()