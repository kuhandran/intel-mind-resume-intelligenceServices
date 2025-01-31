import logging
import torch
import onnxruntime
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from pathlib import Path
from services.utils import quantize_model, export_to_onnx, create_onnx_session

# Logging setup
logging.basicConfig(level=logging.INFO)

# Directories and model info
CACHE_DIR = "./cache"
MODEL_NAME = "deepset/roberta-base-squad2"
ONNX_DIR = Path("onnx")
ONNX_PATH = ONNX_DIR / "roberta-qa.onnx"

def load_models():
    """Loads the tokenizer, model, and ONNX session."""
    logging.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)

    logging.info("Loading and quantizing model...")
    model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
    model = quantize_model(model)

    logging.info("Exporting to ONNX if needed...")
    export_to_onnx(model, ONNX_PATH)

    logging.info("Creating ONNX session...")
    onnx_session = create_onnx_session(ONNX_PATH)

    return tokenizer, onnx_session

# Load models on startup
tokenizer, onnx_session = load_models()
