import logging
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from pathlib import Path

# Logging setup
logging.basicConfig(level=logging.INFO)

# Configuration
CACHE_DIR = Path("./cache")
MODEL_NAME = "google/flan-t5-small"  # Use generative model

def load_models():
    """Loads the tokenizer and model."""
    logging.info("Starting the model loading process...")
    
    # Load tokenizer
    try:
        logging.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=str(CACHE_DIR))
        logging.info("Tokenizer loaded successfully.")
    except Exception as e:
        logging.error(f"Error loading tokenizer: {e}")
        raise

    # Load model
    try:
        logging.info(f"Attempting to load model: {MODEL_NAME}")
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, cache_dir=str(CACHE_DIR))
        logging.info(f"Model '{MODEL_NAME}' loaded successfully.")
    except Exception as e:
        logging.error(f"Error loading model '{MODEL_NAME}': {e}")
        raise

    return tokenizer, model

# Load once and reuse
tokenizer, model = None, None
try:
    tokenizer, model = load_models()
    logging.info("Models loaded and ready for use.")
except Exception as e:
    logging.error(f"Failed to load models: {e}")

def get_model():
    """Returns the loaded tokenizer and model."""
    if tokenizer and model:
        return tokenizer, model
    else:
        raise Exception("Model not loaded properly, unable to return.")