import logging
import torch
import onnxruntime
from transformers import AutoTokenizer
from services.utils import create_onnx_session ,export_to_onnx
from pathlib import Path

# Logging setup
logging.basicConfig(level=logging.INFO)

# Directories and model info
CACHE_DIR = "./cache"
MODEL_NAME = "deepset/roberta-base-squad2"
ONNX_DIR = Path("onnx")
ONNX_PATH = ONNX_DIR / "roberta-qa.onnx"

from transformers import AutoModelForQuestionAnswering

def load_models():
    """Loads the tokenizer and ONNX session."""
    logging.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
    
    # Define the ONNX model path
    onnx_path = Path(CACHE_DIR) / "onnx" / "roberta-qa.onnx"

    # If the ONNX model doesn't exist, export it
    if not onnx_path.exists():
        logging.info("ONNX model not found, exporting...")
        # Load the pre-trained model
        model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME)
        export_to_onnx(model, onnx_path)  # Export the model to ONNX format

    logging.info("Creating ONNX session...")
    onnx_session = create_onnx_session(onnx_path)  # Now, load the ONNX session
    
    return tokenizer, onnx_session

tokenizer, onnx_session = load_models()

def get_answer(question: str, context: str = "Welcome to IntelMind!"):
    """Uses the ONNX model to generate an answer for the given question and context."""
    inputs = tokenizer(question, context, return_tensors="pt", padding='max_length', truncation=True, max_length=512)
    
    input_ids = inputs["input_ids"].cpu().numpy()
    attention_mask = inputs["attention_mask"].cpu().numpy()

    if input_ids.shape[1] != 512:
        raise ValueError("Input question exceeds token limit.")

    ort_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
    ort_outputs = onnx_session.run(None, ort_inputs)

    start_logits, end_logits = ort_outputs
    answer_start = start_logits.argmax()
    answer_end = end_logits.argmax()

    answer_tokens = input_ids[0, answer_start: answer_end + 1]
    answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)

    return answer if answer.strip() else "I'm not sure I understood your question. Can you clarify?"
