import logging
import torch
import random
import onnxruntime
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
from pathlib import Path

# Set up the router for Text Generation API endpoint
router = APIRouter()

# Pydantic model for request payload
class TextPayload(BaseModel):
    question: str
    context: str = None  # Make context optional

# Pydantic model for response
class TextResponse(BaseModel):
    response: str

# Logging setup
logging.basicConfig(level=logging.INFO)

# Initialize the tokenizer and ONNX session for question answering
MODEL_NAME = "deepset/roberta-base-squad2"
ONNX_DIR = Path("./onnx")
ONNX_PATH = ONNX_DIR / "roberta-qa.onnx"

# Initialize the tokenizer globally
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def export_model_to_onnx():
    """Exports the RoBERTa model to ONNX format."""
    logging.info("Exporting RoBERTa model to ONNX format...")
    try:
        # Ensure the ONNX directory exists
        ONNX_DIR.mkdir(parents=True, exist_ok=True)

        # Load the model from HuggingFace transformers
        model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME)

        # Create dummy input for export
        dummy_input = {
            "input_ids": torch.randint(0, 1000, (1, 512)),  # Example input with size 512
            "attention_mask": torch.ones(1, 512),  # Attention mask of ones with size 512
        }

        # Ensure the model is in evaluation mode
        model.eval()

        # Export the model to ONNX format
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
        logging.info(f"Model exported successfully to {ONNX_PATH}")
    except Exception as e:
        logging.error(f"Error during model export: {e}")
        raise HTTPException(status_code=500, detail="Error exporting model to ONNX")

def create_onnx_session():
    """Creates and returns an ONNX Runtime session."""
    options = onnxruntime.SessionOptions()
    options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    return onnxruntime.InferenceSession(str(ONNX_PATH), options)

# Ensure model export if ONNX path does not exist
if not ONNX_PATH.exists():
    export_model_to_onnx()

# Create ONNX session
onnx_session = create_onnx_session()

# Default context
default_context = "Welcome to IntelMind! I am an advanced AI here to assist with processing and understanding professional profiles."

def get_answer(question: str, context: str = None) -> str:
    """Get answer from model based on the question and context."""
    context = context or default_context

    # Tokenize input using the tokenizer
    inputs = tokenizer(question, context, return_tensors="pt", padding='max_length', truncation=True, max_length=512)

    # Convert inputs to numpy arrays for ONNX Runtime
    input_ids = inputs["input_ids"].cpu().numpy()
    attention_mask = inputs["attention_mask"].cpu().numpy()

    # Ensure the input size matches expected dimensions
    if input_ids.shape[1] != 512:
        raise HTTPException(status_code=400, detail="Input question exceeds token limit.")

    # Run inference using ONNX Runtime
    ort_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
    ort_outputs = onnx_session.run(None, ort_inputs)

    # Extract start and end logits
    start_logits, end_logits = ort_outputs

    # Find the most probable start and end positions for the answer
    answer_start = start_logits.argmax()
    answer_end = end_logits.argmax()

    # Decode the answer tokens
    answer_tokens = input_ids[0, answer_start: answer_end + 1]
    answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)

    return answer if answer.strip() else "I'm not sure I understood your question fully. Can you clarify?"

# Endpoint for text generation using RoBERTa (Question Answering)
@router.post("/generate_text/", response_model=TextResponse)
async def generate_text(payload: TextPayload) -> TextResponse:
    try:
        logging.info(f"Received question: {payload.question}")

        # Get the answer based on the provided context
        answer = get_answer(payload.question, payload.context)

        return TextResponse(response=answer)
    except Exception as e:
        logging.error(f"Error generating text: {e}")
        raise HTTPException(status_code=500, detail="Error generating text")