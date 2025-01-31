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

        model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME)
        
        # Create dummy input for export
        dummy_input = {
            "input_ids": torch.randint(0, 1000, (1, 512)),  # Example input
            "attention_mask": torch.ones(1, 512),
        }
        
        # Export the model
        torch.onnx.export(
            model,
            (dummy_input["input_ids"], dummy_input["attention_mask"]),
            str(ONNX_PATH),
            input_names=["input_ids", "attention_mask"],
            output_names=["start_logits", "end_logits"],
            opset_version=14,
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
    
    # Handle simple greetings and farewells
    greetings = ['hi', 'hello', 'hey', 'greetings', 'morning', 'evening', 'yo']
    farewells = ['bye', 'goodbye', 'see you', 'later']

    if any(g in question.lower() for g in greetings):
        return random.choice(["Hello! How can I assist you?", "Hi there! Need help?", "Greetings! What can I do for you?"])
    
    if any(f in question.lower() for f in farewells):
        return "Goodbye! It was a pleasure assisting you."

    # Tokenize input
    inputs = tokenizer(question, context, return_tensors="pt", padding=True, truncation=True, max_length=512)

    # Convert inputs to numpy arrays for ONNX Runtime
    input_ids = inputs["input_ids"].cpu().numpy()
    attention_mask = inputs["attention_mask"].cpu().numpy()

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