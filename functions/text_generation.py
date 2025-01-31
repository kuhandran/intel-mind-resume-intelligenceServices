import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from services.inference import get_answer  # Importing from models.py

# Set up the router for Text Generation API endpoint
router = APIRouter()

# Pydantic model for request payload
class TextPayload(BaseModel):
    question: str
    context: str = None  # Optional context

# Pydantic model for response
class TextResponse(BaseModel):
    response: str

# Default context
default_context = "Welcome to IntelMind! I am an AI assistant for professional profiles."

@router.post("/generate_text/", response_model=TextResponse)
def generate_text(payload: TextPayload) -> TextResponse:
    try:
        logging.info(f"Received question: {payload.question}")
        answer = get_answer(payload.question, payload.context or default_context)
        return TextResponse(response=answer)
    except Exception as e:
        logging.error(f"Error generating text: {e}")
        raise HTTPException(status_code=500, detail="Error generating text")
