import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from transformers import pipeline
import re

# Set up the router for Text Generation API endpoint
router = APIRouter()

# Pydantic model for request payload
class TextPayload(BaseModel):
    text: str

# Helper function to extract name from text
def extract_name(text: str) -> str:
    match = re.search(r"(?:hi|hello|hey)[, ]+my name is (\w+)", text, re.IGNORECASE)
    if match:
        return match.group(1)
    return None

# Endpoint for text generation using DialoGPT-medium
@router.post("/generate_text/")
async def generate_text(payload: TextPayload, text_generator: pipeline):
    try:
        logging.info(f"Request to generate text: {payload.text}")

        # Extract name from the input text
        name = extract_name(payload.text)
        if name:
            # Generate a personalized welcome message
            welcome_message = f"Hi {name}! Welcome to Intel Mind. How can I assist you today?"
            return {"response": welcome_message}
        else:
            # Generate text using DialoGPT-medium
            generated_text = text_generator(
                payload.text,
                max_length=50,
                num_return_sequences=1,
                temperature=0.7,  # Adjust for creativity
                top_k=50,
                top_p=0.95,
                do_sample=True
            )[0]['generated_text']
            return {"response": generated_text}
    except Exception as e:
        logging.error(f"Error generating text: {e}")
        raise HTTPException(status_code=500, detail="Error generating text")