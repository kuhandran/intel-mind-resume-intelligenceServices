import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from transformers import GPT2Tokenizer, GPT2LMHeadModel, pipeline
import re
from typing import List

# Set up the router for Text Generation API endpoint
router = APIRouter()

# Pydantic model for request payload
class TextPayload(BaseModel):
    text: str

# Pydantic model for response
class TextResponse(BaseModel):
    response: str

# Initialize the text generation pipeline outside the route handler
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')
text_generator = pipeline('text-generation', model=model, tokenizer=tokenizer)

# Helper function to extract name from text
def extract_name(text: str) -> str:
    match = re.search(r"(?:hi|hello|hey)[, ]+my name is ([\w\s]+)", text, re.IGNORECASE)
    if match:
        return match.group(1).strip()  # Ensure the name is trimmed of extra spaces
    return None

# Endpoint for text generation using GPT-2
@router.post("/generate_text/", response_model=TextResponse)
async def generate_text(payload: TextPayload) -> TextResponse:
    try:
        logging.info(f"Received text for generation: {payload.text}")

        # Extract name from the input text
        name = extract_name(payload.text)
        if name:
            # Generate a personalized welcome message
            welcome_message = f"Hi {name}! Welcome to Intel Mind. How can I assist you today?"
            return TextResponse(response=welcome_message)
        else:
            # Generate text using GPT-2 model
            generated_text = text_generator(
                payload.text,
                max_length=50,
                num_return_sequences=1,
                temperature=0.7,  # Adjust for creativity
                top_k=50,
                top_p=0.95,
                do_sample=True
            )[0]['generated_text']
            logging.info(f"Generated text: {generated_text}")
            return TextResponse(response=generated_text)
    except Exception as e:
        logging.error(f"Error generating text: {e}")
        raise HTTPException(status_code=500, detail="Error generating text")