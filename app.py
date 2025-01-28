import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from typing import List, AsyncGenerator
from fastapi.middleware.cors import CORSMiddleware
from huggingface_hub import login
from contextlib import asynccontextmanager
import spacy
import torch
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Model identifiers from Hugging Face model hub
DIALOGPT_MODEL = 'microsoft/DialoGPT-medium'  # Using DialoGPT-medium for text generation
SPACY_MODEL = 'en_core_web_sm'  # spaCy model for NER

# Cache directory to manage model storage
CACHE_DIR = './cache'

# Lazy load models
text_generator = None
nlp = None

# Pydantic models for request payloads
class TextPayload(BaseModel):
    text: str

class LocationPayload(BaseModel):
    text: str
    countries_list: List[str]
    cities_list: List[str]

class SkillPayload(BaseModel):
    text: str

# Asynchronous context manager for managing model loading and cleanup
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    global text_generator, nlp

    hf_token = "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"  # Your Hugging Face token
    try:
        login(token=hf_token)
        logging.info("Successfully logged into Hugging Face.")
    except Exception as e:
        logging.error(f"Error logging into Hugging Face: {e}")
        raise HTTPException(status_code=500, detail="Failed to authenticate with Hugging Face")

    logging.info("Starting up and loading models...")

    try:
        # Load text generation model from Hugging Face Hub (DialoGPT-medium)
        tokenizer = AutoTokenizer.from_pretrained(DIALOGPT_MODEL, cache_dir=CACHE_DIR)
        model = AutoModelForCausalLM.from_pretrained(DIALOGPT_MODEL, cache_dir=CACHE_DIR)
        text_generator = pipeline('text-generation', model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)

        # Load spaCy model for NER
        nlp = spacy.load(SPACY_MODEL)

        logging.info("Models loaded successfully.")
    except Exception as e:
        logging.error(f"Error loading models: {e}")
        raise HTTPException(status_code=500, detail="Error loading models")

    yield

    logging.info("Shutting down, cleaning up resources...")
    text_generator = None
    nlp = None
    logging.info("Cleanup complete.")

# FastAPI app setup
app = FastAPI(lifespan=lifespan)

# CORS middleware setup to allow requests from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can specify specific domains here
    allow_credentials=True,
    allow_methods=["*"],  # You can limit methods to GET, POST, etc.
    allow_headers=["*"],  # You can specify specific headers here
)

# Helper function to extract name from text
def extract_name(text: str) -> str:
    """
    Extracts a name from the input text using a simple regex pattern.
    Assumes the input text contains a phrase like "Hi, my name is [Name]".
    """
    match = re.search(r"(?:hi|hello|hey)[, ]+my name is (\w+)", text, re.IGNORECASE)
    if match:
        return match.group(1)
    return None

# Endpoint for text generation using DialoGPT-medium
@app.post("/generate_text/")
async def generate_text(payload: TextPayload):
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

# Endpoint for extracting location entities from text (using spaCy)
@app.post("/extract_locations/")
async def extract_locations(payload: LocationPayload):
    try:
        logging.info(f"Request to extract locations: {payload.text}")
        # Extract named entities using spaCy
        doc = nlp(payload.text)
        locations = [ent.text for ent in doc.ents if ent.label_ in ['GPE', 'LOC']]
        # Filter locations based on provided countries and cities list
        filtered_locations = [location for location in locations if location.lower() in map(str.lower, payload.countries_list + payload.cities_list)]
        return {"locations": filtered_locations}
    except Exception as e:
        logging.error(f"Error extracting locations: {e}")
        raise HTTPException(status_code=500, detail="Error extracting locations")

# Endpoint for extracting skills from text (using spaCy)
@app.post("/extract_skills/")
async def extract_skills(payload: SkillPayload):
    try:
        logging.info(f"Request to extract skills: {payload.text}")
        # Extract named entities using spaCy
        doc = nlp(payload.text)
        skills = [ent.text for ent in doc.ents if ent.label_ in ['SKILL']]  # Add custom labels if needed
        return {"skills": skills}
    except Exception as e:
        logging.error(f"Error extracting skills: {e}")
        raise HTTPException(status_code=500, detail="Error extracting skills")

# To run the app, use:
# uvicorn main:app --reload