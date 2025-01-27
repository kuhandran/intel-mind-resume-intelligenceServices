import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from typing import List, AsyncGenerator
from fastapi.middleware.cors import CORSMiddleware
from huggingface_hub import login
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Model identifiers from Hugging Face model hub
T5_MODEL = 't5-small'  # Using T5-small for text generation
BERT_MODEL = 'dbmdz/bert-large-cased-finetuned-conll03-english'  # BERT model for NER

# Cache directory to manage model storage
CACHE_DIR = './cache'

# Lazy load models
text_generator = None
ner_model = None

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
    global text_generator, ner_model

    hf_token = "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"  # Your Hugging Face token
    try:
        login(token=hf_token)
        logging.info("Successfully logged into Hugging Face.")
    except Exception as e:
        logging.error(f"Error logging into Hugging Face: {e}")
        raise HTTPException(status_code=500, detail="Failed to authenticate with Hugging Face")

    logging.info("Starting up and loading models...")

    try:
        # Load text generation model from Hugging Face Hub (T5-small)
        text_generator = pipeline('text2text-generation', model=T5_MODEL, device=-1)

        # Load NER model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL, cache_dir=CACHE_DIR)
        model = AutoModelForSeq2SeqLM.from_pretrained(BERT_MODEL, cache_dir=CACHE_DIR)
        ner_model = pipeline('ner', model=model, tokenizer=tokenizer, device=-1)

        logging.info("Models loaded successfully.")
    except Exception as e:
        logging.error(f"Error loading models: {e}")
        raise HTTPException(status_code=500, detail="Error loading models")

    yield

    logging.info("Shutting down, cleaning up resources...")
    text_generator = None
    ner_model = None
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

# Endpoint for text generation using T5
@app.post("/generate_text/")
async def generate_text(payload: TextPayload):
    try:
        logging.info(f"Request to generate text: {payload.text}")
        # Prepare input text for T5 and generate text based on the input prompt
        generated_text = text_generator(f"generate: {payload.text}", max_length=50, num_return_sequences=1)[0]['generated_text']
        return {"generated_text": generated_text}
    except Exception as e:
        logging.error(f"Error generating text: {e}")
        raise HTTPException(status_code=500, detail="Error generating text")

# Endpoint for extracting location entities from text (using NER)
@app.post("/extract_locations/")
async def extract_locations(payload: LocationPayload):
    try:
        logging.info(f"Request to extract locations: {payload.text}")
        # Extract named entities
        ner_results = ner_model(payload.text)
        locations = [result['word'] for result in ner_results if result['entity'] in ['B-LOC', 'I-LOC']]
        # Filter locations based on provided countries and cities list
        filtered_locations = [location for location in locations if location.lower() in map(str.lower, payload.countries_list + payload.cities_list)]
        return {"locations": filtered_locations}
    except Exception as e:
        logging.error(f"Error extracting locations: {e}")
        raise HTTPException(status_code=500, detail="Error extracting locations")

# Endpoint for extracting skills from text (using NER)
@app.post("/extract_skills/")
async def extract_skills(payload: SkillPayload):
    try:
        logging.info(f"Request to extract skills: {payload.text}")
        # Extract named entities
        ner_results = ner_model(payload.text)
        skills = [result['word'] for result in ner_results if result['entity'] in ['B-MISC', 'I-MISC']]
        return {"skills": skills}
    except Exception as e:
        logging.error(f"Error extracting skills: {e}")
        raise HTTPException(status_code=500, detail="Error extracting skills")

# To run the app, use:
# uvicorn main:app --reload