import logging
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline
from typing import List


# Initialize FastAPI
app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Paths to local model directories
MODEL_DIR = 'models'

# Lazy load models to optimize memory usage
text_generator = None
ner_model = None

STOP_WORDS = {'a', 'the', 'is', 'in', 'on', 'and'}

class TextPayload(BaseModel):
    text: str

class LocationPayload(BaseModel):
    text: str
    countries_list: List[str]
    cities_list: List[str]

# Define the startup event to load models
@app.on_event("startup")
async def load_models():
    logging.info("Loading models...")
    
    # Load the local models
    global text_generator, ner_model
    
    gpt2_model_path = os.path.join(MODEL_DIR, 'distilgpt2')
    bert_model_path = os.path.join(MODEL_DIR, 'dbmdz/bert-large-cased-finetuned-conll03-english')

    # Load models using the path to local directories
    text_generator = pipeline('text-generation', model=gpt2_model_path)
    ner_model = pipeline('ner', model=bert_model_path)
    
    logging.info("Models loaded.")

@app.post("/generate_text/")
async def generate_text(payload: TextPayload):
    try:
        logging.info(f"Request to generate text: {payload.text}")
        generated_text = text_generator(payload.text, max_length=50, num_return_sequences=1)[0]['generated_text']
        return {"generated_text": generated_text}
    except Exception as e:
        logging.error(f"Error generating text: {e}")
        raise HTTPException(status_code=500, detail="Error generating text")

@app.post("/extract_locations/")
async def extract_locations(payload: LocationPayload):
    try:
        logging.info(f"Request to extract locations: {payload.text}")
        ner_results = ner_model(payload.text)
        locations = [result['word'] for result in ner_results if result['entity'] in ['B-LOC', 'I-LOC']]
        return {"locations": locations}
    except Exception as e:
        logging.error(f"Error extracting locations: {e}")
        raise HTTPException(status_code=500, detail="Error extracting locations")