import logging
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline
from typing import List
from collections import Counter
import re

# Initialize FastAPI
app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load Hugging Face models
text_generator = pipeline('text-generation', model='gpt2')
ner_model = pipeline('ner', model='dbmdz/bert-large-cased-finetuned-conll03-english')

# Example stop words list
STOP_WORDS = {'a', 'the', 'is', 'in', 'on', 'and'}

def extract_location_keywords(text: str, num_keywords: int = 20) -> list:
    words = re.findall(r'\b\w+\b', text.lower())
    filtered_words = [word for word in words if word not in STOP_WORDS]
    word_counts = Counter(filtered_words)
    keywords = [word for word, count in word_counts.most_common(num_keywords)]
    return keywords

class TextPayload(BaseModel):
    text: str

class LocationPayload(BaseModel):
    text: str
    countries_list: List[str]
    cities_list: List[str]

@app.post("/generate_text/")
async def generate_text(payload: TextPayload):
    try:
        logging.info(f"Request to generate text: {payload.text}")
        
        # Use Hugging Face's text generation pipeline
        generated_text = text_generator(payload.text, max_length=50, num_return_sequences=1)[0]['generated_text']
        
        return {"generated_text": generated_text}

    except Exception as e:
        logging.error(f"Error generating text: {e}")
        raise HTTPException(status_code=500, detail="Error generating text")

@app.post("/extract_locations/")
async def extract_locations(payload: LocationPayload):
    try:
        logging.info(f"Request to extract locations: {payload.text}")
        
        # Use Hugging Face's NER pipeline
        ner_results = ner_model(payload.text)
        locations = [result['word'] for result in ner_results if result['entity'] in ['B-LOC', 'I-LOC']]
        
        return {"locations": locations}

    except Exception as e:
        logging.error(f"Error extracting locations: {e}")
        raise HTTPException(status_code=500, detail="Error extracting locations")