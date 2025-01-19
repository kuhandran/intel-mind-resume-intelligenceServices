import logging
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline
from typing import List
from fastapi.middleware.cors import CORSMiddleware
from huggingface_hub import login

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Model identifiers from Hugging Face model hub
GPT2_MODEL = 'distilgpt2'
BERT_MODEL = 'dbmdz/bert-large-cased-finetuned-conll03-english'

# Lazy load models
text_generator = None
ner_model = None
skill_extractor_model = None

STOP_WORDS = {'a', 'the', 'is', 'in', 'on', 'and'}

class TextPayload(BaseModel):
    text: str

class LocationPayload(BaseModel):
    text: str
    countries_list: List[str]
    cities_list: List[str]

class SkillPayload(BaseModel):
    text: str

# Define lifespan context manager to load and unload models
async def lifespan(app: FastAPI):
    global text_generator, ner_model, skill_extractor_model
    
    # Hardcoded Hugging Face authentication token (for testing)
    hf_token = "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"  # Hardcoded for testing purposes
    
    try:
        # Authenticate with Hugging Face API using the provided token
        login(token=hf_token)
        logging.info("Successfully logged into Hugging Face.")
    except Exception as e:
        logging.error(f"Error logging into Hugging Face: {e}")
        raise HTTPException(status_code=500, detail="Failed to authenticate with Hugging Face")

    # Startup logic: Load models
    logging.info("Starting up and loading models...")
    
    try:
        # Load models from Hugging Face Hub
        text_generator = pipeline('text-generation', model=GPT2_MODEL)
        ner_model = pipeline('ner', model=BERT_MODEL)

        # Load a model for skill extraction. We can use a pre-trained model or fine-tune one for skills.
        # For simplicity, we're using the same NER model as a placeholder for skill extraction.
        skill_extractor_model = ner_model  # Placeholder for skill extraction

        logging.info("Models loaded successfully.")
    except Exception as e:
        logging.error(f"Error loading models: {e}")
        raise HTTPException(status_code=500, detail="Error loading models")
    
    yield  # Application runs during this pause
    
    # Shutdown logic: Clean up resources
    logging.info("Shutting down, cleaning up resources...")
    text_generator = None
    ner_model = None
    skill_extractor_model = None
    logging.info("Cleanup complete.")

# Initialize FastAPI with lifespan function
app = FastAPI(lifespan=lifespan)

# Add CORS middleware to allow all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins, change to specific domains if needed
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

@app.post("/generate_text/")
async def generate_text(payload: TextPayload):
    try:
        logging.info(f"Request to generate text: {payload.text}")
        # Generate text based on the input
        generated_text = text_generator(payload.text, max_length=50, num_return_sequences=1)[0]['generated_text']
        return {"generated_text": generated_text}
    except Exception as e:
        logging.error(f"Error generating text: {e}")
        raise HTTPException(status_code=500, detail="Error generating text")

@app.post("/extract_locations/")
async def extract_locations(payload: LocationPayload):
    try:
        logging.info(f"Request to extract locations: {payload.text}")
        # Extract locations using the NER model
        ner_results = ner_model(payload.text)
        locations = [result['word'] for result in ner_results if result['entity'] in ['B-LOC', 'I-LOC']]
        
        # Filter extracted locations by countries and cities
        filtered_locations = [location for location in locations if location.lower() in [x.lower() for x in payload.countries_list + payload.cities_list]]
        return {"locations": filtered_locations}
    except Exception as e:
        logging.error(f"Error extracting locations: {e}")
        raise HTTPException(status_code=500, detail="Error extracting locations")

@app.post("/extract_skills/")
async def extract_skills(payload: SkillPayload):
    try:
        logging.info(f"Request to extract skills: {payload.text}")
        # Extract skills using the skill extractor model (here we use the same NER model as a placeholder)
        ner_results = skill_extractor_model(payload.text)
        
        # Extract possible skill-related words (this is a simplified approach; you can fine-tune a specific skill extraction model)
        skills = [result['word'] for result in ner_results if result['entity'] in ['B-MISC', 'I-MISC']]  # Assuming skills are labeled as MISC
        
        # Return extracted skills
        return {"skills": skills}
    except Exception as e:
        logging.error(f"Error extracting skills: {e}")
        raise HTTPException(status_code=500, detail="Error extracting skills")