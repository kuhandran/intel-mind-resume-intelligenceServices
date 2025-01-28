import logging
from fastapi import FastAPI
from contextlib import asynccontextmanager
from huggingface_hub import login
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import spacy
from fastapi.middleware.cors import CORSMiddleware
from fastapi import HTTPException
import os
import csv
import re

# Import the routers for the APIs from the 'functions' folder
from functions.text_generation import router as text_gen_router
from functions.location_extraction import router as loc_extract_router
from functions.skill_extraction import router as skill_extract_router

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

# Asynchronous context manager for managing model loading and cleanup
@asynccontextmanager
async def lifespan(app: FastAPI):
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

        # Load CSV for cities and countries
        load_csv_to_memory()

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

# Registering the routes from the respective files in the 'functions' folder
app.include_router(text_gen_router)
app.include_router(loc_extract_router)
app.include_router(skill_extract_router)

# CORS middleware setup to allow requests from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can specify specific domains here
    allow_credentials=True,
    allow_methods=["*"],  # You can limit methods to GET, POST, etc.
    allow_headers=["*"],  # You can specify specific headers here
)

# Helper function to load CSV data into memory
def load_csv_to_memory():
    global cities_countries, countries

    CSV_FILE_PATH = './data/cities5000.csv'  # Path to CSV file containing city and country data
    if not os.path.exists(CSV_FILE_PATH):
        logging.error(f"CSV file not found: {CSV_FILE_PATH}")
        raise HTTPException(status_code=500, detail="CSV file not found")

    # Read CSV and process it
    try:
        with open(CSV_FILE_PATH, mode='r') as csv_file:
            csv_reader = csv.reader(csv_file)
            cities_countries = {row[0].lower(): row[1] for row in csv_reader if len(row) > 1}
            # Add countries to a separate set for easy matching
            countries = {row[1].lower() for row in csv_reader if len(row) > 1}

        logging.info(f"CSV data loaded successfully. Number of cities: {len(cities_countries)}")
    except Exception as e:
        logging.error(f"Error loading CSV file: {e}")
        raise HTTPException(status_code=500, detail="Error loading CSV file")

# To run the app, use:
# uvicorn app:app --reload