import logging
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from huggingface_hub import login
import torch
from transformers import GPT2Tokenizer, GPT2Model, pipeline
import spacy
from fastapi.middleware.cors import CORSMiddleware
import os
import csv

# Import the routers for the APIs from the 'functions' folder
from functions.text_generation import router as text_gen_router
from functions.location_extraction import router as loc_extract_router
from functions.skill_extraction import router as skill_extract_router

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Model identifiers from Hugging Face model hub
GPT2_MODEL = 'gpt2'  # Using GPT-2 for text generation
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
        # Load GPT-2 model from Hugging Face Hub
        tokenizer = GPT2Tokenizer.from_pretrained(GPT2_MODEL, cache_dir=CACHE_DIR)
        model = GPT2Model.from_pretrained(GPT2_MODEL, cache_dir=CACHE_DIR)
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
# Helper function to load CSV data into memory or create if not found
def load_csv_to_memory():
    global cities_countries, countries

    CSV_FILE_PATH = './data/cities5000.csv'  # Path to CSV file containing city and country data
    
    # Check if the file exists
    if not os.path.exists(CSV_FILE_PATH):
        logging.warning(f"CSV file not found at {CSV_FILE_PATH}. Creating the file...")
        
        # Create the 'data' directory if it doesn't exist
        os.makedirs(os.path.dirname(CSV_FILE_PATH), exist_ok=True)
        
        # Create an empty CSV file or initialize with some data
        try:
            with open(CSV_FILE_PATH, mode='w', newline='', encoding='utf-8') as csv_file:
                writer = csv.writer(csv_file)
                # Example of writing header (can be customized with your data)
                writer.writerow(['City', 'Country'])
                logging.info(f"CSV file created at {CSV_FILE_PATH}.")
                
            # Set the file permissions (read, write, execute)
            os.chmod(CSV_FILE_PATH, 0o777)  # This gives full permissions (read, write, execute) for everyone
            logging.info(f"File permissions set to 777 for {CSV_FILE_PATH}.")
        
        except Exception as e:
            logging.error(f"Error creating CSV file: {e}")
            raise HTTPException(status_code=500, detail="Error creating CSV file")
    
    # Read and process the CSV file
    try:
        with open(CSV_FILE_PATH, mode='r', newline='', encoding='utf-8') as csv_file:
            csv_reader = csv.reader(csv_file)
            cities_countries = {row[0].lower(): row[1] for row in csv_reader if len(row) > 1}
            # Re-open the file to read it again and get unique countries
            csv_file.seek(0)
            countries = {row[1].lower() for row in csv_reader if len(row) > 1}

        logging.info(f"CSV data loaded successfully. Number of cities: {len(cities_countries)}")
    
    except Exception as e:
        logging.error(f"Error loading CSV file: {e}")
        raise HTTPException(status_code=500, detail="Error loading CSV file")

# To run the app, use:
# uvicorn app:app --reload