import logging
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from huggingface_hub import login
from transformers import GPT2Tokenizer, GPT2ForQuestionAnswering
from fastapi.middleware.cors import CORSMiddleware
import os

# Import the routers for the APIs from the 'functions' folder
from functions.text_generation import router as text_gen_router
from functions.location_extraction import router as loc_extract_router
from functions.skill_extraction import router as skill_extract_router

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Cache directory to manage model storage
CACHE_DIR = "./cache"

# Lazy load models
text_generator = None
qa_model = None
tokenizer = None


# Asynchronous context manager for managing model loading and cleanup
@asynccontextmanager
async def lifespan(app: FastAPI):
    global text_generator, qa_model, tokenizer

    hf_token = "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"  # Your Hugging Face token
    try:
        login(token=hf_token)
        logging.info("Successfully logged into Hugging Face.")
    except Exception as e:
        logging.error(f"Error logging into Hugging Face: {e}")
        raise HTTPException(
            status_code=500, detail="Failed to authenticate with Hugging Face"
        )

    logging.info("Starting up and loading models...")

    try:
        # Load GPT-2 tokenizer and question answering model
        repo_id = "openai-community/gpt2"  # Your model identifier
        tokenizer = GPT2Tokenizer.from_pretrained(repo_id, cache_dir=CACHE_DIR)
        qa_model = GPT2ForQuestionAnswering.from_pretrained(
            repo_id, cache_dir=CACHE_DIR
        )

        logging.info("Models loaded successfully.")

        load_txt_to_csv

        logging.info("Files loaded successfully.")

    except Exception as e:
        logging.error(f"Error loading models: {e}")
        raise HTTPException(status_code=500, detail="Error loading models")

    yield

    logging.info("Shutting down, cleaning up resources...")
    text_generator = None
    qa_model = None
    tokenizer = None
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
def load_txt_to_csv():
    global cities_countries, countries

    TXT_FILE_PATH = "./data/cities1500.txt"  # Path to the txt file
    CSV_FILE_PATH = "./data/cities5000.csv"  # Path to save the CSV file

    # Check if the txt file exists
    if not os.path.exists(TXT_FILE_PATH):
        logging.warning(f"TXT file not found at {TXT_FILE_PATH}.")
        return

    try:
        with open(TXT_FILE_PATH, mode="r", encoding="utf-8") as txt_file:
            lines = txt_file.readlines()

        logging.info(
            f"Successfully read the txt file at {TXT_FILE_PATH}. Parsing data..."
        )

        # Prepare the CSV file for writing
        with open(CSV_FILE_PATH, mode="w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)

            # Writing the header for CSV
            writer.writerow(
                [
                    "Geoname ID",
                    "City",
                    "Alternative City Names",
                    "Latitude",
                    "Longitude",
                    "Feature Class",
                    "Feature Code",
                    "Country Code",
                    "Admin 1 Code",
                    "Admin 2 Code",
                    "Admin 3 Code",
                    "Admin 4 Code",
                    "Population",
                    "Elevation",
                    "Timezone",
                    "Modification Date",
                ]
            )

            # Parse each line and write to CSV
            for line in lines:
                fields = line.strip().split("\t")
                if len(fields) == 17:  # Ensure correct number of columns
                    writer.writerow(fields)
                else:
                    logging.warning(f"Skipping malformed line: {line.strip()}")

        logging.info(f"CSV file successfully created at {CSV_FILE_PATH}.")

        # Set file permissions to make it accessible
        os.chmod(CSV_FILE_PATH, 0o777)
        logging.info(f"File permissions set to 777 for {CSV_FILE_PATH}.")

    except Exception as e:
        logging.error(f"Error processing the TXT file and creating the CSV: {e}")
        raise HTTPException(
            status_code=500, detail="Error processing the TXT file and creating the CSV"
        )


# To run the app, use:
# uvicorn app:app --reload
