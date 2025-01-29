import logging
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from fastapi.middleware.cors import CORSMiddleware
import os
import csv
import pandas as pd

# Import the routers for the APIs from the 'functions' folder
from functions.text_generation import router as text_gen_router
from functions.location_extraction import router as loc_extract_router
from functions.skill_extraction import router as skill_extract_router


# Configure root logger to show messages in console
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Clear existing handlers
if logger.hasHandlers():
    logger.handlers.clear()

# Add console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(console_handler)

# Cache directory to manage model storage
CACHE_DIR = "./cache"

# Global variables for data storage
cities_data = None
countries_data = None

# Lazy load models
qa_model = None
tokenizer = None

def load_txt_to_csv():
    """Convert the cities text file to CSV format and load into memory with detailed logging"""
    TXT_FILE_PATH  = os.path.join(os.path.dirname(__file__), '../data/cities15000.txt')
    CSV_FILE_PATH  = os.path.join(os.path.dirname(__file__), '../data/cities5000.csv')
    LOG_FILE_PATH  = os.path.join(os.path.dirname(__file__), '../data/csv_conversion.log')
    

    # Configure file handler for logging
    file_handler = logging.FileHandler(LOG_FILE_PATH, mode='w')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logging.getLogger().addHandler(file_handler)

    # Check if TXT file exists
    if not os.path.exists(TXT_FILE_PATH):
        logging.error(f"TXT file not found at {TXT_FILE_PATH}.")
        raise HTTPException(status_code=404, detail=f"TXT file not found at {TXT_FILE_PATH}")

    try:
        # First, ensure the data directory exists for logging
        os.makedirs(os.path.dirname(LOG_FILE_PATH), exist_ok=True)

        # Read the text file and log its size
        with open(TXT_FILE_PATH, mode="r", encoding="utf-8") as txt_file:
            lines = txt_file.readlines()
            logging.info(f"Read {len(lines)} lines from {TXT_FILE_PATH}")

        # Create CSV file if it doesn't exist
        if not os.path.exists(CSV_FILE_PATH):
            logging.info(f"CSV file not found at {CSV_FILE_PATH}. Creating new CSV file...")
            
            # Write to CSV with verification
            rows_written = 0
            skipped_rows = 0
            with open(CSV_FILE_PATH, mode="w", newline="", encoding="utf-8") as csv_file:
                writer = csv.writer(csv_file, quoting=csv.QUOTE_ALL)

                # Write header
                header = [
                    "Geoname ID", "City", "Alternative City Names", "Latitude",
                    "Longitude", "Feature Class", "Feature Code", "Country Code",
                    "Admin 1 Code", "Admin 2 Code", "Admin 3 Code", "Admin 4 Code",
                    "Population", "Elevation", "Timezone", "Modification Date"
                ]
                writer.writerow(header)
                rows_written += 1
                logging.info("Header row written successfully")

                # Process each line from the text file
                for index, line in enumerate(lines, 1):
                    fields = line.strip().split("\t")
                    
                    try:
                        # Process fields in the correct order
                        processed_fields = [
                            fields[0],  # Geoname ID
                            fields[1],  # City
                            ",".join(fields[2:len(fields)-13]),  # Alternative Names
                            fields[-13],  # Latitude
                            fields[-12],  # Longitude
                            fields[-11],  # Feature Class
                            fields[-10],  # Feature Code
                            fields[-9],   # Country Code
                            fields[-8],   # Admin 1 Code
                            fields[-7],   # Admin 2 Code
                            fields[-6],   # Admin 3 Code
                            fields[-5],   # Admin 4 Code
                            fields[-4],   # Population
                            fields[-3],   # Elevation
                            fields[-2],   # Timezone
                            fields[-1]    # Modification Date
                        ]
                        
                        if len(processed_fields) == 16:
                            writer.writerow(processed_fields)
                            rows_written += 1
                            if rows_written % 1000 == 0:  # Log every 1000 records
                                logging.info(f"Processed {rows_written} records. Last city added: {processed_fields[1]}")
                            logging.debug(f"Added record {rows_written}: City={processed_fields[1]}, ID={processed_fields[0]}")
                        else:
                            skipped_rows += 1
                            logging.warning(
                                f"Skipped line {index} due to incorrect field count: {len(processed_fields)}. "
                                f"City: {fields[1] if len(fields) > 1 else 'Unknown'}"
                            )
                    except IndexError as e:
                        skipped_rows += 1
                        logging.warning(f"Error processing line {index}: {e}")
                        continue

            # Log final statistics
            logging.info(f"""
            Processing completed:
            - Total rows written: {rows_written}
            - Skipped rows: {skipped_rows}
            - Success rate: {(rows_written/(rows_written+skipped_rows))*100:.2f}%
            """)

            # Verify the file was written
            if os.path.exists(CSV_FILE_PATH):
                file_size = os.path.getsize(CSV_FILE_PATH)
                logging.info(f"CSV file size: {file_size} bytes")
                
                if file_size == 0:
                    logging.error("CSV file was created but is empty!")
                    raise HTTPException(status_code=500, detail="CSV file was created but is empty!")
            else:
                logging.error("CSV file was not created!")
                raise HTTPException(status_code=500, detail="CSV file creation failed!")

        else:
            logging.info(f"CSV file already exists at {CSV_FILE_PATH}.")

        # Load into memory and verify
        global cities_data
        try:
            cities_data = pd.read_csv(CSV_FILE_PATH)
            logging.info(f"""
            Data loaded into memory:
            - DataFrame shape: {cities_data.shape}
            - Memory usage: {cities_data.memory_usage().sum() / 1024**2:.2f} MB
            - Columns: {', '.join(cities_data.columns)}
            """)
            
            # Verify data was actually loaded
            if cities_data.empty:
                logging.error("DataFrame is empty after loading CSV!")
                raise HTTPException(status_code=500, detail="CSV loaded but DataFrame is empty!")
                
            return True
            
        except Exception as e:
            logging.error(f"Error loading CSV into memory: {e}")
            raise HTTPException(status_code=500, detail=f"Error loading CSV: {str(e)}")

    except Exception as e:
        logging.error(f"Error in load_txt_to_csv: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing the TXT file and creating the CSV: {str(e)}"
        )
    finally:
        # Remove the file handler to avoid duplicate logs
        logging.getLogger().removeHandler(file_handler)
        
# Context manager for model loading and cleanup
@asynccontextmanager
async def lifespan(app: FastAPI):
    global qa_model, tokenizer, cities_data

    logging.info("=== Starting Application Initialization ===")
    
    # First, handle the CSV data loading
    logging.info("Step 1: Loading city data...")
    try:
        csv_loaded = load_txt_to_csv()
        if not csv_loaded:
            logging.error("Failed to load city data - CSV conversion failed")
            raise HTTPException(status_code=500, detail="Failed to load city data")
        
        if cities_data is None or cities_data.empty:
            logging.error("City data is empty after loading")
            raise HTTPException(status_code=500, detail="City data is empty after loading")
            
        logging.info(f"Successfully loaded city data. Shape: {cities_data.shape}")
    except Exception as e:
        logging.error(f"Critical error during city data loading: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to load city data: {str(e)}")

    # Then handle Hugging Face authentication and model loading
    logging.info("Step 2: Authenticating with Hugging Face...")
    hf_token = "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"
    try:
        login(token=hf_token)
        logging.info("Successfully authenticated with Hugging Face")
    except Exception as e:
        logging.error(f"Hugging Face authentication failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to authenticate with Hugging Face"
        )

    logging.info("Step 3: Loading ML models...")
    try:
        # Load question-answering model
        model_name = "distilbert-base-cased-distilled-squad"
        
        logging.info(f"Loading tokenizer from {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=CACHE_DIR)
        
        logging.info(f"Loading model from {model_name}...")
        qa_model = AutoModelForQuestionAnswering.from_pretrained(
            model_name,
            cache_dir=CACHE_DIR
        )
        logging.info("ML models loaded successfully")
        
        # Verify models are loaded
        if qa_model is None or tokenizer is None:
            raise ValueError("Models failed to load properly")
            
    except Exception as e:
        logging.error(f"Error loading ML models: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to load ML models: {str(e)}")

    logging.info("=== Application Initialization Complete ===")
    logging.info(f"""
    Initialization Status:
    - Cities Data: {cities_data is not None and not cities_data.empty}
    - Cities Shape: {cities_data.shape if cities_data is not None else 'N/A'}
    - Tokenizer: {tokenizer is not None}
    - QA Model: {qa_model is not None}
    """)

    yield

    # Cleanup
    logging.info("=== Starting Application Shutdown ===")
    qa_model = None
    tokenizer = None
    cities_data = None
    logging.info("Models and data cleared, shutdown complete.")

# FastAPI app setup
app = FastAPI(
    title="Intel Mind Resume Intelligence Services",
    description="API for text generation, location extraction, and skill extraction",
    version="1.0.0",
    lifespan=lifespan
)

# Register routers
app.include_router(text_gen_router)
app.include_router(loc_extract_router)
app.include_router(skill_extract_router)

# CORS middleware setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check endpoint
@app.get("/health")
async def health_check():
    """Check the health status of the application and its components"""
    return {
        "status": "healthy",
        "models_loaded": qa_model is not None and tokenizer is not None,
        "cities_data_loaded": cities_data is not None,
        "cities_data_shape": cities_data.shape if cities_data is not None else None
    }

@app.get("/")
async def root():
    """Root endpoint returning basic API information"""
    return {
        "message": "Welcome to Intel Mind Resume Intelligence Services",
        "version": "1.0.0",
        "endpoints": [
            "/docs",  # Swagger documentation
            "/health",  # Health check
            "/extract-locations",  # Location extraction
            "/extract-skills"  # Skill extraction
        ]
    }

