import logging
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from fastapi.middleware.cors import CORSMiddleware
import os
import csv
import pandas as pd
from fastapi.responses import JSONResponse
import aiofiles
import torch
from transformers import pipeline
import onnxruntime
from transformers.onnx import export
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from pathlib import Path

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
console_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
)
logger.addHandler(console_handler)

# Cache directory to manage model storage
CACHE_DIR = "./cache"

# Global variables for data storage
cities_data = None
countries_data = None

# Lazy load models
qa_model = None
tokenizer = None
onnx_session = None

# Define file paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TXT_FILE_PATH = os.path.join(BASE_DIR, "data/cities15000.txt")
CSV_FILE_PATH = os.path.join(BASE_DIR, "data/cities5000.csv")


async def convert_txt_to_csv():
    """Asynchronously converts a TXT file to a CSV file."""

    # Check if TXT file exists
    if not os.path.exists(TXT_FILE_PATH):
        logging.error(f"Error: TXT file not found -> {TXT_FILE_PATH}")
        return False

    try:
        # Open TXT file asynchronously
        async with aiofiles.open(TXT_FILE_PATH, "r", encoding="utf-8") as txt_file:
            contents = await txt_file.readlines()

        # Now, write the CSV file synchronously using the csv.writer
        with open(CSV_FILE_PATH, "w", encoding="utf-8", newline="") as csv_file:
            writer = csv.writer(csv_file, quoting=csv.QUOTE_ALL)

            # Write the header row
            writer.writerow(
                [
                    "geonameid",
                    "name",
                    "asciiname",
                    "alternatenames",
                    "latitude",
                    "longitude",
                    "feature class",
                    "feature code",
                    "country code",
                    "cc2",
                    "admin1 code",
                    "admin2 code",
                    "admin3 code",
                    "admin4 code",
                    "population",
                    "elevation",
                    "dem",
                    "timezone",
                    "modification date",
                ]
            )

            # Write the data rows
            for line in contents:
                data = line.strip().split("\t")
                if len(data) >= 19:
                    writer.writerow(data[:19])

        logging.info(f"CSV successfully created at: {CSV_FILE_PATH}")
        return True
    except FileNotFoundError:
        logging.error(f"File not found error while processing: {TXT_FILE_PATH}")
    except PermissionError:
        logging.error(
            f"Permission error while trying to access file: {TXT_FILE_PATH} or {CSV_FILE_PATH}"
        )
    except csv.Error as e:
        logging.error(f"CSV error: {e}")
    except Exception as e:
        logging.error(f"Unexpected error: {e}")

    return False


def quantize_model(model):
    """Quantize the model to reduce memory usage."""
    quantized_model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )
    return quantized_model


def export_to_onnx(model, tokenizer, output_dir):
    """Export the model to ONNX format."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Export the model
    onnx_path = output_dir / "model.onnx"
    export(
        model=model,
        tokenizer=tokenizer,
        feature="question-answering",
        opset=12,
        output=onnx_path,
    )
    return onnx_path


def create_onnx_session(onnx_path):
    """Create an ONNX Runtime session for inference."""
    options = onnxruntime.SessionOptions()
    options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = onnxruntime.InferenceSession(str(onnx_path), options)
    return session


# Context manager for model loading and cleanup
@asynccontextmanager
async def lifespan(app: FastAPI):
    global qa_model, tokenizer, onnx_session, cities_data

    logging.info("=== Starting Application Initialization ===")

    # Then handle Hugging Face authentication and model loading
    logging.info("Step 1: Authenticating with Hugging Face...")
    hf_token = "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"
    try:
        login(token=hf_token)
        logging.info("Successfully authenticated with Hugging Face")
    except Exception as e:
        logging.error(f"Hugging Face authentication failed: {str(e)}")
        raise HTTPException(
            status_code=500, detail="Failed to authenticate with Hugging Face"
        )

    logging.info("Step 2: Loading ML models...")
    try:
        # Load question-answering model
        model_name = "deepset/roberta-base-squad2"

        logging.info(f"Loading tokenizer from {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=CACHE_DIR)

        logging.info(f"Loading model from {model_name}...")
        qa_model = AutoModelForQuestionAnswering.from_pretrained(
            model_name, cache_dir=CACHE_DIR
        )

        # Quantize the model to reduce memory usage
        logging.info("Quantizing the model...")
        qa_model = quantize_model(qa_model)

        # Export the model to ONNX format
        logging.info("Exporting the model to ONNX format...")
        onnx_path = export_to_onnx(qa_model, tokenizer, output_dir="./onnx")

        # Create an ONNX Runtime session
        logging.info("Creating ONNX Runtime session...")
        onnx_session = create_onnx_session(onnx_path)

        logging.info("ML models loaded successfully")

        # Verify models are loaded
        if qa_model is None or tokenizer is None or onnx_session is None:
            raise ValueError("Models failed to load properly")

    except Exception as e:
        logging.error(f"Error loading ML models: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to load ML models: {str(e)}"
        )

    logging.info("=== Application Initialization Complete ===")
    logging.info(
        f"""
    Initialization Status:
    - Cities Data: {cities_data is not None and not cities_data.empty}
    - Cities Shape: {cities_data.shape if cities_data is not None else 'N/A'}
    - Tokenizer: {tokenizer is not None}
    - QA Model: {qa_model is not None}
    - ONNX Session: {onnx_session is not None}
    """
    )

    yield

    # Cleanup
    logging.info("=== Starting Application Shutdown ===")
    qa_model = None
    tokenizer = None
    onnx_session = None
    cities_data = None
    logging.info("Models and data cleared, shutdown complete.")


# FastAPI app setup
app = FastAPI(
    title="DeepSeek Resume Intelligence Services",
    description="API for text generation, location extraction, and skill extraction using DeepSeek",
    version="1.0.0",
    lifespan=lifespan,
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
        "cities_data_shape": cities_data.shape if cities_data is not None else None,
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
            "/extract-skills",  # Skill extraction
        ],
    }


@app.get("/convert")
async def convert():
    """API endpoint to trigger conversion."""
    success = await convert_txt_to_csv()

    if success:
        # Log the success messages before returning the response
        logger.info("CSV file created successfully.")
        logger.info(f"CSV file path: {CSV_FILE_PATH}")

        return JSONResponse(
            content={
                "message": "CSV file created successfully.",
                "csv_path": CSV_FILE_PATH,
            },
            status_code=200,
        )
    else:
        # Log the error
        logger.error("Failed to create CSV file.")

        return JSONResponse(
            content={"error": "Failed to create CSV file."}, status_code=500
        )