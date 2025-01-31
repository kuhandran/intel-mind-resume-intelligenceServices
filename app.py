import logging
from fastapi import FastAPI
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware

# Import the consolidated router
from routers import api_router
from services.model_loader import load_models
from utils.logger import setup_logger

# Initialize logger
logger = setup_logger()

# Context manager for application lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.info("Starting application...")

    try:
        load_models()
    except Exception as e:
        logging.error(f"Error loading models: {str(e)}")

    yield

    logging.info("Shutting down application...")


# Initialize FastAPI app
app = FastAPI(
    title="Intel Mind Resume Intelligence Services",
    description="API for text generation, location extraction, and skill extraction",
    version="1.0.0",
    lifespan=lifespan,
)

# Register the main API router
app.include_router(api_router)

# CORS middleware setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)