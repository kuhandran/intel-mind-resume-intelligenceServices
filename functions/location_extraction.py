import logging
import csv
import os
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List

# Set up logging
logging.basicConfig(level=logging.INFO)

# Set up the router for Location Extraction API endpoint
router = APIRouter()

# Path to the cities5000.csv or cities5000.txt file
CITIES_FILE_PATH = os.path.join(os.path.dirname(__file__), '../data/cities5000.csv')

# Pydantic models for request & response
class LocationPayload(BaseModel):
    text: str

class LocationResponse(BaseModel):
    locations: List[str]

# Global variable to store loaded city data
cities_data = None

# Function to load data only when needed
def load_city_country_data():
    global cities_data

    # If data is already loaded, return early
    if cities_data is not None:
        return

    # Check if the file exists before attempting to read
    if not os.path.exists(CITIES_FILE_PATH):
        logging.error(f"City data file not found: {CITIES_FILE_PATH}")
        raise FileNotFoundError(f"City data file not found: {CITIES_FILE_PATH}")

    cities_data = []  # Initialize empty list

    try:
        logging.info(f"Loading city data from {CITIES_FILE_PATH}")
        file_ext = CITIES_FILE_PATH.split('.')[-1].lower()

        if file_ext == 'csv':
            with open(CITIES_FILE_PATH, mode='r', encoding='utf-8') as file:
                reader = csv.reader(file)
                for row in reader:
                    if len(row) >= 2:
                        city, country = row[:2]
                        cities_data.append((city.lower(), country.lower()))
        elif file_ext == 'txt':
            with open(CITIES_FILE_PATH, mode='r', encoding='utf-8') as file:
                for line in file:
                    parts = line.strip().split('\t')  # Assuming tab-separated values
                    if len(parts) >= 2:
                        city, country = parts[:2]
                        cities_data.append((city.lower(), country.lower()))
        else:
            raise ValueError("Unsupported file format. Only CSV and TXT are supported.")

        logging.info(f"Successfully loaded {len(cities_data)} city entries.")

    except Exception as e:
        logging.error(f"Error loading cities data: {e}")
        cities_data = None  # Reset to None to retry on next request
        raise e

# Helper function to search for cities in the text
def search_locations_in_text(text: str) -> List[str]:
    if cities_data is None:
        load_city_country_data()  # Load data only when the API is called

    found_locations = []
    text = text.lower()

    for city, country in cities_data:
        if city in text or country in text:
            found_locations.append(f"{city.capitalize()}, {country.capitalize()}")

    return found_locations

# API Endpoint for extracting locations
@router.post("/extract_locations/", response_model=LocationResponse)
async def extract_locations(payload: LocationPayload) -> LocationResponse:
    try:
        logging.info(f"Processing location extraction request: {payload.text}")

        # Attempt to search locations
        locations = search_locations_in_text(payload.text)

        if locations:
            return LocationResponse(locations=locations)
        else:
            raise HTTPException(status_code=404, detail="No locations found in the text")

    except FileNotFoundError as e:
        logging.error(f"City data file missing: {e}")
        raise HTTPException(status_code=500, detail="City data file not found on the server")
    except HTTPException as http_exc:
        logging.error(f"HTTPException: {http_exc.detail}")
        raise http_exc
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Error extracting locations")