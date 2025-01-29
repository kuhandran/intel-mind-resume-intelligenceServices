import logging
import csv
import os
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List

# Set up the router for Location Extraction API endpoint
router = APIRouter()

# Path to the cities5000.csv or cities5000.txt file
CITIES_FILE_PATH = 'data/cities5000.csv'  # Path to CSV file containing city and country data

# Pydantic model for request payload
class LocationPayload(BaseModel):
    text: str

# Pydantic model for response
class LocationResponse(BaseModel):
    locations: List[str]

# Cache cities and countries data at startup
cities_data = []

# Function to read data from CSV or TXT file
def load_city_country_data(file_path: str):
    global cities_data
    try:
        # Check if the file exists
        logging.info(f"Loading city data from {file_path}")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"{file_path} not found")

        # Check file extension to process accordingly
        file_ext = file_path.split('.')[-1].lower()

        if file_ext == 'csv':
            with open(file_path, mode='r', encoding='utf-8') as file:
                reader = csv.reader(file)
                for row in reader:
                    city, country = row
                    cities_data.append((city.lower(), country.lower()))  # Store cities and countries in lowercase
        elif file_ext == 'txt':
            with open(file_path, mode='r', encoding='utf-8') as file:
                for line in file:
                    city, country = line.strip().split('\t')  # Assuming tab-separated values
                    cities_data.append((city.lower(), country.lower()))  # Store cities and countries in lowercase
        else:
            raise ValueError("Unsupported file format. Only CSV and TXT are supported.")
        
        logging.info(f"Loaded city data from {file_path}")

    except Exception as e:
        logging.error(f"Error loading cities data from {file_path}: {e}")

# Helper function to search for cities and countries in the input text
def search_locations_in_text(text: str) -> List[str]:
    found_locations = []
    text = text.lower()  # Convert text to lowercase for case-insensitive search

    for city, country in cities_data:
        # Check if the city or country exists in the text
        if city in text or country in text:
            found_locations.append(f"{city.capitalize()}, {country.capitalize()}")  # Capitalize for readability

    return found_locations

# Load cities data once at startup (this will now support both CSV and TXT formats)
load_city_country_data(CITIES_FILE_PATH)

# Endpoint for extracting location entities from text
@router.post("/extract_locations/", response_model=LocationResponse)
async def extract_locations(payload: LocationPayload) -> LocationResponse:
    try:
        logging.info(f"Request to extract locations: {payload.text}")

        # Search for cities and countries in the text
        locations = search_locations_in_text(payload.text)

        if locations:
            return LocationResponse(locations=locations)
        else:
            return LocationResponse(locations=["No location found"])

    except Exception as e:
        logging.error(f"Error while processing request: {e}")
        raise HTTPException(status_code=500, detail="Error extracting locations")