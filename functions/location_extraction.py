import logging
import csv
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List

# Set up the router for Location Extraction API endpoint
router = APIRouter()

# Path to the cities5000.csv file
CITIES_CSV_PATH = './data/cities5000.csv'  # Path to CSV file containing city and country data

# Pydantic model for request payload
class LocationPayload(BaseModel):
    text: str

# Pydantic model for response
class LocationResponse(BaseModel):
    locations: List[str]

# Cache cities and countries data at startup
cities_data = []

# Load cities and countries into a list of tuples
def load_city_country_data():
    global cities_data
    try:
        with open(CITIES_CSV_PATH, mode='r', encoding='utf-8') as file:
            reader = csv.reader(file)
            for row in reader:
                city, country = row
                cities_data.append((city.lower(), country.lower()))  # Store cities and countries in lowercase
    except Exception as e:
        logging.error(f"Error loading cities data: {e}")

# Helper function to search for cities and countries in the input text
def search_locations_in_text(text: str) -> List[str]:
    found_locations = []
    text = text.lower()  # Convert text to lowercase for case-insensitive search

    for city, country in cities_data:
        # Check if the city or country exists in the text
        if city in text or country in text:
            found_locations.append(f"{city.capitalize()}, {country.capitalize()}")  # Capitalize for readability

    return found_locations

# Load cities data once at startup
load_city_country_data()

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