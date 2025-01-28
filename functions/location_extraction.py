import logging
import csv
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import re

# Set up the router for Location Extraction API endpoint
router = APIRouter()

# Path to the cities5000.csv file
CITIES_CSV_PATH = './data/cities5000.csv'  # Path to CSV file containing city and country data

# Load cities and countries into a list of tuples
def load_city_country_data():
    cities = []
    try:
        with open(CITIES_CSV_PATH, mode='r', encoding='utf-8') as file:
            reader = csv.reader(file)
            for row in reader:
                city, country = row
                cities.append((city.lower(), country.lower()))  # Store cities and countries in lowercase
    except Exception as e:
        logging.error(f"Error loading cities data: {e}")
    return cities

# Pydantic model for request payload
class LocationPayload(BaseModel):
    text: str

# Helper function to search for cities and countries in the input text
def search_locations_in_text(text: str, cities_data: list) -> list:
    found_locations = []
    text = text.lower()  # Convert text to lowercase for case-insensitive search

    for city, country in cities_data:
        # Check if the city or country exists in the text
        if city in text or country in text:
            found_locations.append((city.capitalize(), country.capitalize()))  # Capitalize the location for readability

    return found_locations

# Endpoint for extracting location entities from text
@router.post("/extract_locations/")
async def extract_locations(payload: LocationPayload):
    try:
        logging.info(f"Request to extract locations: {payload.text}")

        # Load the cities and countries data (this can be cached or loaded once at startup for better performance)
        cities_data = load_city_country_data()

        if not cities_data:
            raise HTTPException(status_code=500, detail="Error loading city and country data")

        # Search for cities and countries in the text
        locations = search_locations_in_text(payload.text, cities_data)

        if locations:
            return {"locations": locations}
        else:
            return {"locations": ["No location found"]}

    except Exception as e:
        logging.error(f"Error while processing request: {e}")
        raise HTTPException(status_code=500, detail="Error extracting locations")