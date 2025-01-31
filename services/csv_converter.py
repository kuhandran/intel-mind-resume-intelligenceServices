import os
import csv
import aiofiles
import logging

# File paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TXT_FILE_PATH = os.path.join(BASE_DIR, "../data/cities15000.txt")
CSV_FILE_PATH = os.path.join(BASE_DIR, "../data/cities5000.csv")
TXT_NATIONS_FILE_PATH = os.path.join(BASE_DIR, "../data/countryInfo.txt")
CSV_NATIONS_FILE_PATH = os.path.join(BASE_DIR, "../data/countries_data.csv")


async def convert_txt_to_csv():
    """Asynchronously converts a TXT file to a CSV file."""

    # Check if TXT file exists
    if not os.path.exists(TXT_FILE_PATH):
        logging.error(f"Error: TXT file not found -> {TXT_FILE_PATH}")
        return False

    try:
        async with aiofiles.open(TXT_FILE_PATH, "r", encoding="utf-8") as txt_file:
            contents = await txt_file.readlines()

        with open(CSV_FILE_PATH, "w", encoding="utf-8", newline="") as csv_file:
            writer = csv.writer(csv_file, quoting=csv.QUOTE_ALL)

            writer.writerow(
                [
                    "geonameid", "name", "asciiname", "alternatenames", "latitude",
                    "longitude", "feature class", "feature code", "country code",
                    "cc2", "admin1 code", "admin2 code", "admin3 code", "admin4 code",
                    "population", "elevation", "dem", "timezone", "modification date",
                ]
            )

            for line in contents:
                data = line.strip().split("\t")
                if len(data) >= 19:
                    writer.writerow(data[:19])

        logging.info(f"CSV successfully created at: {CSV_FILE_PATH}")
        return True
    except Exception as e:
        logging.error(f"Unexpected error: {e}")

    return False





async def convert_nations_txt_to_csv():
    """Asynchronously converts a TXT file to a CSV file."""

    # Check if TXT file exists
    if not os.path.exists(TXT_NATIONS_FILE_PATH):
        logging.error(f"Error: TXT file not found -> {TXT_NATIONS_FILE_PATH}")
        return False

    try:
        # Open the TXT file asynchronously
        async with aiofiles.open(TXT_NATIONS_FILE_PATH, "r", encoding="utf-8") as txt_file:
            contents = await txt_file.readlines()

        # Open the CSV file for writing
        with open(CSV_NATIONS_FILE_PATH, "w", encoding="utf-8", newline="") as csv_file:
            writer = csv.writer(csv_file, quoting=csv.QUOTE_ALL)

            # Define the headers based on the provided sample data
            writer.writerow([
                "ISO", "ISO3", "ISO-Numeric", "fips", "Country", "Capital", "Area(in sq km)", 
                "Population", "Continent", "tld", "CurrencyCode", "CurrencyName", "Phone", 
                "Postal Code Format", "Postal Code Regex", "Languages", "geonameid", "neighbours", 
                "EquivalentFipsCode"
            ])

            # Process each line from the text file
            for line in contents:
                # Remove any leading/trailing spaces and split by tab character
                data = line.strip().split("\t")
                
                # Only write rows that have enough columns (18 columns expected in this case)
                if len(data) >= 18:
                    writer.writerow(data[:18])

        logging.info(f"CSV successfully created Nations at: {CSV_NATIONS_FILE_PATH}")
        return True
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        return False