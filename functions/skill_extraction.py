import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import spacy
from typing import List

# Set up the router for Skill Extraction API endpoint
router = APIRouter()

# Pydantic model for request payload
class SkillPayload(BaseModel):
    text: str

# Pydantic model for response
class SkillResponse(BaseModel):
    skills: List[str]

# Initialize the spaCy model outside the route handler (global variable)
nlp = spacy.load("en_core_web_sm")  # Adjust to your spaCy model if needed

# Endpoint for extracting skills from text
@router.post("/extract_skills/", response_model=SkillResponse)
async def extract_skills(payload: SkillPayload) -> SkillResponse:
    try:
        logging.info(f"Request to extract skills: {payload.text}")
        # Extract named entities using spaCy
        doc = nlp(payload.text)
        # Assuming 'SKILL' is a custom label; adjust if necessary
        skills = [ent.text for ent in doc.ents if ent.label_ == 'SKILL']
        return SkillResponse(skills=skills)
    except Exception as e:
        logging.error(f"Error extracting skills: {e}")
        raise HTTPException(status_code=500, detail="Error extracting skills")