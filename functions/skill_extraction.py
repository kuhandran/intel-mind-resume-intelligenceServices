import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import spacy

# Set up the router for Skill Extraction API endpoint
router = APIRouter()

# Pydantic model for request payload
class SkillPayload(BaseModel):
    text: str

# Endpoint for extracting skills from text
@router.post("/extract_skills/")
async def extract_skills(payload: SkillPayload, nlp: spacy.language):
    try:
        logging.info(f"Request to extract skills: {payload.text}")
        # Extract named entities using spaCy
        doc = nlp(payload.text)
        skills = [ent.text for ent in doc.ents if ent.label_ in ['SKILL']]  # Adjust label if necessary
        return {"skills": skills}
    except Exception as e:
        logging.error(f"Error extracting skills: {e}")
        raise HTTPException(status_code=500, detail="Error extracting skills")