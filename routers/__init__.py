from fastapi import APIRouter
from functions.text_generation import router as text_gen_router
from functions.location_extraction import router as loc_extract_router
from functions.skill_extraction import router as skill_extract_router
from functions.general import router as general_router    

# Create a main router to include all sub-routers
api_router = APIRouter()

api_router.include_router(text_gen_router, prefix="/question-services", tags=["Text Generation"])
api_router.include_router(loc_extract_router, prefix="/location-extraction", tags=["Location Extraction"])
api_router.include_router(skill_extract_router, prefix="/skills-extraction", tags=["Skill Extraction"])
api_router.include_router(general_router, tags=["General"])