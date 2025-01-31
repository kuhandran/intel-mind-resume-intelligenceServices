from fastapi import APIRouter
from fastapi.responses import JSONResponse
from services.csv_converter import convert_txt_to_csv

router = APIRouter()

# Health check endpoint
@router.get("/health")
async def health_check():
    return {
        "status": "healthy",
    }


# Root endpoint
@router.get("/")
async def root():
    return {
        "message": "Welcome to Intel Mind Resume Intelligence Services",
        "version": "1.0.0",
    }


# Convert endpoint
@router.get("/convert")
async def convert():
    success = await convert_txt_to_csv()
    if success:
        return JSONResponse(content={"message": "CSV created successfully"}, status_code=200)
    else:
        return JSONResponse(content={"error": "Failed to create CSV file"}, status_code=500)