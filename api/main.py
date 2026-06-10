# FarmGuard AI — FastAPI Backend
# Exposes the disease detection pipeline as API endpoints
# The Streamlit dashboard calls these endpoints

import os
import sys
import shutil
import uuid
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Add project root to path so we can import predict.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.predict import analyze_leaf, predict_disease

# INITIALIZE FASTAPI APP
app = FastAPI(
    title="FarmGuard AI",
    description="Crop disease detection API for Nigerian farmers",
    version="1.0.0"
)

# CORS MIDDLEWARE
# Allows the Streamlit dashboard to call this API

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# TEMP FOLDER
# Stores uploaded images temporarily during prediction
TEMP_FOLDER = "temp_uploads"
os.makedirs(TEMP_FOLDER, exist_ok=True)

# ROUTES
# Endpoints for health check, prediction, and class listing

@app.get("/")
def root():
    """Root endpoint — confirms API is running."""
    return {
        "message": "FarmGuard AI API is running!",
        "version": "1.0.0",
        "status": "healthy"
    }

@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model": "loaded",
        "api": "running"
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Main prediction endpoint.

    Accepts a leaf image and returns:
    - Disease name
    - Confidence score
    - Severity level
    - AI generated treatment recommendation

    Args:
        file: Uploaded leaf image (JPEG, PNG)

    Returns:
        dict: Complete disease analysis results
    """
    # Validate file is an image
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Only image files accepted. Please upload JPEG or PNG."
        )

    # Save uploaded image temporarily with unique filename
    temp_filename = f"{uuid.uuid4()}_{file.filename}"
    temp_path = os.path.join(TEMP_FOLDER, temp_filename)

    try:
        # Save image to temp folder
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Run full prediction pipeline
        result = analyze_leaf(temp_path)

        return {
            "success": True,
            "disease": result["disease"],
            "confidence": result["confidence"],
            "severity": result["severity"],
            "is_healthy": result["is_healthy"],
            "recommendation": result["recommendation"]
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

    finally:
        # Always delete temp file after prediction
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.get("/classes")
def get_classes():
    """Returns all disease classes the model can detect."""
    return {
        "classes": [
            "Pepper Bacterial Spot",
            "Pepper Healthy",
            "Potato Early Blight",
            "Potato Healthy",
            "Tomato Early Blight",
            "Tomato Healthy"
        ],
        "total": 6
    }