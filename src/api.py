#!/usr/bin/env python3
"""
Clinical Guidelines API Service

This module provides a FastAPI-based web service for generating clinical screening
recommendations based on patient clinical notes using the Guidelines RAG system.
"""

import logging
import json
import os
import time
from typing import Dict, Any, Optional

from fastapi import FastAPI, HTTPException, Request, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
import uvicorn

from .Guidelines import Guidelines

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("guidelines-api")

# Initialize the Guidelines system
try:
    guidelines_system = Guidelines()
    logger.info("Guidelines system initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Guidelines system: {str(e)}")
    raise

# Create FastAPI app
app = FastAPI(
    title="Clinical Guidelines API",
    description="API for generating screening recommendations based on clinical notes",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request and response models
class ClinicalNoteRequest(BaseModel):
    clinical_note: str = Field(
        ..., 
        description="Patient clinical note text",
        min_length=10
    )
    
    @validator('clinical_note')
    def validate_clinical_note(cls, v):
        if len(v.strip()) < 10:
            raise ValueError("Clinical note must contain meaningful content")
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "clinical_note": "Patient is a 68 year old male. The patient smoked 35 years ago, but no longer does. He has had multiple sexual partners in the last year. He has had angina in the past ten years, and required 2 stents 3 years ago."
            }
        }

class RecommendationResponse(BaseModel):
    patient_data: Dict[str, Any]
    recommendations: Dict[str, Any]
    
    class Config:
        schema_extra = {
            "example": {
                "patient_data": {
                    "patient_age": 68,
                    "patient_gender": "male",
                    "past_medical_history": ["angina", "cardiac stents"],
                    "social_history": {"smoking_status": "former smoker"}
                },
                "recommendations": {
                    "recommendations": [
                        {
                            "test_name": "colonoscopy",
                            "justification": "patients age 50 and older should have screening colonoscopy every ten years. He has never had one.",
                            "evidence": "USPTF colonoscopy guidelines, 2018"
                        },
                        {
                            "test_name": "mammogram",
                            "justification": "women age 40 to 74 are asked to have a mammogram every two years. Her last one was 3 years ago.",
                            "evidence": "USPTF mammogram screening guidelines, 2021"
                        }
                    ],
                    "additional_recommendations": [
                        {
                            "test_name": "lipid panel",
                            "justification": "patients with a history of high LDL levels should have a lipid panel every 2 years.",
                            "evidence": "USPTF lipid panel guidelines, 2020"
                        },
                        {
                            "test_name": "flu shot",
                            "justification": "people age 18-80 are asked to have a flu vaccination each year.",
                            "evidence": "USPTF flu vaccination guidelines, 2022"
                        }
                    ]
                }
            }
        }

        

# Middleware for request timing and logging
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    logger.info(f"Request {request.method} {request.url.path} processed in {process_time:.4f} seconds")
    return response

# Error handler
@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An unexpected error occurred. Please try again later."}
    )

# API endpoints
# @app.post("/guidelines/recommendations", response_model=RecommendationResponse)
# async def get_recommendations(request: ClinicalNoteRequest):
#     """
#     Generate screening recommendations based on a clinical note.
    
#     Returns patient data extracted from the note and recommended screening tests.
#     """
#     try:
#         logger.info("Processing recommendation request")
#         results = guidelines_system.generate_recommendations({"clinical_note": request.clinical_note})
#         logger.info("Successfully generated recommendations")
#         logger.info(f"******Results:****** {results}")
#         return results
#     except Exception as e:
#         logger.error(f"Error generating recommendations: {str(e)}", exc_info=True)
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
#             detail=f"Error generating recommendations: {str(e)}"
#         )

@app.post("/guidelines/recommendations", response_model=RecommendationResponse)
async def get_recommendations(request: ClinicalNoteRequest):
    """
    Generate screening recommendations based on a clinical note.
    
    Returns patient data extracted from the note and recommended screening tests.
    """
    try:
        logger.info("Processing recommendation request")
        raw_response = guidelines_system.generate_recommendations({"clinical_note": request.clinical_note})
        
        # Parse the response if it's a string
        if isinstance(raw_response["recommendations"], str):
            try:
                recommendations = json.loads(raw_response["recommendations"])
                raw_response["recommendations"] = recommendations
            except json.JSONDecodeError:
                logger.error("Failed to parse LLM response as JSON")
                logger.error(f"LLM response: {raw_response['recommendations']}")
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail="LLM returned invalid JSON format"
                )
                
        logger.info("Successfully generated recommendations")
        return raw_response
        
    except Exception as e:
        logger.error(f"Error generating recommendations: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=f"Error generating recommendations: {str(e)}"
        )

@app.get("/")
async def health_check():
    """Tests to verify the API is running."""
    return {"status": "hello", "time": time.time()}

# Main entry point
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    
    logger.info(f"Health check called: Guidelines API server on {host}:{port}")
    uvicorn.run(
        "api:app", 
        host=host, 
        port=port, 
        reload=False,  # Set to False in production
        log_level="info"
    )