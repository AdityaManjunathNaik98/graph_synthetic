from fastapi import FastAPI, HTTPException, Header
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from enum import Enum
import json
import uuid
from datetime import datetime
import traceback
import threading

# Import all functions from the original script
from test import (
    download_file_from_url,
    ensure_model_available,
    extract_prompts,
    process_prompts,
    process_dependencies,
    generate_synthetic_data
)

# Define constants (same as in test.py)
DEFAULT_MODEL_NAME = "gpt-oss:20b"
API_BASE_URL = "http://ollama-keda.mobiusdtaas.ai"

app = FastAPI(
    title="Synthetic Data Generation API",
    description="API for generating synthetic data from business process definitions",
    version="1.0.0"
)

# In-memory job storage (use Redis/Database in production)
jobs_storage: Dict[str, Dict[str, Any]] = {}

# Enums
class GenerationMode(str, Enum):
    input = "input"
    both = "both"

# Request Models
class SyntheticDataRequest(BaseModel):
    cdn_url: str = Field(..., description="CDN URL to input JSON file containing business process definitions")
    mode: GenerationMode = Field(..., description="Generation mode: 'input' (only inputs) or 'both' (inputs and outputs)")
    row_count: int = Field(default=100, ge=1, le=10000, description="Number of rows to generate per entity")
    model_name: Optional[str] = Field(default=DEFAULT_MODEL_NAME, description="AI model to use for analysis")

    class Config:
        schema_extra = {
            "example": {
                "cdn_url": "https://cdn-new.gov-cloud.ai/_ENC(...)/file.json",
                "mode": "both",
                "row_count": 100,
                "model_name": "gpt-oss:20b"
            }
        }

# Response Models
class JobResponse(BaseModel):
    job_id: str
    status: str
    message: str
    created_at: str

class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    message: str
    created_at: str
    completed_at: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


# Background processing function that runs in a separate thread
def process_generation_task(
    job_id: str,
    cdn_url: str,
    mode: str,
    row_count: int,
    auth_token: str,
    model_name: str
):
    """Background task to process synthetic data generation - runs in separate thread"""
    try:
        # Update job status
        jobs_storage[job_id]["status"] = "processing"
        jobs_storage[job_id]["message"] = "Downloading input file from CDN..."
        
        # Step 1: Download file
        print(f"[{job_id}] Downloading file from CDN...")
        input_file = download_file_from_url(cdn_url)
        
        # Step 2: Check model availability
        jobs_storage[job_id]["message"] = "Checking AI model availability..."
        print(f"[{job_id}] Checking model availability...")
        if not ensure_model_available(model_name, API_BASE_URL):
            raise Exception("AI model not available")
        
        # Step 3: Extract domain and context
        jobs_storage[job_id]["message"] = "Extracting domain and context from business processes..."
        print(f"[{job_id}] Extracting prompts...")
        prompts, all_entities, input_data = extract_prompts(input_file)
        
        print(f"[{job_id}] Processing prompts with AI...")
        results = process_prompts(prompts, model_name, API_BASE_URL)
        
        results_data = {
            "metadata": {
                "total_prompts": len(prompts),
                "total_business_entities": len(all_entities),
                "model_used": model_name
            },
            "all_business_entities": all_entities,
            "results": results
        }
        
        # Step 4: Process dependencies
        jobs_storage[job_id]["message"] = "Analyzing data dependencies between entities..."
        print(f"[{job_id}] Processing dependencies...")
        dependencies_data = process_dependencies(
            results_data, input_data, mode, model_name, API_BASE_URL
        )
        
        # Step 5: Generate synthetic data
        jobs_storage[job_id]["message"] = "Generating synthetic data via ATSD API..."
        print(f"[{job_id}] Generating synthetic data...")
        synthetic_data_results = generate_synthetic_data(
            dependencies_data, row_count, auth_token, mode
        )
        
        # Update job with final results
        jobs_storage[job_id]["status"] = "completed"
        jobs_storage[job_id]["message"] = "Synthetic data generation completed successfully"
        jobs_storage[job_id]["completed_at"] = datetime.utcnow().isoformat()
        jobs_storage[job_id]["result"] = synthetic_data_results
        
        print(f"[{job_id}] Job completed successfully")
        
    except Exception as e:
        error_msg = str(e)
        error_trace = traceback.format_exc()
        
        jobs_storage[job_id]["status"] = "failed"
        jobs_storage[job_id]["message"] = f"Generation failed: {error_msg}"
        jobs_storage[job_id]["error"] = error_trace
        jobs_storage[job_id]["completed_at"] = datetime.utcnow().isoformat()
        
        print(f"[{job_id}] Job failed: {error_msg}")
        print(error_trace)


# API Endpoints

@app.post("/api/graph_perturbation_synthetic", response_model=JobResponse)
async def graph_perturbation_synthetic(
    request: SyntheticDataRequest,
    authorization: str = Header(..., description="Bearer token for authentication")
):
    """
    Generate synthetic data from business process definitions (Async).
    
    This endpoint immediately returns a job_id and processes in the background.
    
    Steps:
    1. Downloads the business process definition from CDN
    2. Uses AI to extract domain and context
    3. Analyzes dependencies between business entities
    4. Generates synthetic data for all entities
    5. Returns CDN URLs where the generated data can be downloaded
    
    Use GET /api/job/{job_id} to check status and get results.
    
    The Authorization header should contain: Bearer <your-token>
    """
    try:
        # Extract token from Authorization header
        auth_token = authorization.strip()
        
        # Remove "Bearer " prefix if present (case insensitive)
        if auth_token.lower().startswith("bearer "):
            auth_token = auth_token[7:].strip()
        
        # Remove any quotes that might be present
        auth_token = auth_token.strip('"').strip("'").strip()
        
        if not auth_token:
            raise HTTPException(status_code=401, detail="Authorization token is required")
        
        # Generate unique job ID
        job_id = str(uuid.uuid4())
        
        # Initialize job status
        jobs_storage[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "message": "Job queued for processing",
            "created_at": datetime.utcnow().isoformat(),
            "completed_at": None,
            "result": None,
            "error": None,
            "request": {
                "cdn_url": request.cdn_url,
                "mode": request.mode,
                "row_count": request.row_count,
                "model_name": request.model_name
            }
        }
        
        # Start processing in a separate thread (truly async)
        thread = threading.Thread(
            target=process_generation_task,
            args=(
                job_id,
                request.cdn_url,
                request.mode,
                request.row_count,
                auth_token,
                request.model_name
            ),
            daemon=True
        )
        thread.start()
        
        # Return immediately with job_id
        return JobResponse(
            job_id=job_id,
            status="queued",
            message="Job queued successfully. Use GET /api/job/{job_id} to check status.",
            created_at=jobs_storage[job_id]["created_at"]
        )
        
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to queue job: {str(e)}")


@app.get("/api/job/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """
    Get the status and results of a synthetic data generation job.
    
    Status values:
    - queued: Job is waiting to be processed
    - processing: Job is currently being processed (check 'message' for current step)
    - completed: Job completed successfully (check 'result' field for CDN URLs)
    - failed: Job failed (check 'error' field for details)
    """
    if job_id not in jobs_storage:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    job = jobs_storage[job_id]
    
    return JobStatusResponse(
        job_id=job["job_id"],
        status=job["status"],
        message=job["message"],
        created_at=job["created_at"],
        completed_at=job.get("completed_at"),
        result=job.get("result"),
        error=job.get("error")
    )


@app.get("/api/jobs")
async def list_jobs():
    """List all jobs with their current status"""
    return {
        "total_jobs": len(jobs_storage),
        "jobs": [
            {
                "job_id": job["job_id"],
                "status": job["status"],
                "message": job["message"],
                "created_at": job["created_at"],
                "completed_at": job.get("completed_at")
            }
            for job in jobs_storage.values()
        ]
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Synthetic Data Generation API",
        "timestamp": datetime.utcnow().isoformat()
    }


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "service": "Synthetic Data Generation API",
        "version": "1.0.0",
        "endpoints": {
            "generate": "POST /api/graph_perturbation_synthetic (returns immediately with job_id)",
            "job_status": "GET /api/job/{job_id} (check status and get results)",
            "list_jobs": "GET /api/jobs",
            "health": "GET /health",
            "docs": "GET /docs"
        },
        "description": "Generate synthetic data from business process definitions using AI",
        "workflow": [
            "1. POST to /api/graph_perturbation_synthetic - get job_id (< 100ms)",
            "2. Poll GET /api/job/{job_id} until status is 'completed' or 'failed'",
            "3. Get results from 'result' field when status is 'completed'"
        ]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)