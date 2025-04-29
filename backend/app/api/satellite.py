import os
import sys
import logging
import json
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks, Depends
from fastapi.responses import JSONResponse
from typing import List, Optional
import shutil
import tempfile
from datetime import datetime
import uuid

# Add ML directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'ml'))

# Import ML modules
from inference.predict_forest_change import ForestChangePredictor
from inference.estimate_carbon_sequestration import CarbonSequestrationEstimator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

# Initialize ML models
predictor = ForestChangePredictor()
estimator = CarbonSequestrationEstimator(predictor)

# Storage directories
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'uploads')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'results')

# Create directories if they don't exist
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Background task for processing satellite images
def process_satellite_image(image_id: str, bands_dir: str):
    """Process satellite image in the background."""
    try:
        # Create output directory
        output_dir = os.path.join(RESULTS_DIR, image_id)
        os.makedirs(output_dir, exist_ok=True)
        
        # Process image
        results = predictor.process_satellite_image(bands_dir, output_dir)
        
        # Save results
        with open(os.path.join(output_dir, 'results.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Completed processing satellite image {image_id}")
    except Exception as e:
        logger.error(f"Error processing satellite image {image_id}: {str(e)}")
        # Save error information
        with open(os.path.join(output_dir, 'error.json'), 'w') as f:
            json.dump({'error': str(e)}, f)

# Background task for estimating carbon sequestration
def estimate_carbon_sequestration_task(task_id: str, before_dir: str, after_dir: str):
    """Estimate carbon sequestration in the background."""
    try:
        # Create output directory
        output_dir = os.path.join(RESULTS_DIR, task_id)
        os.makedirs(output_dir, exist_ok=True)
        
        # Estimate carbon sequestration
        results = estimator.estimate_carbon_sequestration(before_dir, after_dir, output_dir)
        
        # Save results
        with open(os.path.join(output_dir, 'results.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Completed carbon sequestration estimation {task_id}")
    except Exception as e:
        logger.error(f"Error estimating carbon sequestration {task_id}: {str(e)}")
        # Save error information
        with open(os.path.join(output_dir, 'error.json'), 'w') as f:
            json.dump({'error': str(e)}, f)

@router.post("/upload-satellite-bands")
async def upload_satellite_bands(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    project_id: str = Form(...),
    timestamp: Optional[str] = Form(None)
):
    """
    Upload Sentinel-2 band files for processing.
    
    Args:
        files: List of band files (B02, B03, B04, B08)
        project_id: ID of the project
        timestamp: Optional timestamp for the image
    
    Returns:
        JSON response with task ID and status
    """
    # Generate unique ID for this upload
    image_id = f"{project_id}_{uuid.uuid4()}"
    
    # Create directory for this upload
    upload_dir = os.path.join(UPLOAD_DIR, image_id)
    os.makedirs(upload_dir, exist_ok=True)
    
    # Save uploaded files
    for file in files:
        file_path = os.path.join(upload_dir, file.filename)
        with open(file_path, 'wb') as f:
            shutil.copyfileobj(file.file, f)
    
    # Start background processing
    background_tasks.add_task(process_satellite_image, image_id, upload_dir)
    
    return JSONResponse({
        "task_id": image_id,
        "status": "processing",
        "message": "Satellite bands uploaded and processing started"
    })

@router.get("/forest-change/{task_id}")
async def get_forest_change_results(task_id: str):
    """
    Get results of forest change detection.
    
    Args:
        task_id: ID of the processing task
    
    Returns:
        JSON response with results or status
    """
    results_path = os.path.join(RESULTS_DIR, task_id, 'results.json')
    error_path = os.path.join(RESULTS_DIR, task_id, 'error.json')
    
    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            results = json.load(f)
        return JSONResponse(results)
    elif os.path.exists(error_path):
        with open(error_path, 'r') as f:
            error = json.load(f)
        raise HTTPException(status_code=500, detail=error)
    else:
        return JSONResponse({
            "task_id": task_id,
            "status": "processing",
            "message": "Processing in progress"
        })

@router.post("/estimate-carbon-sequestration")
async def estimate_carbon_sequestration(
    background_tasks: BackgroundTasks,
    before_task_id: str = Form(...),
    after_task_id: str = Form(...),
    project_id: str = Form(...)
):
    """
    Estimate carbon sequestration between two time points.
    
    Args:
        before_task_id: ID of the earlier satellite image processing task
        after_task_id: ID of the later satellite image processing task
        project_id: ID of the project
    
    Returns:
        JSON response with task ID and status
    """
    # Check if both tasks have completed
    before_results_path = os.path.join(RESULTS_DIR, before_task_id, 'results.json')
    after_results_path = os.path.join(RESULTS_DIR, after_task_id, 'results.json')
    
    if not os.path.exists(before_results_path):
        raise HTTPException(status_code=400, detail=f"Results for before_task_id {before_task_id} not found")
    
    if not os.path.exists(after_results_path):
        raise HTTPException(status_code=400, detail=f"Results for after_task_id {after_task_id} not found")
    
    # Generate unique ID for this task
    task_id = f"{project_id}_carbon_{uuid.uuid4()}"
    
    # Get directories with satellite bands
    before_dir = os.path.join(UPLOAD_DIR, before_task_id)
    after_dir = os.path.join(UPLOAD_DIR, after_task_id)
    
    # Start background processing
    background_tasks.add_task(estimate_carbon_sequestration_task, task_id, before_dir, after_dir)
    
    return JSONResponse({
        "task_id": task_id,
        "status": "processing",
        "message": "Carbon sequestration estimation started"
    })

@router.get("/carbon-sequestration/{task_id}")
async def get_carbon_sequestration_results(task_id: str):
    """
    Get results of carbon sequestration estimation.
    
    Args:
        task_id: ID of the estimation task
    
    Returns:
        JSON response with results or status
    """
    results_path = os.path.join(RESULTS_DIR, task_id, 'carbon_sequestration_results.json')
    error_path = os.path.join(RESULTS_DIR, task_id, 'error.json')
    
    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            results = json.load(f)
        return JSONResponse(results)
    elif os.path.exists(error_path):
        with open(error_path, 'r') as f:
            error = json.load(f)
        raise HTTPException(status_code=500, detail=error)
    else:
        return JSONResponse({
            "task_id": task_id,
            "status": "processing",
            "message": "Processing in progress"
        })

@router.get("/explanation/{task_id}")
async def get_explanation(task_id: str):
    """
    Get explanation for a forest change prediction.
    
    Args:
        task_id: ID of the processing task
    
    Returns:
        JSON response with explanation data
    """
    explanation_path = os.path.join(RESULTS_DIR, task_id, 'explanation.tif')
    visualization_path = os.path.join(RESULTS_DIR, task_id, 'visualization.png')
    
    if not os.path.exists(explanation_path):
        raise HTTPException(status_code=404, detail=f"Explanation for task {task_id} not found")
    
    # Return paths to explanation files
    return JSONResponse({
        "task_id": task_id,
        "explanation_path": explanation_path,
        "visualization_path": visualization_path if os.path.exists(visualization_path) else None
    })
