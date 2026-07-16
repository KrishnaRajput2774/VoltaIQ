from fastapi import APIRouter, status, Depends
from app.schemas import SystemHealthOutput, ErrorResponse
from app.services.models_service import ModelService
from app.api.dependencies import get_model_service
from app.utils.data_loader import data_loader
import time
import sys
import platform
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/system", tags=["System Diagnostics"])

# Record service start time
START_TIME = time.time()

@router.get("/health", response_model=SystemHealthOutput, status_code=status.HTTP_200_OK,
            responses={500: {"model": ErrorResponse}})
async def get_system_health(
    model_svc: ModelService = Depends(get_model_service)
) -> SystemHealthOutput:
    """
    Detailed system health status including uptime, ML model load state,
    and dataset availability diagnostics.
    """
    uptime = time.time() - START_TIME
    
    # Model statuses
    model_status = model_svc.get_model_load_status()
    
    # Dataset file presence
    try:
        dataset_status = data_loader.verify_dataset_structure()
    except Exception as e:
        logger.error(f"Error checking dataset structure: {e}")
        dataset_status = {}
        
    return SystemHealthOutput(
        status="healthy",
        uptime_seconds=round(uptime, 2),
        model_load_status=model_status,
        dataset_status=dataset_status
    )

@router.get("/info", status_code=status.HTTP_200_OK)
async def get_system_info():
    """
    Get backend version, environment variables, operating system platform,
    and Python engine configurations.
    """
    return {
        "app_name": "VoltIQ",
        "version": "2.0.0",
        "api_status": "operational",
        "python_version": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "datasets_root": data_loader.datasets_dir
    }
