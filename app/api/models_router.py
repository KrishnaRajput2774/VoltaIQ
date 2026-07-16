from fastapi import APIRouter, status, HTTPException, Depends
from app.schemas import ModelRegistryOutput, ModelInfoOutput, ErrorResponse
from app.services.models_service import ModelService
from app.api.dependencies import get_model_service
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/models", tags=["Model Registry"])

@router.get("/registry", response_model=ModelRegistryOutput, status_code=status.HTTP_200_OK,
            responses={500: {"model": ErrorResponse}})
async def get_model_registry(
    model_svc: ModelService = Depends(get_model_service)
) -> ModelRegistryOutput:
    """
    Fetch the list of all registered models, their metadata, feature schemas,
    and latest performance evaluation metrics.
    """
    try:
        registry_list = model_svc.get_registry_info()
        return ModelRegistryOutput(
            models=[ModelInfoOutput(**m) for m in registry_list]
        )
    except Exception as e:
        logger.error(f"Error getting model registry: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load model registry details: {str(e)}"
        )

@router.get("/{model_id}", response_model=ModelInfoOutput, status_code=status.HTTP_200_OK,
            responses={404: {"model": ErrorResponse}, 500: {"model": ErrorResponse}})
async def get_model_details(
    model_id: str,
    model_svc: ModelService = Depends(get_model_service)
) -> ModelInfoOutput:
    """
    Fetch extensive metadata, training parameters, and metrics for a specific model ID.
    """
    try:
        registry_list = model_svc.get_registry_info()
        match = next((m for m in registry_list if m["model_id"] == model_id), None)
        
        if not match:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model ID '{model_id}' was not found in registry."
            )
            
        return ModelInfoOutput(**match)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting model details for {model_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch model details: {str(e)}"
        )
