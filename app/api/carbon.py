from fastapi import APIRouter, status, HTTPException, Depends
from app.schemas import CarbonFootprintOutput, CarbonAnalysisInput, CarbonAnalysisOutput, ErrorResponse
from app.services.carbon_service import CarbonService
from app.api.dependencies import get_carbon_service
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/carbon", tags=["Carbon Intelligence Tracker"])

@router.get("/metrics", response_model=CarbonFootprintOutput, status_code=status.HTTP_200_OK,
            responses={500: {"model": ErrorResponse}})
async def get_carbon_metrics(
    carbon_svc: CarbonService = Depends(get_carbon_service)
) -> CarbonFootprintOutput:
    """
    Fetch cumulative carbon emissions, simulated EV offset projections,
    and fleet progress toward net-zero targets from fleet carbon data.
    """
    try:
        metrics = carbon_svc.get_carbon_metrics()
        return CarbonFootprintOutput(**metrics)
    except Exception as e:
        logger.error(f"Error getting carbon metrics: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch carbon footprint metrics: {str(e)}"
        )

@router.post("/analysis", response_model=CarbonAnalysisOutput, status_code=status.HTTP_200_OK,
             responses={422: {"model": ErrorResponse}, 500: {"model": ErrorResponse}})
async def analyze_vehicle_carbon(
    payload: CarbonAnalysisInput,
    carbon_svc: CarbonService = Depends(get_carbon_service)
) -> CarbonAnalysisOutput:
    """
    Perform a comparative carbon emissions analysis for a specific vehicle (baseline vs EV scenario)
    over a custom annual distance and fuel type.
    """
    try:
        result = carbon_svc.analyze_vehicle_carbon(
            vehicle_id=payload.vehicle_id,
            annual_distance_km=payload.annual_distance_km,
            fuel_type=payload.fuel_type
        )
        return CarbonAnalysisOutput(**result)
    except Exception as e:
        logger.error(f"Error analyzing vehicle carbon: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to perform carbon analysis: {str(e)}"
        )
