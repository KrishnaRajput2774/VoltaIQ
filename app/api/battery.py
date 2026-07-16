from fastapi import APIRouter, status, HTTPException, Depends
from app.schemas import BatteryTelemetryInput, BatteryPredictionOutput, BatteryRULInput, BatteryRULOutput, ErrorResponse
from app.services.models_service import ModelService
from app.api.dependencies import get_model_service
from app.utils.data_loader import data_loader
import pandas as pd
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/battery", tags=["Battery APM"])

@router.post("/predict", response_model=BatteryPredictionOutput, status_code=status.HTTP_200_OK,
             responses={422: {"model": ErrorResponse}, 500: {"model": ErrorResponse}})
async def predict_battery_health(
    payload: BatteryTelemetryInput,
    model_svc: ModelService = Depends(get_model_service)
) -> BatteryPredictionOutput:
    """
    Predict State of Health (SOH) and Remaining Useful Life (RUL) of an EV battery
    based on incoming cell telemetry metrics (Voltage, Cycle Number, Temp, Capacity).
    """
    try:
        soh, rul, zone = model_svc.predict_soh_and_rul(
            cycle_number=payload.cycle_number,
            voltage_v=payload.voltage_v,
            temperature_c=payload.temperature_c,
            capacity_ah=payload.capacity_ah,
            voltage_sag_v=payload.voltage_sag_v,
            degradation_rate=payload.degradation_rate,
            cycle_normalized=payload.cycle_normalized
        )
        
        # Determine model sources
        status_dict = model_svc.get_model_load_status()
        soh_source = "trained" if status_dict["battery_soh_model"] else "fallback_math"
        rul_source = "trained" if status_dict["battery_rul_model"] else "fallback_math"
        
        return BatteryPredictionOutput(
            battery_id="B0005-PRED",
            state_of_health=soh,
            remaining_useful_life_cycles=rul,
            health_zone=zone,
            soh_model_used=soh_source,
            rul_model_used=rul_source
        )
    except Exception as e:
        logger.error(f"Error predicting battery health: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to execute battery prediction: {str(e)}"
        )

@router.post("/rul", response_model=BatteryRULOutput, status_code=status.HTTP_200_OK,
             responses={422: {"model": ErrorResponse}, 500: {"model": ErrorResponse}})
async def predict_battery_rul(
    payload: BatteryRULInput,
    model_svc: ModelService = Depends(get_model_service)
) -> BatteryRULOutput:
    """
    Predict Remaining Useful Life (RUL) only, using State of Health as input feature.
    """
    try:
        rul, source = model_svc.predict_rul_only(
            cycle_number=payload.cycle_number,
            voltage_v=payload.voltage_v,
            temperature_c=payload.temperature_c,
            capacity_ah=payload.capacity_ah,
            voltage_sag_v=payload.voltage_sag_v,
            degradation_rate=payload.degradation_rate,
            state_of_health=payload.state_of_health,
            cycle_normalized=payload.cycle_normalized
        )
        return BatteryRULOutput(
            remaining_useful_life_cycles=rul,
            rul_model_used=source
        )
    except Exception as e:
        logger.error(f"Error predicting battery RUL: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to execute battery RUL prediction: {str(e)}"
        )

@router.get("/health/{battery_id}", response_model=BatteryPredictionOutput, status_code=status.HTTP_200_OK,
            responses={404: {"model": ErrorResponse}, 500: {"model": ErrorResponse}})
async def get_battery_health_by_id(
    battery_id: str,
    model_svc: ModelService = Depends(get_model_service)
) -> BatteryPredictionOutput:
    """
    Lookup a battery by ID in the database/dataset, extract the latest cycle telemetry,
    and predict SOH and RUL.
    """
    try:
        # Load battery dataframe
        try:
            df = data_loader.load("battery")
        except Exception as e:
            logger.error(f"Failed to load battery dataset: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Battery dataset is unavailable."
            )
            
        # Try matching integer or string ID
        normalized_id = battery_id
        if battery_id.upper().startswith("BAT-"):
            digits = "".join([c for c in battery_id if c.isdigit()])
            if digits:
                normalized_id = int(digits)
        else:
            try:
                normalized_id = int(battery_id)
            except ValueError:
                pass
                
        # Find rows
        battery_rows = df[
            (df["Battery_ID"] == normalized_id) | 
            (df["Battery_ID"].astype(str) == str(battery_id))
        ]
        
        if battery_rows.empty:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Battery ID '{battery_id}' was not found in dataset."
            )
            
        # Get latest cycle row
        latest_row = battery_rows.sort_values("Cycle_Number", ascending=False).iloc[0]
        
        cycle_number = int(latest_row.get("Cycle_Number", 1))
        voltage_v = float(latest_row.get("Voltage_V", 3.7))
        temperature_c = float(latest_row.get("Temperature_C", 25.0))
        capacity_ah = float(latest_row.get("Capacity_Ah", 1.8))
        voltage_sag_v = float(latest_row.get("Voltage_Sag_V", 0.03))
        degradation_rate = float(latest_row.get("Degradation_Rate", -0.002))
        cycle_normalized = float(latest_row.get("Cycle_Normalized", cycle_number / 200.0))
        
        soh, rul, zone = model_svc.predict_soh_and_rul(
            cycle_number=cycle_number,
            voltage_v=voltage_v,
            temperature_c=temperature_c,
            capacity_ah=capacity_ah,
            voltage_sag_v=voltage_sag_v,
            degradation_rate=degradation_rate,
            cycle_normalized=cycle_normalized
        )
        
        status_dict = model_svc.get_model_load_status()
        soh_source = "trained" if status_dict["battery_soh_model"] else "fallback_math"
        rul_source = "trained" if status_dict["battery_rul_model"] else "fallback_math"
        
        return BatteryPredictionOutput(
            battery_id=str(battery_id),
            state_of_health=soh,
            remaining_useful_life_cycles=rul,
            health_zone=zone,
            soh_model_used=soh_source,
            rul_model_used=rul_source
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting battery health: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error looking up battery health: {str(e)}"
        )
