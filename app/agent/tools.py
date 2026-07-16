import logging
import os
import json
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from langchain_core.tools import tool


from app.config import settings
from app.utils.data_loader import data_loader
from app.services.models_service import model_service
from app.services.carbon_service import carbon_service
from app.utils.relationships import RelationshipMapper

logger = logging.getLogger(__name__)


class EmptyInput(BaseModel):
    """Empty schema for tools requiring no input arguments."""
    pass

class VehicleSearchInput(BaseModel):
    vehicle_id: str = Field(..., description="Unique vehicle identifier string or number (e.g. '1', '2', 'VH-0013')")

class BatterySearchInput(BaseModel):
    battery_id: str = Field(..., description="Unique battery identifier name (e.g. 'B0005', 'B0006')")

class BatteryPredictInput(BaseModel):
    cycle_number: int = Field(..., ge=1, description="Telemetry charge/discharge cycle count")
    voltage_v: float = Field(..., ge=0.0, description="Terminal voltage in Volts")
    temperature_c: float = Field(..., description="Cell core temperature in Celsius")
    capacity_ah: float = Field(..., ge=0.0, description="Cell capacity in Ah")
    voltage_sag_v: float = Field(0.0, description="Voltage sag under load in Volts")
    degradation_rate: float = Field(0.0, description="Degradation rate in Ah/cycle")
    cycle_normalized: Optional[float] = Field(None, ge=0.0, le=1.0, description="Normalized cycle index")

class CarbonSavingsInput(BaseModel):
    vehicle_id: str = Field(..., description="Vehicle ID to evaluate")
    annual_distance_km: float = Field(..., ge=0.0, description="Annual distance traveled in km")
    fuel_type: str = Field(..., description="Current fuel type (Diesel/Petrol/CNG)")

@tool
def get_fleet_summary_metrics() -> str:
    """
    Get the overall fleet electrification readiness statistics, including total vehicle count,
    high/medium/low readiness counts, and general transition percentage.
    """
    try:
        df = data_loader.load("fleet_readiness")
        total_v = len(df)
        scores = df["EV_Readiness_Score"].dropna()
        
        high_c = int((scores >= 0.6).sum())
        med_c = int(((scores >= 0.4) & (scores < 0.6)).sum())
        low_c = int((scores < 0.4).sum())
        pct = (high_c / total_v * 100.0) if total_v > 0 else 0.0
        
        result = {
            "total_vehicles": total_v,
            "high_readiness_count": high_c,
            "medium_readiness_count": med_c,
            "low_readiness_count": low_c,
            "readiness_percentage": round(pct, 2),
            "data_source": "Fleet Dataset"
        }
        return json.dumps(result)
    except Exception as e:
        logger.error(f"Error in get_fleet_summary_metrics tool: {str(e)}", exc_info=True)
        return json.dumps({"error": f"Failed to compute fleet summary: {str(e)}"})

@tool
def search_vehicle_feasibility(vehicle_id: str) -> str:
    """
    Look up a specific vehicle by its ID in the fleet electrification database to retrieve its readiness score,
    transition category, recommended EV replacement model, estimated cost, and delivery lead time.
    """
    try:
        df = data_loader.load("fleet_readiness")
        
        normalized_id = vehicle_id
        if vehicle_id.upper().startswith("VH-"):
            digits = "".join([c for c in vehicle_id if c.isdigit()])
            if digits:
                normalized_id = int(digits)
        else:
            try:
                normalized_id = int(vehicle_id)
            except ValueError:
                pass

        match = df[
            (df["Vehicle_ID"] == normalized_id) | 
            (df["Vehicle_ID"].astype(str) == str(vehicle_id))
        ]
        
        if match.empty:
            return json.dumps({"error": f"Vehicle ID '{vehicle_id}' not found in registry database."})
            
        row = match.iloc[0]
        score = float(row["EV_Readiness_Score"])
        v_type = str(row.get("Vehicle_Type", "Van"))
        
        if score >= 0.6:
            cat = "High Readiness"
        elif score >= 0.4:
            cat = "Moderate Readiness"
        else:
            cat = "Low Readiness"
            
        if "truck" in v_type.lower():
            repl = "Rivian EDV"
            cost = 72000.0
            lead = 5
        elif "van" in v_type.lower():
            repl = "Ford E-Transit"
            cost = 45000.0
            lead = 3
        else:
            repl = "Tata Ace EV"
            cost = 15000.0
            lead = 2
            
        result = {
            "vehicle_id": str(vehicle_id),
            "vehicle_type": v_type,
            "ev_readiness_score": round(score, 4),
            "readiness_category": cat,
            "recommended_ev_replacement": repl,
            "estimated_cost_usd": cost,
            "lead_time_months": lead,
            "model_name": "LinearRegression Fleet Model",
            "model_confidence_r2": 0.9999,
            "data_source": "Fleet Dataset"
        }
        return json.dumps(result)
    except Exception as e:
        logger.error(f"Error in search_vehicle_feasibility tool: {str(e)}", exc_info=True)
        return json.dumps({"error": f"Failed to retrieve vehicle data: {str(e)}"})

@tool
def analyze_battery_health(battery_id: str) -> str:
    """
    Get current State of Health (SOH), Remaining Useful Life (RUL) in cycles, and overall health status category
    for a specific battery asset ID (e.g. B0005) by analyzing its latest cycle telemetry.
    """
    try:
        df = data_loader.load("battery")
        
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
                
        battery_rows = df[
            (df["Battery_ID"] == normalized_id) | 
            (df["Battery_ID"].astype(str) == str(battery_id))
        ]
        
        if battery_rows.empty:
            return json.dumps({"error": f"Battery ID '{battery_id}' was not found in database logs."})
            
        latest_row = battery_rows.sort_values("Cycle_Number", ascending=False).iloc[0]
        
        cycle_number = int(latest_row.get("Cycle_Number", 1))
        voltage_v = float(latest_row.get("Voltage_V", 3.7))
        temperature_c = float(latest_row.get("Temperature_C", 25.0))
        capacity_ah = float(latest_row.get("Capacity_Ah", 1.8))
        voltage_sag_v = float(latest_row.get("Voltage_Sag_V", 0.03))
        degradation_rate = float(latest_row.get("Degradation_Rate", -0.002))
        cycle_normalized = float(latest_row.get("Cycle_Normalized", cycle_number / 200.0))
        
        soh, rul, zone = model_service.predict_soh_and_rul(
            cycle_number=cycle_number,
            voltage_v=voltage_v,
            temperature_c=temperature_c,
            capacity_ah=capacity_ah,
            voltage_sag_v=voltage_sag_v,
            degradation_rate=degradation_rate,
            cycle_normalized=cycle_normalized
        )
        
        status_dict = model_service.get_model_load_status()
        soh_model = "GradientBoosting Battery Model" if status_dict["battery_soh_model"] else "Fallback Math Model"
        rul_model = "GradientBoosting Battery Model" if status_dict["battery_rul_model"] else "Fallback Math Model"
        
        result = {
            "battery_id": str(battery_id),
            "latest_logged_cycle": cycle_number,
            "predicted_soh": round(soh, 4),
            "predicted_rul_cycles": rul,
            "health_zone": zone,
            "soh_model_used": soh_model,
            "rul_model_used": rul_model,
            "soh_model_confidence_r2": 0.8009,
            "data_source": "Battery Dataset"
        }
        return json.dumps(result)
    except Exception as e:
        logger.error(f"Error in analyze_battery_health tool: {str(e)}", exc_info=True)
        return json.dumps({"error": f"Failed to analyze battery health: {str(e)}"})

@tool
def predict_custom_battery_telemetry(
    cycle_number: int,
    voltage_v: float,
    temperature_c: float,
    capacity_ah: float,
    voltage_sag_v: float = 0.0,
    degradation_rate: float = 0.0,
    cycle_normalized: Optional[float] = None
) -> str:
    """
    Run predictions using the trained ML model pipeline to calculate the State of Health (SOH) and
    Remaining Useful Life (RUL) for any custom battery telemetry readings.
    """
    try:
        soh, rul, zone = model_service.predict_soh_and_rul(
            cycle_number=cycle_number,
            voltage_v=voltage_v,
            temperature_c=temperature_c,
            capacity_ah=capacity_ah,
            voltage_sag_v=voltage_sag_v,
            degradation_rate=degradation_rate,
            cycle_normalized=cycle_normalized
        )
        
        status_dict = model_service.get_model_load_status()
        soh_model = "GradientBoosting Battery Model" if status_dict["battery_soh_model"] else "Fallback Math Model"
        rul_model = "GradientBoosting Battery Model" if status_dict["battery_rul_model"] else "Fallback Math Model"
        
        result = {
            "predicted_soh": round(soh, 4),
            "predicted_rul_cycles": rul,
            "health_zone": zone,
            "soh_model_used": soh_model,
            "rul_model_used": rul_model,
            "soh_model_confidence_r2": 0.8009
        }
        return json.dumps(result)
    except Exception as e:
        logger.error(f"Error in predict_custom_battery_telemetry tool: {str(e)}", exc_info=True)
        return json.dumps({"error": f"Failed to predict custom battery telemetry: {str(e)}"})

@tool
def get_carbon_metrics_summary() -> str:
    """
    Retrieve the cumulative carbon footprint statistics for the entire fleet, including annual baseline ICE emissions,
    target EV scenario emissions, net CO2 savings, and net-zero progress.
    """
    try:
        metrics = carbon_service.get_carbon_metrics()
        metrics["data_source"] = "Carbon Dataset"
        return json.dumps(metrics)
    except Exception as e:
        logger.error(f"Error in get_carbon_metrics_summary tool: {str(e)}", exc_info=True)
        return json.dumps({"error": f"Failed to get carbon metrics: {str(e)}"})

@tool
def calculate_vehicle_carbon_savings(vehicle_id: str, annual_distance_km: float, fuel_type: str) -> str:
    """
    Compare baseline ICE emissions versus projected EV grid emissions for a specific vehicle over a given annual distance and fuel type.
    """
    try:
        res = carbon_service.analyze_vehicle_carbon(
            vehicle_id=vehicle_id,
            annual_distance_km=annual_distance_km,
            fuel_type=fuel_type
        )
        res["data_source"] = "Carbon Dataset"
        return json.dumps(res)
    except Exception as e:
        logger.error(f"Error in calculate_vehicle_carbon_savings tool: {str(e)}", exc_info=True)
        return json.dumps({"error": f"Failed to calculate carbon savings: {str(e)}"})

@tool
def get_poor_charging_routes() -> str:
    """
    Retrieve fleet route details with poor charging accessibility (Charging Proximity Index is 0.0).
    """
    try:
        raw_routes = data_loader.load("fleet_routes")
        raw_stations = data_loader.load("charging_stations")
        
        # Run relationship mapper index
        mapper = RelationshipMapper({"fleet_routes": raw_routes, "charging_stations": raw_stations})
        cpi_df = mapper.compute_route_charging_proximity()
        
        if cpi_df is None or cpi_df.empty:
            return json.dumps({"error": "Failed to map charging proximity data."})
            
        # Get poor charging routes (cpi = 0)
        poor_routes = cpi_df[cpi_df["Charging_Proximity_Index"] == 0.0]
        
        summary_routes = []
        # Return top 5 sample poor routes with details
        for idx, row in poor_routes.head(5).iterrows():
            summary_routes.append({
                "vehicle_id": str(row.get("Vehicle_ID")),
                "route_id": str(row.get("Route_ID")),
                "start_location": str(row.get("Start_Location")),
                "end_location": str(row.get("End_Location")),
                "distance_km": float(row.get("Distance_km", 0.0)),
                "charging_proximity_index": 0.0,
                "cpi_coverage_category": str(row.get("CPI_Coverage_Category", "No Coverage"))
            })
            
        result = {
            "total_unmapped_routes": len(poor_routes),
            "sample_unmapped_routes": summary_routes,
            "data_source": "Fleet Dataset, Charging Dataset"
        }
        return json.dumps(result)
    except Exception as e:
        logger.error(f"Error in get_poor_charging_routes tool: {str(e)}", exc_info=True)
        return json.dumps({"error": f"Failed to analyze charging routes: {str(e)}"})

# Complete list of tools
ALL_TOOLS = [
    get_fleet_summary_metrics,
    search_vehicle_feasibility,
    analyze_battery_health,
    predict_custom_battery_telemetry,
    get_carbon_metrics_summary,
    calculate_vehicle_carbon_savings,
    get_poor_charging_routes
]
