from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Dict, Any

# --- Error Schemas ---
class ErrorResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "detail": "Requested resource was not found.",
                "request_id": "8f8c7b8c-5544-4321-9abc-def012345678"
            }
        }
    )
    detail: str = Field(..., description="Detailed description of the error")
    request_id: Optional[str] = Field(None, description="Unique UUID associated with the request log")

# --- Battery APM Schemas ---
class BatteryTelemetryInput(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "cycle_number": 50,
                "voltage_v": 3.65,
                "temperature_c": 32.5,
                "capacity_ah": 1.75,
                "voltage_sag_v": 0.03,
                "degradation_rate": -0.002,
                "cycle_normalized": 0.30
            }
        }
    )
    cycle_number: int = Field(..., description="Current charge/discharge cycle count", ge=1)
    voltage_v: float = Field(..., description="Measured terminal voltage in volts", ge=0.0)
    temperature_c: float = Field(..., description="Core cell temperature in Celsius")
    capacity_ah: float = Field(..., description="Measured cell capacity in Ampere-hours", ge=0.0)
    voltage_sag_v: float = Field(0.0, description="Measured voltage sag under load in volts")
    degradation_rate: float = Field(0.0, description="Calculated rate of degradation")
    cycle_normalized: Optional[float] = Field(None, description="Normalized cycle index (cycle/max_cycles)", ge=0.0, le=1.0)

class BatteryPredictionOutput(BaseModel):
    battery_id: Optional[str] = Field(None, description="Identifier of the battery")
    state_of_health: float = Field(..., description="Predicted State of Health (SOH) index [0-1]", ge=0.0, le=1.0)
    remaining_useful_life_cycles: int = Field(..., description="Estimated remaining useful cycles before retirement", ge=0)
    health_zone: str = Field(..., description="Categorized health zone (Healthy, Attention Required, Critical Failure Danger)")
    soh_model_used: str = Field(..., description="Source of SOH model prediction (trained/fallback_math)")
    rul_model_used: str = Field(..., description="Source of RUL model prediction (trained/fallback_math)")

class BatteryRULInput(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "cycle_number": 60,
                "voltage_v": 3.5,
                "temperature_c": 28.0,
                "capacity_ah": 1.6,
                "voltage_sag_v": 0.04,
                "degradation_rate": -0.003,
                "cycle_normalized": 0.35,
                "state_of_health": 0.88
            }
        }
    )
    cycle_number: int = Field(..., description="Current charge/discharge cycle count", ge=1)
    voltage_v: float = Field(..., description="Measured terminal voltage in volts", ge=0.0)
    temperature_c: float = Field(..., description="Core cell temperature in Celsius")
    capacity_ah: float = Field(..., description="Measured cell capacity in Ampere-hours", ge=0.0)
    voltage_sag_v: float = Field(0.0, description="Measured voltage sag under load in volts")
    degradation_rate: float = Field(0.0, description="Calculated rate of degradation")
    cycle_normalized: Optional[float] = Field(None, description="Normalized cycle index", ge=0.0, le=1.0)
    state_of_health: float = Field(..., description="Estimated or calculated SOH", ge=0.0, le=1.0)

class BatteryRULOutput(BaseModel):
    remaining_useful_life_cycles: int = Field(..., description="Estimated remaining cycles before retirement", ge=0)
    rul_model_used: str = Field(..., description="Source of model prediction")

# --- Fleet Electrification Schemas ---
class FleetReadinessInput(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "vehicle_id": "VH-0013"
            }
        }
    )
    vehicle_id: str = Field(..., description="Existing vehicle unique identifier")

class FleetVehicleReadiness(BaseModel):
    vehicle_id: str = Field(..., description="Vehicle unique identifier")
    vehicle_type: str = Field(..., description="Type/class of the vehicle")
    ev_readiness_score: float = Field(..., description="Electrification suitability index [0-1]", ge=0.0, le=1.0)
    readiness_category: str = Field(..., description="Categorized suitability status (e.g. High Readiness)")
    recommended_ev_replacement: str = Field(..., description="Suggested electric model replacement")
    estimated_cost_usd: float = Field(..., description="Estimated vehicle acquisition cost in USD")
    lead_time_months: int = Field(..., description="Delivery lead time in months")

class FleetSummaryOutput(BaseModel):
    total_vehicles: int = Field(..., description="Total active vehicles analyzed in fleet")
    high_readiness_count: int = Field(..., description="Vehicles with score >= 0.6")
    medium_readiness_count: int = Field(..., description="Vehicles with score between 0.4 and 0.6")
    low_readiness_count: int = Field(..., description="Vehicles with score < 0.4")
    readiness_percentage: float = Field(..., description="Percentage of vehicles eligible for transition")
    recommendations: List[FleetVehicleReadiness] = Field(..., description="Sample high-readiness recommendations")

class FleetPredictInput(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "vehicle_age_years": 8,
                "usage_hours": 8000.0,
                "fuel_consumption": 10.5,
                "health_score": 80.0,
                "load_capacity": 5000.0,
                "actual_load": 3000.0,
                "load_utilization_pct": 60.0,
                "fuel_per_hour": 1.25,
                "maintenance_cost": 1200.0,
                "days_since_last_maintenance": 45,
                "failure_history": 2,
                "anomalies_detected": 1,
                "diagnostic_trouble_code_count": 0,
                "predictive_score": 0.85,
                "pcr": 0.55,
                "uir": 0.60,
                "tpi": 0.70,
                "mbf": 250.0,
                "ads": 3.5,
                "ohi": 85.0,
                "cmes": 1500.0,
                "uer": 0.75,
                "vehicle_type": "Light Truck",
                "route_info": "Urban Delivery",
                "road_conditions": "Smooth",
                "weather_conditions": "Clear",
                "brake_condition": "Good"
            }
        }
    )
    vehicle_age_years: int = Field(..., ge=0, description="Age of vehicle in years")
    usage_hours: float = Field(..., ge=0.0, description="Total engine/usage hours")
    fuel_consumption: float = Field(..., ge=0.0, description="Fuel consumption rate (L/100km)")
    health_score: float = Field(..., ge=0.0, le=100.0, description="Vehicle health index")
    load_capacity: float = Field(5000.0, ge=0.0, description="Max load capacity in kg")
    actual_load: float = Field(2500.0, ge=0.0, description="Typical load carried in kg")
    load_utilization_pct: float = Field(50.0, ge=0.0, le=100.0, description="Load capacity utilisation %")
    fuel_per_hour: float = Field(1.5, ge=0.0, description="Fuel burned per hour")
    maintenance_cost: float = Field(1000.0, ge=0.0, description="Annual maintenance cost in USD")
    days_since_last_maintenance: int = Field(30, ge=0, description="Days since last service")
    failure_history: int = Field(0, ge=0, description="Count of historical mechanical failures")
    anomalies_detected: int = Field(0, ge=0, description="Sensor anomalous flag counts")
    diagnostic_trouble_code_count: int = Field(0, ge=0, description="Active trouble codes count")
    predictive_score: float = Field(0.5, ge=0.0, le=1.0, description="Maintenance predictive score")
    pcr: float = Field(0.5, description="Performance Cost Ratio")
    uir: float = Field(0.5, description="Utilisation Idle Ratio")
    tpi: float = Field(0.5, description="Total Performance Index")
    mbf: float = Field(300.0, description="Mean time between failures")
    ads: float = Field(5.0, description="Average Downtime Score")
    ohi: float = Field(80.0, description="Overall Health Index")
    cmes: float = Field(1000.0, description="Cumulative Maintenance Effort Score")
    uer: float = Field(0.5, description="Utilisation Efficiency Ratio")
    vehicle_type: str = Field("Light Truck", description="Vehicle subclass type")
    route_info: str = Field("Urban Delivery", description="Primary route info")
    road_conditions: str = Field("Smooth", description="Typical road condition")
    weather_conditions: str = Field("Clear", description="Typical weather condition")
    brake_condition: str = Field("Good", description="Brake pad health category")

class FleetPredictOutput(BaseModel):
    ev_readiness_score: float = Field(..., ge=0.0, le=1.0, description="Suitability score")
    readiness_category: str = Field(..., description="Suitability category")
    model_used: str = Field(..., description="Algorithm used for prediction")

# --- Carbon Intelligence Schemas ---
class CarbonFootprintOutput(BaseModel):
    baseline_co2_kg: float = Field(..., description="Total annual baseline ICE emissions")
    ev_scenario_co2_kg: float = Field(..., description="Target annual emissions with selected EV replacement")
    annual_savings_kg: float = Field(..., description="Avoided carbon emissions in kg")
    carbon_intensity_reduction_pct: float = Field(..., description="Percentage drop in carbon intensity index")
    net_zero_progress_pct: float = Field(..., description="Current fleet progress toward target net-zero percentage")

class CarbonAnalysisInput(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "vehicle_id": "VH-0013",
                "annual_distance_km": 15000.0,
                "fuel_type": "Diesel"
            }
        }
    )
    vehicle_id: str = Field(..., description="Vehicle ID to evaluate")
    annual_distance_km: float = Field(..., ge=0.0, description="Annual distance travelled in km")
    fuel_type: str = Field("Diesel", description="Fuel type (Diesel/Petrol/CNG)")

class CarbonAnalysisOutput(BaseModel):
    vehicle_id: str = Field(..., description="Vehicle ID analyzed")
    baseline_annual_co2_kg: float = Field(..., description="Estimated ICE baseline CO2 emissions per year")
    ev_projected_co2_kg: float = Field(..., description="Projected grid/EV charging emissions per year")
    net_annual_savings_kg: float = Field(..., description="Net annual saved carbon in kg")
    savings_percentage: float = Field(..., description="Emissions reduction efficiency percentage")

# --- AI Chat Advisor Schemas ---
class ChatQueryInput(BaseModel):
    message: str = Field(..., description="User query relating to fleet performance, battery health, or carbon indices")
    context: Optional[dict] = Field(None, description="Optional telemetry parameters to enhance prompt context")
    chat_history: Optional[List[Dict[str, Any]]] = Field(None, description="Optional conversation memory history logs")

class ChatQueryResponse(BaseModel):
    response: str = Field(..., description="Generated markdown answer from the AI fleet advisor")
    sources: List[str] = Field(default=[], description="Referenced datasets or telemetry records")

# --- System Health & Model Registry Schemas ---
class SystemHealthOutput(BaseModel):
    status: str = Field(..., description="Overall status of the backend API")
    uptime_seconds: float = Field(..., description="Seconds since server process started")
    model_load_status: Dict[str, bool] = Field(..., description="Mapping of model identifier to loaded status")
    dataset_status: Dict[str, bool] = Field(..., description="Mapping of dataset names to file availability status")

class ModelInfoOutput(BaseModel):
    model_id: str = Field(..., description="Unique model registry identifier")
    version: str = Field(..., description="Version of the trained model")
    algorithm: str = Field(..., description="Underlying ML algorithm name")
    training_timestamp: str = Field(..., description="ISO 8601 timestamp of model training completion")
    selected_features: List[str] = Field(..., description="List of features input into model pipeline")
    metrics: Dict[str, Any] = Field(..., description="Performance evaluation metrics summary")
    metadata_path: str = Field(..., description="Path to model metadata file")

class ModelRegistryOutput(BaseModel):
    models: List[ModelInfoOutput] = Field(..., description="List of all registered models")
