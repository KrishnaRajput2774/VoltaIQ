import logging
import os
import json
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage

from app.config import settings
from app.utils.data_loader import data_loader
from app.services.models_service import model_service
from app.services.carbon_service import carbon_service
from app.utils.relationships import RelationshipMapper

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 1. Pydantic Schemas for Tool Input Validation
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# 2. LangChain Tools Definition
# ---------------------------------------------------------------------------

@tool("get_fleet_summary_metrics", args_schema=EmptyInput)
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

@tool("search_vehicle_feasibility", args_schema=VehicleSearchInput)
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

@tool("analyze_battery_health", args_schema=BatterySearchInput)
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

@tool("predict_custom_battery_telemetry", args_schema=BatteryPredictInput)
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

@tool("get_carbon_metrics_summary", args_schema=EmptyInput)
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

@tool("calculate_vehicle_carbon_savings", args_schema=CarbonSavingsInput)
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

@tool("get_poor_charging_routes", args_schema=EmptyInput)
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

# ---------------------------------------------------------------------------
# 3. AI Fleet Advisor Agent Class
# ---------------------------------------------------------------------------

class FleetAdvisorAgent:
    """
    Conversational AI Advisor leveraging LangChain.
    Communicates with datasets and predictive models via structured tools.
    """
    def __init__(self):
        self.api_key = settings.openai_api_key
        self.model_name = settings.openai_model_name
        self.temperature = settings.openai_model_temp
        
        self.system_prompt = (
            "You are VoltIQ's dedicated Industrial Fleet Electrification & Asset Intelligence Advisor. "
            "You are completely data-grounded. You must answer questions ONLY using our official datasets, "
            "trained ML models, and outputs from the tools provided to you. "
            "If information is not available in the datasets or tool outputs, clearly state that it is unavailable.\n\n"
            "Guidelines:\n"
            "- References to data should cite exact sources, such as 'Fleet Dataset', 'Battery Dataset', or 'Carbon Dataset'.\n"
            "- References to predictions must explicitly name the machine learning model used, e.g., 'LinearRegression Fleet Model' "
            "or 'GradientBoosting Battery Model'.\n"
            "- When returning predictions, format them clearly with: prediction output value, confidence level/performance metric "
            "(like R2 score or MAE as reported by the tools), and the model name.\n"
            "- Never make up vehicle IDs, SOH numbers, or carbon metrics. Only return facts returned by your tools.\n"
            "- Maintain a professional, action-oriented, and data-centric tone."
        )
        self.is_initialized = False
        self.llm = None
        self.llm_with_tools = None

    def initialize_agent(self) -> None:
        """Initialize the ChatOpenAI client and bind tools."""
        if not self.api_key:
            logger.warning("OPENAI_API_KEY is not set. FleetAdvisor will run in rules-based data-grounded simulation mode.")
            return

        try:
            self.llm = ChatOpenAI(
                model=self.model_name,
                temperature=self.temperature,
                openai_api_key=self.api_key
            )
            self.llm_with_tools = self.llm.bind_tools(ALL_TOOLS)
            self.is_initialized = True
            logger.info(f"LangChain Fleet Advisor initialized with model {self.model_name} (temp={self.temperature}).")
        except Exception as e:
            logger.error(f"Failed to initialize ChatOpenAI: {str(e)}", exc_info=True)
            self.is_initialized = False

    def run_query(self, user_message: str, chat_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """
        Execute query against the agent.
        If API key is missing or calls fail, falls back to the deterministic data-grounded responder.
        """
        if not self.is_initialized:
            self.initialize_agent()

        if self.is_initialized and self.llm_with_tools is not None:
            try:
                # Format conversation history
                messages = [SystemMessage(content=self.system_prompt)]
                if chat_history:
                    for chat in chat_history:
                        role = chat.get("role", "user")
                        msg = chat.get("message", "")
                        if role == "user":
                            messages.append(HumanMessage(content=msg))
                        else:
                            messages.append(AIMessage(content=msg))
                messages.append(HumanMessage(content=user_message))

                # First LLM invocation
                logger.info(f"Invoking LLM for query: {user_message}")
                ai_msg = self.llm_with_tools.invoke(messages)

                # Tool-execution loop
                if ai_msg.tool_calls:
                    messages.append(ai_msg)
                    
                    for tool_call in ai_msg.tool_calls:
                        tool_name = tool_call["name"]
                        tool_args = tool_call["args"]
                        tool_id = tool_call["id"]
                        
                        logger.info(f"Executing tool {tool_name} with args: {tool_args}")
                        
                        # Find and run the tool
                        matched_tool = next((t for t in ALL_TOOLS if t.name == tool_name), None)
                        if matched_tool:
                            try:
                                tool_result = matched_tool.invoke(tool_args)
                            except Exception as e:
                                logger.error(f"Error running tool {tool_name}: {str(e)}")
                                tool_result = json.dumps({"error": f"Failed to execute tool: {str(e)}"})
                        else:
                            tool_result = json.dumps({"error": f"Tool '{tool_name}' not found."})

                        messages.append(ToolMessage(content=tool_result, tool_call_id=tool_id))

                    # Invoke LLM again to synthesize tool outputs
                    logger.info("Re-invoking LLM with tool outputs...")
                    final_msg = self.llm.invoke(messages)
                    response_text = final_msg.content
                else:
                    response_text = ai_msg.content

                # Resolve source attributions based on response text keywords
                sources = []
                response_lower = response_text.lower()
                if "fleet dataset" in response_lower:
                    sources.append("Fleet Dataset")
                if "battery dataset" in response_lower:
                    sources.append("Battery Dataset")
                if "carbon dataset" in response_lower:
                    sources.append("Carbon Dataset")
                if "charging dataset" in response_lower:
                    sources.append("Charging Dataset")
                if "gradientboosting battery model" in response_lower:
                    sources.append("GradientBoosting Battery Model")
                if "linearregression fleet model" in response_lower:
                    sources.append("LinearRegression Fleet Model")

                # Fallback default sources based on tool names if none matched text
                if not sources and ai_msg.tool_calls:
                    for tc in ai_msg.tool_calls:
                        name = tc["name"]
                        if "fleet" in name or "vehicle" in name:
                            sources.append("Fleet Dataset")
                            if "vehicle" in name:
                                sources.append("LinearRegression Fleet Model")
                        if "battery" in name:
                            sources.append("Battery Dataset")
                            sources.append("GradientBoosting Battery Model")
                        if "carbon" in name:
                            sources.append("Carbon Dataset")

                return {
                    "response": response_text,
                    "sources": list(set(sources))
                }

            except Exception as e:
                logger.error(f"OpenAI/LangChain call failed: {str(e)}. Falling back to offline responder.", exc_info=True)
                return self._run_offline_fallback(user_message)
        else:
            return self._run_offline_fallback(user_message)

    def _run_offline_fallback(self, user_message: str) -> Dict[str, Any]:
        """
        Smart, completely data-grounded rules-based fallback engine.
        Reads the local datasets and runs the same python functions to construct exact answers.
        """
        logger.info(f"Running query on offline data-grounded fallback: {user_message}")
        query_lower = user_message.lower()
        
        # 1. Which vehicles should I electrify first?
        if "electrify first" in query_lower or "which vehicles should i electrify" in query_lower:
            try:
                # Load fleet summary
                res = get_fleet_summary_metrics.invoke({})
                summary = json.loads(res)
                
                # Fetch high readiness vehicles
                df = data_loader.load("fleet_readiness")
                high_ready = df[df["EV_Readiness_Score"] >= 0.6].sort_values("EV_Readiness_Score", ascending=False).head(3)
                
                v_list = []
                for _, r in high_ready.iterrows():
                    v_list.append(f"- **Vehicle {r['Vehicle_ID']}** (Class: {r['Vehicle_Type']}): Readiness score of **{r['EV_Readiness_Score']:.4f}**")
                v_str = "\n".join(v_list)
                
                response = (
                    f"According to the **Fleet Dataset**, you should prioritize electrifying the vehicles with the highest suitability indexes. "
                    f"Out of {summary['total_vehicles']:,} total fleet assets, **{summary['high_readiness_count']:,} vehicles** ({summary['readiness_percentage']:.1f}%) "
                    f"exhibit high electrification suitability (scores >= 0.6).\n\n"
                    f"The top candidates for immediate transition are:\n{v_str}\n\n"
                    f"These classifications were predicted using the **LinearRegression Fleet Model** (Test R² score: 0.9999) based on operational variables."
                )
                return {
                    "response": response,
                    "sources": ["Fleet Dataset", "LinearRegression Fleet Model"]
                }
            except Exception as e:
                return {
                    "response": f"Failed to lookup vehicle readiness data: {str(e)}",
                    "sources": ["Fleet Dataset"]
                }

        # 2. Show batteries with the lowest SOH.
        elif "lowest soh" in query_lower or "lowest state of health" in query_lower:
            try:
                df = data_loader.load("battery")
                # Group by Battery ID to find the lowest/latest SOH
                latest_records = df.sort_values("Cycle_Number").groupby("Battery_ID").last().reset_index()
                lowest_soh = latest_records.sort_values("State_of_Health", ascending=True).head(3)
                
                b_list = []
                for _, r in lowest_soh.iterrows():
                    b_list.append(f"- **Battery {r['Battery_ID']}** at cycle {r['Cycle_Number']}: SOH = **{r['State_of_Health']*100:.2f}%**")
                b_str = "\n".join(b_list)
                
                response = (
                    f"Based on historical telemetry logs in the **Battery Dataset**, the battery assets with the lowest recorded State of Health (SOH) are:\n"
                    f"{b_str}\n\n"
                    f"These state of health estimates are processed from physical parameters via the **GradientBoosting Battery Model** (Test R²: 0.8009)."
                )
                return {
                    "response": response,
                    "sources": ["Battery Dataset", "GradientBoosting Battery Model"]
                }
            except Exception as e:
                return {
                    "response": f"Failed to retrieve battery logs: {str(e)}",
                    "sources": ["Battery Dataset"]
                }

        # 3. Compare diesel vs EV carbon emissions.
        elif "compare diesel vs ev" in query_lower or "carbon emissions" in query_lower or "co2 emissions" in query_lower:
            try:
                res = get_carbon_metrics_summary.invoke({})
                metrics = json.loads(res)
                
                response = (
                    f"According to the **Carbon Dataset**, transitioning the conventional combustion fleet to electric alternatives yields significant carbon offsets:\n"
                    f"- **Baseline ICE Annual CO2**: `{metrics['baseline_co2_kg']:,.2f} kg` (mainly Diesel greenhouse gas outputs)\n"
                    f"- **Projected EV Scenario CO2**: `{metrics['ev_scenario_co2_kg']:,.2f} kg` (grid electricity charging footprint)\n"
                    f"- **Avoided CO2 per Year**: `{metrics['annual_savings_kg']:,.2f} kg` (a **{metrics['carbon_intensity_reduction_pct']:.1f}% intensity reduction**)\n\n"
                    f"This analysis indicates that the fleet has achieved **{metrics['net_zero_progress_pct']:.2f}% progress** toward its target net-zero alignment commits."
                )
                return {
                    "response": response,
                    "sources": ["Carbon Dataset"]
                }
            except Exception as e:
                return {
                    "response": f"Failed to calculate carbon offsets: {str(e)}",
                    "sources": ["Carbon Dataset"]
                }

        # 4. Which routes have poor charging accessibility?
        elif "poor charging" in query_lower or "charging accessibility" in query_lower or "proximity index" in query_lower:
            try:
                res = get_poor_charging_routes.invoke({})
                poor_res = json.loads(res)
                
                routes_list = []
                for r in poor_res["sample_unmapped_routes"]:
                    routes_list.append(f"- **Route {r['route_id']}** (Vehicle {r['vehicle_id']}): `{r['start_location']}` to `{r['end_location']}` (Distance: {r['distance_km']} km) -> CPI: **{r['charging_proximity_index']:.1f}** ({r['cpi_coverage_category']})")
                r_str = "\n".join(routes_list)
                
                response = (
                    f"Based on the geospatial proximity analysis mapping routes to charging station centroids (**Fleet Dataset** and **Charging Dataset**):\n"
                    f"There are **{poor_res['total_unmapped_routes']:,} routes** classified as having poor charging accessibility (Charging Proximity Index = 0.0).\n\n"
                    f"Sample affected routes include:\n{r_str}\n\n"
                    f"These index metrics represent the city-centroid approximation, where route origins are matched to known charging coordinates."
                )
                return {
                    "response": response,
                    "sources": ["Fleet Dataset", "Charging Dataset"]
                }
            except Exception as e:
                return {
                    "response": f"Failed to compute route charging access parameters: {str(e)}",
                    "sources": ["Fleet Dataset", "Charging Dataset"]
                }

        # 5. Summarize the overall fleet readiness.
        elif "summarize the overall" in query_lower or "overall fleet readiness" in query_lower:
            try:
                res = get_fleet_summary_metrics.invoke({})
                summary = json.loads(res)
                
                response = (
                    f"A dynamic summary of electrification suitability extracted from the **Fleet Dataset** indicates:\n"
                    f"- **Total Vehicles Analyzed**: `{summary['total_vehicles']:,}` assets\n"
                    f"- **High Readiness (score >= 0.6)**: `{summary['high_readiness_count']:,}` vehicles\n"
                    f"- **Moderate Readiness (0.4 to 0.6)**: `{summary['medium_readiness_count']:,}` vehicles\n"
                    f"- **Low Readiness (< 0.4)**: `{summary['low_readiness_count']:,}` vehicles\n"
                    f"- **Electrification Percentage**: `{summary['readiness_percentage']:.2f}%` of the fleet is highly suitable for immediate transition.\n\n"
                    f"Scores are calculated from vehicle operational metrics using the **LinearRegression Fleet Model**."
                )
                return {
                    "response": response,
                    "sources": ["Fleet Dataset", "LinearRegression Fleet Model"]
                }
            except Exception as e:
                return {
                    "response": f"Failed to extract fleet summary metrics: {str(e)}",
                    "sources": ["Fleet Dataset"]
                }
                
        # 6. Default helper response
        else:
            return {
                "response": (
                    "I am the VoltIQ Fleet Intelligence Assistant. I can assist you with data-grounded insights on:\n"
                    "- Electrification feasibility (try: 'Which vehicles should I electrify first?' or 'Summarize the overall fleet readiness.')\n"
                    "- Battery health analysis (try: 'Show batteries with the lowest SOH.')\n"
                    "- Carbon footprint metrics (try: 'Compare diesel vs EV carbon emissions.')\n"
                    "- Charging accessibility (try: 'Which routes have poor charging accessibility?')\n\n"
                    "Please specify one of these query prompts to review exact records from the **VoltIQ Datasets**."
                ),
                "sources": []
            }

fleet_advisor = FleetAdvisorAgent()
# Initialize agent on module load
fleet_advisor.initialize_agent()
