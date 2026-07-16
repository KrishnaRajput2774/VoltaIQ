import os
import json
import joblib
import logging
from typing import Optional, Tuple, Dict, Any, List
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class ModelService:
    """
    Service class managing trained machine learning model loading, caching,
    and inference execution for battery SOH/RUL and fleet EV readiness.
    """
    def __init__(self):
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        
        logger.info("Base Dir Name: ", base_dir)
        
        self.battery_soh_model_path = os.path.join(base_dir, "saved_models", "battery", "battery_soh_model.pkl")
        self.battery_rul_model_path = os.path.join(base_dir, "saved_models", "battery", "battery_rul_model.pkl")
        self.fleet_readiness_model_path = os.path.join(base_dir, "saved_models", "fleet", "fleet_readiness_model.pkl")
        
        self.battery_soh_meta_path = os.path.join(base_dir, "saved_models", "battery", "battery_soh_metadata.json")
        self.battery_rul_meta_path = os.path.join(base_dir, "saved_models", "battery", "battery_rul_metadata.json")
        self.fleet_readiness_meta_path = os.path.join(base_dir, "saved_models", "fleet", "fleet_readiness_metadata.json")

        self._soh_model = None
        self._rul_model = None
        self._fleet_model = None
        
        self._soh_meta = None
        self._rul_meta = None
        self._fleet_meta = None

    def load_models(self) -> None:
        """
        Load trained model pickle files and metadata JSONs into memory.
        Does not raise errors; logs warnings or errors to support graceful fallback.
        """
        # Load Battery SOH Model
        try:
            if os.path.exists(self.battery_soh_model_path):
                self._soh_model = joblib.load(self.battery_soh_model_path)
                logger.info("Successfully loaded Battery SOH model.")
            else:
                logger.warning(f"Battery SOH model file not found at {self.battery_soh_model_path}. Fallback math logic will be used.")
            
            if os.path.exists(self.battery_soh_meta_path):
                with open(self.battery_soh_meta_path, "r", encoding="utf-8") as f:
                    self._soh_meta = json.load(f)
        except Exception as e:
            logger.error(f"Error loading Battery SOH model or metadata: {str(e)}")

        # Load Battery RUL Model
        try:
            if os.path.exists(self.battery_rul_model_path):
                self._rul_model = joblib.load(self.battery_rul_model_path)
                logger.info("Successfully loaded Battery RUL model.")
            else:
                logger.warning(f"Battery RUL model file not found at {self.battery_rul_model_path}. Fallback math logic will be used.")
            
            if os.path.exists(self.battery_rul_meta_path):
                with open(self.battery_rul_meta_path, "r", encoding="utf-8") as f:
                    self._rul_meta = json.load(f)
        except Exception as e:
            logger.error(f"Error loading Battery RUL model or metadata: {str(e)}")

        # Load Fleet Readiness Model
        try:
            if os.path.exists(self.fleet_readiness_model_path):
                self._fleet_model = joblib.load(self.fleet_readiness_model_path)
                logger.info("Successfully loaded Fleet Electrification Readiness model.")
            else:
                logger.warning(f"Fleet Readiness model file not found at {self.fleet_readiness_model_path}. Fallback math logic will be used.")
            
            if os.path.exists(self.fleet_readiness_meta_path):
                with open(self.fleet_readiness_meta_path, "r", encoding="utf-8") as f:
                    self._fleet_meta = json.load(f)
        except Exception as e:
            logger.error(f"Error loading Fleet Readiness model or metadata: {str(e)}")

    def get_model_load_status(self) -> Dict[str, bool]:
        """Return load status of each model file."""
        return {
            "battery_soh_model": self._soh_model is not None,
            "battery_rul_model": self._rul_model is not None,
            "fleet_readiness_model": self._fleet_model is not None,
        }

    def predict_soh_and_rul(self, cycle_number: int, voltage_v: float, temperature_c: float, capacity_ah: float, voltage_sag_v: float, degradation_rate: float, cycle_normalized: Optional[float] = None) -> Tuple[float, int, str]:
        """
        Predict SOH and RUL of a battery using loaded models or mathematical fallback.
        Maintains backward compatibility with tests/test_models.py.
        """
        # Load models if not loaded yet
        if self._soh_model is None or self._rul_model is None:
            self.load_models()

        soh_model_used = "fallback_math"
        rul_model_used = "fallback_math"

        # Predict SOH
        if self._soh_model is not None:
            try:
                # Features list expected: Cycle_Number, Voltage_V, Temperature_C, Capacity_Ah, Voltage_Sag_V, Degradation_Rate, Cycle_Normalized
                # The model is built as a pipeline so we pass a pandas DataFrame matching the schema
                cycle_norm = cycle_normalized if cycle_normalized is not None else (cycle_number / 200.0)
                input_data = pd.DataFrame([{
                    "Cycle_Number": cycle_number,
                    "Voltage_V": voltage_v,
                    "Temperature_C": temperature_c,
                    "Capacity_Ah": capacity_ah,
                    "Voltage_Sag_V": voltage_sag_v,
                    "Degradation_Rate": degradation_rate,
                    "Cycle_Normalized": cycle_norm
                }])
                
                soh = float(np.clip(self._soh_model.predict(input_data)[0], 0.0, 1.0))
                soh_model_used = "trained"
            except Exception as e:
                logger.error(f"Error running Battery SOH model prediction: {str(e)}. Falling back to mathematical SOH.")
                soh = self._get_fallback_soh(cycle_number, temperature_c)
        else:
            soh = self._get_fallback_soh(cycle_number, temperature_c)

        # Predict RUL
        if self._rul_model is not None:
            try:
                # Features expected by battery RUL model (includes State_of_Health)
                cycle_norm = cycle_normalized if cycle_normalized is not None else (cycle_number / 200.0)
                input_data = pd.DataFrame([{
                    "Cycle_Number": cycle_number,
                    "Voltage_V": voltage_v,
                    "Temperature_C": temperature_c,
                    "Capacity_Ah": capacity_ah,
                    "Capacity_Fade_Pct": (1.0 - soh) * 100.0,
                    "Voltage_Sag_V": voltage_sag_v,
                    "Degradation_Rate": degradation_rate,
                    "Cycle_Normalized": cycle_norm,
                    "State_of_Health": soh
                }])
                
                # Slicing the dataframe to match the expected features in the model metadata if necessary
                if self._rul_meta and "selected_features" in self._rul_meta:
                    cols = self._rul_meta["selected_features"]
                    input_data = input_data.reindex(columns=cols, fill_value=0.0)
                
                rul = int(max(0, round(self._rul_model.predict(input_data)[0])))
                rul_model_used = "trained"
            except Exception as e:
                logger.error(f"Error running Battery RUL model prediction: {str(e)}. Falling back to mathematical RUL.")
                rul = self._get_fallback_rul(cycle_number)
        else:
            rul = self._get_fallback_rul(cycle_number)

        # Classify health zones
        if soh >= 0.85:
            zone = "Healthy"
        elif soh >= 0.70:
            zone = "Attention Required"
        else:
            zone = "Critical Failure Danger"

        return round(soh, 4), rul, zone

    def predict_soh_only(self, cycle_number: int, voltage_v: float, temperature_c: float, capacity_ah: float, voltage_sag_v: float, degradation_rate: float, cycle_normalized: Optional[float] = None) -> Tuple[float, str]:
        """Predict SOH only."""
        if self._soh_model is None:
            self.load_models()
        if self._soh_model is not None:
            try:
                cycle_norm = cycle_normalized if cycle_normalized is not None else (cycle_number / 200.0)
                input_data = pd.DataFrame([{
                    "Cycle_Number": cycle_number,
                    "Voltage_V": voltage_v,
                    "Temperature_C": temperature_c,
                    "Capacity_Ah": capacity_ah,
                    "Voltage_Sag_V": voltage_sag_v,
                    "Degradation_Rate": degradation_rate,
                    "Cycle_Normalized": cycle_norm
                }])
                soh = float(np.clip(self._soh_model.predict(input_data)[0], 0.0, 1.0))
                return round(soh, 4), "trained"
            except Exception as e:
                logger.error(f"SOH prediction error: {str(e)}")
        return round(self._get_fallback_soh(cycle_number, temperature_c), 4), "fallback_math"

    def predict_rul_only(self, cycle_number: int, voltage_v: float, temperature_c: float, capacity_ah: float, voltage_sag_v: float, degradation_rate: float, state_of_health: float, cycle_normalized: Optional[float] = None) -> Tuple[int, str]:
        """Predict RUL only, using state_of_health as input."""
        if self._rul_model is None:
            self.load_models()
        if self._rul_model is not None:
            try:
                cycle_norm = cycle_normalized if cycle_normalized is not None else (cycle_number / 200.0)
                input_data = pd.DataFrame([{
                    "Cycle_Number": cycle_number,
                    "Voltage_V": voltage_v,
                    "Temperature_C": temperature_c,
                    "Capacity_Ah": capacity_ah,
                    "Capacity_Fade_Pct": (1.0 - state_of_health) * 100.0,
                    "Voltage_Sag_V": voltage_sag_v,
                    "Degradation_Rate": degradation_rate,
                    "Cycle_Normalized": cycle_norm,
                    "State_of_Health": state_of_health
                }])
                if self._rul_meta and "selected_features" in self._rul_meta:
                    cols = self._rul_meta["selected_features"]
                    input_data = input_data.reindex(columns=cols, fill_value=0.0)
                
                rul = int(max(0, round(self._rul_model.predict(input_data)[0])))
                return rul, "trained"
            except Exception as e:
                logger.error(f"RUL prediction error: {str(e)}")
        return self._get_fallback_rul(cycle_number), "fallback_math"

    def predict_fleet_readiness(self, payload_dict: Dict[str, Any]) -> Tuple[float, str, str]:
        """
        Predict EV Electrification Readiness score and category for a vehicle.
        """
        if self._fleet_model is None:
            self.load_models()
        
        if self._fleet_model is not None:
            try:
                # Construct DataFrame from payload_dict
                # We need all the features expected by the pipeline.
                # Features expected: 22 numeric, 5 categorical
                input_data = pd.DataFrame([payload_dict])
                
                pred = self._fleet_model.predict(input_data)[0]
                score = float(np.clip(pred, 0.0, 1.0))
                model_used = "trained"
            except Exception as e:
                logger.error(f"Error predicting fleet readiness: {str(e)}. Using fallback math.")
                score = self._get_fallback_readiness(payload_dict)
                model_used = "fallback_math"
        else:
            score = self._get_fallback_readiness(payload_dict)
            model_used = "fallback_math"
        
        # Categorize readiness score
        if score >= 0.6:
            category = "High Readiness"
        elif score >= 0.4:
            category = "Moderate Readiness"
        elif score >= 0.2:
            category = "Low Readiness"
        else:
            category = "Not Ready"
            
        return round(score, 4), category, model_used

    def get_registry_info(self) -> List[Dict[str, Any]]:
        """Return metadata info for all 3 models."""
        registry = []
        
        # Battery SOH
        registry.append({
            "model_id": "battery_soh_model",
            "version": self._soh_meta.get("model_version", "2.0.0") if self._soh_meta else "2.0.0",
            "algorithm": self._soh_meta.get("algorithm", "GradientBoostingRegressor") if self._soh_meta else "GradientBoostingRegressor",
            "training_timestamp": self._soh_meta.get("training_timestamp", "2026-07-15T18:00:00Z") if self._soh_meta else "Unknown",
            "selected_features": self._soh_meta.get("selected_features", []) if self._soh_meta else ["Cycle_Number", "Voltage_V", "Temperature_C", "Capacity_Ah", "Voltage_Sag_V", "Degradation_Rate", "Cycle_Normalized"],
            "metrics": self._soh_meta.get("evaluation_metrics", {}) if self._soh_meta else {},
            "metadata_path": self.battery_soh_meta_path
        })
        
        # Battery RUL
        registry.append({
            "model_id": "battery_rul_model",
            "version": self._rul_meta.get("model_version", "2.0.0") if self._rul_meta else "2.0.0",
            "algorithm": self._rul_meta.get("algorithm", "GradientBoostingRegressor") if self._rul_meta else "GradientBoostingRegressor",
            "training_timestamp": self._rul_meta.get("training_timestamp", "2026-07-15T18:00:00Z") if self._rul_meta else "Unknown",
            "selected_features": self._rul_meta.get("selected_features", []) if self._rul_meta else ["Cycle_Number", "Voltage_V", "Temperature_C", "Capacity_Ah", "Capacity_Fade_Pct", "Voltage_Sag_V", "Degradation_Rate", "Cycle_Normalized", "State_of_Health"],
            "metrics": self._rul_meta.get("evaluation_metrics", {}) if self._rul_meta else {},
            "metadata_path": self.battery_rul_meta_path
        })
        
        # Fleet Readiness
        registry.append({
            "model_id": "fleet_readiness_model",
            "version": self._fleet_meta.get("model_version", "2.0.0") if self._fleet_meta else "2.0.0",
            "algorithm": self._fleet_meta.get("algorithm", "LinearRegression") if self._fleet_meta else "LinearRegression",
            "training_timestamp": self._fleet_meta.get("training_timestamp", "2026-07-15T18:00:00Z") if self._fleet_meta else "Unknown",
            "selected_features": (self._fleet_meta.get("selected_numeric_features", []) + self._fleet_meta.get("selected_categorical_features", [])) if self._fleet_meta else [],
            "metrics": self._fleet_meta.get("evaluation_metrics", {}) if self._fleet_meta else {},
            "metadata_path": self.fleet_readiness_meta_path
        })
        
        return registry

    def _get_fallback_soh(self, cycle_number: int, temperature_c: float) -> float:
        """Math-based fallback for SOH."""
        soh = max(0.0, 1.0 - (cycle_number * 0.0012) - (0.015 if temperature_c > 35.0 else 0.0))
        return float(soh)

    def _get_fallback_rul(self, cycle_number: int) -> int:
        """Math-based fallback for RUL."""
        return max(0, int(150 - cycle_number))

    def _get_fallback_readiness(self, payload_dict: Dict[str, Any]) -> float:
        """Math-based fallback for Fleet Readiness."""
        age = payload_dict.get("vehicle_age_years", 5)
        fuel = payload_dict.get("fuel_consumption", 12.0)
        health = payload_dict.get("health_score", 85.0)
        
        age_factor  = max(0.0, 1.0 - age / 20.0)
        fuel_factor = max(0.0, 1.0 - fuel / 25.0)
        score       = age_factor * 0.4 + fuel_factor * 0.3 + (health / 100.0) * 0.3
        return float(np.clip(score, 0.0, 1.0))

model_service = ModelService()
