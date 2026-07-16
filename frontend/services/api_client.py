# Re-export helpers from the new ui component module for backward compatibility
from frontend.components.ui import render_error_card, render_empty_state  # noqa: F401

import requests
import time
import streamlit as st
from typing import Optional, Dict, Any, List

API_BASE_URL = "http://127.0.0.1:8000/api/v1"
MAX_RETRIES = 3
TIMEOUT_SECONDS = 3.0


class APIClient:
    """
    Dedicated API client for VoltIQ frontend to communicate with the FastAPI backend.
    Handles timeouts, retries, and catches exceptions to return friendly statuses.
    """
    def __init__(self, base_url: str = API_BASE_URL):
        self.base_url = base_url

    def _request(self, method: str, endpoint: str, json_data: Optional[Dict[str, Any]] = None, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generic request wrapper that implements timeout and retry capability."""
        url = f"{self.base_url}/{endpoint.lstrip('/')}"

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                if method.upper() == "GET":
                    res = requests.get(url, params=params, timeout=TIMEOUT_SECONDS)
                elif method.upper() == "POST":
                    res = requests.post(url, json=json_data, timeout=TIMEOUT_SECONDS)
                else:
                    raise ValueError(f"Unsupported HTTP method: {method}")

                if res.status_code == 200:
                    return {"success": True, "data": res.json(), "status_code": 200}

                try:
                    error_detail = res.json().get("detail", res.text)
                except Exception:
                    error_detail = res.text
                return {"success": False, "error": error_detail, "status_code": res.status_code}

            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                if attempt == MAX_RETRIES:
                    return {
                        "success": False,
                        "error": f"Failed to connect to VoltIQ backend after {MAX_RETRIES} attempts. Connection timeout or server is offline.",
                        "status_code": 503,
                    }
                time.sleep(0.5 * attempt)
            except Exception as e:
                return {"success": False, "error": str(e), "status_code": 500}

        return {"success": False, "error": "Unknown request error occurred", "status_code": 500}

    # --- System Diagnostics ---
    def get_system_health(self) -> Dict[str, Any]:
        return self._request("GET", "system/health")

    def get_system_info(self) -> Dict[str, Any]:
        return self._request("GET", "system/info")

    # --- Fleet Electrification Readiness ---
    def get_fleet_summary(self) -> Dict[str, Any]:
        return self._request("GET", "fleet/summary")

    def get_vehicle_readiness(self, vehicle_id: str) -> Dict[str, Any]:
        return self._request("GET", f"fleet/vehicle/{vehicle_id}")

    def predict_fleet(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self._request("POST", "fleet/predict", json_data=payload)

    # --- Battery APM ---
    def predict_battery(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self._request("POST", "battery/predict", json_data=payload)

    def predict_rul(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self._request("POST", "battery/rul", json_data=payload)

    def get_battery_health_by_id(self, battery_id: str) -> Dict[str, Any]:
        return self._request("GET", f"battery/health/{battery_id}")

    # --- Carbon Tracker ---
    def get_carbon_metrics(self) -> Dict[str, Any]:
        return self._request("GET", "carbon/metrics")

    def analyze_carbon(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self._request("POST", "carbon/analysis", json_data=payload)

    # --- Chat Advisor ---
    def chat_query(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self._request("POST", "chat/query", json_data=payload)

    # --- Model Registry ---
    def get_model_registry(self) -> Dict[str, Any]:
        return self._request("GET", "models/registry")

    def get_model_details(self, model_id: str) -> Dict[str, Any]:
        return self._request("GET", f"models/{model_id}")


# Singleton client instance
api_client = APIClient()
