import unittest
from fastapi.testclient import TestClient
from app.main import app

class TestVoltIQAPI(unittest.TestCase):
    """
    Comprehensive API integration tests for all VoltIQ v1 REST endpoints.
    """
    def setUp(self):
        self.client = TestClient(app)

    # ------------------------------------------------------------------
    # 1. Root & Health Check Endpoints
    # ------------------------------------------------------------------
    def test_read_root(self):
        """Verify root endpoint returns status online and application name."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        json_data = response.json()
        self.assertEqual(json_data["app_name"], "VoltIQ")
        self.assertEqual(json_data["status"], "online")

    def test_health_check(self):
        """Verify health check endpoint returns 200 and healthy status."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        json_data = response.json()
        self.assertEqual(json_data["status"], "healthy")

    # ------------------------------------------------------------------
    # 2. System Diagnostics Endpoints
    # ------------------------------------------------------------------
    def test_system_health(self):
        """Verify system health endpoint returns loaded models and dataset stats."""
        response = self.client.get("/api/v1/system/health")
        self.assertEqual(response.status_code, 200)
        json_data = response.json()
        self.assertEqual(json_data["status"], "healthy")
        self.assertIn("uptime_seconds", json_data)
        self.assertIn("model_load_status", json_data)
        self.assertIn("dataset_status", json_data)
        
        # Ensure SOH model, RUL model, and Fleet readiness status are present
        load_status = json_data["model_load_status"]
        self.assertIn("battery_soh_model", load_status)
        self.assertIn("battery_rul_model", load_status)
        self.assertIn("fleet_readiness_model", load_status)

    def test_system_info(self):
        """Verify system info returns software configurations and metadata."""
        response = self.client.get("/api/v1/system/info")
        self.assertEqual(response.status_code, 200)
        json_data = response.json()
        self.assertEqual(json_data["app_name"], "VoltIQ")
        self.assertEqual(json_data["version"], "2.0.0")
        self.assertIn("python_version", json_data)
        self.assertIn("platform", json_data)

    # ------------------------------------------------------------------
    # 3. Model Registry Endpoints
    # ------------------------------------------------------------------
    def test_model_registry(self):
        """Verify model registry returns all 3 models with their properties."""
        response = self.client.get("/api/v1/models/registry")
        self.assertEqual(response.status_code, 200)
        json_data = response.json()
        self.assertIn("models", json_data)
        models = json_data["models"]
        self.assertEqual(len(models), 3)
        
        model_ids = [m["model_id"] for m in models]
        self.assertIn("battery_soh_model", model_ids)
        self.assertIn("battery_rul_model", model_ids)
        self.assertIn("fleet_readiness_model", model_ids)

    def test_model_details_valid(self):
        """Verify detailed metadata retrieval for SOH model."""
        response = self.client.get("/api/v1/models/battery_soh_model")
        self.assertEqual(response.status_code, 200)
        json_data = response.json()
        self.assertEqual(json_data["model_id"], "battery_soh_model")
        self.assertIn("version", json_data)
        self.assertIn("algorithm", json_data)
        self.assertIn("selected_features", json_data)

    def test_model_details_invalid(self):
        """Verify requesting details for non-existent model ID returns 404."""
        response = self.client.get("/api/v1/models/invalid_model_id")
        self.assertEqual(response.status_code, 404)
        json_data = response.json()
        self.assertIn("detail", json_data)

    # ------------------------------------------------------------------
    # 4. Battery APM Endpoints
    # ------------------------------------------------------------------
    def test_battery_prediction(self):
        """Verify battery predict endpoint handles payload and returns output."""
        payload = {
            "cycle_number": 50,
            "voltage_v": 3.65,
            "temperature_c": 32.5,
            "capacity_ah": 1.75,
            "voltage_sag_v": 0.03,
            "degradation_rate": -0.002,
            "cycle_normalized": 0.30
        }
        response = self.client.post("/api/v1/battery/predict", json=payload)
        self.assertEqual(response.status_code, 200)
        json_data = response.json()
        self.assertIn("state_of_health", json_data)
        self.assertIn("remaining_useful_life_cycles", json_data)
        self.assertIn("health_zone", json_data)
        
        # Validate health zone is correct classification
        soh = json_data["state_of_health"]
        zone = json_data["health_zone"]
        if soh >= 0.85:
            self.assertEqual(zone, "Healthy")
        elif soh >= 0.70:
            self.assertEqual(zone, "Attention Required")
        else:
            self.assertEqual(zone, "Critical Failure Danger")

    def test_battery_prediction_validation_failure(self):
        """Verify battery predict rejects invalid inputs with 422."""
        invalid_payload = {
            "cycle_number": -10,  # invalid count (must be >= 1)
            "voltage_v": -1.5,    # invalid negative voltage
            "temperature_c": 25.0,
            "capacity_ah": 1.8
        }
        response = self.client.post("/api/v1/battery/predict", json=invalid_payload)
        self.assertEqual(response.status_code, 422)

    def test_battery_rul_prediction(self):
        """Verify separate battery RUL prediction endpoint handles input SOH."""
        payload = {
            "cycle_number": 80,
            "voltage_v": 3.4,
            "temperature_c": 29.0,
            "capacity_ah": 1.5,
            "voltage_sag_v": 0.04,
            "degradation_rate": -0.003,
            "state_of_health": 0.78
        }
        response = self.client.post("/api/v1/battery/rul", json=payload)
        self.assertEqual(response.status_code, 200)
        json_data = response.json()
        self.assertIn("remaining_useful_life_cycles", json_data)
        self.assertGreaterEqual(json_data["remaining_useful_life_cycles"], 0)

    def test_battery_health_by_id_valid(self):
        """Verify lookup of existing battery ID fetches telemetry and predicts."""
        response = self.client.get("/api/v1/battery/health/B0005")
        self.assertEqual(response.status_code, 200)
        json_data = response.json()
        self.assertEqual(json_data["battery_id"], "B0005")
        self.assertIn("state_of_health", json_data)
        self.assertIn("remaining_useful_life_cycles", json_data)

    def test_battery_health_by_id_invalid(self):
        """Verify lookup of non-existent battery ID returns 404."""
        response = self.client.get("/api/v1/battery/health/999999")
        self.assertEqual(response.status_code, 404)

    # ------------------------------------------------------------------
    # 5. Fleet Electrification Endpoints
    # ------------------------------------------------------------------
    def test_fleet_summary(self):
        """Verify fleet summary returns correct counts and top recommendations."""
        response = self.client.get("/api/v1/fleet/summary")
        self.assertEqual(response.status_code, 200)
        json_data = response.json()
        self.assertIn("total_vehicles", json_data)
        self.assertIn("high_readiness_count", json_data)
        self.assertIn("recommendations", json_data)
        self.assertGreater(json_data["total_vehicles"], 0)
        self.assertTrue(len(json_data["recommendations"]) > 0)

    def test_fleet_vehicle_readiness_valid(self):
        """Verify lookup of existing vehicle ID returns transition specs."""
        response = self.client.get("/api/v1/fleet/vehicle/1")
        self.assertEqual(response.status_code, 200)
        json_data = response.json()
        self.assertEqual(json_data["vehicle_id"], "1")
        self.assertIn("ev_readiness_score", json_data)
        self.assertIn("recommended_ev_replacement", json_data)

    def test_fleet_vehicle_readiness_invalid(self):
        """Verify lookup of non-existent vehicle ID returns 404."""
        response = self.client.get("/api/v1/fleet/vehicle/99999999")
        self.assertEqual(response.status_code, 404)

    def test_fleet_predict(self):
        """Verify fleet transition score predict handles vehicle features payload."""
        payload = {
            "vehicle_age_years": 5,
            "usage_hours": 3000.0,
            "fuel_consumption": 12.0,
            "health_score": 85.0,
            "load_capacity": 5000.0,
            "actual_load": 2500.0,
            "load_utilization_pct": 50.0,
            "fuel_per_hour": 1.5,
            "maintenance_cost": 1000.0,
            "days_since_last_maintenance": 30,
            "failure_history": 0,
            "anomalies_detected": 0,
            "diagnostic_trouble_code_count": 0,
            "predictive_score": 0.8,
            "pcr": 0.5,
            "uir": 0.5,
            "tpi": 0.5,
            "mbf": 300.0,
            "ads": 5.0,
            "ohi": 80.0,
            "cmes": 1000.0,
            "uer": 0.5,
            "vehicle_type": "Light Truck",
            "route_info": "Urban Delivery",
            "road_conditions": "Smooth",
            "weather_conditions": "Clear",
            "brake_condition": "Good"
        }
        response = self.client.post("/api/v1/fleet/predict", json=payload)
        self.assertEqual(response.status_code, 200)
        json_data = response.json()
        self.assertIn("ev_readiness_score", json_data)
        self.assertIn("readiness_category", json_data)
        self.assertIn("model_used", json_data)
        self.assertGreaterEqual(json_data["ev_readiness_score"], 0.0)

    # ------------------------------------------------------------------
    # 6. Carbon Tracker Endpoints
    # ------------------------------------------------------------------
    def test_carbon_metrics(self):
        """Verify carbon metrics route responds with footprint metrics."""
        response = self.client.get("/api/v1/carbon/metrics")
        self.assertEqual(response.status_code, 200)
        json_data = response.json()
        self.assertIn("baseline_co2_kg", json_data)
        self.assertIn("ev_scenario_co2_kg", json_data)
        self.assertIn("annual_savings_kg", json_data)
        self.assertIn("carbon_intensity_reduction_pct", json_data)
        self.assertIn("net_zero_progress_pct", json_data)

    def test_carbon_analysis(self):
        """Verify individual carbon savings analysis returns comparative emissions."""
        payload = {
            "vehicle_id": "VH-0013",
            "annual_distance_km": 20000.0,
            "fuel_type": "Diesel"
        }
        response = self.client.post("/api/v1/carbon/analysis", json=payload)
        self.assertEqual(response.status_code, 200)
        json_data = response.json()
        self.assertEqual(json_data["vehicle_id"], "VH-0013")
        self.assertIn("baseline_annual_co2_kg", json_data)
        self.assertIn("ev_projected_co2_kg", json_data)
        self.assertIn("net_annual_savings_kg", json_data)

    # ------------------------------------------------------------------
    # 7. Middleware Verification
    # ------------------------------------------------------------------
    def test_middleware_request_id(self):
        """Verify that X-Request-ID response header is returned."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("X-Request-ID", response.headers)

    def test_middleware_process_time(self):
        """Verify that X-Process-Time response header is returned."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("X-Process-Time", response.headers)

if __name__ == "__main__":
    unittest.main()
