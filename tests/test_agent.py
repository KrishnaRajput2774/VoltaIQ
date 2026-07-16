import unittest
import json
from app.agent.tools import (
    fleet_advisor,
    get_fleet_summary_metrics,
    search_vehicle_feasibility,
    analyze_battery_health,
    predict_custom_battery_telemetry,
    get_carbon_metrics_summary,
    calculate_vehicle_carbon_savings,
    get_poor_charging_routes
)

class TestFleetAdvisorAgent(unittest.TestCase):
    """
    Unit tests for the AI Fleet Advisor LangChain tools and query routing.
    """

    def test_tool_fleet_summary(self):
        """Verify get_fleet_summary_metrics tool returns overall fleet ready metrics."""
        res_str = get_fleet_summary_metrics.invoke({})
        res = json.loads(res_str)
        self.assertIn("total_vehicles", res)
        self.assertIn("readiness_percentage", res)
        self.assertEqual(res["data_source"], "Fleet Dataset")

    def test_tool_vehicle_feasibility_valid(self):
        """Verify search_vehicle_feasibility tool returns specs for existing vehicle ID."""
        res_str = search_vehicle_feasibility.invoke({"vehicle_id": "1"})
        res = json.loads(res_str)
        self.assertEqual(res["vehicle_id"], "1")
        self.assertIn("ev_readiness_score", res)
        self.assertEqual(res["data_source"], "Fleet Dataset")
        self.assertEqual(res["model_name"], "LinearRegression Fleet Model")

    def test_tool_vehicle_feasibility_invalid(self):
        """Verify search_vehicle_feasibility tool returns clean error for bad ID."""
        res_str = search_vehicle_feasibility.invoke({"vehicle_id": "99999999"})
        res = json.loads(res_str)
        self.assertIn("error", res)

    def test_tool_battery_health_valid(self):
        """Verify analyze_battery_health tool retrieves telemetry and predicts for valid ID."""
        res_str = analyze_battery_health.invoke({"battery_id": "B0005"})
        res = json.loads(res_str)
        self.assertEqual(res["battery_id"], "B0005")
        self.assertIn("predicted_soh", res)
        self.assertIn("predicted_rul_cycles", res)
        self.assertEqual(res["data_source"], "Battery Dataset")

    def test_tool_battery_health_invalid(self):
        """Verify analyze_battery_health tool returns error for bad ID."""
        res_str = analyze_battery_health.invoke({"battery_id": "B999999"})
        res = json.loads(res_str)
        self.assertIn("error", res)

    def test_tool_predict_custom_battery(self):
        """Verify predict_custom_battery_telemetry tool runs SOH/RUL predictions."""
        payload = {
            "cycle_number": 50,
            "voltage_v": 3.6,
            "temperature_c": 28.0,
            "capacity_ah": 1.7,
            "voltage_sag_v": 0.02,
            "degradation_rate": -0.002
        }
        res_str = predict_custom_battery_telemetry.invoke(payload)
        res = json.loads(res_str)
        self.assertIn("predicted_soh", res)
        self.assertIn("predicted_rul_cycles", res)

    def test_tool_carbon_summary(self):
        """Verify get_carbon_metrics_summary tool returns fleet footprint sums."""
        res_str = get_carbon_metrics_summary.invoke({})
        res = json.loads(res_str)
        self.assertIn("baseline_co2_kg", res)
        self.assertIn("ev_scenario_co2_kg", res)
        self.assertEqual(res["data_source"], "Carbon Dataset")

    def test_tool_vehicle_carbon(self):
        """Verify calculate_vehicle_carbon_savings tool computes offset values."""
        payload = {
            "vehicle_id": "1",
            "annual_distance_km": 15000.0,
            "fuel_type": "Diesel"
        }
        res_str = calculate_vehicle_carbon_savings.invoke(payload)
        res = json.loads(res_str)
        self.assertEqual(res["vehicle_id"], "1")
        self.assertIn("baseline_annual_co2_kg", res)
        self.assertIn("ev_projected_co2_kg", res)

    def test_tool_poor_charging_routes(self):
        """Verify get_poor_charging_routes tool finds route coordinates with no coverage."""
        res_str = get_poor_charging_routes.invoke({})
        res = json.loads(res_str)
        self.assertIn("total_unmapped_routes", res)
        self.assertIn("sample_unmapped_routes", res)
        self.assertEqual(res["data_source"], "Fleet Dataset, Charging Dataset")

    def test_agent_fallback_fleet_readiness(self):
        """Verify fallback response for fleet readiness keywords."""
        result = fleet_advisor.run_query("Which vehicles should I electrify first?")
        self.assertIn("Fleet Dataset", result["sources"])
        self.assertIn("LinearRegression Fleet Model", result["sources"])
        self.assertIn("readiness", result["response"].lower())

    def test_agent_fallback_battery(self):
        """Verify fallback response for battery SOH keywords."""
        result = fleet_advisor.run_query("Show batteries with the lowest SOH.")
        self.assertIn("Battery Dataset", result["sources"])
        self.assertIn("GradientBoosting Battery Model", result["sources"])
        self.assertIn("soh", result["response"].lower())

    def test_agent_fallback_carbon(self):
        """Verify fallback response for carbon emissions keywords."""
        result = fleet_advisor.run_query("Compare diesel vs EV carbon emissions.")
        self.assertIn("Carbon Dataset", result["sources"])
        self.assertIn("co2", result["response"].lower())

    def test_agent_fallback_charging(self):
        """Verify fallback response for routes charging accessibility keywords."""
        result = fleet_advisor.run_query("Which routes have poor charging accessibility?")
        self.assertIn("Fleet Dataset", result["sources"])
        self.assertIn("Charging Dataset", result["sources"])
        self.assertIn("charging", result["response"].lower())

    def test_agent_fallback_summary(self):
        """Verify fallback response for overall summary queries."""
        result = fleet_advisor.run_query("Summarize the overall fleet readiness.")
        self.assertIn("Fleet Dataset", result["sources"])
        self.assertIn("LinearRegression Fleet Model", result["sources"])
        self.assertIn("vehicles", result["response"].lower())

if __name__ == "__main__":
    unittest.main()
