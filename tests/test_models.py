import unittest
from app.services.models_service import model_service

class TestModelService(unittest.TestCase):
    """
    Unit tests for ModelService loading and predictions.
    """
    def setUp(self):
        # Keep track of original models and methods to restore them after each test
        self.original_soh = model_service._soh_model
        self.original_rul = model_service._rul_model
        self.original_fleet = model_service._fleet_model
        self.original_load_models = model_service.load_models

    def tearDown(self):
        # Restore models and methods
        model_service._soh_model = self.original_soh
        model_service._rul_model = self.original_rul
        model_service._fleet_model = self.original_fleet
        model_service.load_models = self.original_load_models

    def test_fallback_predictions_healthy(self):
        """
        Verify mathematical calculations for healthy cell telemetry.
        """
        # Force fallback by setting models to None and mocking load_models to do nothing
        model_service._soh_model = None
        model_service._rul_model = None
        model_service.load_models = lambda: None
        
        soh, rul, zone = model_service.predict_soh_and_rul(
            cycle_number=10,
            voltage_v=3.8,
            temperature_c=25.0,
            capacity_ah=1.85,
            voltage_sag_v=0.0,
            degradation_rate=0.0
        )
        self.assertTrue(soh > 0.9)
        self.assertEqual(rul, 140)
        self.assertEqual(zone, "Healthy")

    def test_fallback_predictions_degraded(self):
        """
        Verify mathematical calculations return degraded zones for high cycle counts.
        """
        # Force fallback by setting models to None and mocking load_models to do nothing
        model_service._soh_model = None
        model_service._rul_model = None
        model_service.load_models = lambda: None
        
        soh, rul, zone = model_service.predict_soh_and_rul(
            cycle_number=300,
            voltage_v=3.2,
            temperature_c=40.0,
            capacity_ah=1.2,
            voltage_sag_v=-0.05,
            degradation_rate=-0.015
        )
        self.assertTrue(soh < 0.7)
        self.assertEqual(rul, 0)
        self.assertEqual(zone, "Critical Failure Danger")
        
if __name__ == "__main__":
    unittest.main()
