"""
VoltIQ – tests/test_data.py
============================
Unit tests for the full Data Engineering & Data Wrangling layer (Page 2).

Tests cover:
    TestDataLoader         — DataLoader file resolution and load_all()
    TestDataValidation     — DataValidator schema and null checks
    TestDataCleaning       — DataCleaner in-memory transformations
    TestFeatureEngineering — FeatureEngineer derived column creation
    TestDataProfiling      — DataProfiler quality score computation
    TestRelationships      — RelationshipMapper registry and join availability
"""

import os
import unittest
from unittest.mock import patch

import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Helpers — build minimal test DataFrames
# ---------------------------------------------------------------------------

def _battery_df(n: int = 10) -> pd.DataFrame:
    return pd.DataFrame({
        "Battery_ID":                  [f"B{i}" for i in range(n)],
        "Cycle_Number":                list(range(n)),
        "Voltage_V":                   [3.7 + 0.01 * i for i in range(n)],
        "Temperature_C":               [25.0] * n,
        "Capacity_Ah":                 [100.0 - i for i in range(n)],
        "State_of_Health":             [1.0 - 0.01 * i for i in range(n)],
        "Remaining_Useful_Life_Cycles": [200 - 5 * i for i in range(n)],
        "Capacity_Fade_Pct":           [0.01 * i for i in range(n)],
        "Voltage_Sag_V":               [0.05] * n,
        "Degradation_Rate":            [-0.002] * n,
        "Health_Zone":                 ["Healthy"] * n,
        "Temperature_Zone":            ["Moderate"] * n,
        "Is_End_of_Life":              [0] * n,
    })


def _fleet_df(n: int = 10) -> pd.DataFrame:
    return pd.DataFrame({
        "Vehicle_ID":          [f"V{i}" for i in range(n)],
        "Make_and_Model":      ["Tata Ace"] * n,
        "Vehicle_Type":        ["Light Truck"] * n,
        "Year_of_Manufacture": [2018] * n,
        "Vehicle_Age_Years":   list(range(1, n + 1)),
        "Usage_Hours":         [1000.0 + 100 * i for i in range(n)],
        "Load_Capacity":       [5000.0] * n,
        "Actual_Load":         [3000.0] * n,
        "Fuel_Consumption":    [8.0 + 0.5 * i for i in range(n)],
        "Health_Score":        [90.0 - i for i in range(n)],
        "EV_Readiness_Score":  [0.9 - 0.05 * i for i in range(n)],
    })


def _carbon_df(n: int = 10) -> pd.DataFrame:
    return pd.DataFrame({
        "Vehicle_ID":               [f"V{i}" for i in range(n)],
        "Vehicle_Make_Model":       ["Tata Ace"] * n,
        "Vehicle_Type":             ["Light Truck"] * n,
        "Vehicle_Age_Years":        list(range(1, n + 1)),
        "Fuel_Consumption_L100km":  [8.0] * n,
        "Annual_CO2_Emissions_kg":  [10000.0 + 500 * i for i in range(n)],
        "EV_Readiness_Score":       [0.8 - 0.05 * i for i in range(n)],
        "Route_Type":               ["Urban"] * n,
        "Total_Usage_Hours":        [2000.0] * n,
        "Annual_Maintenance_Cost_USD": [3000.0] * n,
        "Vehicle_Health_Score":     [85.0] * n,
        "EV_Scenario_CO2_kg":       [5000.0] * n,
        "Annual_CO2_Saving_kg":     [5000.0 + 500 * i for i in range(n)],
        "Net_Zero_Progress_Pct":    [30.0 + 5 * i for i in range(n)],
    })


def _charging_sessions_df(n: int = 5) -> pd.DataFrame:
    return pd.DataFrame({
        "User_ID":                 [f"U{i}" for i in range(n)],
        "Vehicle_Model":           ["BYD Atto"] * n,
        "Battery_Capacity_kWh":    [75.0] * n,
        "Charging_Station_ID":     [f"CS{i}" for i in range(n)],
        "Session_Start_Time":      ["2024-01-01 08:00:00"] * n,
        "Session_End_Time":        ["2024-01-01 10:00:00"] * n,
        "Energy_Consumed_kWh":     [30.0 + i for i in range(n)],
        "Charging_Duration_hrs":   [2.0] * n,
        "Charging_Rate_kW":        [50.0] * n,
        "Charging_Cost_USD":       [6.0 + i for i in range(n)],
        "SOC_Start_Pct":           [20.0] * n,
        "SOC_End_Pct":             [80.0] * n,
        "Temperature_C":           [22.0] * n,
        "Charger_Type":            ["DC Fast Charger"] * n,
    })


def _weather_df(n: int = 8) -> pd.DataFrame:
    return pd.DataFrame({
        "Date":              ["2024-01-15"] * n,
        "Year":              [2024] * n,
        "Month":             [1] * n,
        "Season":            ["Winter"] * n,
        "City":              ["Mumbai"] * n,
        "Temperature_C":     [20.0 + i for i in range(n)],
        "Humidity_Pct":      [60.0] * n,
        "Precipitation_mm":  [0.0] * n,
        "Wind_Speed_kmh":    [15.0 + i for i in range(n)],
        "Weather_Condition": ["Clear"] * n,
    })


# ---------------------------------------------------------------------------
# Test suite
# ---------------------------------------------------------------------------

class TestDataLoader(unittest.TestCase):
    """DataLoader file resolution and load_all tests."""

    def test_settings_loaded(self):
        """Config settings are loadable."""
        from app.config import settings
        self.assertEqual(settings.app_name, "VoltIQ")
        self.assertIsNotNone(settings.database_url)

    def test_datasets_directory_exists(self):
        """datasets_dir resolves to an existing directory."""
        from app.utils.data_loader import data_loader
        self.assertTrue(
            os.path.isdir(data_loader.datasets_dir),
            f"datasets_dir not found: {data_loader.datasets_dir}",
        )

    def test_all_subdirectories_exist(self):
        """All required dataset subdirectories are present."""
        from app.utils.data_loader import data_loader
        for sub in ["battery", "carbon", "charging", "fleet", "reference", "weather"]:
            sub_path = os.path.join(data_loader.datasets_dir, sub)
            self.assertTrue(
                os.path.isdir(sub_path),
                f"Missing subdirectory: {sub}",
            )

    def test_verify_dataset_structure_returns_dict(self):
        """verify_dataset_structure() returns dict with expected keys."""
        from app.utils.data_loader import DataLoader, DATASET_REGISTRY
        loader = DataLoader()
        status = loader.verify_dataset_structure()
        self.assertIsInstance(status, dict)
        for key in DATASET_REGISTRY:
            self.assertIn(key, status)

    def test_load_all_returns_eight_dataframes(self):
        """load_all() returns all 8 registered datasets."""
        from app.utils.data_loader import data_loader
        dfs = data_loader.load_all()
        self.assertEqual(len(dfs), 8, f"Expected 8 datasets, got {len(dfs)}")
        for key, df in dfs.items():
            self.assertIsInstance(df, pd.DataFrame, f"{key} is not a DataFrame")
            self.assertGreater(len(df), 0, f"{key} is empty")

    def test_load_single_key(self):
        """load(key) loads a single dataset correctly."""
        from app.utils.data_loader import data_loader
        df = data_loader.load("battery")
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(len(df), 0)

    def test_load_invalid_key_raises(self):
        """load() raises KeyError for unknown keys."""
        from app.utils.data_loader import data_loader
        with self.assertRaises(KeyError):
            data_loader.load("nonexistent_dataset")

    def test_get_file_path_returns_string(self):
        """get_file_path() returns an absolute path string."""
        from app.utils.data_loader import data_loader
        path = data_loader.get_file_path("battery")
        self.assertIsInstance(path, str)
        self.assertTrue(os.path.isabs(path))


class TestDataValidation(unittest.TestCase):
    """DataValidator schema and constraint tests using synthetic data."""

    def setUp(self):
        self._battery_df = _battery_df()

    def test_validate_battery_passes_clean_data(self):
        """Validator passes for a well-formed battery DataFrame."""
        from app.utils.validation import DataValidator
        validator = DataValidator(dataframes={"battery": self._battery_df})
        report = validator.validate("battery", self._battery_df)
        self.assertEqual(report["errors"], [])

    def test_validate_detects_missing_column(self):
        """Validator raises an error when required columns are missing."""
        from app.utils.validation import DataValidator
        bad_df = self._battery_df.drop(columns=["State_of_Health"])
        validator = DataValidator(dataframes={"battery": bad_df})
        report = validator.validate("battery", bad_df)
        self.assertIn("State_of_Health", str(report["errors"]))
        self.assertFalse(report["passed"])

    def test_validate_detects_duplicates(self):
        """Validator counts duplicate rows correctly."""
        from app.utils.validation import DataValidator
        dup_df = pd.concat([self._battery_df, self._battery_df], ignore_index=True)
        validator = DataValidator(dataframes={"battery": dup_df})
        report = validator.validate("battery", dup_df)
        self.assertEqual(report["duplicate_rows"], len(self._battery_df))

    def test_validate_detects_nulls(self):
        """Validator identifies null values in numeric columns."""
        from app.utils.validation import DataValidator
        null_df = self._battery_df.copy()
        null_df.loc[0, "Voltage_V"] = None
        validator = DataValidator(dataframes={"battery": null_df})
        report = validator.validate("battery", null_df)
        self.assertIn("Voltage_V", report["null_summary"])

    def test_validate_all_real_datasets(self):
        """All 8 real datasets pass validation."""
        from app.utils.data_loader import data_loader
        from app.utils.validation import DataValidator
        dfs = data_loader.load_all()
        validator = DataValidator(dataframes=dfs)
        reports = validator.validate_all()
        for key, report in reports.items():
            self.assertEqual(
                report["errors"], [],
                f"Dataset '{key}' has validation errors: {report['errors']}",
            )


class TestDataCleaning(unittest.TestCase):
    """DataCleaner in-memory transformation tests."""

    def test_clean_battery_adds_outlier_columns(self):
        """clean_battery() adds _outlier flag columns."""
        from app.utils.cleaning import DataCleaner
        df = _battery_df(30)
        cleaner = DataCleaner()
        result = cleaner.clean_battery(df)
        self.assertIn("Voltage_V_outlier", result.columns)

    def test_clean_removes_duplicates(self):
        """DataCleaner removes duplicate rows."""
        from app.utils.cleaning import DataCleaner, drop_full_duplicates
        df = pd.concat([_fleet_df(5), _fleet_df(5)], ignore_index=True)
        clean = drop_full_duplicates(df, "test")
        self.assertEqual(len(clean), 5)

    def test_fill_numeric_nulls_median(self):
        """fill_numeric_nulls imputes missing values with median."""
        from app.utils.cleaning import fill_numeric_nulls
        df = pd.DataFrame({"A": [1.0, 2.0, None, 4.0]})
        result = fill_numeric_nulls(df, ["A"], "median")
        self.assertAlmostEqual(result["A"].iloc[2], 2.0, places=1)
        self.assertEqual(result["A"].isnull().sum(), 0)

    def test_flag_iqr_outliers(self):
        """flag_iqr_outliers adds boolean flag column."""
        from app.utils.cleaning import flag_iqr_outliers
        df = pd.DataFrame({"X": [1, 2, 3, 4, 1000]})
        result = flag_iqr_outliers(df, ["X"])
        self.assertIn("X_outlier", result.columns)
        self.assertTrue(result["X_outlier"].iloc[-1])

    def test_clean_never_modifies_original(self):
        """Cleaning operations return new DataFrames (original unchanged)."""
        from app.utils.cleaning import DataCleaner
        original = _battery_df()
        col_count_before = len(original.columns)
        cleaner = DataCleaner()
        _ = cleaner.clean_battery(original)
        self.assertEqual(len(original.columns), col_count_before)

    def test_coordinate_validation(self):
        """flag_invalid_coordinates marks out-of-bounds rows."""
        from app.utils.cleaning import flag_invalid_coordinates
        df = pd.DataFrame({
            "Latitude":  [37.77, 200.0, -91.0],
            "Longitude": [-122.41, 50.0, 180.0],
        })
        result = flag_invalid_coordinates(df)
        self.assertTrue(result["invalid_coordinates"].iloc[1])
        self.assertFalse(result["invalid_coordinates"].iloc[0])

    def test_clean_all_returns_all_keys(self):
        """clean_all() returns every input key."""
        from app.utils.data_loader import data_loader
        from app.utils.cleaning import DataCleaner
        dfs = data_loader.load_all()
        cleaner = DataCleaner()
        clean = cleaner.clean_all(dfs)
        for key in dfs:
            self.assertIn(key, clean)


class TestFeatureEngineering(unittest.TestCase):
    """FeatureEngineer derived column tests."""

    def test_battery_soh_category_created(self):
        """engineer_battery() creates SOH_Category column."""
        from app.utils.feature_engineering import FeatureEngineer
        df = _battery_df(20)
        fe = FeatureEngineer()
        result = fe.engineer_battery(df)
        self.assertIn("SOH_Category", result.columns)
        self.assertFalse(result["SOH_Category"].isnull().all())

    def test_battery_age_indicator_created(self):
        """engineer_battery() creates Battery_Age_Indicator column."""
        from app.utils.feature_engineering import FeatureEngineer
        df = _battery_df(20)
        fe = FeatureEngineer()
        result = fe.engineer_battery(df)
        self.assertIn("Battery_Age_Indicator", result.columns)

    def test_fleet_readiness_label_created(self):
        """engineer_fleet_readiness() creates Readiness_Label column."""
        from app.utils.feature_engineering import FeatureEngineer
        df = _fleet_df(10)
        fe = FeatureEngineer()
        result = fe.engineer_fleet_readiness(df)
        self.assertIn("Readiness_Label", result.columns)

    def test_fleet_vehicle_age_category(self):
        """Vehicle_Age_Category bins correctly for different ages."""
        from app.utils.feature_engineering import FeatureEngineer
        df = _fleet_df(4)
        df["Vehicle_Age_Years"] = [1, 5, 10, 20]
        fe = FeatureEngineer()
        result = fe.engineer_fleet_readiness(df)
        cats = result["Vehicle_Age_Category"].tolist()
        self.assertIn("New (0-3 yrs)", cats[0])
        self.assertIn("Old (15+ yrs)", cats[3])

    def test_carbon_saving_potential_non_negative(self):
        """Carbon_Saving_Potential_kg must be >= 0."""
        from app.utils.feature_engineering import FeatureEngineer
        df = _carbon_df(10)
        fe = FeatureEngineer()
        result = fe.engineer_carbon(df)
        self.assertIn("Carbon_Saving_Potential_kg", result.columns)
        self.assertTrue((result["Carbon_Saving_Potential_kg"] >= 0).all())

    def test_ev_priority_score_range(self):
        """EV_Priority_Score is normalised to [0, 1]."""
        from app.utils.feature_engineering import FeatureEngineer
        df = _carbon_df(20)
        fe = FeatureEngineer()
        result = fe.engineer_carbon(df)
        score = result["EV_Priority_Score"]
        self.assertTrue((score >= 0).all())
        self.assertTrue((score <= 1).all())

    def test_weather_climate_severity_range(self):
        """Climate_Severity_Index values fall between 0 and 1."""
        from app.utils.feature_engineering import FeatureEngineer
        df = _weather_df(10)
        fe = FeatureEngineer()
        result = fe.engineer_weather(df)
        self.assertIn("Climate_Severity_Index", result.columns)
        csi = result["Climate_Severity_Index"].dropna()
        self.assertTrue((csi >= 0).all())
        self.assertTrue((csi <= 1).all())

    def test_charging_soc_delta_computed(self):
        """SOC_Delta_Computed = SOC_End - SOC_Start."""
        from app.utils.feature_engineering import FeatureEngineer
        df = _charging_sessions_df(5)
        fe = FeatureEngineer()
        result = fe.engineer_charging_sessions(df)
        expected = df["SOC_End_Pct"].iloc[0] - df["SOC_Start_Pct"].iloc[0]
        self.assertAlmostEqual(result["SOC_Delta_Computed"].iloc[0], expected)

    def test_fast_charging_flag_dc_charger(self):
        """Fast_Charging_Flag is True for DC Fast Charger sessions."""
        from app.utils.feature_engineering import FeatureEngineer
        df = _charging_sessions_df(3)
        fe = FeatureEngineer()
        result = fe.engineer_charging_sessions(df)
        self.assertIn("Fast_Charging_Flag", result.columns)
        self.assertTrue(result["Fast_Charging_Flag"].all())

    def test_engineer_all_returns_all_keys(self):
        """engineer_all() returns entries for all input keys."""
        from app.utils.data_loader import data_loader
        from app.utils.cleaning import DataCleaner
        from app.utils.feature_engineering import FeatureEngineer
        dfs = data_loader.load_all()
        clean = DataCleaner().clean_all(dfs)
        eng = FeatureEngineer().engineer_all(clean)
        for key in clean:
            self.assertIn(key, eng)


class TestDataProfiling(unittest.TestCase):
    """DataProfiler quality score and column statistics tests."""

    def test_profile_returns_required_keys(self):
        """profile() dict contains all expected top-level keys."""
        from app.utils.profiling import DataProfiler
        profiler = DataProfiler()
        result = profiler.profile("battery", _battery_df())
        for key in ["rows", "cols", "memory_mb", "total_nulls",
                    "total_dupes", "columns", "quality_score"]:
            self.assertIn(key, result)

    def test_profile_quality_score_range(self):
        """Quality score is between 0 and 100."""
        from app.utils.profiling import DataProfiler
        profiler = DataProfiler()
        result = profiler.profile("battery", _battery_df())
        self.assertGreaterEqual(result["quality_score"], 0)
        self.assertLessEqual(result["quality_score"], 100)

    def test_profile_all_real_datasets(self):
        """Profiling runs on all 8 real datasets without errors."""
        from app.utils.data_loader import data_loader
        from app.utils.profiling import DataProfiler
        dfs = data_loader.load_all()
        profiler = DataProfiler()
        profiles = profiler.profile_all(dfs)
        self.assertEqual(len(profiles), len(dfs))

    def test_to_dataframe_returns_dataframe(self):
        """to_dataframe() returns a non-empty DataFrame."""
        from app.utils.profiling import DataProfiler
        profiler = DataProfiler()
        profiles = profiler.profile_all({"battery": _battery_df()})
        df = profiler.to_dataframe(profiles)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(len(df), 0)

    def test_numeric_column_has_statistics(self):
        """Numeric columns include mean, std, min, max in profile."""
        from app.utils.profiling import DataProfiler
        profiler = DataProfiler()
        result = profiler.profile("battery", _battery_df())
        numeric_info = result["columns"]["Voltage_V"]
        for stat in ["mean", "std", "min", "max", "median"]:
            self.assertIn(stat, numeric_info)


class TestRelationships(unittest.TestCase):
    """RelationshipMapper registry and join availability tests."""

    def test_registry_contains_six_relationships(self):
        """RELATIONSHIP_REGISTRY has 6 entries."""
        from app.utils.relationships import RELATIONSHIP_REGISTRY
        self.assertEqual(len(RELATIONSHIP_REGISTRY), 6)

    def test_all_joins_available_with_real_data(self):
        """All 6 joins are marked available when real datasets are loaded."""
        from app.utils.data_loader import data_loader
        from app.utils.relationships import RelationshipMapper
        dfs = data_loader.load_all()
        mapper = RelationshipMapper(dfs)
        avail = mapper.availability_check()
        self.assertEqual(sum(avail.values()), len(avail))

    def test_join_fleet_carbon_returns_dataframe(self):
        """join_fleet_carbon() returns a non-empty merged DataFrame."""
        from app.utils.data_loader import data_loader
        from app.utils.relationships import RelationshipMapper
        dfs = data_loader.load_all()
        mapper = RelationshipMapper(dfs)
        result = mapper.join_fleet_carbon()
        self.assertIsNotNone(result)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertGreater(len(result), 0)

    def test_join_returns_none_for_missing_dataset(self):
        """join returns None when a required dataset is absent."""
        from app.utils.relationships import RelationshipMapper
        mapper = RelationshipMapper({"battery": _battery_df()})
        result = mapper.join_fleet_carbon()
        self.assertIsNone(result)

    def test_describe_all_returns_list(self):
        """describe_all() returns a list of dicts."""
        from app.utils.relationships import RelationshipMapper
        mapper = RelationshipMapper({})
        result = mapper.describe_all()
        self.assertIsInstance(result, list)
        for item in result:
            self.assertIn("name", item)
            self.assertIn("cardinality", item)


if __name__ == "__main__":
    unittest.main(verbosity=2)
