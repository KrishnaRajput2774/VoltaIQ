"""
VoltIQ – Data Validation
========================
Validates every VoltIQ dataset against expected schema, data-type,
uniqueness, and value constraints.

Validation checks performed per dataset:
    1. File existence (delegated to DataLoader)
    2. Required column presence
    3. Expected data types
    4. Duplicate row detection
    5. Missing / null value analysis
    6. Invalid numeric ranges
    7. Inconsistent categorical values
    8. Unique-identifier integrity

Usage:
    from app.utils.validation import DataValidator
    validator = DataValidator()
    report = validator.validate_all()
"""

import logging
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Schema definitions — required columns & basic type expectations per dataset
# ---------------------------------------------------------------------------
SCHEMAS: Dict[str, Dict[str, Any]] = {
    "battery": {
        "required_columns": [
            "Battery_ID", "Cycle_Number", "Voltage_V", "Temperature_C",
            "Capacity_Ah", "State_of_Health", "Remaining_Useful_Life_Cycles",
            "Capacity_Fade_Pct", "Voltage_Sag_V", "Degradation_Rate",
            "Health_Zone", "Temperature_Zone", "Is_End_of_Life",
        ],
        "numeric_cols": [
            "Cycle_Number", "Voltage_V", "Temperature_C", "Capacity_Ah",
            "State_of_Health", "Remaining_Useful_Life_Cycles",
            "Capacity_Fade_Pct", "Voltage_Sag_V", "Degradation_Rate",
        ],
        "categorical_cols": ["Health_Zone", "Temperature_Zone"],
        "unique_id_col": None,  # composite key (Battery_ID + Cycle_Number)
        "numeric_ranges": {
            "State_of_Health": (0.0, 1.0),
            "Voltage_V":       (0.0, 10.0),
            "Temperature_C":   (-50.0, 100.0),
            "Capacity_Ah":     (0.0, 20.0),
        },
        "expected_categories": {
            "Health_Zone": {"Healthy", "Degrading", "Critical", "End of Life"},
            "Temperature_Zone": {"Cold", "Moderate", "Warm", "Hot"},
        },
    },

    "carbon": {
        "required_columns": [
            "Vehicle_ID", "Vehicle_Make_Model", "Vehicle_Type",
            "Vehicle_Age_Years", "Fuel_Consumption_L100km",
            "Annual_CO2_Emissions_kg", "EV_Readiness_Score",
            "Route_Type", "Total_Usage_Hours", "Annual_Maintenance_Cost_USD",
            "Vehicle_Health_Score", "EV_Scenario_CO2_kg",
            "Annual_CO2_Saving_kg", "Net_Zero_Progress_Pct",
        ],
        "numeric_cols": [
            "Vehicle_Age_Years", "Fuel_Consumption_L100km",
            "Annual_CO2_Emissions_kg", "EV_Readiness_Score",
            "Total_Usage_Hours", "Annual_Maintenance_Cost_USD",
            "Vehicle_Health_Score", "EV_Scenario_CO2_kg",
            "Annual_CO2_Saving_kg", "Net_Zero_Progress_Pct",
        ],
        "categorical_cols": ["Vehicle_Type", "Route_Type"],
        "unique_id_col": "Vehicle_ID",
        "numeric_ranges": {
            "EV_Readiness_Score":    (0.0, 1.0),
            "Net_Zero_Progress_Pct": (0.0, 100.0),
            "Vehicle_Health_Score":  (0.0, 100.0),
        },
        "expected_categories": {
            "Vehicle_Type": {
                "Refrigerated Truck", "Light Truck", "Medium Truck",
                "Heavy Truck", "Van",
            },
        },
    },

    "charging_sessions": {
        "required_columns": [
            "User_ID", "Vehicle_Model", "Battery_Capacity_kWh",
            "Charging_Station_ID", "Session_Start_Time", "Session_End_Time",
            "Energy_Consumed_kWh", "Charging_Duration_hrs",
            "Charging_Rate_kW", "Charging_Cost_USD",
            "SOC_Start_Pct", "SOC_End_Pct", "Temperature_C",
            "Charger_Type",
        ],
        "numeric_cols": [
            "Battery_Capacity_kWh", "Energy_Consumed_kWh",
            "Charging_Duration_hrs", "Charging_Rate_kW",
            "Charging_Cost_USD", "SOC_Start_Pct", "SOC_End_Pct",
            "Temperature_C",
        ],
        "categorical_cols": ["Charger_Type"],
        "unique_id_col": "User_ID",
        "numeric_ranges": {
            "SOC_Start_Pct": (0.0, 100.0),
            "SOC_End_Pct":   (0.0, 100.0),
            "Charging_Rate_kW": (0.0, 500.0),
        },
        "expected_categories": {
            "Charger_Type": {"Level 1", "Level 2", "DC Fast Charger"},
        },
    },

    "charging_stations": {
        "required_columns": [
            "Station_ID", "Station_Name", "City", "State", "Latitude",
            "Longitude", "Charger_Type", "Power_kW", "Total_Chargers",
            "Availability_Pct",
        ],
        "numeric_cols": [
            "Latitude", "Longitude", "Power_kW",
            "Total_Chargers", "Availability_Pct",
        ],
        "categorical_cols": ["Charger_Type"],
        "unique_id_col": "Station_ID",
        "numeric_ranges": {
            "Latitude":         (-90.0, 90.0),
            "Longitude":        (-180.0, 180.0),
            "Availability_Pct": (0.0, 100.0),
            "Power_kW":         (0.0, 1000.0),
        },
        "expected_categories": {},
    },

    "fleet_readiness": {
        "required_columns": [
            "Vehicle_ID", "Make_and_Model", "Vehicle_Type",
            "Year_of_Manufacture", "Vehicle_Age_Years", "Usage_Hours",
            "Load_Capacity", "Actual_Load", "Fuel_Consumption",
            "Health_Score", "EV_Readiness_Score",
        ],
        "numeric_cols": [
            "Vehicle_Age_Years", "Usage_Hours", "Load_Capacity",
            "Actual_Load", "Fuel_Consumption", "Health_Score",
            "EV_Readiness_Score",
        ],
        "categorical_cols": ["Vehicle_Type"],
        "unique_id_col": "Vehicle_ID",
        "numeric_ranges": {
            "EV_Readiness_Score": (0.0, 1.0),
            "Health_Score":       (0.0, 100.0),
        },
        "expected_categories": {},
    },

    "fleet_routes": {
        "required_columns": [
            "Vehicle_ID", "Vehicle_Type", "Route_ID", "Date",
            "Distance_km", "Average_Speed_kmh", "Travel_Time_hrs",
            "Payload_kg", "Idle_Time_min", "Estimated_Fuel_L",
        ],
        "numeric_cols": [
            "Distance_km", "Average_Speed_kmh", "Travel_Time_hrs",
            "Payload_kg", "Idle_Time_min", "Estimated_Fuel_L",
        ],
        "categorical_cols": ["Vehicle_Type"],
        "unique_id_col": None,
        "numeric_ranges": {
            "Distance_km":      (0.0, 20000.0),
            "Average_Speed_kmh": (0.0, 250.0),
        },
        "expected_categories": {},
    },

    "carbon_reference": {
        "required_columns": [
            "Manufacturer", "Vehicle_Model", "Vehicle_Class",
            "Engine_Size_L", "Fuel_Type",
            "CO2_Emissions_g_per_km", "Estimated_Annual_CO2_kg",
        ],
        "numeric_cols": [
            "Engine_Size_L", "CO2_Emissions_g_per_km",
            "Estimated_Annual_CO2_kg",
        ],
        "categorical_cols": ["Fuel_Type", "Vehicle_Class"],
        "unique_id_col": None,
        "numeric_ranges": {
            "CO2_Emissions_g_per_km": (0.0, 1500.0),
        },
        "expected_categories": {},
    },

    "weather": {
        "required_columns": [
            "Date", "Year", "Month", "Season", "City",
            "Temperature_C", "Humidity_Pct", "Precipitation_mm",
            "Wind_Speed_kmh", "Weather_Condition",
        ],
        "numeric_cols": [
            "Year", "Month", "Temperature_C", "Humidity_Pct",
            "Precipitation_mm", "Wind_Speed_kmh",
        ],
        "categorical_cols": ["Season", "Weather_Condition"],
        "unique_id_col": None,
        "numeric_ranges": {
            "Temperature_C":   (-60.0, 60.0),
            "Humidity_Pct":    (0.0, 100.0),
            "Wind_Speed_kmh":  (0.0, 400.0),
        },
        "expected_categories": {},
    },
}


# ---------------------------------------------------------------------------
# Validation result dataclass (dict-based for simplicity / JSON-serialisable)
# ---------------------------------------------------------------------------
def _empty_result(key: str) -> Dict[str, Any]:
    return {
        "dataset":            key,
        "row_count":          0,
        "col_count":          0,
        "missing_columns":    [],
        "type_issues":        {},
        "duplicate_rows":     0,
        "null_summary":       {},
        "range_violations":   {},
        "category_anomalies": {},
        "id_duplicates":      0,
        "passed":             False,
        "warnings":           [],
        "errors":             [],
    }


class DataValidator:
    """
    Validates each VoltIQ dataset against its registered schema.

    Parameters
    ----------
    dataframes : dict[str, pd.DataFrame], optional
        Pre-loaded dataframes keyed by registry key.
        If omitted, DataLoader is used to load all datasets.
    """

    def __init__(
        self,
        dataframes: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> None:
        if dataframes is not None:
            self._dfs = dataframes
        else:
            from app.utils.data_loader import data_loader
            self._dfs = data_loader.load_all()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate_all(self) -> Dict[str, Dict[str, Any]]:
        """
        Run validation on every loaded dataset.

        Returns
        -------
        dict[str, dict]
            Full validation report keyed by dataset name.
        """
        reports: Dict[str, Dict[str, Any]] = {}
        for key in SCHEMAS:
            if key not in self._dfs:
                result = _empty_result(key)
                result["errors"].append(
                    f"Dataset '{key}' was not loaded — file may be missing."
                )
                logger.error("Validation skipped for missing dataset: %s", key)
                reports[key] = result
                continue
            reports[key] = self.validate(key, self._dfs[key])
        logger.info(
            "validate_all() complete. Passed: %d / %d",
            sum(1 for r in reports.values() if r["passed"]),
            len(reports),
        )
        return reports

    def validate(self, key: str, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate a single dataframe against its schema definition.

        Parameters
        ----------
        key : str
            Registry key matching an entry in SCHEMAS.
        df : pd.DataFrame
            The dataframe to validate.

        Returns
        -------
        dict
            Detailed validation result.
        """
        schema = SCHEMAS.get(key, {})
        result = _empty_result(key)
        result["row_count"] = len(df)
        result["col_count"] = len(df.columns)
        logger.info("Validating dataset: %s (%d rows)", key, len(df))

        self._check_required_columns(df, schema, result)
        self._check_data_types(df, schema, result)
        self._check_duplicates(df, result)
        self._check_nulls(df, schema, result)
        self._check_numeric_ranges(df, schema, result)
        self._check_categories(df, schema, result)
        self._check_unique_id(df, schema, result)

        result["passed"] = len(result["errors"]) == 0
        level = logging.INFO if result["passed"] else logging.WARNING
        logger.log(
            level,
            "Validation %s for '%s' — errors=%d warnings=%d",
            "PASSED" if result["passed"] else "FAILED",
            key,
            len(result["errors"]),
            len(result["warnings"]),
        )
        return result

    # ------------------------------------------------------------------
    # Internal check methods
    # ------------------------------------------------------------------

    def _check_required_columns(
        self,
        df: pd.DataFrame,
        schema: Dict[str, Any],
        result: Dict[str, Any],
    ) -> None:
        required: List[str] = schema.get("required_columns", [])
        missing = [c for c in required if c not in df.columns]
        result["missing_columns"] = missing
        if missing:
            msg = f"Missing required columns: {missing}"
            result["errors"].append(msg)
            logger.error(msg)

    def _check_data_types(
        self,
        df: pd.DataFrame,
        schema: Dict[str, Any],
        result: Dict[str, Any],
    ) -> None:
        issues: Dict[str, str] = {}
        for col in schema.get("numeric_cols", []):
            if col not in df.columns:
                continue
            if not pd.api.types.is_numeric_dtype(df[col]):
                issues[col] = str(df[col].dtype)
                result["warnings"].append(
                    f"Column '{col}' expected numeric but found {df[col].dtype}"
                )
        result["type_issues"] = issues

    def _check_duplicates(
        self,
        df: pd.DataFrame,
        result: Dict[str, Any],
    ) -> None:
        n_dup = int(df.duplicated().sum())
        result["duplicate_rows"] = n_dup
        if n_dup > 0:
            result["warnings"].append(f"{n_dup} fully duplicate rows detected.")
            logger.warning("%d duplicate rows detected.", n_dup)

    def _check_nulls(
        self,
        df: pd.DataFrame,
        schema: Dict[str, Any],
        result: Dict[str, Any],
    ) -> None:
        watched = (
            schema.get("required_columns", [])
            + schema.get("numeric_cols", [])
        )
        null_summary: Dict[str, int] = {}
        for col in dict.fromkeys(watched):   # preserve order, deduplicate
            if col not in df.columns:
                continue
            n_null = int(df[col].isnull().sum())
            if n_null > 0:
                null_summary[col] = n_null
                result["warnings"].append(
                    f"Column '{col}' has {n_null} null values."
                )
        result["null_summary"] = null_summary

    def _check_numeric_ranges(
        self,
        df: pd.DataFrame,
        schema: Dict[str, Any],
        result: Dict[str, Any],
    ) -> None:
        violations: Dict[str, int] = {}
        for col, (lo, hi) in schema.get("numeric_ranges", {}).items():
            if col not in df.columns:
                continue
            series = pd.to_numeric(df[col], errors="coerce")
            n_viol = int(((series < lo) | (series > hi)).sum())
            if n_viol > 0:
                violations[col] = n_viol
                result["warnings"].append(
                    f"Column '{col}': {n_viol} values outside range [{lo}, {hi}]."
                )
        result["range_violations"] = violations

    def _check_categories(
        self,
        df: pd.DataFrame,
        schema: Dict[str, Any],
        result: Dict[str, Any],
    ) -> None:
        anomalies: Dict[str, List[str]] = {}
        for col, expected_set in schema.get("expected_categories", {}).items():
            if col not in df.columns or not expected_set:
                continue
            actual_vals = set(df[col].dropna().unique())
            unexpected = list(actual_vals - expected_set)
            if unexpected:
                anomalies[col] = unexpected
                result["warnings"].append(
                    f"Column '{col}' contains unexpected categories: {unexpected}"
                )
        result["category_anomalies"] = anomalies

    def _check_unique_id(
        self,
        df: pd.DataFrame,
        schema: Dict[str, Any],
        result: Dict[str, Any],
    ) -> None:
        id_col: Optional[str] = schema.get("unique_id_col")
        if not id_col or id_col not in df.columns:
            return
        n_dup_ids = int(df[id_col].duplicated().sum())
        result["id_duplicates"] = n_dup_ids
        if n_dup_ids > 0:
            result["warnings"].append(
                f"Unique-ID column '{id_col}' has {n_dup_ids} duplicate values."
            )
            logger.warning(
                "ID column '%s' has %d duplicates.", id_col, n_dup_ids
            )


def print_validation_report(reports: Dict[str, Dict[str, Any]]) -> None:
    """
    Pretty-print a validation report dictionary to stdout.

    Parameters
    ----------
    reports : dict
        Output of DataValidator.validate_all().
    """
    for key, r in reports.items():
        status = "✅ PASS" if r["passed"] else "❌ FAIL"
        print(f"\n{'='*60}")
        print(f"  {status}  Dataset: {key}")
        print(f"{'='*60}")
        print(f"  Rows: {r['row_count']:,}  |  Cols: {r['col_count']}")
        if r["missing_columns"]:
            print(f"  ❌ Missing columns: {r['missing_columns']}")
        if r["duplicate_rows"]:
            print(f"  ⚠  Duplicate rows: {r['duplicate_rows']:,}")
        if r["null_summary"]:
            print(f"  ⚠  Null values:")
            for col, n in r["null_summary"].items():
                print(f"       {col}: {n:,}")
        if r["range_violations"]:
            print(f"  ⚠  Range violations: {r['range_violations']}")
        if r["category_anomalies"]:
            print(f"  ⚠  Category anomalies: {r['category_anomalies']}")
        if r["errors"]:
            print(f"  ❌ Errors:")
            for e in r["errors"]:
                print(f"       {e}")
