"""
VoltIQ – Data Cleaning
======================
Reusable, in-memory preprocessing functions for all VoltIQ datasets.

Design rules
------------
* NEVER overwrite source CSV files.
* All operations return new DataFrames (or annotated copies).
* Outliers are flagged via a new boolean column, never silently removed.
* Every public function is logged.

Usage:
    from app.utils.cleaning import DataCleaner
    cleaner = DataCleaner()
    clean_battery = cleaner.clean_battery(battery_df)
    clean_fleet   = cleaner.clean_fleet_readiness(fleet_df)
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Standalone utility functions (reusable across all datasets)
# ---------------------------------------------------------------------------

def drop_full_duplicates(df: pd.DataFrame, name: str = "") -> pd.DataFrame:
    """
    Remove fully identical rows.

    Parameters
    ----------
    df : pd.DataFrame
    name : str
        Dataset label used in log messages.

    Returns
    -------
    pd.DataFrame
        Deduplicated copy.
    """
    n_before = len(df)
    df_clean = df.drop_duplicates()
    n_removed = n_before - len(df_clean)
    if n_removed:
        logger.warning("[%s] Removed %d duplicate rows.", name, n_removed)
    else:
        logger.info("[%s] No duplicate rows found.", name)
    return df_clean.reset_index(drop=True)


def fill_numeric_nulls(
    df: pd.DataFrame,
    columns: List[str],
    strategy: str = "median",
    name: str = "",
) -> pd.DataFrame:
    """
    Fill missing numeric values using mean, median, or zero.

    Parameters
    ----------
    df : pd.DataFrame
    columns : list[str]
        Columns to impute.
    strategy : {'mean', 'median', 'zero'}
        Imputation strategy (default: 'median').
    name : str

    Returns
    -------
    pd.DataFrame
    """
    df = df.copy()
    for col in columns:
        if col not in df.columns:
            continue
        n_null = df[col].isnull().sum()
        if n_null == 0:
            continue
        if strategy == "mean":
            fill_val = df[col].mean()
        elif strategy == "zero":
            fill_val = 0.0
        else:  # median (default)
            fill_val = df[col].median()
        df[col] = df[col].fillna(fill_val)
        logger.info(
            "[%s] Filled %d nulls in '%s' with %s (%.4f).",
            name, n_null, col, strategy, fill_val,
        )
    return df


def fill_categorical_nulls(
    df: pd.DataFrame,
    columns: List[str],
    fill_value: str = "Unknown",
    name: str = "",
) -> pd.DataFrame:
    """
    Fill missing categorical / string columns with a placeholder value.

    Parameters
    ----------
    df : pd.DataFrame
    columns : list[str]
    fill_value : str
    name : str

    Returns
    -------
    pd.DataFrame
    """
    df = df.copy()
    for col in columns:
        if col not in df.columns:
            continue
        n_null = df[col].isnull().sum()
        if n_null:
            df[col] = df[col].fillna(fill_value)
            logger.info(
                "[%s] Filled %d nulls in '%s' with '%s'.",
                name, n_null, col, fill_value,
            )
    return df


def normalize_strings(
    df: pd.DataFrame,
    columns: List[str],
    name: str = "",
) -> pd.DataFrame:
    """
    Strip whitespace and apply title-case normalisation to string columns.

    Returns
    -------
    pd.DataFrame
    """
    df = df.copy()
    for col in columns:
        if col not in df.columns:
            continue
        df[col] = df[col].astype(str).str.strip().str.title()
    logger.info("[%s] Normalised string columns: %s", name, columns)
    return df


def convert_to_datetime(
    df: pd.DataFrame,
    columns: List[str],
    fmt: Optional[str] = None,
    name: str = "",
) -> pd.DataFrame:
    """
    Parse date/datetime columns using pd.to_datetime.

    Parameters
    ----------
    df : pd.DataFrame
    columns : list[str]
    fmt : str, optional
        strftime format string; None lets pandas infer.
    name : str

    Returns
    -------
    pd.DataFrame
    """
    df = df.copy()
    for col in columns:
        if col not in df.columns:
            continue
        try:
            df[col] = pd.to_datetime(df[col], format=fmt, errors="coerce")
            n_failed = df[col].isnull().sum()
            if n_failed:
                logger.warning(
                    "[%s] %d rows could not be parsed as datetime in '%s'.",
                    name, n_failed, col,
                )
            else:
                logger.info("[%s] Parsed datetime column '%s'.", name, col)
        except Exception as exc:
            logger.error("[%s] Failed to convert '%s': %s", name, col, exc)
    return df


def coerce_numeric(
    df: pd.DataFrame,
    columns: List[str],
    name: str = "",
) -> pd.DataFrame:
    """
    Force-cast columns to numeric, coercing non-parseable values to NaN.

    Returns
    -------
    pd.DataFrame
    """
    df = df.copy()
    for col in columns:
        if col not in df.columns:
            continue
        before_nulls = df[col].isnull().sum()
        df[col] = pd.to_numeric(df[col], errors="coerce")
        new_nulls = df[col].isnull().sum() - before_nulls
        if new_nulls > 0:
            logger.warning(
                "[%s] Coerced %d non-numeric values to NaN in '%s'.",
                name, new_nulls, col,
            )
    return df


def flag_iqr_outliers(
    df: pd.DataFrame,
    columns: List[str],
    multiplier: float = 1.5,
    name: str = "",
) -> pd.DataFrame:
    """
    Detect outliers using the IQR method and add a boolean flag column.

    Adds a new column ``<col>_outlier`` (True if outlier) for each column.
    Original values are NEVER removed.

    Parameters
    ----------
    df : pd.DataFrame
    columns : list[str]
    multiplier : float
        IQR fence multiplier (default 1.5 = standard Tukey fences).
    name : str

    Returns
    -------
    pd.DataFrame
    """
    df = df.copy()
    for col in columns:
        if col not in df.columns:
            continue
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lo = q1 - multiplier * iqr
        hi = q3 + multiplier * iqr
        flag_col = f"{col}_outlier"
        df[flag_col] = (df[col] < lo) | (df[col] > hi)
        n_out = int(df[flag_col].sum())
        if n_out:
            logger.warning(
                "[%s] %d outliers flagged in '%s' (IQR ×%.1f).",
                name, n_out, col, multiplier,
            )
    return df


def flag_invalid_coordinates(
    df: pd.DataFrame,
    lat_col: str = "Latitude",
    lon_col: str = "Longitude",
    name: str = "",
) -> pd.DataFrame:
    """
    Flag rows where latitude or longitude are outside valid geographic bounds.

    Adds a boolean column ``invalid_coordinates``.

    Returns
    -------
    pd.DataFrame
    """
    df = df.copy()
    if lat_col not in df.columns or lon_col not in df.columns:
        logger.warning("[%s] Coordinate columns not found; skipping.", name)
        return df
    lat_invalid = (df[lat_col] < -90) | (df[lat_col] > 90)
    lon_invalid = (df[lon_col] < -180) | (df[lon_col] > 180)
    df["invalid_coordinates"] = lat_invalid | lon_invalid
    n_invalid = int(df["invalid_coordinates"].sum())
    if n_invalid:
        logger.warning(
            "[%s] %d rows have invalid coordinates.", name, n_invalid
        )
    else:
        logger.info("[%s] All coordinates are within valid bounds.", name)
    return df


def clip_to_range(
    df: pd.DataFrame,
    col_ranges: Dict[str, Tuple[float, float]],
    name: str = "",
) -> pd.DataFrame:
    """
    Clip numeric columns to a valid [min, max] range (in-place on copy).

    Unlike outlier removal, this caps extreme values to the boundary.
    Use with caution — prefer flag_iqr_outliers for exploratory work.

    Parameters
    ----------
    df : pd.DataFrame
    col_ranges : dict[str, (float, float)]
        e.g. {"SOC_Start_Pct": (0.0, 100.0)}
    name : str

    Returns
    -------
    pd.DataFrame
    """
    df = df.copy()
    for col, (lo, hi) in col_ranges.items():
        if col not in df.columns:
            continue
        before_min, before_max = df[col].min(), df[col].max()
        df[col] = df[col].clip(lo, hi)
        logger.info(
            "[%s] Clipped '%s' from [%.2f, %.2f] to [%.2f, %.2f].",
            name, col, before_min, before_max, lo, hi,
        )
    return df


# ---------------------------------------------------------------------------
# Dataset-specific cleaners
# ---------------------------------------------------------------------------

class DataCleaner:
    """
    Orchestrates in-memory cleaning pipelines for every VoltIQ dataset.

    All methods return a clean copy of the input DataFrame.
    Source CSV files are never modified.
    """

    # ------------------------------------------------------------------
    # Battery
    # ------------------------------------------------------------------
    def clean_battery(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean battery degradation data.

        Steps:
            1. Drop full duplicates.
            2. Coerce numeric columns.
            3. Fill nulls (median strategy).
            4. Clip SOH to [0, 1].
            5. Normalise categorical columns.
            6. Flag outliers in key telemetry columns.

        Returns
        -------
        pd.DataFrame
        """
        name = "battery"
        logger.info("[%s] Starting cleaning pipeline.", name)
        numeric_cols = [
            "Cycle_Number", "Voltage_V", "Temperature_C", "Capacity_Ah",
            "State_of_Health", "Remaining_Useful_Life_Cycles",
            "Capacity_Fade_Pct", "Voltage_Sag_V", "Degradation_Rate",
        ]
        df = drop_full_duplicates(df, name)
        df = coerce_numeric(df, numeric_cols, name)
        df = fill_numeric_nulls(df, numeric_cols, "median", name)
        df = clip_to_range(df, {"State_of_Health": (0.0, 1.0)}, name)
        df = normalize_strings(df, ["Health_Zone", "Temperature_Zone"], name)
        df = flag_iqr_outliers(
            df, ["Voltage_V", "Temperature_C", "Capacity_Ah"], name=name
        )
        logger.info("[%s] Cleaning complete. Shape: %s", name, df.shape)
        return df

    # ------------------------------------------------------------------
    # Carbon Intelligence
    # ------------------------------------------------------------------
    def clean_carbon(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean fleet carbon intelligence data.

        Steps:
            1. Drop duplicates.
            2. Coerce numeric columns.
            3. Fill nulls.
            4. Clip readiness scores & progress percentages.
            5. Normalise string columns.
            6. Flag outlier CO2 values.

        Returns
        -------
        pd.DataFrame
        """
        name = "carbon"
        logger.info("[%s] Starting cleaning pipeline.", name)
        numeric_cols = [
            "Vehicle_Age_Years", "Fuel_Consumption_L100km",
            "Annual_CO2_Emissions_kg", "EV_Readiness_Score",
            "Total_Usage_Hours", "Annual_Maintenance_Cost_USD",
            "Vehicle_Health_Score", "EV_Scenario_CO2_kg",
            "Annual_CO2_Saving_kg", "Net_Zero_Progress_Pct",
        ]
        df = drop_full_duplicates(df, name)
        df = coerce_numeric(df, numeric_cols, name)
        df = fill_numeric_nulls(df, numeric_cols, "median", name)
        df = clip_to_range(
            df,
            {
                "EV_Readiness_Score":    (0.0, 1.0),
                "Net_Zero_Progress_Pct": (0.0, 100.0),
                "Vehicle_Health_Score":  (0.0, 100.0),
            },
            name,
        )
        df = normalize_strings(df, ["Vehicle_Type", "Route_Type"], name)
        df = flag_iqr_outliers(
            df, ["Annual_CO2_Emissions_kg", "Annual_CO2_Saving_kg"], name=name
        )
        logger.info("[%s] Cleaning complete. Shape: %s", name, df.shape)
        return df

    # ------------------------------------------------------------------
    # Charging Sessions
    # ------------------------------------------------------------------
    def clean_charging_sessions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean EV charging session data.

        Steps:
            1. Drop duplicates.
            2. Parse datetime columns.
            3. Coerce numeric columns.
            4. Fill nulls.
            5. Clip SOC to [0, 100].
            6. Normalise categorical columns.
            7. Flag outlier charging rates.

        Returns
        -------
        pd.DataFrame
        """
        name = "charging_sessions"
        logger.info("[%s] Starting cleaning pipeline.", name)
        numeric_cols = [
            "Battery_Capacity_kWh", "Energy_Consumed_kWh",
            "Charging_Duration_hrs", "Charging_Rate_kW",
            "Charging_Cost_USD", "SOC_Start_Pct", "SOC_End_Pct",
            "Temperature_C",
        ]
        date_cols = ["Session_Start_Time", "Session_End_Time"]
        df = drop_full_duplicates(df, name)
        df = convert_to_datetime(df, date_cols, name=name)
        df = coerce_numeric(df, numeric_cols, name)
        df = fill_numeric_nulls(df, numeric_cols, "median", name)
        df = clip_to_range(
            df,
            {"SOC_Start_Pct": (0.0, 100.0), "SOC_End_Pct": (0.0, 100.0)},
            name,
        )
        df = normalize_strings(df, ["Charger_Type", "User_Type"], name)
        df = flag_iqr_outliers(df, ["Charging_Rate_kW", "Energy_Consumed_kWh"], name=name)
        logger.info("[%s] Cleaning complete. Shape: %s", name, df.shape)
        return df

    # ------------------------------------------------------------------
    # Charging Stations
    # ------------------------------------------------------------------
    def clean_charging_stations(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean EV charging station data.

        Steps:
            1. Drop duplicates.
            2. Coerce numeric columns.
            3. Fill nulls.
            4. Validate and flag invalid coordinates.
            5. Normalise string columns.

        Returns
        -------
        pd.DataFrame
        """
        name = "charging_stations"
        logger.info("[%s] Starting cleaning pipeline.", name)
        numeric_cols = [
            "Latitude", "Longitude", "Power_kW",
            "Total_Chargers", "Availability_Pct",
        ]
        df = drop_full_duplicates(df, name)
        df = coerce_numeric(df, numeric_cols, name)
        df = fill_numeric_nulls(df, ["Power_kW", "Availability_Pct"], "median", name)
        df = flag_invalid_coordinates(df, "Latitude", "Longitude", name)
        df = normalize_strings(df, ["Charger_Type", "City", "State"], name)
        logger.info("[%s] Cleaning complete. Shape: %s", name, df.shape)
        return df

    # ------------------------------------------------------------------
    # Fleet Electrification Readiness
    # ------------------------------------------------------------------
    def clean_fleet_readiness(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean fleet electrification readiness data.

        Steps:
            1. Drop duplicates.
            2. Coerce numeric columns.
            3. Fill nulls.
            4. Clip readiness and health scores.
            5. Normalise string columns.
            6. Flag outlier fuel consumption.

        Returns
        -------
        pd.DataFrame
        """
        name = "fleet_readiness"
        logger.info("[%s] Starting cleaning pipeline.", name)
        numeric_cols = [
            "Vehicle_Age_Years", "Usage_Hours", "Load_Capacity",
            "Actual_Load", "Fuel_Consumption", "Health_Score",
            "EV_Readiness_Score",
        ]
        df = drop_full_duplicates(df, name)
        df = coerce_numeric(df, numeric_cols, name)
        df = fill_numeric_nulls(df, numeric_cols, "median", name)
        df = clip_to_range(
            df,
            {
                "EV_Readiness_Score": (0.0, 1.0),
                "Health_Score":       (0.0, 100.0),
            },
            name,
        )
        df = normalize_strings(
            df, ["Vehicle_Type", "Route_Info", "Road_Conditions"], name
        )
        df = flag_iqr_outliers(df, ["Fuel_Consumption", "Usage_Hours"], name=name)
        logger.info("[%s] Cleaning complete. Shape: %s", name, df.shape)
        return df

    # ------------------------------------------------------------------
    # Fleet Routes
    # ------------------------------------------------------------------
    def clean_fleet_routes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean fleet route data.

        Steps:
            1. Drop duplicates.
            2. Parse date column.
            3. Coerce numeric columns.
            4. Fill nulls.
            5. Normalise string columns.
            6. Flag outlier distances and speeds.

        Returns
        -------
        pd.DataFrame
        """
        name = "fleet_routes"
        logger.info("[%s] Starting cleaning pipeline.", name)
        numeric_cols = [
            "Distance_km", "Average_Speed_kmh", "Travel_Time_hrs",
            "Payload_kg", "Idle_Time_min", "Estimated_Fuel_L",
        ]
        df = drop_full_duplicates(df, name)
        df = convert_to_datetime(df, ["Date"], name=name)
        df = coerce_numeric(df, numeric_cols, name)
        df = fill_numeric_nulls(df, numeric_cols, "median", name)
        df = normalize_strings(
            df, ["Vehicle_Type", "Road_Type", "Traffic_Level"], name
        )
        df = flag_iqr_outliers(
            df, ["Distance_km", "Average_Speed_kmh"], name=name
        )
        logger.info("[%s] Cleaning complete. Shape: %s", name, df.shape)
        return df

    # ------------------------------------------------------------------
    # Carbon Reference
    # ------------------------------------------------------------------
    def clean_carbon_reference(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean carbon emissions reference table.

        Returns
        -------
        pd.DataFrame
        """
        name = "carbon_reference"
        logger.info("[%s] Starting cleaning pipeline.", name)
        numeric_cols = [
            "Engine_Size_L", "CO2_Emissions_g_per_km",
            "Estimated_Annual_CO2_kg",
        ]
        df = drop_full_duplicates(df, name)
        df = coerce_numeric(df, numeric_cols, name)
        df = fill_numeric_nulls(df, numeric_cols, "median", name)
        df = normalize_strings(df, ["Fuel_Type", "Vehicle_Class", "Manufacturer"], name)
        logger.info("[%s] Cleaning complete. Shape: %s", name, df.shape)
        return df

    # ------------------------------------------------------------------
    # Weather
    # ------------------------------------------------------------------
    def clean_weather(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean weather data.

        Steps:
            1. Drop duplicates.
            2. Parse date column.
            3. Coerce numeric columns.
            4. Fill nulls.
            5. Clip temperature and humidity to valid ranges.
            6. Normalise string columns.
            7. Flag outlier wind speed values.

        Returns
        -------
        pd.DataFrame
        """
        name = "weather"
        logger.info("[%s] Starting cleaning pipeline.", name)
        numeric_cols = [
            "Year", "Month", "Temperature_C", "Humidity_Pct",
            "Precipitation_mm", "Wind_Speed_kmh",
        ]
        df = drop_full_duplicates(df, name)
        df = convert_to_datetime(df, ["Date"], name=name)
        df = coerce_numeric(df, numeric_cols, name)
        df = fill_numeric_nulls(df, numeric_cols, "median", name)
        df = clip_to_range(
            df,
            {
                "Humidity_Pct":  (0.0, 100.0),
                "Temperature_C": (-60.0, 60.0),
            },
            name,
        )
        df = normalize_strings(
            df, ["Season", "Weather_Condition", "City", "Country"], name
        )
        df = flag_iqr_outliers(df, ["Wind_Speed_kmh", "Precipitation_mm"], name=name)
        logger.info("[%s] Cleaning complete. Shape: %s", name, df.shape)
        return df

    # ------------------------------------------------------------------
    # Convenience: clean all
    # ------------------------------------------------------------------
    def clean_all(
        self,
        dfs: Dict[str, pd.DataFrame],
    ) -> Dict[str, pd.DataFrame]:
        """
        Run all dataset-specific cleaning pipelines.

        Parameters
        ----------
        dfs : dict[str, pd.DataFrame]
            Raw dataframes keyed by registry key.

        Returns
        -------
        dict[str, pd.DataFrame]
            Cleaned dataframes with the same keys.
        """
        dispatch = {
            "battery":          self.clean_battery,
            "carbon":           self.clean_carbon,
            "charging_sessions": self.clean_charging_sessions,
            "charging_stations": self.clean_charging_stations,
            "fleet_readiness":  self.clean_fleet_readiness,
            "fleet_routes":     self.clean_fleet_routes,
            "carbon_reference": self.clean_carbon_reference,
            "weather":          self.clean_weather,
        }
        cleaned: Dict[str, pd.DataFrame] = {}
        for key, fn in dispatch.items():
            if key in dfs:
                try:
                    cleaned[key] = fn(dfs[key])
                except Exception as exc:
                    logger.error("Cleaning failed for '%s': %s", key, exc)
            else:
                logger.warning("Dataset '%s' not available for cleaning.", key)
        logger.info(
            "clean_all() complete. Cleaned %d / %d datasets.",
            len(cleaned), len(dispatch),
        )
        return cleaned
