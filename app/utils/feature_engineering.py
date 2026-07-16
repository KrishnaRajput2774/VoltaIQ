"""
VoltIQ – Feature Engineering  (Phase 1: Data Preparation)
===========================================================
PHASE SCOPE
-----------
This module is a Phase 1 (Environment & Data Wrangling) deliverable.
It creates derived columns that PREPARE the datasets for downstream
use in Phase 2 (ML training), Phase 3 (API), and Phase 4 (dashboard).

No machine learning training, model fitting, prediction, or evaluation
is performed here. The outputs are additional columns attached to
in-memory DataFrames — no model files are written.

Purpose of each derived column
-------------------------------
These columns are created during Phase 1 so that Phase 2 ML training
can work with clean, semantically meaningful input features without
needing to repeat binning or normalisation logic. They are not ML
outputs; they are ML inputs prepared in advance.

Design rules
------------
* Input DataFrames are NEVER mutated — all functions return new copies.
* No scikit-learn .fit(), .predict(), or .transform() calls are made.
* No model files (.pkl, .joblib) are read or written.
* All column derivations use only pandas and NumPy arithmetic/binning.

Derived features per dataset
-----------------------------
Fleet Readiness
    Vehicle_Age_Category       — age band for readiness segmentation
    Maintenance_Cost_Band      — Low/Medium/High cost tier
    Fuel_Efficiency_Category   — efficiency label from fuel consumption
    Load_Utilization_Category  — load ratio label
    Readiness_Label            — human-readable EV readiness label

Battery
    Capacity_Loss_Pct          — alias of Capacity_Fade_Pct for clarity
    Battery_Age_Indicator      — cycle-count age band
    SOH_Category               — health quality label
    Degradation_Severity       — rate-of-degradation label

Charging Sessions
    Session_Duration_Computed_hrs     — end-time minus start-time
    Charging_Efficiency_Computed_Pct  — charge added / energy consumed
    Fast_Charging_Flag                — True for DC Fast Charger sessions
    Cost_Per_kWh_Computed             — cost / energy consumed
    SOC_Delta_Computed                — SOC_End - SOC_Start

Carbon Intelligence
    Carbon_Saving_Potential_kg  — baseline minus EV scenario CO2
    Net_Zero_Progress_Category  — progress label
    High_Emitter_Flag           — above 75th-percentile emissions
    EV_Priority_Score           — composite readiness x savings score (0-1)

Weather
    Temperature_Category     — binned temperature label
    Climate_Severity_Index   — weighted severity composite (0-1)
    Extreme_Weather_Flag     — True if Climate_Severity_Index > 0.7

Usage:
    from app.utils.feature_engineering import FeatureEngineer
    fe = FeatureEngineer()
    battery_fe = fe.engineer_battery(clean_battery_df)   # Phase 1 prep
    all_dfs    = fe.engineer_all(cleaned_dfs_dict)       # all datasets
"""

import logging
from typing import Dict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper: safe binning with labelled categories
# ---------------------------------------------------------------------------

def _safe_cut(
    series: pd.Series,
    bins: list,
    labels: list,
    col_name: str,
    default: str = "Unknown",
) -> pd.Series:
    """Apply pd.cut safely, filling out-of-range values with *default*."""
    result = pd.cut(series, bins=bins, labels=labels, right=True)
    return result.astype(str).replace("nan", default)


# ---------------------------------------------------------------------------
# FeatureEngineer class
# ---------------------------------------------------------------------------

class FeatureEngineer:
    """
    Derives engineered features for every VoltIQ dataset.

    All methods return a *new* DataFrame with additional columns appended.
    Original columns are preserved unchanged.
    """

    # ------------------------------------------------------------------
    # Fleet Electrification Readiness
    # ------------------------------------------------------------------
    def engineer_fleet_readiness(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Derive fleet-level features for electrification readiness modelling.

        New columns
        -----------
        Vehicle_Age_Category : str
            Binned vehicle age — 'New (0–3 yrs)', 'Mid-Life (4–8 yrs)',
            'Aging (9–14 yrs)', 'Old (15+ yrs)'.
        Maintenance_Cost_Band : str
            Low / Medium / High based on Maintenance_Cost quartiles.
        Fuel_Efficiency_Category : str
            Low / Moderate / High efficiency derived from Fuel_Consumption.
        Load_Utilization_Category : str
            Under-loaded / Optimal / Over-loaded based on Load_Utilization_Pct.
        Readiness_Label : str
            Human-readable readiness from EV_Readiness_Score.

        Returns
        -------
        pd.DataFrame
        """
        name = "fleet_readiness"
        logger.info("[%s] Engineering features.", name)
        df = df.copy()

        # Vehicle Age Category
        if "Vehicle_Age_Years" in df.columns:
            df["Vehicle_Age_Category"] = _safe_cut(
                df["Vehicle_Age_Years"],
                bins=[-1, 3, 8, 14, 200],
                labels=["New (0-3 yrs)", "Mid-Life (4-8 yrs)",
                        "Aging (9-14 yrs)", "Old (15+ yrs)"],
                col_name="Vehicle_Age_Category",
            )

        # Maintenance Cost Band (quantile-based)
        if "Maintenance_Cost" in df.columns:
            try:
                df["Maintenance_Cost_Band"] = pd.qcut(
                    df["Maintenance_Cost"],
                    q=3,
                    labels=["Low", "Medium", "High"],
                    duplicates="drop",
                ).astype(str)
            except Exception as exc:
                logger.warning("[%s] Maintenance_Cost_Band skipped: %s", name, exc)

        # Fuel Efficiency Category (lower L/100km = more efficient)
        if "Fuel_Consumption" in df.columns:
            df["Fuel_Efficiency_Category"] = _safe_cut(
                df["Fuel_Consumption"],
                bins=[-1, 6, 10, 15, 1000],
                labels=["High Efficiency", "Moderate Efficiency",
                        "Low Efficiency", "Very Low Efficiency"],
                col_name="Fuel_Efficiency_Category",
            )

        # Load Utilization Category
        if "Load_Utilization_Pct" in df.columns:
            df["Load_Utilization_Category"] = _safe_cut(
                df["Load_Utilization_Pct"],
                bins=[-1, 40, 70, 90, 101],
                labels=["Under-loaded", "Optimal", "Near-capacity", "Over-loaded"],
                col_name="Load_Utilization_Category",
            )

        # Readiness Label
        if "EV_Readiness_Score" in df.columns:
            df["Readiness_Label"] = _safe_cut(
                df["EV_Readiness_Score"],
                bins=[-0.01, 0.33, 0.66, 1.01],
                labels=["Low Readiness", "Medium Readiness", "High Readiness"],
                col_name="Readiness_Label",
            )

        logger.info("[%s] Features engineered. Total cols: %d", name, len(df.columns))
        return df

    # ------------------------------------------------------------------
    # Battery Degradation
    # ------------------------------------------------------------------
    def engineer_battery(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Derive battery health and lifecycle features.

        New columns
        -----------
        Capacity_Loss_Pct : float
            Percentage capacity lost from initial rated capacity.
            = Capacity_Fade_Pct (already present, aliased for clarity).
        Battery_Age_Indicator : str
            'Early Cycles' / 'Mid Cycles' / 'Late Cycles' / 'End of Life'
            based on Cycle_Number quantile.
        Charge_Rate_Category : str
            'Slow' / 'Moderate' / 'Fast' derived from Voltage_V proxy.
        SOH_Category : str
            'Excellent (>90%)' / 'Good (80-90%)' /
            'Fair (70-80%)' / 'Poor (<70%)'.
        Degradation_Severity : str
            Derived from Degradation_Rate magnitude.

        Returns
        -------
        pd.DataFrame
        """
        name = "battery"
        logger.info("[%s] Engineering features.", name)
        df = df.copy()

        # Capacity Loss (alias for downstream ML clarity)
        if "Capacity_Fade_Pct" in df.columns:
            df["Capacity_Loss_Pct"] = df["Capacity_Fade_Pct"]

        # Battery Age Indicator
        if "Cycle_Number" in df.columns:
            df["Battery_Age_Indicator"] = _safe_cut(
                df["Cycle_Number"],
                bins=[-1, 30, 80, 140, 10000],
                labels=["Early Cycles", "Mid Cycles", "Late Cycles", "End of Life"],
                col_name="Battery_Age_Indicator",
            )

        # SOH Category
        if "State_of_Health" in df.columns:
            df["SOH_Category"] = _safe_cut(
                df["State_of_Health"],
                bins=[-0.01, 0.70, 0.80, 0.90, 1.01],
                labels=["Poor (<70%)", "Fair (70-80%)",
                        "Good (80-90%)", "Excellent (>90%)"],
                col_name="SOH_Category",
            )

        # Degradation Severity
        if "Degradation_Rate" in df.columns:
            abs_rate = df["Degradation_Rate"].abs()
            df["Degradation_Severity"] = _safe_cut(
                abs_rate,
                bins=[-0.001, 0.003, 0.007, 0.012, 1.0],
                labels=["Minimal", "Moderate", "Significant", "Critical"],
                col_name="Degradation_Severity",
            )

        # Temperature Zone (rederive if not already present)
        if "Temperature_C" in df.columns and "Temperature_Zone" not in df.columns:
            df["Temperature_Zone"] = _safe_cut(
                df["Temperature_C"],
                bins=[-100, 10, 25, 35, 100],
                labels=["Cold", "Moderate", "Warm", "Hot"],
                col_name="Temperature_Zone",
            )

        logger.info("[%s] Features engineered. Total cols: %d", name, len(df.columns))
        return df

    # ------------------------------------------------------------------
    # Charging Sessions
    # ------------------------------------------------------------------
    def engineer_charging_sessions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Derive charging session performance features.

        New columns
        -----------
        Session_Duration_Computed_hrs : float
            Computed from Session_End_Time − Session_Start_Time (if parsed).
        Charging_Efficiency_Computed_Pct : float
            = (Charge_Added_kWh / Energy_Consumed_kWh) × 100.
        Fast_Charging_Flag : bool
            True if Charger_Type == 'DC Fast Charger'.
        Cost_Per_kWh_Computed : float
            = Charging_Cost_USD / Energy_Consumed_kWh.
        SOC_Delta_Computed : float
            = SOC_End_Pct − SOC_Start_Pct.

        Returns
        -------
        pd.DataFrame
        """
        name = "charging_sessions"
        logger.info("[%s] Engineering features.", name)
        df = df.copy()

        # Session Duration
        if "Session_Start_Time" in df.columns and "Session_End_Time" in df.columns:
            try:
                start = pd.to_datetime(df["Session_Start_Time"], errors="coerce")
                end   = pd.to_datetime(df["Session_End_Time"],   errors="coerce")
                df["Session_Duration_Computed_hrs"] = (
                    (end - start).dt.total_seconds() / 3600
                ).round(4)
            except Exception as exc:
                logger.warning("[%s] Session duration calc failed: %s", name, exc)

        # Charging Efficiency
        if "Charge_Added_kWh" in df.columns and "Energy_Consumed_kWh" in df.columns:
            mask = df["Energy_Consumed_kWh"] > 0
            df["Charging_Efficiency_Computed_Pct"] = np.where(
                mask,
                (df["Charge_Added_kWh"] / df["Energy_Consumed_kWh"]) * 100,
                np.nan,
            ).round(2)

        # Fast Charging Flag
        if "Charger_Type" in df.columns:
            df["Fast_Charging_Flag"] = (
                df["Charger_Type"].str.upper().str.contains("DC", na=False)
            )

        # Cost per kWh
        if "Charging_Cost_USD" in df.columns and "Energy_Consumed_kWh" in df.columns:
            mask = df["Energy_Consumed_kWh"] > 0
            df["Cost_Per_kWh_Computed"] = np.where(
                mask,
                df["Charging_Cost_USD"] / df["Energy_Consumed_kWh"],
                np.nan,
            ).round(4)

        # SOC Delta
        if "SOC_Start_Pct" in df.columns and "SOC_End_Pct" in df.columns:
            df["SOC_Delta_Computed"] = (
                df["SOC_End_Pct"] - df["SOC_Start_Pct"]
            ).round(2)

        logger.info("[%s] Features engineered. Total cols: %d", name, len(df.columns))
        return df

    # ------------------------------------------------------------------
    # Carbon Intelligence
    # ------------------------------------------------------------------
    def engineer_carbon(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Derive carbon offset and net-zero tracking features.

        New columns
        -----------
        Carbon_Saving_Potential_kg : float
            = Annual_CO2_Emissions_kg − EV_Scenario_CO2_kg.
        Net_Zero_Progress_Category : str
            'Behind Target' / 'On Track' / 'Leading'.
        High_Emitter_Flag : bool
            True if Annual_CO2_Emissions_kg > 75th-percentile.
        EV_Priority_Score : float
            Composite score = EV_Readiness_Score × Carbon_Saving_Potential_kg,
            normalised to [0, 1].

        Returns
        -------
        pd.DataFrame
        """
        name = "carbon"
        logger.info("[%s] Engineering features.", name)
        df = df.copy()

        # Carbon Saving Potential
        if "Annual_CO2_Emissions_kg" in df.columns and "EV_Scenario_CO2_kg" in df.columns:
            df["Carbon_Saving_Potential_kg"] = (
                df["Annual_CO2_Emissions_kg"] - df["EV_Scenario_CO2_kg"]
            ).clip(lower=0)

        # Net Zero Progress Category
        if "Net_Zero_Progress_Pct" in df.columns:
            df["Net_Zero_Progress_Category"] = _safe_cut(
                df["Net_Zero_Progress_Pct"],
                bins=[-0.1, 20, 50, 100.1],
                labels=["Behind Target", "On Track", "Leading"],
                col_name="Net_Zero_Progress_Category",
            )

        # High Emitter Flag
        if "Annual_CO2_Emissions_kg" in df.columns:
            threshold = df["Annual_CO2_Emissions_kg"].quantile(0.75)
            df["High_Emitter_Flag"] = df["Annual_CO2_Emissions_kg"] > threshold

        # EV Priority Score (composite, normalised)
        if (
            "EV_Readiness_Score" in df.columns
            and "Carbon_Saving_Potential_kg" in df.columns
        ):
            raw = (
                df["EV_Readiness_Score"]
                * df["Carbon_Saving_Potential_kg"]
            )
            r_min, r_max = raw.min(), raw.max()
            df["EV_Priority_Score"] = (
                ((raw - r_min) / (r_max - r_min)).round(4)
                if r_max > r_min
                else 0.0
            )

        logger.info("[%s] Features engineered. Total cols: %d", name, len(df.columns))
        return df

    # ------------------------------------------------------------------
    # Weather
    # ------------------------------------------------------------------
    def engineer_weather(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Derive climate impact features for battery and route modelling.

        New columns
        -----------
        Temperature_Category : str
            'Freezing (<0°C)' / 'Cold (0–10°C)' / 'Mild (10–25°C)' /
            'Warm (25–35°C)' / 'Hot (>35°C)'.
        Climate_Severity_Index : float
            Weighted composite of Temperature deviation, Humidity, Wind, and
            Precipitation. Range: [0, 1].  Higher = more severe.
        Extreme_Weather_Flag : bool
            True if Climate_Severity_Index > 0.7.

        Returns
        -------
        pd.DataFrame
        """
        name = "weather"
        logger.info("[%s] Engineering features.", name)
        df = df.copy()

        # Temperature Category
        if "Temperature_C" in df.columns:
            df["Temperature_Category"] = _safe_cut(
                df["Temperature_C"],
                bins=[-100, 0, 10, 25, 35, 100],
                labels=["Freezing (<0C)", "Cold (0-10C)", "Mild (10-25C)",
                        "Warm (25-35C)", "Hot (>35C)"],
                col_name="Temperature_Category",
            )

        # Climate Severity Index
        cols_needed = ["Temperature_C", "Humidity_Pct", "Wind_Speed_kmh",
                       "Precipitation_mm"]
        if all(c in df.columns for c in cols_needed):
            # Normalise each component to [0,1] then combine with weights
            def _norm(series: pd.Series) -> pd.Series:
                lo, hi = series.min(), series.max()
                return (series - lo) / (hi - lo + 1e-9)

            temp_dev = _norm(df["Temperature_C"].abs())   # deviation from 0
            humidity = _norm(df["Humidity_Pct"])
            wind     = _norm(df["Wind_Speed_kmh"])
            precip   = _norm(df["Precipitation_mm"])

            df["Climate_Severity_Index"] = (
                0.35 * temp_dev
                + 0.20 * humidity
                + 0.25 * wind
                + 0.20 * precip
            ).round(4)

        # Extreme Weather Flag
        if "Climate_Severity_Index" in df.columns:
            df["Extreme_Weather_Flag"] = df["Climate_Severity_Index"] > 0.7

        logger.info("[%s] Features engineered. Total cols: %d", name, len(df.columns))
        return df

    # ------------------------------------------------------------------
    # Convenience: engineer all
    # ------------------------------------------------------------------
    def engineer_all(
        self,
        dfs: Dict[str, pd.DataFrame],
    ) -> Dict[str, pd.DataFrame]:
        """
        Run all feature-engineering pipelines in one call.

        Parameters
        ----------
        dfs : dict[str, pd.DataFrame]
            Cleaned DataFrames keyed by registry key.

        Returns
        -------
        dict[str, pd.DataFrame]
            Feature-enriched DataFrames.
        """
        dispatch = {
            "fleet_readiness":   self.engineer_fleet_readiness,
            "battery":           self.engineer_battery,
            "charging_sessions": self.engineer_charging_sessions,
            "carbon":            self.engineer_carbon,
            "weather":           self.engineer_weather,
        }
        engineered: Dict[str, pd.DataFrame] = {}
        for key, fn in dispatch.items():
            if key in dfs:
                try:
                    engineered[key] = fn(dfs[key])
                except Exception as exc:
                    logger.error(
                        "Feature engineering failed for '%s': %s", key, exc
                    )
                    engineered[key] = dfs[key]   # pass through unchanged
            else:
                logger.warning(
                    "Dataset '%s' not available for feature engineering.", key
                )
        # Pass-through datasets that don't have specific pipelines
        for key in dfs:
            if key not in engineered:
                engineered[key] = dfs[key]

        logger.info(
            "engineer_all() complete. Processed %d datasets.",
            len(engineered),
        )
        return engineered
