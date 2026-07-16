"""
VoltIQ – Data Loader
====================
Responsible for loading all 8 canonical datasets from VoltIQ/datasets/.

Datasets loaded:
    battery/ev_battery_degradation.csv
    carbon/fleet_carbon_intelligence.csv
    charging/ev_charging_sessions.csv
    charging/ev_charging_station_data.csv
    fleet/fleet_electrification_readiness.csv
    fleet/fleet_route_data.csv
    reference/carbon_emissions_reference.csv
    weather/weather_data.csv

Usage:
    from app.utils.data_loader import DataLoader
    loader = DataLoader()
    dfs = loader.load_all()
    battery_df = loader.load("battery")
"""

import os
import logging
from typing import Dict, Optional

import pandas as pd

from app.config import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset registry — single source of truth for all file paths
# ---------------------------------------------------------------------------
DATASET_REGISTRY: Dict[str, str] = {
    "battery":              "battery/ev_battery_degradation.csv",
    "carbon":               "carbon/fleet_carbon_intelligence.csv",
    "charging_sessions":    "charging/ev_charging_sessions.csv",
    "charging_stations":    "charging/ev_charging_station_data.csv",
    "fleet_readiness":      "fleet/fleet_electrification_readiness.csv",
    "fleet_routes":         "fleet/fleet_route_data.csv",
    "carbon_reference":     "reference/carbon_emissions_reference.csv",
    "weather":              "weather/weather_data.csv",
}


class DataLoader:
    """
    Centralised data loader for all VoltIQ datasets.

    Resolves every file path relative to the configured datasets directory
    (settings.datasets_dir, defaulting to ``VoltIQ/datasets/``).

    Parameters
    ----------
    datasets_dir : str, optional
        Override the base datasets directory. Defaults to settings.datasets_dir.
    """

    def __init__(self, datasets_dir: Optional[str] = None) -> None:
        base = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        root = datasets_dir or settings.datasets_dir
        self.datasets_dir: str = os.path.abspath(os.path.join(base, root))
        self._cache: Dict[str, pd.DataFrame] = {}
        logger.info("DataLoader initialised. datasets_dir=%s", self.datasets_dir)
        self._verify_root()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self, key: str) -> pd.DataFrame:
        """
        Load a single dataset by its registry key.

        Parameters
        ----------
        key : str
            One of the keys defined in DATASET_REGISTRY.

        Returns
        -------
        pd.DataFrame
            Loaded dataframe.

        Raises
        ------
        KeyError
            If *key* is not found in DATASET_REGISTRY.
        FileNotFoundError
            If the corresponding CSV file does not exist on disk.
        """
        if key not in DATASET_REGISTRY:
            raise KeyError(
                f"Unknown dataset key '{key}'. "
                f"Valid keys: {list(DATASET_REGISTRY.keys())}"
            )
        if key in self._cache:
            logger.info("Returning cached dataset: %s", key)
            return self._cache[key]
        rel_path = DATASET_REGISTRY[key]
        full_path = os.path.join(self.datasets_dir, rel_path)
        df = self._read_csv(key, full_path)
        self._cache[key] = df
        return df

    def load_all(self) -> Dict[str, pd.DataFrame]:
        """
        Load every registered dataset and return them in a dictionary.

        Returns
        -------
        dict[str, pd.DataFrame]
            Mapping of registry key → loaded DataFrame.
            Keys that fail to load are omitted; errors are logged.
        """
        result: Dict[str, pd.DataFrame] = {}
        for key in DATASET_REGISTRY:
            try:
                result[key] = self.load(key)
            except FileNotFoundError as exc:
                logger.error("MISSING FILE — %s: %s", key, exc)
            except Exception as exc:
                logger.error("LOAD ERROR — %s: %s", key, exc)
        logger.info(
            "load_all() complete. Loaded %d / %d datasets.",
            len(result),
            len(DATASET_REGISTRY),
        )
        return result

    def verify_dataset_structure(self) -> Dict[str, bool]:
        """
        Check which dataset files exist on disk without loading them.

        Returns
        -------
        dict[str, bool]
            Mapping of registry key → True if file exists, False otherwise.
        """
        status: Dict[str, bool] = {}
        for key, rel_path in DATASET_REGISTRY.items():
            full_path = os.path.join(self.datasets_dir, rel_path)
            exists = os.path.isfile(full_path)
            status[key] = exists
            level = logging.INFO if exists else logging.WARNING
            logger.log(
                level,
                "[%s] %s — %s",
                "OK" if exists else "MISSING",
                key,
                full_path,
            )
        return status

    def get_file_path(self, key: str) -> str:
        """Return the absolute path for a dataset key without loading it."""
        if key not in DATASET_REGISTRY:
            raise KeyError(f"Unknown dataset key '{key}'.")
        return os.path.join(self.datasets_dir, DATASET_REGISTRY[key])

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _read_csv(self, key: str, full_path: str) -> pd.DataFrame:
        """Read a CSV, log the outcome, and return the DataFrame."""
        if not os.path.isfile(full_path):
            raise FileNotFoundError(
                f"Dataset '{key}' not found at expected path: {full_path}. "
                "Do NOT create or substitute a replacement dataset."
            )
        try:
            df = pd.read_csv(full_path, low_memory=False)
            logger.info(
                "Loaded %-22s | rows=%-7d | cols=%d | path=%s",
                key,
                len(df),
                len(df.columns),
                full_path,
            )
            return df
        except Exception as exc:
            logger.error("Failed to read '%s' from %s: %s", key, full_path, exc)
            raise

    def _verify_root(self) -> None:
        """Warn if the datasets root directory does not exist."""
        if not os.path.isdir(self.datasets_dir):
            logger.warning(
                "datasets_dir does not exist: %s — "
                "Data loading will fail until the directory is present.",
                self.datasets_dir,
            )


# ---------------------------------------------------------------------------
# Module-level singleton (used by other modules via import)
# ---------------------------------------------------------------------------
data_loader = DataLoader()
