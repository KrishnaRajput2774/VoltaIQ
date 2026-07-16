"""
VoltIQ – Dataset Profiling
===========================
Generates summary statistics and data-quality profiles for all datasets.
Designed for EDA notebooks and the automated data-quality report.

Produces per-dataset:
    • Shape, memory usage
    • Column-level: dtype, null count, null %, unique count, min/max/mean/std
    • Categorical: top-5 value frequencies
    • Numeric: skewness, kurtosis
    • Overall quality score (0–100)

Usage:
    from app.utils.profiling import DataProfiler
    profiler = DataProfiler()
    profiles = profiler.profile_all(dfs)
    profiler.print_summary(profiles)
    report_df = profiler.to_dataframe(profiles)
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DataProfiler:
    """
    Generates statistical profiles and quality scores for VoltIQ datasets.

    Parameters
    ----------
    top_n_categories : int
        Number of top category frequencies to record (default 5).
    """

    def __init__(self, top_n_categories: int = 5) -> None:
        self.top_n = top_n_categories

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def profile(self, key: str, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Profile a single DataFrame.

        Parameters
        ----------
        key : str
            Dataset identifier (used in log messages).
        df : pd.DataFrame

        Returns
        -------
        dict
            Comprehensive profile dictionary.
        """
        logger.info("Profiling dataset: %s (%d rows × %d cols)", key, *df.shape)
        profile: Dict[str, Any] = {
            "dataset":       key,
            "rows":          len(df),
            "cols":          len(df.columns),
            "memory_mb":     round(df.memory_usage(deep=True).sum() / 1024**2, 3),
            "total_nulls":   int(df.isnull().sum().sum()),
            "total_dupes":   int(df.duplicated().sum()),
            "columns":       {},
            "quality_score": 0.0,
        }

        for col in df.columns:
            profile["columns"][col] = self._profile_column(df[col])

        profile["quality_score"] = self._compute_quality_score(df, profile)
        logger.info(
            "Profile complete: %s — quality_score=%.1f",
            key, profile["quality_score"],
        )
        return profile

    def profile_all(
        self,
        dfs: Dict[str, pd.DataFrame],
    ) -> Dict[str, Dict[str, Any]]:
        """
        Profile every dataset in *dfs*.

        Returns
        -------
        dict[str, dict]
            Mapping of dataset key → profile dict.
        """
        profiles: Dict[str, Dict[str, Any]] = {}
        for key, df in dfs.items():
            try:
                profiles[key] = self.profile(key, df)
            except Exception as exc:
                logger.error("Profiling failed for '%s': %s", key, exc)
        logger.info("profile_all() complete. Profiled %d datasets.", len(profiles))
        return profiles

    def to_dataframe(
        self,
        profiles: Dict[str, Dict[str, Any]],
    ) -> pd.DataFrame:
        """
        Convert column-level profile data to a flat DataFrame (one row per column).

        Useful for exporting to CSV or displaying in a notebook.

        Returns
        -------
        pd.DataFrame
        """
        rows: List[Dict[str, Any]] = []
        for ds_key, profile in profiles.items():
            for col_name, col_info in profile["columns"].items():
                row = {"dataset": ds_key, "column": col_name}
                row.update(col_info)
                rows.append(row)
        return pd.DataFrame(rows)

    def print_summary(
        self,
        profiles: Dict[str, Dict[str, Any]],
        show_columns: bool = False,
    ) -> None:
        """
        Pretty-print dataset-level summaries to stdout.

        Parameters
        ----------
        profiles : dict
        show_columns : bool
            If True, also print column-level statistics.
        """
        for key, p in profiles.items():
            print(f"\n{'═'*65}")
            print(f"  Dataset : {key}")
            print(f"  Shape   : {p['rows']:,} rows × {p['cols']} cols")
            print(f"  Memory  : {p['memory_mb']:.2f} MB")
            print(f"  Nulls   : {p['total_nulls']:,}")
            print(f"  Dupes   : {p['total_dupes']:,}")
            print(f"  Quality : {p['quality_score']:.1f} / 100")
            if show_columns:
                print(f"\n  {'Column':<35} {'Dtype':<12} {'Nulls':>7} {'Uniques':>9}")
                print(f"  {'-'*65}")
                for col, info in p["columns"].items():
                    print(
                        f"  {col:<35} {info['dtype']:<12} "
                        f"{info['null_count']:>7} {info['unique_count']:>9}"
                    )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _profile_column(self, series: pd.Series) -> Dict[str, Any]:
        """Return a dict of statistics for a single column."""
        info: Dict[str, Any] = {
            "dtype":        str(series.dtype),
            "null_count":   int(series.isnull().sum()),
            "null_pct":     round(series.isnull().mean() * 100, 2),
            "unique_count": int(series.nunique(dropna=True)),
        }

        if pd.api.types.is_numeric_dtype(series):
            desc = series.describe()
            info.update({
                "min":      round(float(desc.get("min", np.nan)), 4),
                "max":      round(float(desc.get("max", np.nan)), 4),
                "mean":     round(float(desc.get("mean", np.nan)), 4),
                "std":      round(float(desc.get("std", np.nan)), 4),
                "median":   round(float(series.median()), 4),
                "skewness": round(float(series.skew()), 4),
                "kurtosis": round(float(series.kurtosis()), 4),
            })
        else:
            top = series.value_counts(dropna=True).head(self.top_n)
            info["top_values"] = {str(k): int(v) for k, v in top.items()}

        return info

    def _compute_quality_score(
        self,
        df: pd.DataFrame,
        profile: Dict[str, Any],
    ) -> float:
        """
        Compute a 0–100 data-quality score for a dataset.

        Score components
        ----------------
        Completeness (40 pts) : penalised proportional to null rate.
        Uniqueness   (30 pts) : penalised proportional to duplicate rate.
        Consistency  (30 pts) : penalised for columns where null_pct > 20 %.
        """
        rows = max(profile["rows"], 1)
        total_cells = rows * max(profile["cols"], 1)

        # Completeness
        null_rate = profile["total_nulls"] / total_cells
        completeness = 40.0 * (1 - null_rate)

        # Uniqueness
        dupe_rate = profile["total_dupes"] / rows
        uniqueness = 30.0 * (1 - dupe_rate)

        # Consistency (fraction of columns with <20% nulls)
        high_null_cols = sum(
            1 for info in profile["columns"].values()
            if info["null_pct"] > 20.0
        )
        consistency = 30.0 * (1 - high_null_cols / max(profile["cols"], 1))

        return round(completeness + uniqueness + consistency, 1)
