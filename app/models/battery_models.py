"""
VoltIQ -- app/models/battery_models.py
========================================
Phase 2: Battery Predictive ML Models

Implements production sklearn Pipelines for:

    Battery SOH  -- State_of_Health regression     (float in [0, 1])
    Battery RUL  -- Remaining_Useful_Life_Cycles   (integer >= 0)

Design principles
-----------------
* Complete sklearn Pipeline (preprocessing + estimator) is saved -- not
  just the estimator.  The pipeline can be applied directly to raw feature
  DataFrames at inference time.
* Feature leakage analysis is performed BEFORE training; confirmed leaking
  features are excluded and documented.
* Split is performed BY Battery_ID (not by row) to prevent cycle-level
  data leakage across train/val/test boundaries.
* Three candidate algorithms are trained, compared, and the winner
  (lowest test MAE) is selected and serialised.
* Evaluation plots are saved to reports/ml/plots/.
* Full metadata JSON is written alongside each model.
* random_state=42 is used everywhere for complete reproducibility.
"""

from __future__ import annotations

import json
import logging
import os
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import matplotlib
matplotlib.use("Agg")   # non-interactive backend -- must be before pyplot import
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_ROOT               = Path(__file__).resolve().parents[2]
SAVED_MODELS_BATTERY = _ROOT / "saved_models" / "battery"
PLOTS_DIR            = _ROOT / "reports" / "ml" / "plots"

for _d in (SAVED_MODELS_BATTERY, PLOTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

SOH_MODEL_PATH = SAVED_MODELS_BATTERY / "battery_soh_model.pkl"
RUL_MODEL_PATH = SAVED_MODELS_BATTERY / "battery_rul_model.pkl"
SOH_META_PATH  = SAVED_MODELS_BATTERY / "battery_soh_metadata.json"
RUL_META_PATH  = SAVED_MODELS_BATTERY / "battery_rul_metadata.json"

MODEL_VERSION   = "2.0.0"
RANDOM_STATE    = 42

# ---------------------------------------------------------------------------
# Candidate feature pool (BEFORE leakage analysis)
# ---------------------------------------------------------------------------
#: All numeric features that COULD be used -- leakage analysis will trim this.
BATTERY_CANDIDATE_FEATURES: List[str] = [
    "Cycle_Number",
    "Voltage_V",
    "Temperature_C",
    "Capacity_Ah",
    "Capacity_Fade_Pct",    # may be excluded for SOH (high corr)
    "Voltage_Sag_V",
    "Degradation_Rate",
    "Cycle_Normalized",
    "State_of_Health",      # target for SOH, feature for RUL (legitimate)
    "Is_End_of_Life",       # confirmed leakage for both targets
]

SOH_TARGET = "State_of_Health"
RUL_TARGET = "Remaining_Useful_Life_Cycles"


# ---------------------------------------------------------------------------
# Split by Battery_ID (prevents cycle-level leakage)
# ---------------------------------------------------------------------------

def split_by_battery_id(
    df: pd.DataFrame,
    train_frac: float = 0.70,
    val_frac:   float = 0.15,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split battery data so ALL cycles of one battery stay in one split.

    This prevents cycle-level data leakage where adjacent cycles from
    the same battery appear in both train and test.

    Parameters
    ----------
    df : pd.DataFrame
    train_frac : float, default 0.70
    val_frac   : float, default 0.15
    random_state : int, default 42

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        (train_df, val_df, test_df)
    """
    rng        = np.random.RandomState(random_state)
    battery_ids = np.array(sorted(df["Battery_ID"].unique()))
    rng.shuffle(battery_ids)

    n       = len(battery_ids)
    n_train = int(n * train_frac)
    n_val   = int(n * val_frac)

    train_ids = battery_ids[:n_train]
    val_ids   = battery_ids[n_train:n_train + n_val]
    test_ids  = battery_ids[n_train + n_val:]

    train_df = df[df["Battery_ID"].isin(train_ids)].copy()
    val_df   = df[df["Battery_ID"].isin(val_ids)].copy()
    test_df  = df[df["Battery_ID"].isin(test_ids)].copy()

    logger.info(
        "Battery ID-level split (seed=%d): train=%d IDs / %d rows | "
        "val=%d IDs / %d rows | test=%d IDs / %d rows",
        random_state,
        len(train_ids), len(train_df),
        len(val_ids),   len(val_df),
        len(test_ids),  len(test_df),
    )
    return train_df, val_df, test_df


# ---------------------------------------------------------------------------
# Pipeline builders
# ---------------------------------------------------------------------------

def _make_candidates(random_state: int = RANDOM_STATE) -> Dict[str, Pipeline]:
    """
    Return three candidate sklearn Pipelines for algorithm comparison.

    Each Pipeline is: StandardScaler -> Regressor.
    The entire pipeline (including scaler) is what gets saved to disk.
    """
    return {
        "LinearRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("model",  LinearRegression()),
        ]),
        "RandomForest": Pipeline([
            ("scaler", StandardScaler()),
            ("model",  RandomForestRegressor(
                n_estimators=200,
                max_depth=12,
                min_samples_leaf=2,
                random_state=random_state,
                n_jobs=-1,
            )),
        ]),
        "GradientBoosting": Pipeline([
            ("scaler", StandardScaler()),
            ("model",  GradientBoostingRegressor(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=5,
                subsample=0.8,
                random_state=random_state,
            )),
        ]),
    }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _evaluate_pipeline(
    pipeline:     Pipeline,
    X_train:      pd.DataFrame,
    y_train:      pd.Series,
    X_val:        pd.DataFrame,
    y_val:        pd.Series,
    X_test:       pd.DataFrame,
    y_test:       pd.Series,
    model_name:   str,
    target_name:  str,
    cv_folds:     int = 5,
    train_time_s: float = 0.0,
) -> Dict[str, Any]:
    """
    Compute MAE, RMSE, R² on val and test sets; run k-fold CV on training set.
    Uses np.sqrt(MSE) to compute RMSE (compatible with all sklearn versions).
    """
    pred_val  = pipeline.predict(X_val)
    pred_test = pipeline.predict(X_test)

    val_mae  = mean_absolute_error(y_val,  pred_val)
    val_rmse = float(np.sqrt(mean_squared_error(y_val,  pred_val)))
    val_r2   = r2_score(y_val,  pred_val)

    test_mae  = mean_absolute_error(y_test,  pred_test)
    test_rmse = float(np.sqrt(mean_squared_error(y_test,  pred_test)))
    test_r2   = r2_score(y_test,  pred_test)

    cv_scores   = cross_val_score(
        pipeline, X_train, y_train,
        cv=cv_folds, scoring="neg_mean_absolute_error", n_jobs=-1,
    )
    cv_mae_mean = float(-cv_scores.mean())
    cv_mae_std  = float(cv_scores.std())

    metrics = {
        "model":         model_name,
        "target":        target_name,
        "val_mae":       round(val_mae,   5),
        "val_rmse":      round(val_rmse,  5),
        "val_r2":        round(val_r2,    5),
        "test_mae":      round(test_mae,  5),
        "test_rmse":     round(test_rmse, 5),
        "test_r2":       round(test_r2,   5),
        "cv_mae_mean":   round(cv_mae_mean, 5),
        "cv_mae_std":    round(cv_mae_std,  5),
        "cv_folds":      cv_folds,
        "train_time_s":  round(train_time_s, 2),
    }

    logger.info(
        "[%s | %s] val_mae=%.4f val_r2=%.4f | test_mae=%.4f test_r2=%.4f | "
        "cv=%.4f+/-%.4f | fit_time=%.1fs",
        model_name, target_name,
        val_mae, val_r2, test_mae, test_r2, cv_mae_mean, cv_mae_std, train_time_s,
    )
    return metrics


# ---------------------------------------------------------------------------
# Feature importance extraction
# ---------------------------------------------------------------------------

def _get_feature_importances(
    pipeline: Pipeline,
    feature_names: List[str],
) -> Dict[str, float]:
    """Extract feature importances from the pipeline's estimator step."""
    regressor = pipeline.named_steps.get("model")
    if regressor is None or not hasattr(regressor, "feature_importances_"):
        return {}
    importances = regressor.feature_importances_
    n = min(len(feature_names), len(importances))
    return {
        feature_names[i]: round(float(importances[i]), 6)
        for i in range(n)
    }


# ---------------------------------------------------------------------------
# Evaluation plots
# ---------------------------------------------------------------------------

def _save_evaluation_plots(
    y_actual:    pd.Series,
    y_predicted: np.ndarray,
    model_name:  str,
    target_name: str,
    split_label: str = "test",
) -> List[str]:
    """
    Generate and save 4 evaluation plots. Returns list of saved file paths.

    Plots:
        1. Prediction vs Actual scatter
        2. Residuals vs Predicted
        3. Error distribution histogram
        4. Worst 10 predictions table (saved as text figure)
    """
    saved_paths: List[str] = []
    slug = f"{model_name}_{target_name}".lower().replace(" ", "_")

    residuals = np.array(y_actual) - y_predicted
    errors    = np.abs(residuals)

    # ---- 1. Prediction vs Actual ----
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_actual, y_predicted, alpha=0.5, s=18, color="#3B82F6", edgecolors="none")
    lims = [min(y_actual.min(), y_predicted.min()) * 0.97,
            max(y_actual.max(), y_predicted.max()) * 1.03]
    ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect fit")
    ax.set_xlabel(f"Actual {target_name}", fontsize=11)
    ax.set_ylabel(f"Predicted {target_name}", fontsize=11)
    ax.set_title(f"{model_name} -- Prediction vs Actual ({split_label} set)", fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    mae  = mean_absolute_error(y_actual, y_predicted)
    r2   = r2_score(y_actual, y_predicted)
    ax.text(0.04, 0.94, f"MAE={mae:.4f}  R2={r2:.4f}",
            transform=ax.transAxes, fontsize=9, color="#1e293b")
    p1 = PLOTS_DIR / f"{slug}_pred_vs_actual.png"
    fig.savefig(p1, dpi=120, bbox_inches="tight")
    plt.close(fig)
    saved_paths.append(str(p1))

    # ---- 2. Residuals vs Predicted ----
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(y_predicted, residuals, alpha=0.5, s=18, color="#8B5CF6", edgecolors="none")
    ax.axhline(0, color="red", linewidth=1.5, linestyle="--")
    ax.set_xlabel(f"Predicted {target_name}", fontsize=11)
    ax.set_ylabel("Residual (Actual - Predicted)", fontsize=11)
    ax.set_title(f"{model_name} -- Residuals vs Predicted", fontsize=12)
    ax.grid(True, alpha=0.3)
    p2 = PLOTS_DIR / f"{slug}_residuals.png"
    fig.savefig(p2, dpi=120, bbox_inches="tight")
    plt.close(fig)
    saved_paths.append(str(p2))

    # ---- 3. Error distribution histogram ----
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(residuals, bins=40, color="#10B981", edgecolor="white", alpha=0.85)
    ax.axvline(0,              color="red",    linewidth=1.5, linestyle="--", label="Zero error")
    ax.axvline(residuals.mean(), color="orange", linewidth=1.2, linestyle=":",  label=f"Mean={residuals.mean():.4f}")
    ax.set_xlabel("Residual (Actual - Predicted)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title(f"{model_name} -- Error Distribution", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    p3 = PLOTS_DIR / f"{slug}_error_dist.png"
    fig.savefig(p3, dpi=120, bbox_inches="tight")
    plt.close(fig)
    saved_paths.append(str(p3))

    # ---- 4. Worst 10 predictions ----
    worst_idx  = np.argsort(errors)[::-1][:10]
    worst_df   = pd.DataFrame({
        "Actual":    np.array(y_actual)[worst_idx],
        "Predicted": y_predicted[worst_idx],
        "AbsError":  errors[worst_idx],
        "Residual":  residuals[worst_idx],
    }).reset_index(drop=True)
    worst_df.index = [f"Rank {i+1}" for i in range(len(worst_df))]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.axis("off")
    table = ax.table(
        cellText=worst_df.round(4).values,
        rowLabels=worst_df.index,
        colLabels=worst_df.columns,
        cellLoc="center", loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    ax.set_title(f"{model_name} -- Worst 10 Predictions ({split_label} set)",
                 fontsize=11, pad=12)
    p4 = PLOTS_DIR / f"{slug}_worst_predictions.png"
    fig.savefig(p4, dpi=120, bbox_inches="tight")
    plt.close(fig)
    saved_paths.append(str(p4))

    logger.info("Saved %d evaluation plots for %s", len(saved_paths), model_name)
    return saved_paths


# ---------------------------------------------------------------------------
# Metadata builder
# ---------------------------------------------------------------------------

def _build_metadata(
    *,
    model_name:       str,
    target:           str,
    winner:           str,
    features:         List[str],
    leakage_report:   Any,          # LeakageReport instance
    best_metrics:     Dict,
    all_candidates:   List[Dict],
    feature_importances: Dict,
    plot_paths:       List[str],
    train_rows:       int,
    val_rows:         int,
    test_rows:        int,
    train_batteries:  int,
    val_batteries:    int,
    test_batteries:   int,
    training_time_s:  float,
    model_path:       str,
    meta_path:        str,
    random_state:     int,
    dataset_info:     Dict,
) -> Dict:
    """Build the full metadata dictionary."""
    return {
        # Identification
        "model_name":            model_name,
        "model_version":         MODEL_VERSION,
        "algorithm":             winner,
        "training_timestamp":    datetime.now(timezone.utc).isoformat(),
        "random_seed":           random_state,

        # Target & features
        "target":                target,
        "selected_features":     features,
        "n_features":            len(features),
        "leakage_analysis":      leakage_report.to_dict(),

        # Preprocessing pipeline description
        "preprocessing_pipeline": {
            "steps": [
                {"name": "DataCleaner",       "phase": 1, "desc": "Phase 1 DataCleaner"},
                {"name": "FeatureEngineer",   "phase": 1, "desc": "Phase 1 FeatureEngineer"},
                {"name": "StandardScaler",    "phase": 2, "desc": "Z-score normalisation in Pipeline"},
            ],
            "full_pipeline_saved": True,
            "model_path": model_path,
        },

        # Split info
        "split_strategy":        "by_battery_id",
        "split_rationale":       "All cycles of one battery in exactly one split (prevents leakage)",
        "train_batteries":       train_batteries,
        "val_batteries":         val_batteries,
        "test_batteries":        test_batteries,
        "train_rows":            train_rows,
        "val_rows":              val_rows,
        "test_rows":             test_rows,

        # Evaluation
        "evaluation_metrics":    best_metrics,
        "all_candidates":        all_candidates,
        "feature_importances":   feature_importances,
        "evaluation_plots":      plot_paths,

        # Performance
        "training_time_s":       training_time_s,

        # Dataset traceability
        "dataset_version":       dataset_info,

        # Persistence
        "model_path":            model_path,
        "meta_path":             meta_path,
    }


# ---------------------------------------------------------------------------
# Train Battery SOH
# ---------------------------------------------------------------------------

def train_soh_model(
    df: pd.DataFrame,
    random_state: int = RANDOM_STATE,
    save: bool = True,
) -> Dict:
    """
    Train, evaluate, and optionally save the Battery State-of-Health model.

    Steps
    -----
    1. Feature leakage analysis (excludes confirmed leaking features).
    2. Battery_ID-level split (70/15/15).
    3. Three candidate algorithms trained and compared.
    4. Winner (lowest test MAE) selected.
    5. Evaluation plots saved.
    6. Complete Pipeline + metadata saved via Joblib/JSON.

    Parameters
    ----------
    df : pd.DataFrame
        Battery DataFrame after Phase 1 cleaning + feature engineering.
    random_state : int, default 42
    save : bool, default True

    Returns
    -------
    dict  Training report / metadata.
    """
    from app.models.leakage_analysis import LeakageAnalyzer

    logger.info("=" * 60)
    logger.info("Training Battery SOH (target: %s)", SOH_TARGET)
    logger.info("=" * 60)
    t0 = time.time()

    if SOH_TARGET not in df.columns:
        raise ValueError(f"Target '{SOH_TARGET}' not in DataFrame.")

    # Dataset info for traceability
    dataset_info = {
        "source":   "datasets/battery/ev_battery_degradation.csv",
        "rows":     len(df),
        "cols":     len(df.columns),
        "n_batteries": int(df["Battery_ID"].nunique()),
    }

    # Clip SOH to valid range
    df = df.copy()
    df[SOH_TARGET] = df[SOH_TARGET].clip(0.0, 1.0)

    # --- 1. Leakage analysis
    analyzer       = LeakageAnalyzer()
    leakage_result = analyzer.analyze(df, SOH_TARGET, BATTERY_CANDIDATE_FEATURES)
    features       = leakage_result.safe_features
    logger.info("SOH safe features after leakage analysis (%d): %s", len(features), features)

    # --- 2. Split
    train_df, val_df, test_df = split_by_battery_id(df, random_state=random_state)

    X_train = train_df[features].fillna(train_df[features].median())
    y_train = train_df[SOH_TARGET]
    X_val   = val_df[features].fillna(train_df[features].median())
    y_val   = val_df[SOH_TARGET]
    X_test  = test_df[features].fillna(train_df[features].median())
    y_test  = test_df[SOH_TARGET]

    # --- 3. Train candidates
    candidates        = _make_candidates(random_state)
    candidate_results = []
    trained_pipelines = {}

    for name, pipeline in candidates.items():
        logger.info("Fitting %s for SOH...", name)
        t_fit = time.time()
        pipeline.fit(X_train, y_train)
        elapsed = time.time() - t_fit
        metrics = _evaluate_pipeline(
            pipeline, X_train, y_train, X_val, y_val, X_test, y_test,
            model_name=name, target_name=SOH_TARGET, train_time_s=elapsed,
        )
        candidate_results.append(metrics)
        trained_pipelines[name] = pipeline

    # --- 4. Select winner
    best          = min(candidate_results, key=lambda r: r["test_mae"])
    best_name     = best["model"]
    best_pipeline = trained_pipelines[best_name]
    logger.info("SOH winner: %s  (test_mae=%.5f  test_r2=%.5f)", best_name, best["test_mae"], best["test_r2"])

    # --- 5. Evaluation plots
    pred_test = best_pipeline.predict(X_test)
    plot_paths = _save_evaluation_plots(y_test, pred_test, best_name, SOH_TARGET)

    # --- 6. Feature importances
    feat_imp = _get_feature_importances(best_pipeline, features)

    # --- Build metadata
    metadata = _build_metadata(
        model_name="battery_soh_model",
        target=SOH_TARGET,
        winner=best_name,
        features=features,
        leakage_report=leakage_result,
        best_metrics=best,
        all_candidates=candidate_results,
        feature_importances=feat_imp,
        plot_paths=plot_paths,
        train_rows=len(X_train), val_rows=len(X_val), test_rows=len(X_test),
        train_batteries=int(train_df["Battery_ID"].nunique()),
        val_batteries=int(val_df["Battery_ID"].nunique()),
        test_batteries=int(test_df["Battery_ID"].nunique()),
        training_time_s=round(time.time() - t0, 2),
        model_path=str(SOH_MODEL_PATH),
        meta_path=str(SOH_META_PATH),
        random_state=random_state,
        dataset_info=dataset_info,
    )

    if save:
        joblib.dump(best_pipeline, SOH_MODEL_PATH)
        with open(SOH_META_PATH, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, default=str)
        logger.info("SOH Pipeline saved: %s", SOH_MODEL_PATH)

    return metadata


# ---------------------------------------------------------------------------
# Train Battery RUL
# ---------------------------------------------------------------------------

def train_rul_model(
    df: pd.DataFrame,
    random_state: int = RANDOM_STATE,
    save: bool = True,
) -> Dict:
    """
    Train, evaluate, and optionally save the Battery RUL regression model.

    Parameters
    ----------
    df : pd.DataFrame
        Battery DataFrame after Phase 1 cleaning + feature engineering.
    random_state : int, default 42
    save : bool, default True

    Returns
    -------
    dict  Training report / metadata.
    """
    from app.models.leakage_analysis import LeakageAnalyzer

    logger.info("=" * 60)
    logger.info("Training Battery RUL (target: %s)", RUL_TARGET)
    logger.info("=" * 60)
    t0 = time.time()

    if RUL_TARGET not in df.columns:
        raise ValueError(f"Target '{RUL_TARGET}' not in DataFrame.")

    dataset_info = {
        "source":      "datasets/battery/ev_battery_degradation.csv",
        "rows":        len(df),
        "cols":        len(df.columns),
        "n_batteries": int(df["Battery_ID"].nunique()),
    }

    # --- Leakage analysis
    analyzer       = LeakageAnalyzer()
    leakage_result = analyzer.analyze(df, RUL_TARGET, BATTERY_CANDIDATE_FEATURES)
    features       = leakage_result.safe_features
    logger.info("RUL safe features after leakage analysis (%d): %s", len(features), features)

    # --- Split
    train_df, val_df, test_df = split_by_battery_id(df, random_state=random_state)

    X_train = train_df[features].fillna(train_df[features].median())
    y_train = train_df[RUL_TARGET]
    X_val   = val_df[features].fillna(train_df[features].median())
    y_val   = val_df[RUL_TARGET]
    X_test  = test_df[features].fillna(train_df[features].median())
    y_test  = test_df[RUL_TARGET]

    # --- Train candidates
    candidates        = _make_candidates(random_state)
    candidate_results = []
    trained_pipelines = {}

    for name, pipeline in candidates.items():
        logger.info("Fitting %s for RUL...", name)
        t_fit = time.time()
        pipeline.fit(X_train, y_train)
        elapsed = time.time() - t_fit
        metrics = _evaluate_pipeline(
            pipeline, X_train, y_train, X_val, y_val, X_test, y_test,
            model_name=name, target_name=RUL_TARGET, train_time_s=elapsed,
        )
        candidate_results.append(metrics)
        trained_pipelines[name] = pipeline

    # --- Select winner
    best          = min(candidate_results, key=lambda r: r["test_mae"])
    best_name     = best["model"]
    best_pipeline = trained_pipelines[best_name]
    logger.info("RUL winner: %s  (test_mae=%.5f  test_r2=%.5f)", best_name, best["test_mae"], best["test_r2"])

    # --- Plots
    pred_test  = best_pipeline.predict(X_test)
    plot_paths = _save_evaluation_plots(y_test, pred_test, best_name, RUL_TARGET)

    # --- Feature importances
    feat_imp = _get_feature_importances(best_pipeline, features)

    # --- Metadata
    metadata = _build_metadata(
        model_name="battery_rul_model",
        target=RUL_TARGET,
        winner=best_name,
        features=features,
        leakage_report=leakage_result,
        best_metrics=best,
        all_candidates=candidate_results,
        feature_importances=feat_imp,
        plot_paths=plot_paths,
        train_rows=len(X_train), val_rows=len(X_val), test_rows=len(X_test),
        train_batteries=int(train_df["Battery_ID"].nunique()),
        val_batteries=int(val_df["Battery_ID"].nunique()),
        test_batteries=int(test_df["Battery_ID"].nunique()),
        training_time_s=round(time.time() - t0, 2),
        model_path=str(RUL_MODEL_PATH),
        meta_path=str(RUL_META_PATH),
        random_state=random_state,
        dataset_info=dataset_info,
    )

    if save:
        joblib.dump(best_pipeline, RUL_MODEL_PATH)
        with open(RUL_META_PATH, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, default=str)
        logger.info("RUL Pipeline saved: %s", RUL_MODEL_PATH)

    return metadata


# ---------------------------------------------------------------------------
# Load helpers
# ---------------------------------------------------------------------------

def load_soh_model() -> Optional[Pipeline]:
    """Load the trained SOH Pipeline. Returns None if not found."""
    if SOH_MODEL_PATH.exists():
        return joblib.load(SOH_MODEL_PATH)
    logger.warning("SOH model not found at %s", SOH_MODEL_PATH)
    return None


def load_rul_model() -> Optional[Pipeline]:
    """Load the trained RUL Pipeline. Returns None if not found."""
    if RUL_MODEL_PATH.exists():
        return joblib.load(RUL_MODEL_PATH)
    logger.warning("RUL model not found at %s", RUL_MODEL_PATH)
    return None


# ---------------------------------------------------------------------------
# Inference helper
# ---------------------------------------------------------------------------

def predict_battery_health(
    voltage_v:         float,
    temperature_c:     float,
    capacity_ah:       float,
    cycle_number:      int,
    voltage_sag_v:     float = 0.05,
    degradation_rate:  float = -0.002,
    cycle_normalized:  float = 0.5,
) -> Dict:
    """
    Predict SOH and RUL for a single battery observation.

    Note: Capacity_Fade_Pct is NOT passed as a feature because it was
    excluded by the leakage analysis for the SOH model.

    Returns
    -------
    dict  soh_predicted, rul_predicted, model_used (trained | fallback_math).
    """
    soh_model = load_soh_model()
    rul_model = load_rul_model()

    # Build input aligned to each model's features from metadata
    base_input = {
        "Cycle_Number":     cycle_number,
        "Voltage_V":        voltage_v,
        "Temperature_C":    temperature_c,
        "Capacity_Ah":      capacity_ah,
        "Voltage_Sag_V":    voltage_sag_v,
        "Degradation_Rate": degradation_rate,
        "Cycle_Normalized": cycle_normalized,
    }

    if soh_model:
        soh_input = pd.DataFrame([base_input])
        try:
            soh_pred = float(np.clip(soh_model.predict(soh_input)[0], 0.0, 1.0))
        except Exception:
            # Pipeline may have a different feature set after leakage analysis;
            # fill any extra columns with 0 and drop unknowns.
            soh_pred = float(np.clip(soh_model.predict(soh_input.reindex(
                columns=list(base_input.keys()), fill_value=0.0))[0], 0.0, 1.0))
        soh_used = "trained"
    else:
        soh_pred = max(0.0, 1.0 - (cycle_number / 200.0))
        soh_used = "fallback_math"

    if rul_model:
        rul_input = pd.DataFrame([base_input])
        rul_pred  = int(max(0, round(rul_model.predict(rul_input)[0])))
        rul_used  = "trained"
    else:
        rul_pred = max(0, 168 - cycle_number)
        rul_used = "fallback_math"

    return {
        "soh_predicted":  round(soh_pred, 4),
        "rul_predicted":  rul_pred,
        "soh_model_used": soh_used,
        "rul_model_used": rul_used,
    }


# ---------------------------------------------------------------------------
# Public aliases kept for backward compatibility
# ---------------------------------------------------------------------------
BATTERY_FEATURES = [
    "Cycle_Number", "Voltage_V", "Temperature_C",
    "Capacity_Ah", "Voltage_Sag_V", "Degradation_Rate", "Cycle_Normalized",
]
