"""
VoltIQ -- app/models/fleet_models.py
======================================
Phase 2: Fleet Electrification Readiness ML Model

Implements a production sklearn Pipeline for:

    FleetReadinessPipeline -- predicts EV_Readiness_Score (regression)
    Target range: [0.07, 0.69]

Design principles
-----------------
* Complete sklearn Pipeline (ColumnTransformer + estimator) is saved.
* Feature leakage analysis is run BEFORE training.
* Stratified 70/15/15 split by Vehicle_Type to preserve class distribution.
* Three candidate algorithms compared; winner selected by lowest test MAE.
* Evaluation plots saved to reports/ml/plots/.
* Expanded metadata JSON written with training timestamp, random seed, etc.
* All RMSE computations use np.sqrt(mean_squared_error(...)) for
  compatibility with scikit-learn 1.4+ which removed the squared=False
  parameter.
* Logging uses ASCII-only characters for Windows CP1252 compatibility.
* random_state=42 used everywhere for complete reproducibility.
"""

from __future__ import annotations

import json
import logging
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_ROOT              = Path(__file__).resolve().parents[2]
SAVED_MODELS_FLEET = _ROOT / "saved_models" / "fleet"
PLOTS_DIR          = _ROOT / "reports" / "ml" / "plots"

for _d in (SAVED_MODELS_FLEET, PLOTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

FLEET_MODEL_PATH = SAVED_MODELS_FLEET / "fleet_readiness_model.pkl"
FLEET_META_PATH  = SAVED_MODELS_FLEET / "fleet_readiness_metadata.json"

MODEL_VERSION = "2.0.0"
RANDOM_STATE  = 42

# ---------------------------------------------------------------------------
# Feature configuration
# ---------------------------------------------------------------------------

#: All numeric candidate features BEFORE leakage analysis
FLEET_NUMERIC_FEATURES: List[str] = [
    "Vehicle_Age_Years",
    "Usage_Hours",
    "Load_Capacity",
    "Actual_Load",
    "Load_Utilization_Pct",
    "Fuel_Consumption",
    "Fuel_per_Hour",
    "Health_Score",
    "Maintenance_Cost",
    "Days_Since_Last_Maintenance",
    "Failure_History",
    "Anomalies_Detected",
    "Diagnostic_Trouble_Code_Count",
    "Predictive_Score",
    "PCR",
    "UIR",
    "TPI",
    "MBF",
    "ADS",
    "OHI",
    "CMES",
    "UER",
    # Phase 1 derived -- potentially leaking; leakage analysis will decide
    "EV_Priority_Score",
]

#: Categorical features to one-hot encode
FLEET_CATEGORICAL_FEATURES: List[str] = [
    "Vehicle_Type",
    "Route_Info",
    "Road_Conditions",
    "Weather_Conditions",
    "Brake_Condition",
]

FLEET_TARGET = "EV_Readiness_Score"


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def _filter_available(
    df: pd.DataFrame,
    columns: List[str],
    label: str,
) -> List[str]:
    """Return only columns present in df; log any missing."""
    available = [c for c in columns if c in df.columns]
    missing   = set(columns) - set(available)
    if missing:
        logger.warning("[%s] Columns absent from DataFrame: %s", label, sorted(missing))
    return available


def prepare_fleet_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Select, clean, and validate columns needed for fleet ML training.

    - Drops rows with null target.
    - Fills numeric nulls with column median.
    - Fills categorical nulls with 'Unknown'.
    - Returns a copy; original is never modified.
    """
    df = df.copy()

    if FLEET_TARGET not in df.columns:
        raise ValueError(f"Target '{FLEET_TARGET}' not found in DataFrame.")

    all_cols  = FLEET_NUMERIC_FEATURES + FLEET_CATEGORICAL_FEATURES + [FLEET_TARGET]
    keep_cols = [c for c in all_cols if c in df.columns]

    # Preserve Vehicle_Type for stratified split even if not in feature list
    if "Vehicle_Type" in df.columns and "Vehicle_Type" not in keep_cols:
        keep_cols.append("Vehicle_Type")

    df = df[keep_cols].copy()
    df = df.dropna(subset=[FLEET_TARGET])

    num_cols = [c for c in FLEET_NUMERIC_FEATURES if c in df.columns]
    cat_cols = [c for c in FLEET_CATEGORICAL_FEATURES if c in df.columns]

    for col in num_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    for col in cat_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna("Unknown")

    return df


# ---------------------------------------------------------------------------
# Pipeline factory
# ---------------------------------------------------------------------------

def _make_preprocessor(
    numeric_cols:     List[str],
    categorical_cols: List[str],
) -> ColumnTransformer:
    """
    Build a ColumnTransformer:
      - numeric  -> StandardScaler
      - categorical -> OneHotEncoder(handle_unknown='ignore')
    """
    transformers: list = []
    if numeric_cols:
        transformers.append(("num", StandardScaler(), numeric_cols))
    if categorical_cols:
        transformers.append((
            "cat",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            categorical_cols,
        ))
    return ColumnTransformer(transformers=transformers, remainder="drop")


def _make_fleet_candidates(
    preprocessor:  ColumnTransformer,
    random_state:  int = RANDOM_STATE,
) -> Dict[str, Pipeline]:
    """Return three candidate full Pipelines (preprocessor + estimator)."""
    return {
        "LinearRegression": Pipeline([
            ("prep",  preprocessor),
            ("model", LinearRegression()),
        ]),
        "RandomForest": Pipeline([
            ("prep",  preprocessor),
            ("model", RandomForestRegressor(
                n_estimators=200,
                max_depth=15,
                min_samples_leaf=5,
                random_state=random_state,
                n_jobs=-1,
            )),
        ]),
        "GradientBoosting": Pipeline([
            ("prep",  preprocessor),
            ("model", GradientBoostingRegressor(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                random_state=random_state,
            )),
        ]),
    }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _evaluate_fleet(
    pipeline:    Pipeline,
    X_train:     pd.DataFrame,
    y_train:     pd.Series,
    X_val:       pd.DataFrame,
    y_val:       pd.Series,
    X_test:      pd.DataFrame,
    y_test:      pd.Series,
    model_name:  str,
    cv_folds:    int = 5,
    train_time_s: float = 0.0,
) -> Dict[str, Any]:
    """
    Compute MAE, RMSE (via np.sqrt(MSE)), R2 on val + test;
    run cross-validation on a subsample of training data.

    RMSE is computed as np.sqrt(mean_squared_error(...)) to maintain
    compatibility with scikit-learn >= 1.4 where squared=False was removed.
    """
    pred_val  = pipeline.predict(X_val)
    pred_test = pipeline.predict(X_test)

    val_mae  = mean_absolute_error(y_val,  pred_val)
    val_rmse = float(np.sqrt(mean_squared_error(y_val,  pred_val)))
    val_r2   = r2_score(y_val,  pred_val)

    test_mae  = mean_absolute_error(y_test,  pred_test)
    test_rmse = float(np.sqrt(mean_squared_error(y_test,  pred_test)))
    test_r2   = r2_score(y_test,  pred_test)

    # CV on a 50k-row subsample for speed (175k train rows)
    subsample_n = min(len(X_train), 50_000)
    rng         = np.random.RandomState(RANDOM_STATE)
    idx         = rng.choice(len(X_train), subsample_n, replace=False)

    cv_scores   = cross_val_score(
        pipeline,
        X_train.iloc[idx], y_train.iloc[idx],
        cv=cv_folds, scoring="neg_mean_absolute_error", n_jobs=-1,
    )
    cv_mae_mean = float(-cv_scores.mean())
    cv_mae_std  = float(cv_scores.std())

    metrics = {
        "model":         model_name,
        "target":        FLEET_TARGET,
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
        "[%s | %s] val_mae=%.5f val_r2=%.5f | test_mae=%.5f test_r2=%.5f | "
        "cv=%.5f+/-%.5f | fit=%.1fs",
        model_name, FLEET_TARGET,
        val_mae, val_r2, test_mae, test_r2, cv_mae_mean, cv_mae_std, train_time_s,
    )
    return metrics


# ---------------------------------------------------------------------------
# Feature importance extraction
# ---------------------------------------------------------------------------

def _get_feature_importances(
    pipeline:         Pipeline,
    numeric_cols:     List[str],
    categorical_cols: List[str],
) -> Dict[str, float]:
    """
    Extract feature importances for tree-based estimators.
    Reconstructs feature names from the ColumnTransformer OHE step.
    """
    regressor = pipeline.named_steps.get("model")
    if regressor is None or not hasattr(regressor, "feature_importances_"):
        return {}

    try:
        prep = pipeline.named_steps["prep"]
        ohe_names: List[str] = []
        if categorical_cols:
            ohe = prep.named_transformers_.get("cat")
            if ohe is not None and hasattr(ohe, "get_feature_names_out"):
                ohe_names = list(ohe.get_feature_names_out(categorical_cols))
        all_names  = numeric_cols + ohe_names
        importances = regressor.feature_importances_
        n = min(len(all_names), len(importances))
        raw = {all_names[i]: float(importances[i]) for i in range(n)}
        # Return top 20, sorted descending
        return {
            k: round(v, 6)
            for k, v in sorted(raw.items(), key=lambda x: x[1], reverse=True)[:20]
        }
    except Exception as exc:
        logger.warning("Feature importance extraction failed: %s", exc)
        return {}


# ---------------------------------------------------------------------------
# Evaluation plots
# ---------------------------------------------------------------------------

def _save_evaluation_plots(
    y_actual:    pd.Series,
    y_predicted: np.ndarray,
    model_name:  str,
    target_name: str = FLEET_TARGET,
    split_label: str = "test",
) -> List[str]:
    """
    Generate and save 4 evaluation plots to reports/ml/plots/.
    Uses ASCII-safe labels throughout (no special Unicode chars).

    Returns list of saved file paths.
    """
    saved_paths: List[str] = []
    slug = f"fleet_{model_name}_{target_name}".lower().replace(" ", "_")

    y_arr     = np.array(y_actual, dtype=float)
    residuals = y_arr - y_predicted
    errors    = np.abs(residuals)
    mae       = mean_absolute_error(y_arr, y_predicted)
    r2        = r2_score(y_arr, y_predicted)

    # ---- 1. Prediction vs Actual ----
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_arr, y_predicted, alpha=0.3, s=10, color="#3B82F6", edgecolors="none")
    lo = min(y_arr.min(), y_predicted.min()) * 0.97
    hi = max(y_arr.max(), y_predicted.max()) * 1.03
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.5, label="Perfect fit")
    ax.set_xlabel(f"Actual {target_name}", fontsize=11)
    ax.set_ylabel(f"Predicted {target_name}", fontsize=11)
    ax.set_title(f"{model_name} -- Prediction vs Actual ({split_label} set)", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.text(0.04, 0.94, f"MAE={mae:.5f}  R2={r2:.5f}",
            transform=ax.transAxes, fontsize=8, color="#1e293b")
    p1 = PLOTS_DIR / f"{slug}_pred_vs_actual.png"
    fig.savefig(p1, dpi=120, bbox_inches="tight")
    plt.close(fig)
    saved_paths.append(str(p1))

    # ---- 2. Residuals vs Predicted ----
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(y_predicted, residuals, alpha=0.3, s=10, color="#8B5CF6", edgecolors="none")
    ax.axhline(0, color="red", linewidth=1.5, linestyle="--")
    ax.set_xlabel(f"Predicted {target_name}", fontsize=11)
    ax.set_ylabel("Residual (Actual - Predicted)", fontsize=11)
    ax.set_title(f"{model_name} -- Residuals vs Predicted", fontsize=11)
    ax.grid(True, alpha=0.3)
    p2 = PLOTS_DIR / f"{slug}_residuals.png"
    fig.savefig(p2, dpi=120, bbox_inches="tight")
    plt.close(fig)
    saved_paths.append(str(p2))

    # ---- 3. Error distribution ----
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(residuals, bins=60, color="#10B981", edgecolor="white", alpha=0.85)
    ax.axvline(0,               color="red",    linewidth=1.5, linestyle="--", label="Zero error")
    ax.axvline(float(residuals.mean()), color="orange", linewidth=1.2, linestyle=":",
               label=f"Mean={residuals.mean():.5f}")
    ax.set_xlabel("Residual (Actual - Predicted)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title(f"{model_name} -- Error Distribution", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    p3 = PLOTS_DIR / f"{slug}_error_dist.png"
    fig.savefig(p3, dpi=120, bbox_inches="tight")
    plt.close(fig)
    saved_paths.append(str(p3))

    # ---- 4. Worst 10 predictions ----
    worst_idx = np.argsort(errors)[::-1][:10]
    worst_data = [
        [f"{y_arr[i]:.4f}", f"{y_predicted[i]:.4f}", f"{errors[i]:.4f}", f"{residuals[i]:.4f}"]
        for i in worst_idx
    ]
    rank_labels = [f"Rank {i+1}" for i in range(len(worst_data))]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.axis("off")
    tbl = ax.table(
        cellText=worst_data,
        rowLabels=rank_labels,
        colLabels=["Actual", "Predicted", "AbsError", "Residual"],
        cellLoc="center", loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.2, 1.5)
    ax.set_title(f"{model_name} -- Worst 10 Predictions ({split_label} set)",
                 fontsize=10, pad=12)
    p4 = PLOTS_DIR / f"{slug}_worst_predictions.png"
    fig.savefig(p4, dpi=120, bbox_inches="tight")
    plt.close(fig)
    saved_paths.append(str(p4))

    logger.info("Saved %d evaluation plots for fleet %s", len(saved_paths), model_name)
    return saved_paths


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train_fleet_readiness_model(
    df:           pd.DataFrame,
    random_state: int = RANDOM_STATE,
    save:         bool = True,
) -> Dict:
    """
    Train, evaluate, and optionally save the Fleet EV Readiness model.

    Steps
    -----
    1. Feature leakage analysis (excludes target-derived columns).
    2. Prepare clean feature DataFrame.
    3. Stratified 70/15/15 split by Vehicle_Type.
    4. Three candidate Pipelines trained and compared.
    5. Winner selected by lowest test MAE.
    6. Evaluation plots saved.
    7. Complete Pipeline + expanded metadata saved.

    Parameters
    ----------
    df : pd.DataFrame
        Fleet readiness DataFrame after Phase 1 cleaning + feature engineering.
    random_state : int, default 42
    save : bool, default True

    Returns
    -------
    dict  Full training report / metadata.
    """
    from app.models.leakage_analysis import LeakageAnalyzer

    logger.info("=" * 60)
    logger.info("Training Fleet Readiness model  (target: %s)", FLEET_TARGET)
    logger.info("=" * 60)
    t0 = time.time()

    if FLEET_TARGET not in df.columns:
        raise ValueError(f"Target '{FLEET_TARGET}' not found in DataFrame.")

    dataset_info = {
        "source": "datasets/fleet/fleet_electrification_readiness.csv",
        "rows":   len(df),
        "cols":   len(df.columns),
    }

    # --- 1. Leakage analysis on numeric candidates
    all_numeric_candidates = _filter_available(df, FLEET_NUMERIC_FEATURES, "numeric_candidates")
    analyzer       = LeakageAnalyzer(high_corr_threshold=0.98, moderate_corr_threshold=0.90)
    leakage_result = analyzer.analyze(df, FLEET_TARGET, all_numeric_candidates)
    safe_numeric   = leakage_result.safe_features

    # Categorical features are not subject to Pearson correlation leakage analysis
    safe_cat = _filter_available(df, FLEET_CATEGORICAL_FEATURES, "categorical")

    logger.info(
        "Fleet safe numeric features after leakage analysis (%d): %s",
        len(safe_numeric), safe_numeric,
    )
    if leakage_result.confirmed_leakage:
        logger.info("Fleet CONFIRMED LEAKAGE excluded: %s", leakage_result.confirmed_leakage)
    if leakage_result.high_correlation_risk:
        logger.warning("Fleet HIGH CORR RISK excluded: %s", leakage_result.high_correlation_risk)

    # --- 2. Prepare data
    # Build a trimmed DataFrame with only safe features + categoricals + target
    keep_cols = safe_numeric + safe_cat + [FLEET_TARGET]
    if "Vehicle_Type" not in keep_cols and "Vehicle_Type" in df.columns:
        keep_cols.append("Vehicle_Type")   # needed for stratified split

    df_prep = df[[c for c in keep_cols if c in df.columns]].copy()
    df_prep = df_prep.dropna(subset=[FLEET_TARGET])

    for col in safe_numeric:
        if col in df_prep.columns and df_prep[col].isnull().any():
            df_prep[col] = df_prep[col].fillna(df_prep[col].median())
    for col in safe_cat:
        if col in df_prep.columns and df_prep[col].isnull().any():
            df_prep[col] = df_prep[col].fillna("Unknown")

    feature_cols = [c for c in (safe_numeric + safe_cat) if c in df_prep.columns]
    logger.info("Prepared fleet data: %d rows x %d feature cols", len(df_prep), len(feature_cols))

    # --- 3. Stratified 70/15/15 split
    strat_col = "Vehicle_Type" if "Vehicle_Type" in df_prep.columns else None
    X_all = df_prep[feature_cols]
    y_all = df_prep[FLEET_TARGET]

    try:
        X_temp, X_test, y_temp, y_test = train_test_split(
            X_all, y_all,
            test_size=0.15,
            random_state=random_state,
            stratify=df_prep[strat_col] if strat_col else None,
        )
        val_frac = 0.15 / 0.85   # 15% of total
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_frac,
            random_state=random_state,
            stratify=df_prep.loc[X_temp.index, strat_col] if strat_col else None,
        )
    except Exception as exc:
        logger.warning("Stratified split failed (%s); falling back to random split.", exc)
        X_temp, X_test, y_temp, y_test = train_test_split(
            X_all, y_all, test_size=0.15, random_state=random_state
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.15 / 0.85, random_state=random_state
        )

    logger.info(
        "Fleet split: train=%d | val=%d | test=%d",
        len(X_train), len(X_val), len(X_test),
    )

    # --- 4. Build preprocessor and candidate pipelines
    actual_num = [c for c in safe_numeric  if c in feature_cols]
    actual_cat = [c for c in safe_cat      if c in feature_cols]
    preprocessor = _make_preprocessor(actual_num, actual_cat)
    candidates   = _make_fleet_candidates(preprocessor, random_state)

    candidate_results: List[Dict] = []
    trained_pipelines: Dict[str, Pipeline] = {}

    for name, pipeline in candidates.items():
        logger.info("Fitting fleet candidate: %s...", name)
        t_fit = time.time()
        try:
            pipeline.fit(X_train, y_train)
            elapsed = time.time() - t_fit
            metrics = _evaluate_fleet(
                pipeline, X_train, y_train, X_val, y_val, X_test, y_test,
                model_name=name, train_time_s=elapsed,
            )
            candidate_results.append(metrics)
            trained_pipelines[name] = pipeline
        except Exception as exc:
            logger.error("Candidate %s failed: %s", name, exc, exc_info=True)

    if not candidate_results:
        raise RuntimeError("All fleet model candidates failed to train.")

    # --- 5. Select winner
    best          = min(candidate_results, key=lambda r: r["test_mae"])
    best_name     = best["model"]
    best_pipeline = trained_pipelines[best_name]
    logger.info(
        "Fleet winner: %s  (test_mae=%.5f  test_r2=%.5f)",
        best_name, best["test_mae"], best["test_r2"],
    )

    # --- 6. Evaluation plots
    pred_test  = best_pipeline.predict(X_test)
    plot_paths: List[str] = []
    try:
        plot_paths = _save_evaluation_plots(y_test, pred_test, best_name)
    except Exception as exc:
        logger.warning("Plot generation failed: %s", exc)

    # --- 7. Feature importances
    feat_imp: Dict[str, float] = {}
    try:
        feat_imp = _get_feature_importances(best_pipeline, actual_num, actual_cat)
    except Exception as exc:
        logger.warning("Feature importance extraction failed: %s", exc)

    # --- 8. Build metadata
    ts = datetime.now(timezone.utc).isoformat()
    metadata: Dict[str, Any] = {
        # Identification
        "model_name":            "fleet_readiness_model",
        "model_version":         MODEL_VERSION,
        "algorithm":             best_name,
        "training_timestamp":    ts,
        "random_seed":           random_state,

        # Target & features
        "target":                FLEET_TARGET,
        "selected_numeric_features":     actual_num,
        "selected_categorical_features": actual_cat,
        "n_features_total":      len(actual_num) + len(actual_cat),
        "leakage_analysis":      leakage_result.to_dict(),

        # Preprocessing
        "preprocessing_pipeline": {
            "steps": [
                {"name": "DataCleaner",      "phase": 1, "desc": "Phase 1 cleaning"},
                {"name": "FeatureEngineer",  "phase": 1, "desc": "Phase 1 feature engineering"},
                {"name": "StandardScaler",   "phase": 2, "desc": "Numeric scaling in Pipeline"},
                {"name": "OneHotEncoder",    "phase": 2, "desc": "Categorical OHE in Pipeline"},
            ],
            "full_pipeline_saved": True,
            "model_path": str(FLEET_MODEL_PATH),
        },

        # Split
        "split_strategy":  "stratified_random_by_vehicle_type",
        "train_rows":      len(X_train),
        "val_rows":        len(X_val),
        "test_rows":       len(X_test),

        # Evaluation
        "evaluation_metrics":        best,
        "all_candidates":            candidate_results,
        "top20_feature_importances": feat_imp,
        "evaluation_plots":          plot_paths,

        # Dataset traceability
        "dataset_version": dataset_info,

        # Performance
        "training_time_s":  round(time.time() - t0, 2),
        "model_path":       str(FLEET_MODEL_PATH),
        "meta_path":        str(FLEET_META_PATH),
    }

    if save:
        try:
            joblib.dump(best_pipeline, FLEET_MODEL_PATH)
            logger.info("Fleet Pipeline saved: %s", FLEET_MODEL_PATH)
        except Exception as exc:
            logger.error("Failed to save fleet model: %s", exc, exc_info=True)
            raise

        try:
            with open(FLEET_META_PATH, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, default=str)
            logger.info("Fleet metadata saved: %s", FLEET_META_PATH)
        except Exception as exc:
            logger.error("Failed to save fleet metadata: %s", exc, exc_info=True)

    return metadata


# ---------------------------------------------------------------------------
# Load helper
# ---------------------------------------------------------------------------

def load_fleet_readiness_model() -> Optional[Pipeline]:
    """Load the saved fleet readiness Pipeline. Returns None if not found."""
    if FLEET_MODEL_PATH.exists():
        try:
            return joblib.load(FLEET_MODEL_PATH)
        except Exception as exc:
            logger.error("Failed to load fleet model: %s", exc)
            return None
    logger.warning("Fleet readiness model not found at %s", FLEET_MODEL_PATH)
    return None


# ---------------------------------------------------------------------------
# Inference helper
# ---------------------------------------------------------------------------

def predict_ev_readiness(
    vehicle_age_years:  int,
    usage_hours:        float,
    fuel_consumption:   float,
    health_score:       float,
    vehicle_type:       str = "Light Truck",
    route_info:         str = "Urban Delivery",
    road_conditions:    str = "Smooth",
    weather_conditions: str = "Clear",
    brake_condition:    str = "Good",
    **kwargs: Any,
) -> Dict:
    """
    Predict EV Readiness Score for a single fleet vehicle.

    Extra numeric features default to 0.0 unless supplied via kwargs.
    The function builds a DataFrame matching the trained model's feature set.
    """
    fleet_model = load_fleet_readiness_model()

    # Build row with sensible defaults
    row: Dict[str, Any] = {col: 0.0 for col in FLEET_NUMERIC_FEATURES}
    row.update({
        "Vehicle_Age_Years":   vehicle_age_years,
        "Usage_Hours":         usage_hours,
        "Fuel_Consumption":    fuel_consumption,
        "Health_Score":        health_score,
        "Vehicle_Type":        vehicle_type,
        "Route_Info":          route_info,
        "Road_Conditions":     road_conditions,
        "Weather_Conditions":  weather_conditions,
        "Brake_Condition":     brake_condition,
    })
    row.update(kwargs)

    all_feature_cols = FLEET_NUMERIC_FEATURES + FLEET_CATEGORICAL_FEATURES
    input_df = pd.DataFrame([{c: row[c] for c in all_feature_cols if c in row}])

    if fleet_model:
        try:
            score     = float(np.clip(fleet_model.predict(input_df)[0], 0.0, 1.0))
            model_used = "trained"
        except Exception as exc:
            logger.warning("Fleet model predict failed: %s -- using fallback", exc)
            fleet_model = None

    if not fleet_model:
        age_factor  = max(0.0, 1.0 - vehicle_age_years / 20.0)
        fuel_factor = max(0.0, 1.0 - fuel_consumption / 25.0)
        score       = round(age_factor * 0.4 + fuel_factor * 0.3 +
                            (health_score / 100.0) * 0.3, 4)
        model_used  = "fallback_math"

    if score >= 0.6:
        category = "High Readiness"
    elif score >= 0.4:
        category = "Moderate Readiness"
    elif score >= 0.2:
        category = "Low Readiness"
    else:
        category = "Not Ready"

    return {
        "ev_readiness_score": round(score, 4),
        "readiness_category": category,
        "model_used":         model_used,
    }
