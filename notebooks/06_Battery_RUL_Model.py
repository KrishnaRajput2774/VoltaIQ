#!/usr/bin/env python
"""
VoltIQ -- Phase 2 Notebook 06: Battery RUL Regression Model
=============================================================
Documents the Battery Remaining Useful Life (RUL) training process.

Dataset : datasets/battery/ev_battery_degradation.csv
Target  : Remaining_Useful_Life_Cycles (integer in [0, 167])
Task    : Regression

Run from VoltIQ root:
    python notebooks/06_Battery_RUL_Model.py
"""

from __future__ import annotations
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(".").resolve()))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import time

print("=" * 60)
print("Phase 2 | Notebook 06 | Battery RUL Regression")
print("=" * 60)

# ----------------------------------------------------------------
# Step 1: Load data via Phase 1 pipeline
# ----------------------------------------------------------------
print("\nStep 1 -- Load and prepare data")
print("-" * 50)

from app.utils.data_loader import data_loader
from app.utils.cleaning import DataCleaner
from app.utils.feature_engineering import FeatureEngineer

raw   = data_loader.load("battery")
clean = DataCleaner().clean_battery(raw)
eng   = FeatureEngineer().engineer_battery(clean)

print(f"Shape: {eng.shape}")
print(f"\nTarget (Remaining_Useful_Life_Cycles) statistics:")
print(eng["Remaining_Useful_Life_Cycles"].describe().round(2))

# ----------------------------------------------------------------
# Step 2: Feature leakage analysis
# ----------------------------------------------------------------
print("\nStep 2 -- Feature leakage analysis for RUL")
print("-" * 50)

from app.models.leakage_analysis import LeakageAnalyzer
from app.models.battery_models import BATTERY_CANDIDATE_FEATURES, RUL_TARGET

analyzer = LeakageAnalyzer(high_corr_threshold=0.98, moderate_corr_threshold=0.90)
leakage  = analyzer.analyze(eng, RUL_TARGET, BATTERY_CANDIDATE_FEATURES)

print(f"Candidates         : {leakage.n_candidates}")
print(f"Confirmed excluded : {len(leakage.confirmed_leakage)}  {leakage.confirmed_leakage}")
print(f"High corr excluded : {len(leakage.high_correlation_risk)}  {leakage.high_correlation_risk}")
print(f"Safe admitted      : {leakage.n_safe}  {leakage.safe_features}")

print("\nTop correlations with Remaining_Useful_Life_Cycles:")
sorted_corr = sorted(
    [(f, r) for f, r in leakage.correlations.items() if not np.isnan(r)],
    key=lambda x: abs(x[1]), reverse=True
)
for f, r in sorted_corr[:10]:
    print(f"  {f:38s}  r={r:+.4f}")

features = leakage.safe_features

# ----------------------------------------------------------------
# Step 3: Battery_ID-level split
# ----------------------------------------------------------------
print("\nStep 3 -- Battery_ID-level split (70 / 15 / 15)")
print("-" * 50)

from app.models.battery_models import split_by_battery_id

train_df, val_df, test_df = split_by_battery_id(eng, random_state=42)

X_train = train_df[features].fillna(train_df[features].median())
y_train = train_df[RUL_TARGET]
X_val   = val_df[features].fillna(train_df[features].median())
y_val   = val_df[RUL_TARGET]
X_test  = test_df[features].fillna(train_df[features].median())
y_test  = test_df[RUL_TARGET]

print(f"Train: {len(X_train)} rows  |  Val: {len(X_val)} rows  |  Test: {len(X_test)} rows")

# ----------------------------------------------------------------
# Step 4: Candidate comparison
# ----------------------------------------------------------------
print("\nStep 4 -- Train 3 candidate algorithms")
print("-" * 50)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

candidates = {
    "LinearRegression": Pipeline([
        ("scaler", StandardScaler()),
        ("model",  LinearRegression()),
    ]),
    "RandomForest": Pipeline([
        ("scaler", StandardScaler()),
        ("model",  RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)),
    ]),
    "GradientBoosting": Pipeline([
        ("scaler", StandardScaler()),
        ("model",  GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=5, random_state=42)),
    ]),
}

results = []
for name, pipe in candidates.items():
    print(f"  Fitting {name}...", end=" ", flush=True)
    t0 = time.time()
    pipe.fit(X_train, y_train)
    elapsed   = time.time() - t0
    preds     = pipe.predict(X_test)
    test_mae  = mean_absolute_error(y_test, preds)
    test_rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    test_r2   = r2_score(y_test, preds)
    results.append({
        "Algorithm": name,
        "Test MAE":  round(test_mae,  4),
        "Test RMSE": round(test_rmse, 4),
        "Test R2":   round(test_r2,   4),
        "Fit (s)":   round(elapsed,   1),
    })
    print(f"done ({elapsed:.1f}s)  MAE={test_mae:.4f}  R2={test_r2:.4f}")

print("\nAlgorithm comparison:")
print(pd.DataFrame(results).to_string(index=False))

# ----------------------------------------------------------------
# Step 5: Train and save production RUL model
# ----------------------------------------------------------------
print("\nStep 5 -- Train production RUL model and save")
print("-" * 50)

from app.models.battery_models import train_rul_model

report = train_rul_model(eng, save=True, random_state=42)

print(f"Winner       : {report['winner']}")
print(f"Test MAE     : {report['best_metrics']['test_mae']} cycles")
print(f"Test RMSE    : {report['best_metrics']['test_rmse']}")
print(f"Test R2      : {report['best_metrics']['test_r2']}")
print(f"CV MAE       : {report['best_metrics']['cv_mae_mean']} +/- {report['best_metrics']['cv_mae_std']}")
print(f"Model saved  : {report['model_path']}")

# Step 6: Worst predictions analysis (printed)
print("\nStep 6 -- Worst predictions on test set")
print("-" * 50)

from app.models.battery_models import load_rul_model
rul_pipeline = load_rul_model()
if rul_pipeline:
    preds    = rul_pipeline.predict(X_test)
    errors   = np.abs(y_test.values - preds)
    worst_idx = np.argsort(errors)[::-1][:5]
    worst_df  = pd.DataFrame({
        "Actual_RUL": y_test.values[worst_idx],
        "Predicted":  preds[worst_idx].round(1),
        "AbsError":   errors[worst_idx].round(2),
    })
    print(worst_df.to_string(index=False))

print("\n[Notebook 06] COMPLETE")
