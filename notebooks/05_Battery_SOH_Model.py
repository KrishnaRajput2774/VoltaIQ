#!/usr/bin/env python
"""
VoltIQ -- Phase 2 Notebook 05: Battery SOH Regression Model
=============================================================
Documents the Battery State-of-Health (SOH) training process step by step.

Dataset : datasets/battery/ev_battery_degradation.csv
Target  : State_of_Health (float in [0.0, 1.0])
Task    : Regression

Run from VoltIQ root:
    python notebooks/05_Battery_SOH_Model.py
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

print("=" * 60)
print("Phase 2 | Notebook 05 | Battery SOH Regression")
print("=" * 60)

# ----------------------------------------------------------------
# Step 1: Load data via Phase 1 pipeline
# ----------------------------------------------------------------
print("\nStep 1 -- Load and prepare data (Phase 1 pipeline)")
print("-" * 50)

from app.utils.data_loader import data_loader
from app.utils.cleaning import DataCleaner
from app.utils.feature_engineering import FeatureEngineer

raw   = data_loader.load("battery")
clean = DataCleaner().clean_battery(raw)
eng   = FeatureEngineer().engineer_battery(clean)

print(f"Battery dataset shape after Phase 1 pipeline: {eng.shape}")
print(f"Unique Battery_IDs: {eng['Battery_ID'].nunique()}")
print(f"\nTarget (State_of_Health) statistics:")
print(eng["State_of_Health"].describe().round(4))

# ----------------------------------------------------------------
# Step 2: Feature leakage analysis
# ----------------------------------------------------------------
print("\nStep 2 -- Feature leakage analysis")
print("-" * 50)

from app.models.leakage_analysis import LeakageAnalyzer, CONFIRMED_LEAKAGE_MAP
from app.models.battery_models import BATTERY_CANDIDATE_FEATURES, SOH_TARGET

eng[SOH_TARGET] = eng[SOH_TARGET].clip(0.0, 1.0)
analyzer = LeakageAnalyzer(high_corr_threshold=0.98, moderate_corr_threshold=0.90)
leakage  = analyzer.analyze(eng, SOH_TARGET, BATTERY_CANDIDATE_FEATURES)

print(f"Candidates evaluated      : {leakage.n_candidates}")
print(f"Confirmed leakage excluded: {len(leakage.confirmed_leakage)}  -- {leakage.confirmed_leakage}")
print(f"High corr risk excluded   : {len(leakage.high_correlation_risk)}  -- {leakage.high_correlation_risk}")
print(f"Safe features admitted    : {leakage.n_safe}  -- {leakage.safe_features}")

print("\nFeature correlations with target:")
for f, r in sorted(leakage.correlations.items(), key=lambda x: abs(x[1]) if not np.isnan(x[1]) else 0, reverse=True):
    if not np.isnan(r):
        print(f"  {f:35s}  r={r:+.4f}")

features = leakage.safe_features

# ----------------------------------------------------------------
# Step 3: Battery_ID-level split
# ----------------------------------------------------------------
print("\nStep 3 -- Battery_ID-level split (70 / 15 / 15)")
print("-" * 50)

from app.models.battery_models import split_by_battery_id

train_df, val_df, test_df = split_by_battery_id(eng, random_state=42)

X_train = train_df[features].fillna(train_df[features].median())
y_train = train_df[SOH_TARGET]
X_val   = val_df[features].fillna(train_df[features].median())
y_val   = val_df[SOH_TARGET]
X_test  = test_df[features].fillna(train_df[features].median())
y_test  = test_df[SOH_TARGET]

print(f"Train: {len(X_train)} rows ({train_df['Battery_ID'].nunique()} batteries)")
print(f"Val  : {len(X_val)} rows ({val_df['Battery_ID'].nunique()} batteries)")
print(f"Test : {len(X_test)} rows ({test_df['Battery_ID'].nunique()} batteries)")

# ----------------------------------------------------------------
# Step 4: Train and compare 3 candidates
# ----------------------------------------------------------------
print("\nStep 4 -- Train 3 candidate algorithms")
print("-" * 50)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import time

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
        "Test MAE":  round(test_mae,  5),
        "Test RMSE": round(test_rmse, 5),
        "Test R2":   round(test_r2,   5),
        "Fit (s)":   round(elapsed,   1),
    })
    print(f"done ({elapsed:.1f}s)  MAE={test_mae:.5f}  R2={test_r2:.5f}")

print("\nAlgorithm comparison:")
print(pd.DataFrame(results).to_string(index=False))

# ----------------------------------------------------------------
# Step 5: Train and save production SOH model
# ----------------------------------------------------------------
print("\nStep 5 -- Train production SOH model and save")
print("-" * 50)

from app.models.battery_models import train_soh_model

report = train_soh_model(eng, save=True, random_state=42)

print(f"Winner       : {report['winner']}")
print(f"Test MAE     : {report['best_metrics']['test_mae']}")
print(f"Test RMSE    : {report['best_metrics']['test_rmse']}")
print(f"Test R2      : {report['best_metrics']['test_r2']}")
print(f"CV MAE       : {report['best_metrics']['cv_mae_mean']} +/- {report['best_metrics']['cv_mae_std']}")
print(f"Model saved  : {report['model_path']}")
print(f"Metadata     : {report['meta_path']}")
print(f"Plots        : {len(report.get('evaluation_plots', []))} files")

print("\n[Notebook 05] COMPLETE")
