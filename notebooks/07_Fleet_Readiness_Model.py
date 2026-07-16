#!/usr/bin/env python
"""
VoltIQ -- Phase 2 Notebook 07: Fleet Electrification Readiness Model
======================================================================
Documents the Fleet EV Readiness Score regression training process.

Dataset : datasets/fleet/fleet_electrification_readiness.csv
Target  : EV_Readiness_Score (float in [0.07, 0.69])
Task    : Regression
Rows    : 250,000

Run from VoltIQ root:
    python notebooks/07_Fleet_Readiness_Model.py
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
from sklearn.model_selection import train_test_split
import time

print("=" * 60)
print("Phase 2 | Notebook 07 | Fleet Readiness Regression")
print("=" * 60)

# ----------------------------------------------------------------
# Step 1: Load data via Phase 1 pipeline
# ----------------------------------------------------------------
print("\nStep 1 -- Load fleet readiness dataset")
print("-" * 50)

from app.utils.data_loader import data_loader
from app.utils.cleaning import DataCleaner
from app.utils.feature_engineering import FeatureEngineer

raw   = data_loader.load("fleet_readiness")
clean = DataCleaner().clean_fleet_readiness(raw)
eng   = FeatureEngineer().engineer_fleet_readiness(clean)

print(f"Shape: {eng.shape}")
print(f"\nTarget (EV_Readiness_Score) statistics:")
print(eng["EV_Readiness_Score"].describe().round(4))
print(f"\nVehicle_Type distribution:")
print(eng["Vehicle_Type"].value_counts())

# ----------------------------------------------------------------
# Step 2: Feature leakage analysis
# ----------------------------------------------------------------
print("\nStep 2 -- Feature leakage analysis")
print("-" * 50)

from app.models.leakage_analysis import LeakageAnalyzer
from app.models.fleet_models import FLEET_NUMERIC_FEATURES, FLEET_TARGET, _filter_available

all_numeric = _filter_available(eng, FLEET_NUMERIC_FEATURES, "numeric_candidates")
analyzer = LeakageAnalyzer(high_corr_threshold=0.98, moderate_corr_threshold=0.90)
leakage  = analyzer.analyze(eng, FLEET_TARGET, all_numeric)

print(f"Candidates         : {leakage.n_candidates}")
print(f"Confirmed excluded : {len(leakage.confirmed_leakage)}  {leakage.confirmed_leakage}")
print(f"High corr excluded : {len(leakage.high_correlation_risk)}  {leakage.high_correlation_risk}")
print(f"Moderate flagged   : {len(leakage.moderate_correlation)}  {leakage.moderate_correlation}")
print(f"Safe admitted      : {leakage.n_safe}")

print("\nTop 15 correlations with EV_Readiness_Score:")
sorted_corr = sorted(
    [(f, r) for f, r in leakage.correlations.items() if not np.isnan(r)],
    key=lambda x: abs(x[1]), reverse=True
)
for f, r in sorted_corr[:15]:
    print(f"  {f:40s}  r={r:+.5f}")

# ----------------------------------------------------------------
# Step 3: Prepare features + stratified split
# ----------------------------------------------------------------
print("\nStep 3 -- Prepare features + stratified split (70/15/15)")
print("-" * 50)

from app.models.fleet_models import (
    FLEET_CATEGORICAL_FEATURES, _make_preprocessor, _make_fleet_candidates
)

safe_num = leakage.safe_features
safe_cat = _filter_available(eng, FLEET_CATEGORICAL_FEATURES, "categorical")
feat_cols = safe_num + safe_cat

df_prep = eng[feat_cols + [FLEET_TARGET, "Vehicle_Type"]].dropna(subset=[FLEET_TARGET]).copy()
for col in safe_num:
    if col in df_prep.columns and df_prep[col].isnull().any():
        df_prep[col] = df_prep[col].fillna(df_prep[col].median())
for col in safe_cat:
    if col in df_prep.columns and df_prep[col].isnull().any():
        df_prep[col] = df_prep[col].fillna("Unknown")

X_all = df_prep[feat_cols]
y_all = df_prep[FLEET_TARGET]

X_tmp, X_test, y_tmp, y_test = train_test_split(
    X_all, y_all, test_size=0.15, random_state=42, stratify=df_prep["Vehicle_Type"]
)
X_train, X_val, y_train, y_val = train_test_split(
    X_tmp, y_tmp, test_size=0.15/0.85, random_state=42,
    stratify=df_prep.loc[X_tmp.index, "Vehicle_Type"]
)

print(f"Train: {len(X_train)}  |  Val: {len(X_val)}  |  Test: {len(X_test)}")
print(f"Numeric features: {len(safe_num)}  |  Categorical: {len(safe_cat)}")

# ----------------------------------------------------------------
# Step 4: Candidate comparison
# ----------------------------------------------------------------
print("\nStep 4 -- Train 3 candidate algorithms")
print("-" * 50)

preprocessor = _make_preprocessor(safe_num, safe_cat)
candidates   = _make_fleet_candidates(preprocessor, random_state=42)

results = []
for name, pipe in candidates.items():
    print(f"  Fitting {name} on {len(X_train):,} rows...", end=" ", flush=True)
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
# Step 5: Train and save production model
# ----------------------------------------------------------------
print("\nStep 5 -- Train production fleet readiness model and save")
print("-" * 50)

from app.models.fleet_models import train_fleet_readiness_model

report = train_fleet_readiness_model(eng, save=True, random_state=42)

print(f"Winner       : {report['winner']}")
print(f"Test MAE     : {report['best_metrics']['test_mae']}")
print(f"Test R2      : {report['best_metrics']['test_r2']}")
print(f"Model saved  : {report['model_path']}")
print(f"Plots        : {len(report.get('evaluation_plots', []))} files")

# Step 6: Top feature importances
print("\nStep 6 -- Top feature importances")
print("-" * 50)

top10 = dict(list(report.get("top20_feature_importances", {}).items())[:10])
if top10:
    for feat, imp in top10.items():
        bar = "#" * int(imp * 200)
        print(f"  {feat:40s}  {imp:.6f}  {bar}")
else:
    print("  (Feature importances not available for this algorithm)")

print("\n[Notebook 07] COMPLETE")
