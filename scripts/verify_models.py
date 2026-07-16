"""
VoltIQ Phase 2 -- Inference verification script
================================================
Loads each trained model from disk and runs a single prediction
to confirm the full Pipeline (preprocessing + estimator) is
operational end-to-end.

Run from VoltIQ root:
    python scripts/verify_models.py
"""
from __future__ import annotations
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(".").resolve()))

import numpy as np
import pandas as pd
import joblib

ROOT = Path(".")
PASS = "[PASS]"
FAIL = "[FAIL]"

results = []

# ---------------------------------------------------------------
# 1. Battery SOH Pipeline
# ---------------------------------------------------------------
print("=" * 60)
print("Verifying: battery_soh_model.pkl")
print("=" * 60)
soh_path = ROOT / "saved_models" / "battery" / "battery_soh_model.pkl"
soh_meta  = ROOT / "saved_models" / "battery" / "battery_soh_metadata.json"

try:
    pipeline = joblib.load(soh_path)
    print(f"  Loaded  : {soh_path.name} ({soh_path.stat().st_size / 1024:.1f} KB)")

    meta = json.loads(soh_meta.read_text(encoding="utf-8"))
    features = meta["selected_features"]
    print(f"  Features: {features}")

    # Build a sample row matching the model's feature set
    sample = pd.DataFrame([{f: 0.5 for f in features}])
    # Set realistic values
    if "Cycle_Number"     in features: sample["Cycle_Number"]     = 50
    if "Voltage_V"        in features: sample["Voltage_V"]        = 3.6
    if "Temperature_C"    in features: sample["Temperature_C"]    = 25.0
    if "Capacity_Ah"      in features: sample["Capacity_Ah"]      = 1.8
    if "Voltage_Sag_V"    in features: sample["Voltage_Sag_V"]    = 0.03
    if "Degradation_Rate" in features: sample["Degradation_Rate"] = -0.002
    if "Cycle_Normalized" in features: sample["Cycle_Normalized"] = 0.30

    pred = pipeline.predict(sample)[0]
    pred_clipped = float(np.clip(pred, 0.0, 1.0))
    print(f"  Prediction (SOH): {pred_clipped:.4f}")
    assert 0.0 <= pred_clipped <= 1.0, "SOH out of [0, 1]"
    print(f"  Algorithm : {meta.get('algorithm', meta.get('winner', '?'))}")
    print(f"  Test MAE  : {meta['evaluation_metrics']['test_mae']}")
    print(f"  Test R2   : {meta['evaluation_metrics']['test_r2']}")
    print(f"  {PASS} Battery SOH model inference OK")
    results.append(("Battery SOH", True, pred_clipped))
except Exception as exc:
    print(f"  {FAIL} Battery SOH: {exc}")
    results.append(("Battery SOH", False, None))

# ---------------------------------------------------------------
# 2. Battery RUL Pipeline
# ---------------------------------------------------------------
print()
print("=" * 60)
print("Verifying: battery_rul_model.pkl")
print("=" * 60)
rul_path = ROOT / "saved_models" / "battery" / "battery_rul_model.pkl"
rul_meta  = ROOT / "saved_models" / "battery" / "battery_rul_metadata.json"

try:
    pipeline = joblib.load(rul_path)
    print(f"  Loaded  : {rul_path.name} ({rul_path.stat().st_size / 1024:.1f} KB)")

    meta     = json.loads(rul_meta.read_text(encoding="utf-8"))
    features = meta["selected_features"]
    print(f"  Features: {features}")

    sample = pd.DataFrame([{f: 0.5 for f in features}])
    if "Cycle_Number"       in features: sample["Cycle_Number"]       = 50
    if "Voltage_V"          in features: sample["Voltage_V"]          = 3.6
    if "Temperature_C"      in features: sample["Temperature_C"]      = 25.0
    if "Capacity_Ah"        in features: sample["Capacity_Ah"]        = 1.8
    if "Voltage_Sag_V"      in features: sample["Voltage_Sag_V"]      = 0.03
    if "Degradation_Rate"   in features: sample["Degradation_Rate"]   = -0.002
    if "Cycle_Normalized"   in features: sample["Cycle_Normalized"]   = 0.30
    if "State_of_Health"    in features: sample["State_of_Health"]    = 0.90
    if "Capacity_Fade_Pct"  in features: sample["Capacity_Fade_Pct"]  = 10.0

    pred     = pipeline.predict(sample)[0]
    pred_int = int(max(0, round(pred)))
    print(f"  Prediction (RUL): {pred_int} cycles")
    assert pred_int >= 0, "RUL is negative"
    print(f"  Algorithm : {meta.get('algorithm', meta.get('winner', '?'))}")
    print(f"  Test MAE  : {meta['evaluation_metrics']['test_mae']}")
    print(f"  Test R2   : {meta['evaluation_metrics']['test_r2']}")
    print(f"  {PASS} Battery RUL model inference OK")
    results.append(("Battery RUL", True, pred_int))
except Exception as exc:
    print(f"  {FAIL} Battery RUL: {exc}")
    results.append(("Battery RUL", False, None))

# ---------------------------------------------------------------
# 3. Fleet Readiness Pipeline
# ---------------------------------------------------------------
print()
print("=" * 60)
print("Verifying: fleet_readiness_model.pkl")
print("=" * 60)
fleet_path = ROOT / "saved_models" / "fleet" / "fleet_readiness_model.pkl"
fleet_meta  = ROOT / "saved_models" / "fleet" / "fleet_readiness_metadata.json"

try:
    pipeline = joblib.load(fleet_path)
    print(f"  Loaded  : {fleet_path.name} ({fleet_path.stat().st_size / 1024:.1f} KB)")

    meta     = json.loads(fleet_meta.read_text(encoding="utf-8"))
    num_feats = meta.get("selected_numeric_features", [])
    cat_feats = meta.get("selected_categorical_features", [])
    features  = num_feats + cat_feats
    print(f"  Numeric features  : {len(num_feats)}")
    print(f"  Categorical feats : {len(cat_feats)}")

    # Build a realistic sample row
    row = {f: 0.0 for f in num_feats}
    row.update({
        "Vehicle_Age_Years":         8,
        "Usage_Hours":               8000.0,
        "Fuel_Consumption":          10.5,
        "Health_Score":              80.0,
        "Maintenance_Cost":          600.0,
        "Predictive_Score":          0.75,
        "Vehicle_Type":              "Light Truck",
        "Route_Info":                "Urban Delivery",
        "Road_Conditions":           "Smooth",
        "Weather_Conditions":        "Clear",
        "Brake_Condition":           "Good",
    })
    sample = pd.DataFrame([row])[features]

    pred       = pipeline.predict(sample)[0]
    pred_score = float(np.clip(pred, 0.0, 1.0))
    print(f"  Prediction (EV_Readiness_Score): {pred_score:.4f}")
    assert 0.0 <= pred_score <= 1.0, "Score out of [0, 1]"
    print(f"  Algorithm : {meta.get('algorithm', meta.get('winner', '?'))}")
    print(f"  Test MAE  : {meta['evaluation_metrics']['test_mae']}")
    print(f"  Test R2   : {meta['evaluation_metrics']['test_r2']}")
    print(f"  {PASS} Fleet readiness model inference OK")
    results.append(("Fleet Readiness", True, pred_score))
except Exception as exc:
    print(f"  {FAIL} Fleet Readiness: {exc}")
    results.append(("Fleet Readiness", False, None))

# ---------------------------------------------------------------
# Summary
# ---------------------------------------------------------------
print()
print("=" * 60)
print("INFERENCE VERIFICATION SUMMARY")
print("=" * 60)
all_pass = True
for name, ok, val in results:
    status = PASS if ok else FAIL
    val_str = f"  prediction={val}" if ok else ""
    print(f"  {status} {name}{val_str}")
    if not ok:
        all_pass = False

print()
if all_pass:
    print("All 3 model Pipelines loaded and ran inference successfully.")
    sys.exit(0)
else:
    print("One or more models failed inference verification.")
    sys.exit(1)
