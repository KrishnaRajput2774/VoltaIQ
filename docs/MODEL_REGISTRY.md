# VoltIQ Model Registry

**Last Updated**: 2026-07-15  
**Phase**: Phase 2 — Predictive Machine Learning  
**Maintainer**: VoltIQ Platform Engineering

---

## Registry Overview

This registry tracks all production ML models in the VoltIQ platform.
All models are serialised as complete `sklearn.Pipeline` objects (preprocessing
+ estimator) via Joblib and are ready for direct inference on raw feature DataFrames.

---

## Model Entries

### Model 1 — Battery State of Health

| Property | Value |
|---|---|
| **Model Name** | `battery_soh_model` |
| **Version** | 2.0.0 |
| **Status** | Production |
| **Training Date** | 2026-07-15 |
| **Target Variable** | `State_of_Health` |
| **Target Type** | Regression (float in [0.0, 1.0]) |
| **Dataset** | `datasets/battery/ev_battery_degradation.csv` |
| **Training Rows** | 1,119 (23 batteries) |
| **Algorithm** | GradientBoostingRegressor |
| **Test MAE** | 0.02306 |
| **Test R2** | 0.80089 |
| **Random Seed** | 42 |
| **Pipeline File** | `saved_models/battery/battery_soh_model.pkl` |
| **Metadata File** | `saved_models/battery/battery_soh_metadata.json` |
| **Model Card** | `docs/model_cards/battery_soh_model_card.md` |
| **Training Report** | `reports/ml/battery_soh_report.md` |
| **Inference** | `from app.models.battery_models import predict_battery_health` |

---

### Model 2 — Battery Remaining Useful Life

| Property | Value |
|---|---|
| **Model Name** | `battery_rul_model` |
| **Version** | 2.0.0 |
| **Status** | Production (see limitations) |
| **Training Date** | 2026-07-15 |
| **Target Variable** | `Remaining_Useful_Life_Cycles` |
| **Target Type** | Regression (non-negative integer) |
| **Dataset** | `datasets/battery/ev_battery_degradation.csv` |
| **Training Rows** | 1,119 (23 batteries) |
| **Algorithm** | GradientBoostingRegressor |
| **Test MAE** | 24.54 cycles |
| **Test R2** | -0.428 (small-sample artifact — see model card) |
| **Random Seed** | 42 |
| **Pipeline File** | `saved_models/battery/battery_rul_model.pkl` |
| **Metadata File** | `saved_models/battery/battery_rul_metadata.json` |
| **Model Card** | `docs/model_cards/battery_rul_model_card.md` |
| **Training Report** | `reports/ml/battery_rul_report.md` |
| **Inference** | `from app.models.battery_models import predict_battery_health` |

---

### Model 3 — Fleet EV Electrification Readiness

| Property | Value |
|---|---|
| **Model Name** | `fleet_readiness_model` |
| **Version** | 2.0.0 |
| **Status** | Production |
| **Training Date** | 2026-07-15 |
| **Target Variable** | `EV_Readiness_Score` |
| **Target Type** | Regression (float in [0.07, 0.69]) |
| **Dataset** | `datasets/fleet/fleet_electrification_readiness.csv` |
| **Training Rows** | 175,000 (70% stratified split) |
| **Algorithm** | LinearRegression |
| **Test MAE** | 0.00017 |
| **Test R2** | 0.99990 |
| **Target Note** | Deterministic composite index (confirmed by audit — no leakage) |
| **Random Seed** | 42 |
| **Pipeline File** | `saved_models/fleet/fleet_readiness_model.pkl` |
| **Metadata File** | `saved_models/fleet/fleet_readiness_metadata.json` |
| **Model Card** | `docs/model_cards/fleet_readiness_model_card.md` |
| **Target Audit** | `docs/model_cards/fleet_readiness_target_audit.md` |
| **Training Report** | `reports/ml/fleet_readiness_report.md` |
| **Inference** | `from app.models.fleet_models import predict_ev_readiness` |

---

## Inference API Summary

### Battery Models (SOH + RUL)

```python
from app.models.battery_models import predict_battery_health

result = predict_battery_health(
    voltage_v=3.6,
    temperature_c=25.0,
    capacity_ah=1.8,
    cycle_number=50,
    voltage_sag_v=0.03,
    degradation_rate=-0.002,
    cycle_normalized=0.30,
)
# Returns: {"soh_predicted": 0.8534, "rul_predicted": 112,
#           "soh_model_used": "trained", "rul_model_used": "trained"}
```

### Fleet Readiness Model

```python
from app.models.fleet_models import predict_ev_readiness

result = predict_ev_readiness(
    vehicle_age_years=8,
    usage_hours=8000,
    fuel_consumption=10.5,
    health_score=80.0,
    vehicle_type="Light Truck",
    route_info="Urban Delivery",
)
# Returns: {"ev_readiness_score": 0.2479,
#           "readiness_category": "Low Readiness",
#           "model_used": "trained"}
```

---

## Model Lifecycle

| Stage | Battery SOH | Battery RUL | Fleet Readiness |
|---|---|---|---|
| Development | Phase 2 | Phase 2 | Phase 2 |
| Validation | 78/78 tests pass | 78/78 tests pass | 78/78 tests pass |
| Inference verified | PASS (SOH=0.8534) | PASS (112 cycles) | PASS (score=0.2479) |
| API integration | Phase 3 (planned) | Phase 3 (planned) | Phase 3 (planned) |
| Dashboard integration | Phase 4 (planned) | Phase 4 (planned) | Phase 4 (planned) |
| Retraining schedule | Quarterly / on new data | Quarterly / on new data | On formula change |

---

## Retraining Instructions

To retrain all models:

```bash
# From VoltIQ root, with virtual environment activated
python scripts/train_all_models.py
```

To retrain a specific model:

```python
from app.utils.data_loader import data_loader
from app.utils.cleaning import DataCleaner
from app.utils.feature_engineering import FeatureEngineer
from app.models.battery_models import train_soh_model

raw   = data_loader.load("battery")
clean = DataCleaner().clean_battery(raw)
eng   = FeatureEngineer().engineer_battery(clean)
report = train_soh_model(eng, save=True)
```

---

## Versioning Policy

- **Major version** (X.0.0): Change in feature set, target variable, or split strategy.
- **Minor version** (0.X.0): Hyperparameter tuning, algorithm change.
- **Patch version** (0.0.X): Bug fixes, metadata updates only.

Current version: **2.0.0**  
Previous version: 1.0.0 (pre-Phase 2 skeleton — no trained models)

---

_VoltIQ Model Registry | Last Updated 2026-07-15_
