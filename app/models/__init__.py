"""
VoltIQ -- app/models/__init__.py
==================================
Phase 2 Machine Learning exports.

Public API:
    Battery models: train_soh_model, train_rul_model,
                    load_soh_model, load_rul_model,
                    predict_battery_health
    Fleet models:   train_fleet_readiness_model,
                    load_fleet_readiness_model,
                    predict_ev_readiness
    Leakage:        LeakageAnalyzer, LeakageReport
"""

from app.models.battery_models import (
    BATTERY_FEATURES,
    BATTERY_CANDIDATE_FEATURES,
    SOH_TARGET,
    RUL_TARGET,
    RANDOM_STATE,
    split_by_battery_id,
    train_soh_model,
    train_rul_model,
    load_soh_model,
    load_rul_model,
    predict_battery_health,
)

from app.models.fleet_models import (
    FLEET_TARGET,
    FLEET_NUMERIC_FEATURES,
    FLEET_CATEGORICAL_FEATURES,
    prepare_fleet_data,
    train_fleet_readiness_model,
    load_fleet_readiness_model,
    predict_ev_readiness,
)

from app.models.leakage_analysis import (
    LeakageAnalyzer,
    LeakageReport,
    CONFIRMED_LEAKAGE_MAP,
)

__all__ = [
    # Battery
    "BATTERY_FEATURES",
    "BATTERY_CANDIDATE_FEATURES",
    "SOH_TARGET",
    "RUL_TARGET",
    "RANDOM_STATE",
    "split_by_battery_id",
    "train_soh_model",
    "train_rul_model",
    "load_soh_model",
    "load_rul_model",
    "predict_battery_health",
    # Fleet
    "FLEET_TARGET",
    "FLEET_NUMERIC_FEATURES",
    "FLEET_CATEGORICAL_FEATURES",
    "prepare_fleet_data",
    "train_fleet_readiness_model",
    "load_fleet_readiness_model",
    "predict_ev_readiness",
    # Leakage
    "LeakageAnalyzer",
    "LeakageReport",
    "CONFIRMED_LEAKAGE_MAP",
]
