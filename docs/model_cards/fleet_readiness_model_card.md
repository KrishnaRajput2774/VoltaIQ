# Model Card: Fleet EV Electrification Readiness Regression

**Model ID**: `fleet_readiness_model`  
**Version**: 2.0.0  
**Phase**: VoltIQ Phase 2 — Predictive Machine Learning  
**Last Trained**: 2026-07-15  
**Status**: Production

---

## 1. Purpose

Predicts the **EV Readiness Score** for each fleet vehicle — a composite
index in [0.0, 1.0] that represents a vehicle's suitability for
electrification based on its operational, mechanical, and usage
characteristics.

This model enables fleet managers to rank and prioritise vehicles for
EV transition, identify high-readiness segments, and quantify the
electrification opportunity across a large mixed fleet.

---

## 2. Dataset

| Property | Value |
|---|---|
| Source file | `datasets/fleet/fleet_electrification_readiness.csv` |
| Rows | 250,000 |
| Columns | 51 (raw) + 7 engineered = 58 total |
| Target variable | `EV_Readiness_Score` (float in [0.07, 0.69]) |
| Target mean | 0.254 |
| Target std | 0.102 |
| Split strategy | Stratified by `Vehicle_Type` (70 / 15 / 15) |
| Train rows | 175,000 |
| Validation rows | 37,500 |
| Test rows | 37,500 |

---

## 3. Target Variable — Detailed Analysis

### EV_Readiness_Score is a Deterministic Composite Business Index

> [!IMPORTANT]
> The R2 = 0.9999 and MAE = 0.00017 are **fully explained** and
> **do NOT indicate data leakage**. The target is a deterministic
> weighted linear combination of the input feature columns.

**Audit results** (run against raw dataset, no Phase 1 preprocessing):

```
R2 of LinearRegression(composite index cols only) -> EV_Readiness_Score:  0.9732
R2 of LinearRegression(all 22 model features)     -> EV_Readiness_Score:  0.9999
```

This is conclusive: the `EV_Readiness_Score` in the dataset is
**intentionally constructed** as a weighted linear function of the
operational and telemetry columns. The model has learnt this formula.

### Correlation Structure

The strongest individual correlations with the target are:

| Feature | Pearson r |
|---|---|
| `Maintenance_Cost` | +0.822 |
| `Health_Score` | -0.802 |
| `Predictive_Score` | +0.799 |
| `OHI` | +0.797 |
| `CMES` | +0.795 |
| `MBF` | +0.791 |
| `Days_Since_Last_Maintenance` | +0.766 |
| `Fuel_Consumption` | +0.764 |
| `TPI` | +0.728 |

No single feature achieves r > 0.82, but a linear combination of 22
features achieves R2 = 0.9999. This is the hallmark of a dataset where
the target is a sum of weighted contributions from all feature columns —
a **composite business score design pattern** commonly seen in fleet
analytics, sustainability reporting, and asset management systems.

### Why This is Not Leakage

Data leakage occurs when features that are unavailable at inference time
(i.e., they depend on the target or on future information) are included
in training. In this dataset:

1. All 22 features are **independent physical measurements** of the vehicle
   (age, usage, fuel consumption, health telemetry, maintenance history).
2. The `EV_Readiness_Score` is the **output label** computed from these
   measurements — the correct design for a scoring index.
3. A predictive model that learns this formula perfectly is doing exactly
   what it should: approximating the business scoring function.
4. At inference time, the same physical measurements are available, and
   the model correctly predicts the score.

This is equivalent to a model trained to predict a credit score from
financial attributes, or an energy efficiency rating from building
measurements — high R2 is the expected and desired outcome.

---

## 4. Features

### Numeric Features (22)

| Feature | Description |
|---|---|
| `Vehicle_Age_Years` | Age of the vehicle in years |
| `Usage_Hours` | Total operational hours |
| `Load_Capacity` | Maximum rated load (kg) |
| `Actual_Load` | Typical operational load (kg) |
| `Load_Utilization_Pct` | Actual/capacity ratio (%) |
| `Fuel_Consumption` | Average fuel consumption (L/100km) |
| `Fuel_per_Hour` | Fuel consumption per operational hour |
| `Health_Score` | Composite health metric (0-100) |
| `Maintenance_Cost` | Annual maintenance cost |
| `Days_Since_Last_Maintenance` | Recency of last service |
| `Failure_History` | Count of historical failures |
| `Anomalies_Detected` | Sensor anomaly count |
| `Diagnostic_Trouble_Code_Count` | OBD-II fault codes |
| `Predictive_Score` | Predictive maintenance score |
| `PCR` | Performance-Cost Ratio index |
| `UIR` | Utilisation-to-Idle Ratio |
| `TPI` | Total Performance Index |
| `MBF` | Mean time Between Failures |
| `ADS` | Average Downtime Score |
| `OHI` | Overall Health Index |
| `CMES` | Cumulative Maintenance Effort Score |
| `UER` | Utilisation Efficiency Ratio |

### Categorical Features (5)

| Feature | Cardinality | Description |
|---|---|---|
| `Vehicle_Type` | ~6 classes | Light Truck, Heavy Truck, etc. |
| `Route_Info` | ~4 classes | Urban Delivery, Highway, etc. |
| `Road_Conditions` | ~4 classes | Smooth, Rough, Off-Road, etc. |
| `Weather_Conditions` | ~4 classes | Clear, Rain, Snow, etc. |
| `Brake_Condition` | ~4 classes | Good, Worn, Critical, etc. |

### Features Excluded by Leakage Analysis

Leakage analysis found **0 leaking features** among the 22 numeric
candidates. The one absent feature `EV_Priority_Score` was an engineered
column that did not survive the Phase 1 pipeline for this dataset variant.

---

## 5. Algorithm

| Property | Value |
|---|---|
| Winner | **LinearRegression** |
| Pipeline | ColumnTransformer(StandardScaler + OneHotEncoder) → LinearRegression |
| random_state | 42 (applied to splits and CV; LR is deterministic) |

### Why LinearRegression Won

The target is a linear composite of the features (confirmed by audit).
LinearRegression perfectly captures this structure. Tree-based methods
add unnecessary complexity and fit noise:

| Algorithm | Val MAE | Test MAE | Test R2 | Fit Time |
|---|---|---|---|---|
| **LinearRegression (winner)** | 0.00017 | **0.00017** | **0.99990** | 1.3s |
| RandomForest | 0.00168 | 0.00167 | 0.99912 | 152.9s |
| GradientBoosting | 0.00155 | 0.00154 | 0.99955 | 663.4s |

LinearRegression is 100x faster than RandomForest and 500x faster than
GradientBoosting while delivering superior accuracy. It is the
definitively correct algorithm for this target structure.

---

## 6. Performance Metrics

| Metric | Value | Interpretation |
|---|---|---|
| Test MAE | **0.00017** | Average error of 0.017 percentage points |
| Test RMSE | 0.00022 | Well within measurement noise |
| Test R2 | **0.99990** | Explains 99.99% of score variance |
| CV MAE (5-fold) | 0.00018 ± 0.00001 | Highly stable across folds |

---

## 7. Preprocessing Pipeline

```
Raw CSV (250,000 rows x 51 cols)
  --> Phase 1 DataCleaner
       (null imputation, outlier flagging, string normalisation,
        Readiness_Score clipped to [0, 1])
  --> Phase 1 FeatureEngineer
       (Vehicle_Age_Category, Maintenance_Cost_Band,
        Fuel_Efficiency_Category, Readiness_Label)
  --> Leakage analysis (0 features excluded)
  --> ColumnTransformer:
       StandardScaler (22 numeric features)
       OneHotEncoder  (5 categorical features, handle_unknown='ignore')
  --> LinearRegression
```

---

## 8. Limitations

1. **Composite score design**: The model replicates the formula used to
   generate `EV_Readiness_Score`. If the scoring formula changes (e.g.,
   weights are updated), the model must be retrained on data generated
   with the new formula.

2. **Linear extrapolation**: LinearRegression will extrapolate beyond
   the training data range [0.07, 0.69]. Predictions should be clipped
   to this range at inference time.

3. **OHE unknown categories**: The OneHotEncoder uses
   `handle_unknown='ignore'` — unseen categories at inference time will
   produce zero-vector encoding (neutral contribution).

4. **Assumes stable feature distributions**: If operational conditions
   shift significantly (e.g., all-electric routes), the feature
   distributions will change and the model should be retrained.

---

## 9. Intended Usage

**Appropriate uses:**
- Ranking fleet vehicles by EV readiness for transition planning
- Identifying low-readiness clusters for targeted improvements
- Monitoring fleet readiness over time as maintenance is performed
- Powering the AI Fleet Advisor's electrification recommendations

**Inappropriate uses:**
- Computing readiness for vehicles outside the training Vehicle_Type distribution
- Using raw score as a regulatory or compliance metric without human review

---

## 10. Known Assumptions

- `EV_Readiness_Score` is a deterministic composite of the 22 input
  features (confirmed by audit; R2 of linear reconstruction = 0.9999)
- The scoring formula is stable over the dataset's time period
- Feature scales are consistent across the fleet (normalised by scaler)

---

## 11. Model Artifacts

| Artifact | Path |
|---|---|
| Pipeline (.pkl) | `saved_models/fleet/fleet_readiness_model.pkl` |
| Metadata JSON | `saved_models/fleet/fleet_readiness_metadata.json` |
| Training report | `reports/ml/fleet_readiness_report.md` |
| Target audit | `docs/model_cards/fleet_readiness_target_audit.md` |
| Pred vs Actual | `reports/ml/plots/fleet_linearregressionev_readiness_score_pred_vs_actual.png` |

---

_VoltIQ Model Card | Fleet Readiness | v2.0.0_
