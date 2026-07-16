# Model Card: Battery State-of-Health (SOH) Regression

**Model ID**: `battery_soh_model`  
**Version**: 2.0.0  
**Phase**: VoltIQ Phase 2 — Predictive Machine Learning  
**Last Trained**: 2026-07-15  
**Status**: Production

---

## 1. Purpose

Predicts the **State of Health (SOH)** of an EV battery cell given
real-time telemetry readings. SOH represents remaining capacity relative
to the original rated capacity, expressed as a float in [0.0, 1.0].

This model is the primary tool for proactive battery fleet management —
it enables maintenance teams to identify cells approaching end-of-life
before they cause operational failures.

---

## 2. Dataset

| Property | Value |
|---|---|
| Source file | `datasets/battery/ev_battery_degradation.csv` |
| Rows | 1,415 (cycle-level observations) |
| Unique batteries | 34 |
| Target variable | `State_of_Health` (float in [0.0, 1.0]) |
| Target range | [0.03, 1.00] |
| Split strategy | **By Battery_ID** — all cycles of one battery stay in one split |
| Train batteries | 23 (1,119 rows) |
| Validation batteries | 5 (179 rows) |
| Test batteries | 6 (117 rows) |

> The Battery_ID-level split is critical. Row-level splits would cause
> cycle-level data leakage, where adjacent cycles from the same battery
> appear in both train and test — artificially inflating performance.

---

## 3. Features

| Feature | Type | Description |
|---|---|---|
| `Cycle_Number` | int | Charge-discharge cycle count |
| `Voltage_V` | float | Terminal voltage (V) |
| `Temperature_C` | float | Cell temperature (degrees C) |
| `Capacity_Ah` | float | Measured capacity (Ah) |
| `Voltage_Sag_V` | float | Voltage drop under load (V) |
| `Degradation_Rate` | float | Rate of capacity decline per cycle |
| `Cycle_Normalized` | float | Cycle_Number / max_cycles (Phase 1 engineered) |

### Features Excluded by Leakage Analysis

| Feature | Reason for Exclusion |
|---|---|
| `State_of_Health` | Target itself — cannot be used as input |
| `Capacity_Fade_Pct` | Derived from `State_of_Health`: `(1 - SOH) * 100` |
| `Is_End_of_Life` | Binary derived from SOH threshold |
| `Health_Zone` | Category derived from SOH threshold |
| `SOH_Category` | Category derived from SOH threshold |

---

## 4. Algorithm

| Property | Value |
|---|---|
| Winner | **GradientBoostingRegressor** |
| n_estimators | 300 |
| learning_rate | 0.05 |
| max_depth | 5 |
| subsample | 0.8 |
| random_state | 42 |
| Pipeline | StandardScaler → GradientBoostingRegressor |

### Algorithm Comparison

| Algorithm | Test MAE | Test R2 | Fit Time |
|---|---|---|---|
| LinearRegression | -- | -- | <0.1s |
| RandomForest | -- | -- | ~0.2s |
| **GradientBoosting (winner)** | **0.02306** | **0.80089** | ~2.0s |

---

## 5. Performance Metrics

| Metric | Value |
|---|---|
| Test MAE | **0.02306** (SOH units) |
| Test RMSE | 0.03671 |
| Test R2 | **0.80089** |
| CV MAE (5-fold) | see metadata JSON |

### Interpretation

A MAE of 0.023 means the model's average prediction error is ±2.3
percentage points of battery health. This is operationally meaningful:
a battery at 0.70 SOH would be predicted in the range [0.677, 0.723].

---

## 6. Preprocessing Pipeline

```
Raw CSV
  --> Phase 1 DataCleaner (null imputation, outlier flagging, type enforcement)
  --> Phase 1 FeatureEngineer (Cycle_Normalized, SOH_Category, etc.)
  --> Leakage analysis (exclude 5 target-derived columns)
  --> StandardScaler (fitted on training data only)
  --> GradientBoostingRegressor
```

The complete sklearn Pipeline (scaler + estimator) is serialised together.
No separate preprocessing step is needed at inference time.

---

## 7. Limitations

1. **Small dataset**: 34 batteries is a very small corpus. Performance on
   batteries with chemistries or usage patterns outside the training
   distribution is unknown.

2. **No temporal context**: The model treats each cycle independently. It
   does not model the time series trajectory of degradation — only
   point-in-time telemetry.

3. **Test set size**: 6 test batteries (117 rows). Statistical confidence
   intervals on reported metrics are wide.

4. **Single chemistry assumption**: The training data may represent one
   battery chemistry. Generalisation to LFP vs NMC vs NCA cells is not
   validated.

5. **No uncertainty quantification**: Point predictions only. Confidence
   intervals are not natively provided.

---

## 8. Intended Usage

**Appropriate uses:**
- Fleet health dashboards showing battery SOH trends
- Triggering maintenance alerts when predicted SOH < threshold
- Ranking batteries by health for replacement prioritisation
- Input feature for the RUL model (legitimate — SOH is a predictor of
  remaining life, not derived from it)

**Inappropriate uses:**
- Warranty decisions for individual cells without human review
- Safety-critical shutdown decisions without additional sensor validation
- Predicting SOH for battery chemistries outside the training distribution

---

## 9. Known Assumptions

- Battery telemetry is recorded at consistent cycle boundaries
- `Capacity_Ah` is measured under standardised discharge conditions
- Temperature is ambient or cell-level; coolant effects are not modelled
- Degradation is monotonic (SOH does not recover between cycles)

---

## 10. Model Artifacts

| Artifact | Path |
|---|---|
| Pipeline (.pkl) | `saved_models/battery/battery_soh_model.pkl` |
| Metadata JSON | `saved_models/battery/battery_soh_metadata.json` |
| Training report | `reports/ml/battery_soh_report.md` |
| Pred vs Actual | `reports/ml/plots/gradientsoh_pred_vs_actual.png` |
| Residuals | `reports/ml/plots/gradientsoh_residuals.png` |
| Error distribution | `reports/ml/plots/gradientsoh_error_dist.png` |
| Worst predictions | `reports/ml/plots/gradientsoh_worst_predictions.png` |

---

## 11. Ethical Considerations

Battery health predictions affect maintenance scheduling and asset
replacement decisions. Systematically underestimating SOH could lead to
premature battery replacement (cost); overestimating could result in
unplanned failures (safety). The model should be used with operational
safety margins and not as the sole decision-making input.

---

_VoltIQ Model Card | Battery SOH | v2.0.0_
