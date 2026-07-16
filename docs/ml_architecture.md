# VoltIQ ML Pipeline Architecture

**Document**: `docs/ml_architecture.md`  
**Version**: 2.0.0  
**Phase**: Phase 2 — Predictive Machine Learning  
**Last Updated**: 2026-07-15

---

## Architecture Overview

The VoltIQ ML pipeline transforms raw CSV datasets into production-ready
inference-capable sklearn Pipelines through eight sequential stages.

```
+------------------------------------------------------------------+
|                    VOLTIQ ML PIPELINE                            |
+------------------------------------------------------------------+

 STAGE 1: DATASET
 +-------------------------------------------------+
 |  datasets/battery/ev_battery_degradation.csv   |
 |  datasets/fleet/fleet_electrification_...csv   |
 |  (8 provided datasets -- no new data created)  |
 +-------------------------------------------------+
               |
               v

 STAGE 2: DATA CLEANING  (app/utils/cleaning.py)
 +-------------------------------------------------+
 |  DataCleaner.clean_battery()                   |
 |  DataCleaner.clean_fleet_readiness()           |
 |                                                |
 |  Operations:                                   |
 |  - Remove duplicate rows                       |
 |  - Parse and validate datetime columns         |
 |  - Impute numeric nulls with column median     |
 |  - Clip values to physical bounds              |
 |  - Flag IQR outliers (non-destructive)         |
 |  - Normalise string columns (lowercase, strip) |
 |  - Enforce schema column presence              |
 |                                                |
 |  Guarantee: Source CSVs are NEVER modified.    |
 |  All operations return new DataFrames.         |
 +-------------------------------------------------+
               |
               v

 STAGE 3: FEATURE ENGINEERING  (app/utils/feature_engineering.py)
 +-------------------------------------------------+
 |  FeatureEngineer.engineer_battery()            |
 |  FeatureEngineer.engineer_fleet_readiness()    |
 |                                                |
 |  Battery new columns:                          |
 |  - Cycle_Normalized = Cycle_Number / max       |
 |  - SOH_Category (Healthy/Attention/Critical)   |
 |  - Age_at_Cycle (Phase 1 derived)              |
 |                                                |
 |  Fleet new columns:                            |
 |  - Vehicle_Age_Category (binned)               |
 |  - Maintenance_Cost_Band (quantile)            |
 |  - Fuel_Efficiency_Category                    |
 |  - Load_Utilization_Category                   |
 |  - Readiness_Label                             |
 +-------------------------------------------------+
               |
               v

 STAGE 4: LEAKAGE ANALYSIS  (app/models/leakage_analysis.py)
 +-------------------------------------------------+
 |  LeakageAnalyzer.analyze(df, target, features) |
 |                                                |
 |  Checks:                                       |
 |  1. CONFIRMED_LEAKAGE_MAP -- columns that are  |
 |     definitionally derived from the target     |
 |     (e.g. SOH_Category, Is_End_of_Life)        |
 |                                                |
 |  2. Pearson correlation screen:               |
 |     - |r| >= 0.98 --> HIGH RISK (excluded)     |
 |     - 0.90 <= |r| < 0.98 --> MODERATE (logged) |
 |     - |r| < 0.90 --> SAFE (admitted)           |
 |                                                |
 |  Outputs: LeakageReport.safe_features          |
 |                                                |
 |  Battery SOH: 7 safe / 5 excluded              |
 |  Battery RUL: 9 safe / 1 excluded              |
 |  Fleet:       22 safe / 0 excluded             |
 +-------------------------------------------------+
               |
               v

 STAGE 5: TRAIN / VALIDATION / TEST SPLIT
 +-------------------------------------------------+
 |  Battery models:                               |
 |  - Split BY Battery_ID (not by row)            |
 |  - Prevents cycle-level data leakage           |
 |  - 70% train / 15% val / 15% test batteries   |
 |  - random_state=42                             |
 |                                                |
 |  Fleet model:                                  |
 |  - Stratified by Vehicle_Type                  |
 |  - 70% / 15% / 15% row-level split             |
 |  - random_state=42                             |
 +-------------------------------------------------+
               |
               v

 STAGE 6: CANDIDATE MODEL TRAINING
 +-------------------------------------------------+
 |  Three algorithms compared per model:          |
 |                                                |
 |  Battery Pipeline:                             |
 |    StandardScaler -> {LR | RF | GBR}           |
 |                                                |
 |  Fleet Pipeline:                               |
 |    ColumnTransformer(                          |
 |      StandardScaler (numeric)                  |
 |      OneHotEncoder  (categorical)              |
 |    ) -> {LR | RF | GBR}                        |
 |                                                |
 |  Winner selection: lowest Test MAE             |
 |  random_state=42 for all estimators            |
 +-------------------------------------------------+
               |
               v

 STAGE 7: EVALUATION
 +-------------------------------------------------+
 |  Per model, per split (val + test):            |
 |  - MAE, RMSE (np.sqrt(MSE)), R2               |
 |  - 5-fold cross-validation on training set     |
 |                                                |
 |  Artifacts generated (4 plots per model):      |
 |  - Prediction vs Actual scatter plot           |
 |  - Residuals vs Predicted plot                 |
 |  - Error distribution histogram                |
 |  - Worst 10 predictions table                  |
 |                                                |
 |  Markdown training reports generated           |
 |  Expanded metadata JSON written                |
 +-------------------------------------------------+
               |
               v

 STAGE 8: SERIALIZATION  (joblib)
 +-------------------------------------------------+
 |  joblib.dump(complete_pipeline, path)          |
 |                                                |
 |  Saves the COMPLETE sklearn Pipeline:          |
 |  - Fitted scaler (StandardScaler)              |
 |  - Fitted encoder (OneHotEncoder, fleet only)  |
 |  - Fitted estimator (GBR or LinearRegression)  |
 |                                                |
 |  Battery:                                      |
 |    saved_models/battery/battery_soh_model.pkl  |
 |    saved_models/battery/battery_rul_model.pkl  |
 |                                                |
 |  Fleet:                                        |
 |    saved_models/fleet/fleet_readiness_model.pkl|
 +-------------------------------------------------+
               |
               v

 STAGE 9: INFERENCE
 +-------------------------------------------------+
 |  pipeline = joblib.load(path)                  |
 |  prediction = pipeline.predict(X_new)          |
 |                                                |
 |  The loaded Pipeline automatically applies:   |
 |  - The same scaling fitted on training data    |
 |  - The same encoding fitted on training data   |
 |  - The trained estimator                       |
 |                                                |
 |  No manual preprocessing needed at inference  |
 +-------------------------------------------------+
```

---

## Component Map

```
VoltIQ/
  app/
    utils/
      data_loader.py          Stage 1 -- Dataset loading & path resolution
      cleaning.py             Stage 2 -- DataCleaner class
      feature_engineering.py  Stage 3 -- FeatureEngineer class
    models/
      leakage_analysis.py     Stage 4 -- LeakageAnalyzer, LeakageReport
      battery_models.py       Stages 5-9 -- Battery SOH & RUL
      fleet_models.py         Stages 5-9 -- Fleet Readiness
      __init__.py             Public API exports

  scripts/
    train_all_models.py       Orchestrates Stages 1-9 for all 3 models
    verify_models.py          Post-training inference verification

  saved_models/
    battery/
      battery_soh_model.pkl        Stage 8 output
      battery_rul_model.pkl        Stage 8 output
      battery_soh_metadata.json    Stage 7 output
      battery_rul_metadata.json    Stage 7 output
    fleet/
      fleet_readiness_model.pkl    Stage 8 output
      fleet_readiness_metadata.json Stage 7 output

  reports/ml/
    battery_soh_report.md          Stage 7 output
    battery_rul_report.md          Stage 7 output
    fleet_readiness_report.md      Stage 7 output
    phase2_ml_summary.md           Stage 7 output
    plots/
      *_pred_vs_actual.png         Stage 7 output (3 files)
      *_residuals.png              Stage 7 output (3 files)
      *_error_dist.png             Stage 7 output (3 files)
      *_worst_predictions.png      Stage 7 output (3 files)

  docs/
    MODEL_REGISTRY.md
    ml_architecture.md (this file)
    model_cards/
      battery_soh_model_card.md
      battery_rul_model_card.md
      fleet_readiness_model_card.md
      fleet_readiness_target_audit.md
```

---

## Design Principles

### 1. No Leakage by Design

The split strategy (Battery_ID-level for battery models, stratified for fleet)
ensures no information from test batteries appears during training or evaluation.

### 2. Complete Pipeline Serialization

The entire `sklearn.Pipeline` is saved — not just the estimator. This means:
- Scaling is automatically applied at inference using the training distribution
- No preprocessing code needs to be replicated or maintained separately
- The pipeline is a self-contained unit from raw features to prediction

### 3. Reproducibility

`random_state=42` is used for:
- Train/val/test splits
- Random Forest and Gradient Boosting estimators
- Cross-validation (implicit via sklearn)
- Battery ID shuffling

### 4. Graceful Degradation

Every training step uses fallback logic:
- Missing columns are reported, not silently dropped
- Training failures for one model do not abort others
- Inference functions have math-based fallbacks if models are not found

### 5. Compatibility

- RMSE computed as `np.sqrt(mean_squared_error(...))` — compatible with
  scikit-learn >= 1.4 which removed the `squared=False` parameter
- Logging uses ASCII-only separator characters — compatible with Windows
  CP1252 and all terminal encodings

---

## Data Flow Diagram (Simplified)

```
[Raw CSV]
    |
    +--> DataCleaner ---------> [Clean DataFrame]
                                      |
                                      +--> FeatureEngineer --> [Engineered DataFrame]
                                                                       |
                                                          LeakageAnalyzer --> [Safe Features]
                                                                       |
                                                               [ID-level / Stratified Split]
                                                               /           \
                                                          [Train]         [Val + Test]
                                                              |
                                                    [Candidate 1: LR]
                                                    [Candidate 2: RF]   --> Compare --> [Winner Pipeline]
                                                    [Candidate 3: GBR]                      |
                                                                                     [Evaluation Plots]
                                                                                     [Metadata JSON]
                                                                                     [Markdown Report]
                                                                                     [.pkl Serialization]
                                                                                             |
                                                                                     [Inference Ready]
```

---

_VoltIQ ML Architecture Document | v2.0.0 | 2026-07-15_
