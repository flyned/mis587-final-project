# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Project Overview

Commercial real estate data pipeline and ML project for CoStar property data.

**Status:** Complete - Data pipeline, ML models, and Streamlit dashboard all implemented.

## Quick Commands

```bash
# Run Streamlit app
cd streamlit_app && streamlit run app.py

# Data cleaning pipeline
python main.py  # Outputs ../data/clean_data.csv

# Create train/val/test splits
python create_splits_v2.py  # Creates ../data/{train,val,test}.csv

# Train models
python quick_test_local.py        # Quick RF test (~5 min)
python train_neural_network.py    # Train neural network (~5 min)

# Install dependencies
pip install -r requirements.txt
```

## Code Architecture

### Pipeline Flow

```
Raw Excel → Data Cleaning → Train/Val/Test → Feature Engineering → Models → Streamlit
     ↓           ↓               ↓                  ↓                 ↓          ↓
excel_sheets/  main.py   create_splits_v2.py  feature_engineering  modeling*   app.py
```

### Source Code (src/)

| File | Purpose |
|------|---------|
| `load_raw_data.py` | Load Excel files from ../data/excel_sheets/ |
| `column_types.py` | Centralized column categorizations |
| `clean_numeric.py` | Custom parsers for messy numeric fields |
| `clean_categorical.py` | Fill nulls, consolidate rare categories |
| `feature_engineering.py` | 6-phase feature engineering pipeline |
| `modeling.py` | RF, XGBoost, LightGBM models |
| `modeling_deep.py` | TensorFlow/Keras neural network |
| `dom_model.py` | Days on Market prediction model |
| `opportunity.py` | Investment opportunity scoring |
| `interpretation.py` | SHAP, feature importance analysis |
| `visualization.py` | Plotting functions |
| `model_validators.py` | Model validation utilities |
| `validators.py` | Data validation utilities |
| `utils.py` | Helper functions for data I/O |

### Streamlit App (streamlit_app/)

| Page | Purpose |
|------|---------|
| `app.py` | Main entry point |
| `pages/1_Rent_Predictor.py` | Predict rent for properties |
| `pages/2_Model_Insights.py` | Feature importance, SHAP values |
| `pages/3_Batch_Prediction.py` | Bulk predictions from CSV |
| `pages/4_About.py` | Project documentation |
| `pages/5_Market_Comparison.py` | Market analysis |
| `pages/6_Market_Timing.py` | Days on market analysis |
| `pages/7_Opportunities.py` | Investment opportunities |

### Models (models/)

| Model | File Pattern | Validation R² |
|-------|--------------|---------------|
| Random Forest | `rf_model_*.pkl` | 0.676 |
| Neural Network | `nn_model_*.keras` | 0.762 |
| Days on Market | `dom_model_*.pkl` | 97.8% within 7 days |

## Data Directory Structure

**All data paths use `../data/` (one level up from project root)**

```
../data/
├── excel_sheets/        # 40+ raw CoStar export files (.xlsx)
├── clean_data.csv       # Cleaned data (15,467 rows × 87 cols)
├── train.csv            # Training set (7,520 rows, 60%)
├── val.csv              # Validation set (2,507 rows, 20%)
└── test.csv             # Test set (2,507 rows, 20%)
```

```
models/
├── rf_model_local_*.pkl            # Random Forest model
├── rf_metadata_local_*.json        # RF metadata
├── nn_model_*.keras                # Neural Network model
├── nn_metadata_*.json              # NN metadata with scaler
├── dom_model_*.pkl                 # Days on Market model
└── dom_metadata_*.json             # DOM metadata
```

## Feature Engineering Pipeline

6 phases in `src/feature_engineering.py`:

1. **impute_missing_values()** - Structural zeros, median, random sampling
2. **create_geospatial_features()** - Distance to market center, property density
3. **create_temporal_features()** - Building age, years since sale, construction eras
4. **create_numerical_features()** - Log transforms, ratio features, interactions
5. **encode_categorical_features()** - Target encoding (CV), frequency, one-hot
6. **treat_outliers()** - Winsorization at 1st/99th percentile

**Result:** 78 raw features → 194 engineered features

## Key Design Decisions

### Data Quality
- Deduplicate by Property Address before splitting (prevents leakage)
- Drop columns with >95% nulls or zero variance
- Consolidate rare categories (<50 occurrences → 'other')

### Train/Val/Test Split
- 60/20/20 stratified by Market Name
- Zero data leakage validated
- Use `create_splits_v2.py` only (v1 had leakage)

### Feature Engineering
- Target encoding uses 5-fold CV to prevent overfitting
- Log transforms for right-skewed features (RBA, Land Area)
- Winsorization caps outliers (not removes)

### Modeling
- Neural network: 256→128→64→1 with ReLU and Dropout
- Random Forest: 200 trees, default parameters
- Model metadata uses `validation_metrics` and `test_metrics` keys

## Critical Warnings

1. **Never use `create_splits.py`** - Deleted due to data leakage
2. **Data leakage prevention** - Always deduplicate by Property Address first
3. **Geographic leakage** - Use grouped K-fold by Market Name
4. **Target encoding leakage** - Always use cross-validation
5. **Test set sanctity** - Never tune hyperparameters on test set

## Metadata Structure

Model metadata (JSON) uses this structure for Streamlit compatibility:

```json
{
  "model_type": "RandomForestRegressor",
  "training_samples": 7520,
  "validation_samples": 2507,
  "test_samples": 2507,
  "feature_count": 194,
  "validation_metrics": {
    "mae": 1.20,
    "rmse": 1.61,
    "r2": 0.676,
    "mape": 10.99,
    "within_10pct": 0.645
  },
  "test_metrics": {
    "mae": 1.21,
    "rmse": 1.64,
    "r2": 0.653,
    "mape": 10.99,
    "within_10pct": 0.638
  }
}
```

## Documentation Files

| File | Description |
|------|-------------|
| `README.md` | Project overview and quick start |
| `CLAUDE.md` | This file - code architecture guide |
| `PROJECT_STATUS.md` | Detailed progress tracking |
| `planml.md` | Original 8-week ML roadmap |
| `FINAL_REPORT.md` | Complete academic report |
