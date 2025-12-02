# MIS587 Final Project - Status

**Last Updated:** December 1, 2024
**Status:** Complete

---

## Completed Phases

### Phase 1: Data Pipeline
- Data cleaning (40+ Excel files → clean_data.csv)
- Train/Val/Test splits (60/20/20 stratified by Market)
- Zero data leakage validated

### Phase 2: Feature Engineering
- 6-phase pipeline (imputation, geospatial, temporal, numerical, categorical, outliers)
- 78 raw features → 194 engineered features

### Phase 3: Modeling
- Baseline models (Ridge, Lasso, Decision Tree)
- Advanced models (Random Forest, XGBoost, LightGBM)
- Deep learning (TensorFlow/Keras neural network)
- Days on Market prediction model

### Phase 4: Deployment
- Streamlit dashboard with 7 pages
- Rent predictor, model insights, batch predictions
- Market comparison, timing analysis, opportunities

---

## Latest Models

| Model | Validation R² | Test R² | MAE |
|-------|--------------|---------|-----|
| Neural Network | 0.762 | 0.615 | $0.99/SF |
| Random Forest | 0.676 | 0.656 | $1.20/SF |
| Days on Market | - | - | 97.8% within 7 days |

---

## Key Metrics

| Metric | Value |
|--------|-------|
| Dataset | 12,534 unique properties |
| Features | 194 engineered |
| Target | Rent/SF/Yr ($12.43 mean) |
| Best Model | Neural Network |
| Accuracy | 72% within ±10% |

---

## Files

**Core Source:**
- `src/` - All Python modules
- `streamlit_app/` - Web dashboard

**Data:**
- `../data/train.csv` - 7,520 training samples
- `../data/val.csv` - 2,507 validation samples
- `../data/test.csv` - 2,507 test samples

**Models:**
- `models/rf_model_*.pkl` - Random Forest
- `models/nn_model_*.keras` - Neural Network
- `models/dom_model_*.pkl` - Days on Market

---

## How to Run

```bash
# Start dashboard
cd streamlit_app && streamlit run app.py

# Train models
python quick_test_local.py
python train_neural_network.py
```
