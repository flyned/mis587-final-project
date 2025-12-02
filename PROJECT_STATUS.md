# MIS587 Final Project - Status
## Massachusetts Industrial Properties Price Prediction and Market Analysis

**Last Updated:** December 1, 2025
**Status:** Complete - All Four Proposal Objectives Achieved

**Team 2:**
- Alex Siracusa (Lead Data Analyst)
- Martin Thulani Milanzi (Risk Analyst)
- Shrey Sharma (Project Manager)
- Faisal Yaseen (Subject Matter Expert)

**Sponsor:** Lornell Real Estate (Todd Lornell - Principal/Founder)

---

## Proposal Objectives Status (October 6, 2025)

| Objective | Status | Achievement |
|-----------|--------|-------------|
| 1. Accurate Price Prediction | ✅ Complete | R² = 0.762 (Neural Network) |
| 2. Market Timing Forecast | ✅ Complete | 97.8% accuracy within 7 days |
| 3. Value Attribution Analysis | ✅ Complete | SHAP analysis implemented |
| 4. Opportunity Identification | ✅ Complete | Investment scoring in Streamlit |

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
- Ensemble models (Random Forest, XGBoost, LightGBM)
- Deep learning (TensorFlow/Keras neural network)
- Days on Market prediction model

### Phase 4: Deployment
- 7-page Streamlit dashboard covering all objectives
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
| Accuracy | 72.4% within ±10% |
| DOM Accuracy | 97.8% within 7 days |

---

## Files

**Core Source:**
- `src/` - All Python modules
- `streamlit_app/` - 7-page web dashboard

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
