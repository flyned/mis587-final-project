# Massachusetts Industrial Properties Price Prediction and Market Analysis

**MIS587 - Business Applications in Machine Learning**

Complete data pipeline and machine learning solution for CoStar commercial real estate data with interactive Streamlit dashboard.

**Team 2:**
- Alex Siracusa (Lead Data Analyst)
- Martin Thulani Milanzi (Risk Analyst)
- Shrey Sharma (Project Manager)
- Faisal Yaseen (Subject Matter Expert)

**Sponsor:** Lornell Real Estate (Todd Lornell - Principal/Founder)

**Status:** Complete - All four proposal objectives achieved with 7-page Streamlit app deployed

## Quick Start

### Run the Streamlit App
```bash
cd streamlit_app && streamlit run app.py
```
Access at http://localhost:8501

### Train Models
```bash
# Quick test (5 min, ~400MB RAM)
python quick_test_local.py

# Train neural network (~5 min)
python train_neural_network.py
```

### Make Predictions
```bash
python predict.py \
  --model ./models/rf_model_local_20251130_225851.pkl \
  --metadata ./models/rf_metadata_local_20251130_225851.json \
  --data ../data/test.csv \
  --output ./predictions.csv
```

---

## Project Structure

```
MIS587FinalProject/
├── src/                          # Core source code
│   ├── feature_engineering.py    # Feature engineering pipeline
│   ├── modeling.py               # ML models (RF, XGBoost, LightGBM)
│   ├── modeling_deep.py          # Neural network (TensorFlow/Keras)
│   ├── dom_model.py              # Days on Market prediction
│   ├── opportunity.py            # Investment opportunity scoring
│   ├── interpretation.py         # SHAP & feature importance
│   ├── visualization.py          # Plotting functions
│   └── ...                       # Data loading & cleaning modules
│
├── streamlit_app/                # Interactive web dashboard
│   ├── app.py                    # Main app entry point
│   └── pages/                    # 7 dashboard pages
│       ├── 1_Rent_Predictor.py   # Predict rent for properties
│       ├── 2_Model_Insights.py   # Feature importance & model analysis
│       ├── 3_Batch_Prediction.py # Bulk predictions from CSV
│       ├── 4_About.py            # Project documentation
│       ├── 5_Market_Comparison.py # Market analysis
│       ├── 6_Market_Timing.py    # Days on market analysis
│       └── 7_Opportunities.py    # Investment opportunities
│
├── models/                       # Trained model artifacts
│   ├── rf_model_*.pkl            # Random Forest model
│   ├── nn_model_*.keras          # Neural Network model
│   └── dom_model_*.pkl           # Days on Market model
│
├── figures/                      # Generated visualizations
├── validation_reports/           # Validation JSON reports
│
└── ../data/                      # Data files (one level up)
    ├── excel_sheets/             # 40+ raw CoStar exports
    ├── clean_data.csv            # Cleaned data (15,467 rows)
    ├── train.csv                 # Training set (7,520 rows, 60%)
    ├── val.csv                   # Validation set (2,507 rows, 20%)
    └── test.csv                  # Test set (2,507 rows, 20%)
```

---

## Installation

```bash
pip install -r requirements.txt
```

**Key Dependencies:**
- pandas, numpy, scikit-learn
- tensorflow (for neural network)
- xgboost, lightgbm
- streamlit (for web dashboard)
- shap (for model interpretation)

---

## Model Performance

### Neural Network (Best Validation Performance)
| Metric | Validation | Test |
|--------|-----------|------|
| **R²** | 0.762 | 0.615 |
| **MAE** | $0.99/SF/Yr | $1.04/SF/Yr |
| **MAPE** | 8.7% | 9.2% |
| **Within ±10%** | 72.4% | 71.0% |

### Random Forest (More Stable)
| Metric | Validation | Test |
|--------|-----------|------|
| **R²** | 0.640 | 0.623 |
| **MAE** | $1.24/SF/Yr | $1.24/SF/Yr |
| **MAPE** | 11.3% | 11.2% |
| **Within ±10%** | 64.2% | 64.0% |

**Business Interpretation:**
- Average rent in dataset: $12.43/SF/Yr
- Neural network error: ~$1/SF/Yr (8% of average)
- 71-72% of predictions within ±10% of actual values

---

## Feature Engineering

**6 Major Transformations (78 raw → 194 engineered features):**

1. **Temporal Features** - Building age, years since sale, cyclical month encoding, construction eras
2. **Geospatial Features** - Distance to market center, property density, nearest competitor
3. **Numerical Transformations** - Log transforms, ratio features, interaction features
4. **Missing Value Imputation** - Structural zeros, median, random sampling with indicators
5. **Outlier Treatment** - Winsorization at 1st/99th percentiles with flag indicators
6. **Categorical Encoding** - Target encoding (5-fold CV), frequency encoding, one-hot encoding

---

## Streamlit Dashboard Features

1. **Rent Predictor** - Input property features to get rent prediction with confidence intervals
2. **Model Insights** - Feature importance, SHAP values, model comparison
3. **Batch Prediction** - Upload CSV for bulk predictions
4. **About** - Project documentation and methodology
5. **Market Comparison** - Analyze rent trends across markets
6. **Market Timing** - Days on market analysis and predictions
7. **Opportunities** - Identify undervalued investment opportunities

---

## Project Objectives (from October 6, 2025 Proposal)

| Objective | Status | Achievement |
|-----------|--------|-------------|
| 1. Accurate Price Prediction | ✅ Complete | R² = 0.762 (Neural Network) |
| 2. Market Timing Forecast | ✅ Complete | 97.8% accuracy within 7 days |
| 3. Value Attribution Analysis | ✅ Complete | SHAP analysis implemented |
| 4. Opportunity Identification | ✅ Complete | Investment scoring system |

---

## Key Results Summary

| Metric | Value |
|--------|-------|
| **Dataset** | 12,534 unique properties |
| **Features** | 78 raw → 194 engineered |
| **Target** | Rent/SF/Yr ($12.43/SF/Yr mean) |
| **Best Model** | Neural Network (R² = 0.762 validation) |
| **DOM Model** | 97.8% accuracy within 7 days |
| **Prediction Error** | ~$1/SF/Yr (8% average error) |
| **Accuracy Band** | 72.4% within ±10% |

---

## Documentation

| File | Description |
|------|-------------|
| `CLAUDE.md` | Code architecture and development guide |
| `PROJECT_STATUS.md` | Detailed progress tracking |
| `planml.md` | Original 8-week ML roadmap |
| `FINAL_REPORT.md` | Complete academic report |
| `ACADEMIC_SUBMISSION_REPORT.md` | Academic submission details |
| `PRESENTATION_SLIDES.md` | PowerPoint presentation outline |
| `PRESENTATION_TALKING_POINTS.md` | Presentation talking points |

---

## Academic Project

**MIS587 Final Project - Massachusetts Industrial Properties Price Prediction and Market Analysis**

Presented to: Lornell Real Estate

---

## License

Academic project for MIS587 - Fall 2025.
