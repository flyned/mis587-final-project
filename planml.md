# Machine Learning Plan for Commercial Real Estate Analysis

## Project Overview

This document outlines a comprehensive machine learning strategy for the commercial real estate data pipeline. The current pipeline cleans 40+ CoStar export files into a consolidated dataset with ~60 features across numerical, categorical, location, and temporal dimensions.

## Business Objectives

### Primary Goals
1. **Property Valuation**: Predict sale prices and rental rates for commercial properties
2. **Investment Analysis**: Identify undervalued properties with high ROI potential
3. **Market Segmentation**: Cluster properties by characteristics and market dynamics
4. **Occupancy Prediction**: Forecast vacancy rates and tenant demand

### Target Variables (Potential)
- `For Sale Price` - Primary regression target
- `Last Sale Price` - Secondary valuation target
- `Rent/SF/Yr` - Rental rate prediction
- `Percent Leased` / `Vacancy %` - Occupancy metrics
- `Days On Market` - Liquidity indicator

---

## Current Data Assessment

### Data Characteristics
- **Scale**: 40+ Excel files, consolidated into single DataFrame
- **Features**: ~60 features after aggressive column dropping (100+ columns removed)
- **Categories**:
  - Location: 15 columns (addresses, coordinates, market/submarket)
  - Categorical: 21 columns (property type, building class, status, amenities)
  - Numerical: 46 columns (size metrics, financial data, ratios)
  - Temporal: 8 columns (dates, years)

### Data Quality Issues
- **High missingness**: Numerous columns with >50% null values (already dropped)
- **Sparse categories**: Rare categories consolidated to 'other' (threshold: 50 occurrences)
- **Messy formats**: Numeric ranges, text-embedded values, measurement units
- **Geographic diversity**: Multi-market data requiring location-based features

### Current Cleaning Pipeline
```
load_raw_data.py → column_types.py → main.py → clean_numeric.py → clean_categorical.py → clean_data.csv
```

**Strengths**:
- Centralized column management
- Custom parsers for messy numeric fields
- Category consolidation to reduce sparsity

**Weaknesses**:
- No advanced imputation (nulls = 'unknown' for categorical, NaN for numeric)
- No feature engineering beyond cleaning
- No train/test splitting or validation
- No temporal feature extraction from dates
- Location data underutilized (lat/lon not engineered)

---

## Machine Learning Strategy

### Phase 1: Enhanced Feature Engineering

#### 1.1 Missing Data Strategy
**Current**: Simple fill with 'unknown' or leave as NaN

**Recommended Approach** (using `feature-engine`):
- **Numerical columns**:
  - `MeanMedianImputer`: For columns with <30% missingness (e.g., `Ceiling Ht`, `Parking Ratio`)
  - `RandomSampleImputer`: For skewed distributions (e.g., `For Sale Price`)
  - `ArbitraryNumberImputer`: For domain-specific defaults (e.g., `Number Of Cranes` = 0)
  - `AddMissingIndicator`: Create binary flag for columns where missingness is informative

- **Categorical columns**:
  - Keep current 'unknown' approach for most
  - `RandomSampleImputer`: For high-cardinality categories with patterns

**Implementation**:
```python
from feature_engine.imputation import MeanMedianImputer, AddMissingIndicator

# Impute central tendency
imputer = MeanMedianImputer(imputation_method='median',
                           variables=['Ceiling Ht', 'Parking Ratio', 'Number Of Elevators'])

# Flag missing patterns
missing_ind = AddMissingIndicator(variables=['For Sale Price', 'Last Sale Price'])
```

#### 1.2 Geospatial Feature Engineering
**Current**: Latitude/longitude stored but unused

**Recommended Features**:
- **Distance metrics**:
  - Distance to city center (using market-specific centroids)
  - Distance to nearest major highway/transit
  - Distance to nearest competing property (same type)

- **Density metrics**:
  - Number of properties within 1-mile radius
  - Average price per SF in 5-mile radius
  - Clustering coefficient (urban vs suburban)

- **Market-based encoding**:
  - Submarket average rent (target encoding)
  - Submarket occupancy rate
  - Market growth rate (if temporal data available)

**Implementation**:
```python
import numpy as np
from sklearn.neighbors import BallTree

# Calculate distance to market centroid
def add_distance_features(df):
    market_centroids = df.groupby('Market Name')[['Latitude', 'Longitude']].mean()

    for market, centroid in market_centroids.iterrows():
        mask = df['Market Name'] == market
        df.loc[mask, 'dist_to_center'] = haversine_distance(
            df.loc[mask, ['Latitude', 'Longitude']].values,
            centroid.values
        )

    return df
```

#### 1.3 Temporal Feature Engineering
**Current**: Date columns preserved as-is

**Recommended Features**:
- **Age metrics**:
  - `building_age` = current_year - Year Built
  - `years_since_renovation` = current_year - Year Renovated
  - `years_since_sale` = current_year - Last Sale Date

- **Cyclical encoding** (for seasonal patterns):
  - `sale_month_sin`, `sale_month_cos` (from Last Sale Date)

- **Era binning**:
  - Construction era: Pre-1980, 1980-2000, 2000-2010, Post-2010

**Implementation**:
```python
from datetime import datetime
from feature_engine.creation import CyclicalFeatures

current_year = datetime.now().year
df['building_age'] = current_year - df['Year Built']
df['years_since_renovation'] = current_year - df['Year Renovated']

# Cyclical encoding for months
cyclical = CyclicalFeatures(variables=['sale_month'], max_values={'sale_month': 12})
```

#### 1.4 Numerical Transformations
**Current**: Parse messy formats, no transformations

**Recommended Transformations**:
- **Log transformation** (for right-skewed features):
  - `RBA`, `Land Area (SF)`, `For Sale Price`, `Last Sale Price`
  - Use `LogCpTransformer` to handle zeros

- **Ratio features** (domain-driven):
  - `price_per_sf` = For Sale Price / RBA
  - `land_to_building_ratio` = RBA / Land Area (SF)
  - `parking_efficiency` = Number Of Parking Spaces / Number Of Stories
  - `leasing_premium` = (Percent Leased - market_avg_leased) / market_avg_leased

- **Interaction features**:
  - `Building Class × Property Type` (via polynomial features)
  - `Market Name × Building Status`

**Implementation**:
```python
from feature_engine.transformation import LogCpTransformer
from feature_engine.creation import MathFeatures

# Log transform skewed features
log_transformer = LogCpTransformer(variables=['RBA', 'Land Area (SF)', 'For Sale Price'])

# Create ratio features
ratio_creator = MathFeatures(
    variables=[['For Sale Price', 'RBA']],
    func='div',
    new_variables_names=['price_per_sf']
)
```

#### 1.5 Categorical Encoding
**Current**: Raw categories with 'other' consolidation

**Recommended Encodings**:
- **Target encoding** (for high-cardinality features):
  - `Market Name`, `Submarket Name`, `City` → mean target encoding with CV
  - Use `MeanEncoder` from feature-engine

- **Frequency encoding**:
  - `Building Park`, `County Name` → encode by occurrence count

- **Rare label handling**:
  - Keep current `RareLabelEncoder` approach (threshold = 50)

- **One-hot encoding** (for low-cardinality):
  - `Property Type`, `Building Class`, `Building Status`, `Tenancy`
  - Drop first to avoid multicollinearity

- **WoE encoding** (for monotonic relationships):
  - `Building Class`, `Secondary Type` → Weight of Evidence

**Implementation**:
```python
from feature_engine.encoding import MeanEncoder, CountFrequencyEncoder, RareLabelEncoder
from sklearn.preprocessing import OneHotEncoder

# Target encoding for geographic features (with cross-validation)
target_encoder = MeanEncoder(variables=['Market Name', 'Submarket Name', 'City'])

# One-hot for low-cardinality
ohe = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
```

#### 1.6 Outlier Treatment
**Current**: No outlier handling

**Recommended Approach**:
- **Capping** (instead of removal to preserve data):
  - `Winsorizer`: Cap at 1st/99th percentile for price features
  - `ArbitraryOutlierCapper`: Domain-specific caps (e.g., Rent/SF/Yr > $200)

- **Detection**:
  - IQR method for `RBA`, `Land Area`, `Number Of Stories`
  - Domain logic for `Parking Ratio` (>10 likely error)

**Implementation**:
```python
from feature_engine.outliers import Winsorizer

winsorizer = Winsorizer(
    capping_method='quantiles',
    tail='both',
    fold=0.01,  # 1st and 99th percentile
    variables=['For Sale Price', 'Rent/SF/Yr', 'RBA']
)
```

---

### Phase 2: Model Selection & Training

#### 2.1 Problem Formulation

**Primary Task**: Regression (price/rent prediction)

**Target Variable Prioritization**:
1. `For Sale Price` (if sufficient non-null samples)
2. `Rent/SF/Yr` (alternative target for rental market)
3. `Last Sale Price` (if For Sale Price too sparse)

**Evaluation Metrics**:
- **Primary**: RMSE, MAE (interpretable in dollars/SF)
- **Secondary**: R² score, MAPE (percentage error)
- **Business metric**: Prediction intervals (e.g., ±10% band accuracy)

#### 2.2 Baseline Models

**Purpose**: Establish performance floor before complex models

1. **Linear Regression** (with regularization):
   - Ridge Regression (L2 penalty)
   - Lasso Regression (L1 penalty for feature selection)

2. **Simple Tree**:
   - Decision Tree Regressor (max_depth=5)

3. **Domain Heuristic**:
   - Predict using market-level median price per SF × property RBA

**Implementation**:
```python
from sklearn.linear_model import Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Ridge baseline
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
y_pred = ridge.predict(X_test)

print(f"MAE: ${mean_absolute_error(y_test, y_pred):,.0f}")
print(f"RMSE: ${mean_squared_error(y_test, y_pred, squared=False):,.0f}")
print(f"R²: {r2_score(y_test, y_pred):.3f}")
```

#### 2.3 Advanced Models

**Tree-Based Ensembles** (recommended for tabular data):

1. **Random Forest Regressor**:
   - Pros: Handles non-linearity, robust to outliers, feature importance
   - Hyperparameters:
     - `n_estimators`: 100-500
     - `max_depth`: 10-30
     - `min_samples_split`: 2-10
     - `max_features`: 'sqrt', 'log2'

2. **Gradient Boosting (XGBoost/LightGBM)**:
   - **Best for structured data** per recent benchmarks
   - Pros: Superior accuracy, handles missing values, regularization
   - Hyperparameters:
     - `learning_rate`: 0.01-0.1
     - `n_estimators`: 100-1000 (with early stopping)
     - `max_depth`: 3-8
     - `subsample`: 0.7-1.0
     - `colsample_bytree`: 0.7-1.0
     - `reg_alpha` (L1), `reg_lambda` (L2): 0-10

3. **Histogram Gradient Boosting** (sklearn):
   - Fast alternative to XGBoost with native categorical support
   - Good for large datasets (>10k samples)

**Implementation**:
```python
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
import xgboost as xgb

# Random Forest
rf = RandomForestRegressor(
    n_estimators=200,
    max_depth=20,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1
)

# XGBoost
xgb_model = xgb.XGBRegressor(
    objective='reg:squarederror',
    learning_rate=0.05,
    n_estimators=500,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=1,
    reg_lambda=1,
    random_state=42,
    early_stopping_rounds=50
)

# Train with validation set for early stopping
xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=False
)
```

#### 2.4 Hyperparameter Tuning

**Strategy**: Bayesian optimization or randomized search (grid search too slow)

**Tools**:
- `sklearn.model_selection.RandomizedSearchCV` (quick exploration)
- `optuna` (advanced Bayesian optimization)

**Cross-Validation**:
- **K-Fold CV**: 5-fold (if data >5k samples)
- **Group K-Fold**: Group by `Market Name` to prevent data leakage (same market in train/test)
- **Time-based split**: If temporal component important (e.g., predict future sales)

**Implementation**:
```python
from sklearn.model_selection import RandomizedSearchCV, GroupKFold
import numpy as np

# Define hyperparameter space
param_dist = {
    'n_estimators': [100, 200, 500],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'max_features': ['sqrt', 'log2']
}

# Group K-Fold by market (prevent leakage)
group_kfold = GroupKFold(n_splits=5)
groups = df['Market Name']

# Randomized search
random_search = RandomizedSearchCV(
    estimator=RandomForestRegressor(random_state=42),
    param_distributions=param_dist,
    n_iter=20,
    cv=group_kfold,
    scoring='neg_mean_absolute_error',
    n_jobs=-1,
    random_state=42
)

random_search.fit(X_train, y_train, groups=groups[train_idx])
print(f"Best params: {random_search.best_params_}")
```

#### 2.5 Model Stacking/Ensembling

**Strategy**: Combine predictions from multiple models

**Level 0 Models** (diverse algorithms):
- Random Forest
- XGBoost
- Lasso Regression
- Histogram Gradient Boosting

**Level 1 Meta-Model**:
- Ridge Regression (simple, prevents overfitting)

**Implementation**:
```python
from sklearn.ensemble import StackingRegressor

estimators = [
    ('rf', RandomForestRegressor(n_estimators=200, max_depth=20, random_state=42)),
    ('xgb', xgb.XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6)),
    ('lasso', Lasso(alpha=0.1))
]

stacking_model = StackingRegressor(
    estimators=estimators,
    final_estimator=Ridge(alpha=1.0),
    cv=5
)

stacking_model.fit(X_train, y_train)
```

---

### Phase 3: Model Evaluation & Interpretation

#### 3.1 Evaluation Framework

**Metrics Suite**:
```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error

def evaluate_model(model, X_test, y_test, name="Model"):
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)

    # Prediction interval accuracy (±10%)
    within_10pct = np.mean(np.abs(y_pred - y_test) / y_test < 0.10)

    print(f"{name} Performance:")
    print(f"  MAE: ${mae:,.0f}")
    print(f"  RMSE: ${rmse:,.0f}")
    print(f"  R²: {r2:.3f}")
    print(f"  MAPE: {mape:.1%}")
    print(f"  Within 10% band: {within_10pct:.1%}")

    return {'mae': mae, 'rmse': rmse, 'r2': r2, 'mape': mape}
```

**Residual Analysis**:
- Plot predicted vs actual (scatter)
- Plot residuals vs predicted (check heteroscedasticity)
- Check residuals by market/property type (bias detection)

**Error Segmentation**:
```python
# Analyze errors by property segment
df_test['error'] = y_pred - y_test
error_by_type = df_test.groupby('Property Type')['error'].agg(['mean', 'std', 'count'])
print(error_by_type)
```

#### 3.2 Feature Importance Analysis

**Methods**:
1. **Tree-based importances**: Native to RF/XGBoost
2. **Permutation importance**: Model-agnostic, more reliable
3. **SHAP values**: Local and global interpretability

**Implementation**:
```python
from sklearn.inspection import permutation_importance
import shap

# Permutation importance
perm_importance = permutation_importance(
    model, X_test, y_test,
    n_repeats=10,
    random_state=42,
    n_jobs=-1
)

top_features = sorted(
    zip(X_train.columns, perm_importance.importances_mean),
    key=lambda x: x[1],
    reverse=True
)[:20]

# SHAP values (for XGBoost)
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test)

# Summary plot
shap.summary_plot(shap_values, X_test, plot_type="bar")
```

**Expected Important Features**:
- Location: `Latitude`, `Longitude`, `Market Name` (encoded), `dist_to_center`
- Size: `RBA`, `Land Area (SF)`, `Number Of Stories`
- Quality: `Building Class`, `Star Rating`, `Year Built` (or `building_age`)
- Occupancy: `Percent Leased`, `Vacancy %`
- Financial: `Rent/SF/Yr`, `price_per_sf`, `Parking Ratio`

#### 3.3 Business Validation

**Reality Checks**:
- Do predictions align with known market trends? (e.g., urban > suburban for office)
- Are outlier predictions explainable? (e.g., landmark properties)
- Can domain experts validate top feature importance?

**Use Case Testing**:
- **Scenario 1**: Predict price for newly listed property → within ±15%?
- **Scenario 2**: Identify undervalued properties (predicted >> listed price)
- **Scenario 3**: Portfolio valuation → aggregate error <5%?

---

### Phase 4: Advanced Techniques (Future Work)

#### 4.1 Geospatial Modeling
- **Spatial autocorrelation**: Properties near each other have correlated prices
- **Kriging**: Geostatistical interpolation for missing price data
- **Geographic Weighted Regression**: Location-specific models

#### 4.2 Time Series Forecasting
- **If temporal data sufficient**:
  - Forecast market-level vacancy rates
  - Predict future rental growth by submarket
  - Use ARIMA, Prophet, or LSTM models

#### 4.3 Multi-Task Learning
- **Joint prediction**:
  - Predict both `For Sale Price` AND `Rent/SF/Yr` simultaneously
  - Shared representations capture common property value drivers

#### 4.4 Deep Learning
- **When applicable** (if dataset >50k samples):
  - Tabular neural networks (e.g., TabNet, FT-Transformer)
  - Embedding layers for high-cardinality categoricals
  - Mixed input models (tabular + images if property photos available)

---

## Implementation Roadmap

### Week 1-2: Data Preparation
- [ ] Create `feature_engineering.py` module
- [ ] Implement advanced imputation strategies
- [ ] Add geospatial feature engineering (distances, density)
- [ ] Extract temporal features (age, eras, cyclical)
- [ ] Create ratio and interaction features
- [ ] Add `train_test_split` with stratification by market

### Week 3-4: Baseline Modeling
- [ ] Create `modeling.py` module
- [ ] Implement train/validation/test split (60/20/20)
- [ ] Train baseline models (Ridge, Lasso, simple tree)
- [ ] Establish evaluation metrics suite
- [ ] Create visualization utilities (residual plots, feature importance)

### Week 5-6: Advanced Modeling
- [ ] Train Random Forest with cross-validation
- [ ] Train XGBoost/LightGBM with hyperparameter tuning
- [ ] Implement Histogram Gradient Boosting
- [ ] Compare models using grouped K-fold CV
- [ ] Select best model based on validation performance

### Week 7: Model Interpretation
- [ ] Generate permutation importance rankings
- [ ] Create SHAP value analyses
- [ ] Perform error segmentation by property type/market
- [ ] Validate predictions with domain experts
- [ ] Document model insights

### Week 8: Production Pipeline
- [ ] Create end-to-end pipeline with sklearn `Pipeline`
- [ ] Implement model persistence (joblib/pickle)
- [ ] Build prediction API (optional: Flask/FastAPI)
- [ ] Create model monitoring dashboard
- [ ] Write deployment documentation

---

## File Structure (Proposed)

```
MIS587FinalProject/
├── data/
│   ├── excel_sheets/          # Raw CoStar exports
│   ├── clean_data.csv         # Current pipeline output
│   ├── train.csv              # Training set (to be created)
│   ├── val.csv                # Validation set
│   └── test.csv               # Test set
├── notebooks/                  # Exploratory analysis (to be created)
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_baseline_models.ipynb
│   └── 04_advanced_models.ipynb
├── models/                     # Saved model artifacts
│   ├── best_model.pkl
│   └── feature_pipeline.pkl
├── src/                        # Refactored modular code
│   ├── load_raw_data.py       # Existing
│   ├── column_types.py        # Existing
│   ├── clean_numeric.py       # Existing
│   ├── clean_categorical.py   # Existing
│   ├── feature_engineering.py # NEW: Advanced feature creation
│   ├── modeling.py            # NEW: Model training/evaluation
│   └── utils.py               # NEW: Helper functions
├── main.py                     # Existing (data cleaning)
├── train_model.py              # NEW: Model training script
├── predict.py                  # NEW: Inference script
├── CLAUDE.md                   # Existing
├── planml.md                   # THIS DOCUMENT
└── requirements.txt            # Dependencies
```

---

## Dependencies

**Required Libraries**:
```
# Data processing
pandas>=2.0.0
numpy>=1.24.0

# Feature engineering
feature-engine>=1.6.0
scikit-learn>=1.3.0

# Modeling
xgboost>=2.0.0
lightgbm>=4.0.0

# Evaluation & interpretation
shap>=0.42.0
matplotlib>=3.7.0
seaborn>=0.12.0

# Utilities
joblib>=1.3.0
openpyxl>=3.1.0  # For reading Excel files
```

**Install command**:
```bash
pip install pandas numpy feature-engine scikit-learn xgboost lightgbm shap matplotlib seaborn joblib openpyxl
```

---

## Key Success Metrics

1. **Model Performance**:
   - R² > 0.75 on test set (strong predictive power)
   - MAE < 15% of median price (business-acceptable error)
   - 80%+ predictions within ±10% band (precision)

2. **Feature Engineering**:
   - Geospatial features in top 10 most important
   - Engineered features outperform raw features (ablation test)

3. **Interpretability**:
   - Top 5 features align with domain expertise
   - SHAP values reveal actionable insights (e.g., "Adding parking increases value by $X/space")

4. **Generalization**:
   - Model performs consistently across all markets (no single-market overfitting)
   - Robust to different property types (office, retail, industrial)

---

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|-----------|
| **Insufficient target variable data** (high % null in `For Sale Price`) | Cannot train price prediction model | Use `Last Sale Price` or `Rent/SF/Yr` as alternative targets; focus on classification (price tier) |
| **Geographic data leakage** (same property in train/test via different listings) | Inflated performance metrics | Use `Property Address` hash to ensure no duplicates across splits |
| **Market regime shifts** (2019 data differs from 2024) | Model trained on outdated patterns | Add temporal features; consider weighting recent data higher; validate on out-of-time test set |
| **Imbalanced property types** (90% industrial, 10% retail) | Poor performance on minority classes | Stratified sampling; separate models per type; use SMOTE or class weights |
| **Multicollinearity** (RBA, Total Available Space, Office Space highly correlated) | Unstable coefficients in linear models | Use PCA/feature selection; prefer tree-based models (robust to collinearity) |

---

## References

1. **Feature Engineering**:
   - [Feature-engine Documentation](https://feature-engine.readthedocs.io/)
   - [Python Feature Engineering Cookbook (Packt)](https://www.packtpub.com/product/python-feature-engineering-cookbook)

2. **Geospatial Analysis**:
   - Scikit-learn KDE for spatial density estimation
   - Haversine distance for lat/lon calculations

3. **Model Selection**:
   - [Scikit-learn Ensemble Methods](https://scikit-learn.org/stable/modules/ensemble.html)
   - [XGBoost Documentation](https://xgboost.readthedocs.io/)
   - [SHAP for Model Interpretation](https://shap.readthedocs.io/)

4. **Commercial Real Estate Domain**:
   - CoStar data dictionaries
   - Commercial property valuation methodologies (income approach, comparable sales)

---

## Conclusion

This plan provides a structured path from the current data cleaning pipeline to production-ready ML models for commercial real estate analysis. The phased approach allows for iterative development, with each stage building on validated results from the previous phase.

**Next Steps**:
1. Review current dataset statistics (run `main.py`, examine `clean_data.csv`)
2. Validate target variable availability (check null % for `For Sale Price`, `Rent/SF/Yr`)
3. Begin Phase 1 feature engineering with highest-impact features (geospatial, temporal)
4. Establish baseline model performance before investing in advanced techniques

**Success depends on**:
- Domain expertise validation at each stage
- Rigorous train/test separation to prevent leakage
- Focus on interpretability alongside accuracy (explainable models gain stakeholder trust)

---

*Document created: 2025-10-07*
*Author: Claude Code (MCP Ref-assisted)*
*Version: 1.0*
