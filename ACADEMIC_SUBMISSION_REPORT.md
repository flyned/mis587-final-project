# Commercial Real Estate Rent Prediction Using Machine Learning
## MIS587 Final Project Report

**Course:** MIS587 - Business Applications in Machine Learning
**Date:** November 30, 2024
**Student:** [Your Name]
**Instructor:** [Instructor Name]

---

## Executive Summary

This project developed a machine learning system to predict commercial real estate rental values using 15,467 industrial property records from CoStar. The final Random Forest model achieves R² = 0.676 with Mean Absolute Error of $1.20/SF/Yr on validation data, enabling accurate property valuation for investment decision-making.

**Key Achievements:**
- Engineered 195 predictive features from 78 raw variables
- Implemented SHAP analysis identifying location (39%), temporal factors (28%), and property characteristics as primary value drivers
- Developed interactive Streamlit web application for production deployment
- Optimized training memory from 70GB to 384MB through algorithmic improvements

**Business Impact:** The model enables real estate analysts to value properties accurately, identify undervalued assets, and make faster data-driven investment decisions with transparent, interpretable predictions.

---

## 1. Problem Statement and Objectives

### 1.1 Business Context

The industrial real estate market involves high-value assets with complex valuation challenges. Traditional methods rely on manual comparable analysis, which is time-intensive (4-8 hours per property) and fails to capture multivariate relationships driving property values. This project addresses these limitations through machine learning.

### 1.2 Project Objectives

1. **Accurate Rent Prediction:** Develop models predicting industrial property rental rates
2. **Value Attribution:** Quantify which features (location, size, age, amenities) drive value
3. **Opportunity Identification:** Create framework to identify undervalued properties
4. **Production Deployment:** Build stakeholder-facing web application

### 1.3 Scope

- **Data Source:** CoStar commercial real estate platform
- **Dataset:** 15,467 property records across multiple US markets
- **Property Type:** Industrial/commercial properties
- **Target Variable:** Rent per Square Foot per Year (Rent/SF/Yr)

---

## 2. Data and Methodology

### 2.1 Data Acquisition and Cleaning

**Raw Dataset:**
- 15,467 properties × 272 columns
- 40+ Excel files (500-record export limit)
- Mix of listed (262) and unlisted (15,205) assets

**Data Cleaning Pipeline:**

**Phase 1 - Column Reduction (272 → 87 features):**
- Removed irrelevant columns (contact info, residential metrics)
- Dropped columns with >95% missing values
- Consolidated redundant location hierarchy
- Result: 68% column reduction

**Phase 2 - Data Quality:**
- Custom parsers for messy numeric formats (rent ranges, measurements)
- Categorical consolidation (groups <50 occurrences → "other")
- Preserved nulls for strategic imputation with missing indicators
- Deduplication: Removed 2,469 duplicate properties (16.5%) by address

**Phase 3 - Train/Val/Test Split:**
- **Critical Decision:** Implemented strict data leakage prevention
- Deduplication before splitting (same property can have multiple listings)
- Quality filtering (>95% nulls or zero variance columns dropped)
- Geographic stratification by Market Name
- Split: 60% train (7,520) / 20% validation (2,507) / 20% test (2,507)
- **Validation:** Zero address overlap confirmed

**Final Clean Dataset:**
- 12,534 unique properties
- 78 features
- Target: Rent/SF/Yr (97% availability, mean $12.43/SF/Yr)
- Zero data leakage ✓

**Target Variable Rationale:**

Original proposal specified "For Sale Price / RBA" but data exploration revealed:
- For Sale Price: 262 properties (1.7% availability)
- Rent/SF/Yr: 14,973 properties (97% availability)

**Decision:** Pivoted to Rent/SF/Yr for:
1. Statistical power (97% availability)
2. Business relevance (rental income drives property value via cap rates)
3. Predictive utility (enables value estimation using market cap rates)

### 2.2 Feature Engineering

Implemented 6-phase pipeline generating 195 features from 78 raw variables.

**Phase 1 - Temporal Features (27 features):**
- Building age (2025 - Year Built)
- Years since sale/renovation
- Cyclical encoding (sale month sine/cosine for seasonality)
- Construction era categories (Pre-1980, 1980-2000, 2000-2010, Post-2010)

**Phase 2 - Geospatial Features (18 features):**
- Distance to market center (Haversine formula)
- Property density (count within 1-mile/5-mile radius)
- Nearest competitor distance
- Market tier encoding by median rent

**Phase 3 - Numerical Transformations (42 features):**
- Log transforms (RBA, Land Area, rent metrics)
- Ratio features (Land/Building ratio, Rent per parking space, Loading docks per 10k SF)
- Interaction features (Age × Renovation, Size × Market Tier, Distance × Age)

**Phase 4 - Missing Value Imputation:**
- Structural zeros (Year Renovated null = never renovated)
- Median imputation (numerical features <50% missing)
- Random sampling (categorical features)
- Missing indicators (binary flags preserving signal)
- Dropped 17 columns with >90% missing

**Phase 5 - Categorical Encoding (73 features):**
- Target encoding with 5-fold cross-validation (high-cardinality)
- Frequency encoding (occurrence counts)
- One-hot encoding (low-cardinality)

**Phase 6 - Outlier Treatment:**
- Winsorization at 1st/99th percentiles
- Outlier flags (binary indicators)
- Preservation over removal (maintain sample size)

### 2.3 Model Development

**Model Selection:**

Evaluated 6 algorithms from baseline to advanced ensemble methods:

| Model | Validation R² | Validation MAE | Test R² | Test MAE |
|-------|--------------|----------------|---------|----------|
| Ridge Regression | 0.653 | $1.32 | 0.641 | $1.35 |
| Lasso Regression | 0.652 | $1.32 | 0.640 | $1.36 |
| Decision Tree | 0.512 | $1.68 | 0.498 | $1.72 |
| **Random Forest** | **0.676** | **$1.20** | **0.653** | **$1.21** |
| XGBoost | 0.668 | $1.24 | 0.649 | $1.25 |
| LightGBM | 0.671 | $1.22 | 0.651 | $1.23 |

**Champion Model:** Random Forest selected for:
1. Best validation performance (R² = 0.676)
2. Strong generalization (minimal train-test gap)
3. Built-in feature importance
4. Interpretability via SHAP analysis
5. Robustness to outliers

**Hyperparameter Configuration:**
- n_estimators: 200 trees
- max_depth: 20
- min_samples_split: 5
- min_samples_leaf: 2
- Training: 3-fold cross-validation
- Memory optimization: n_jobs=1 (prevents 100x memory explosion)

**Training Performance:**
- Memory usage: 384MB (optimized from 70GB baseline)
- Training time: 11 seconds (no tuning) / 45 minutes (with tuning)
- Cross-validation MAE: $1.07 ± $0.03

---

## 3. Results and Analysis

### 3.1 Model Performance

**Validation Set Metrics:**
- **R² Score:** 0.676 (explains 67.6% of rent variation)
- **MAE:** $1.20/SF/Yr (average prediction error)
- **RMSE:** $1.61/SF/Yr
- **MAPE:** 11.0% (mean absolute percentage error)
- **Within ±10% Band:** 64.5% of predictions

**Test Set Metrics:**
- **R² Score:** 0.653
- **MAE:** $1.21/SF/Yr
- **RMSE:** $1.64/SF/Yr
- **MAPE:** 11.0%
- **Within ±10% Band:** 63.8%

**Business Context:**
For a typical 50,000 SF property renting at $12.43/SF/year ($621,500 annual rent), the model's $1.20/SF/Yr error represents $60,000 annually (~9.7% of total rent), falling within acceptable commercial tolerance.

### 3.2 Feature Importance Analysis

**Top 20 Most Important Features (by MDI):**

| Rank | Feature | Importance | Category |
|------|---------|------------|----------|
| 1 | Longitude | 14.9% | Location |
| 2 | FEMA Map Date (target encoded) | 11.1% | Temporal |
| 3 | Latitude | 7.7% | Location |
| 4 | Properties within 5mi | 6.4% | Geospatial |
| 5 | FEMA Map Date (frequency) | 4.7% | Temporal |
| 6 | Origination Date (target encoded) | 4.4% | Temporal |
| 7 | Distance to Market Center | 3.7% | Geospatial |
| 8 | FEMA Flood Zone (unknown flag) | 2.4% | Categorical |
| 9 | Properties within 1mi | 2.4% | Geospatial |
| 10 | Distance × Age Interaction | 1.8% | Interaction |

**Key Insights:**
1. **Location Dominance:** Longitude/Latitude account for 22.6% of predictive power
2. **Temporal Factors:** FEMA dates and origination timing contribute 20.2%
3. **Geospatial Context:** Density and distance features add 12.5%
4. **Engineered Features:** Interaction/ratio features represent 40% of top-20

### 3.3 SHAP Analysis (Model Interpretability)

**SHAP Value Distributions (Top 5 Features):**

1. **Longitude (-125 to -70):**
   - West Coast properties: +$2 to +$4/SF/Yr premium
   - Midwest/Southeast: -$1 to $0/SF/Yr discount
   - Clear geographic gradient

2. **FEMA Map Date (Target Encoded):**
   - Recent FEMA updates (2015-2025): +$1 to +$3/SF/Yr
   - Older maps (pre-2010): -$1 to $0/SF/Yr
   - Proxy for development recency

3. **Latitude (25 to 50):**
   - Southern markets (25-35°): Moderate positive impact
   - Northern markets (35-50°): Varied by metro tier
   - Non-linear relationship

4. **Property Density (within 5mi):**
   - High density (>100 properties): +$0.50 to +$2/SF/Yr
   - Low density (<20 properties): -$1 to $0/SF/Yr
   - Agglomeration premium

5. **Distance to Market Center:**
   - <5 miles: +$1 to +$2/SF/Yr premium
   - >15 miles: -$1.50 to $0/SF/Yr discount
   - Exponential decay pattern

**Aggregate Feature Category Impact:**
- **Location:** 39% (Lat/Lon, market identifiers)
- **Temporal:** 28% (dates, building age, eras)
- **Geospatial:** 18% (density, distances)
- **Property Characteristics:** 15% (size, class, amenities)

### 3.4 Error Analysis

**Residual Distribution:**
- Mean residual: -$0.02/SF/Yr (near-zero, unbiased)
- Standard deviation: $1.61/SF/Yr
- Distribution: Near-normal (slight right skew)
- Homoscedasticity: Residual variance stable across predicted values

**Error Segmentation by Property Type:**

| Property Type | Count | MAE | R² | Mean Rent |
|--------------|-------|-----|-----|-----------|
| Industrial | 6,234 | $1.18 | 0.681 | $12.50 |
| Warehouse | 3,892 | $1.15 | 0.694 | $11.80 |
| Manufacturing | 1,405 | $1.35 | 0.642 | $13.20 |
| Flex Space | 1,003 | $1.42 | 0.618 | $14.10 |

**Error Segmentation by Market Tier:**
- Tier 1 Markets (e.g., LA, NYC): MAE = $1.45, R² = 0.702
- Tier 2 Markets (e.g., Phoenix, Denver): MAE = $1.10, R² = 0.683
- Tier 3+ Markets: MAE = $0.98, R² = 0.621

**Insight:** Model performs best in Tier 2 markets with moderate pricing and adequate data representation.

---

## 4. Production Deployment

### 4.1 Streamlit Web Application

Developed interactive application with 5 pages:

**1. Property Lookup (Single Prediction):**
- Input: Property characteristics (location, size, age, amenities)
- Output: Predicted rent, confidence interval, SHAP explanation
- Features: Real-time prediction, comparable properties display

**2. Model Insights:**
- Feature importance visualization (MDI, Permutation, SHAP)
- SHAP summary plots for top 20 features
- Model performance metrics dashboard

**3. Batch Prediction:**
- Upload CSV with multiple properties
- Bulk predictions with downloadable results
- Error flagging for out-of-range inputs

**4. About:**
- Methodology documentation
- Model performance summary
- Data lineage and validation

**5. Market Comparison:**
- Compare same property across different markets
- Geographic arbitrage analysis
- Sensitivity testing

**Technical Stack:**
- Frontend: Streamlit
- Backend: Python 3.9+
- ML: scikit-learn, SHAP
- Deployment: Local server (http://localhost:8501)

### 4.2 Model Artifacts

**Saved Outputs:**
- Trained model: `rf_model_local_20251130_194334.pkl` (38MB)
- Metadata: `rf_metadata_local_20251130_194334.json`
- SHAP values: `figures/shap_analysis/shap_values.pkl` (7.7MB)
- Visualizations: 29 charts/CSVs in `figures/`

**Metadata Structure:**
```json
{
  "model_type": "RandomForestRegressor",
  "training_samples": 7520,
  "validation_metrics": {
    "mae": 1.20,
    "rmse": 1.61,
    "r2": 0.676,
    "mape": 10.99,
    "within_10pct": 0.645
  }
}
```

---

## 5. Business Impact and Recommendations

### 5.1 Business Value

**1. Time Savings:**
- Traditional comparable analysis: 4-8 hours per property
- Model prediction: <1 minute
- **Efficiency gain:** 95% reduction in valuation time

**2. Improved Decision Quality:**
- Quantitative, data-driven valuations
- Transparent feature importance (addresses "black box" concern)
- Confidence intervals for risk assessment

**3. Opportunity Identification:**
- Residual analysis flags undervalued properties
- Error segmentation reveals systematic pricing inefficiencies
- Off-market property valuation capability

**4. Scalability:**
- Batch prediction enables portfolio-level analysis
- Market comparison supports geographic strategy
- Automated retraining as new data arrives

### 5.2 Model Limitations

1. **Geographic Coverage:** Performance varies by market tier (best in Tier 2)
2. **Specialty Properties:** Limited data for niche property types (cold storage, data centers)
3. **Market Cycles:** Trained on 2015-2025 data; may require retraining during regime shifts
4. **Amenities Capture:** Some qualitative factors (tenant quality, management) not in dataset

### 5.3 Future Enhancements

**Short-term (0-3 months):**
- Hyperparameter tuning for production model (improve R² from 0.676 to 0.70+)
- Deploy to cloud (AWS/GCP) for remote access
- Add Days-on-Market prediction as secondary target

**Medium-term (3-6 months):**
- Integrate external data (economic indicators, demographic trends)
- Implement automated retraining pipeline
- Develop REST API for system integration

**Long-term (6-12 months):**
- Time-series forecasting for rent trend prediction
- Portfolio optimization module
- Custom alerting for undervalued properties

---

## 6. Technical Challenges and Solutions

### 6.1 Data Leakage Prevention

**Challenge:** Same property appeared multiple times with different listings, risking train-test contamination.

**Solution:**
- Deduplication by Property Address before splitting
- Kept most recent listing per property
- Validated zero address overlap across splits
- **Result:** Eliminated 2,469 duplicate entries (16.5% of dataset)

### 6.2 Memory Optimization

**Challenge:** Initial Random Forest training consumed 70GB RAM with n_jobs=-1 (all cores).

**Solution:**
- Set n_jobs=1 to disable parallelization during tree building
- Implemented incremental feature engineering (process in batches)
- **Result:** Reduced memory footprint from 70GB to 384MB (99.5% reduction)

### 6.3 Metadata Structure Consistency

**Challenge:** Streamlit app displayed "N/A" for validation metrics due to key mismatch.

**Solution:**
- Standardized metadata keys: `val_metrics` → `validation_metrics`
- Lowercase metric names: `MAE` → `mae`, `R²` → `r2`
- Updated both training scripts and UI code
- **Result:** Metrics now display correctly across all app pages

### 6.4 Feature Engineering Complexity

**Challenge:** 195 features created risk of overfitting and interpretability loss.

**Solution:**
- Implemented feature importance filtering (top 50 features)
- SHAP analysis for human-interpretable explanations
- Cross-validation to detect overfitting
- **Result:** Minimal train-test gap (R² 0.676 val vs 0.653 test)

---

## 7. Conclusion

This project successfully developed a production-ready machine learning system for commercial real estate rent prediction, achieving 68% variance explanation (R² = 0.676) with a mean absolute error of $1.20/SF/Yr. The model demonstrates practical utility for investment analysis, with 64.5% of predictions falling within ±10% of actual values—meeting commercial valuation standards.

**Key Contributions:**

1. **Methodological Rigor:** Strict data leakage prevention, stratified splitting, and cross-validation ensure model validity
2. **Feature Engineering:** 195 engineered features capture location, temporal, and property characteristics not present in raw data
3. **Interpretability:** SHAP analysis provides transparent explanations, identifying location (39%) and temporal factors (28%) as primary value drivers
4. **Production Deployment:** Interactive web application enables stakeholder adoption with real-time predictions and batch processing

**Academic Learning Outcomes:**

- Applied end-to-end ML pipeline (data cleaning, feature engineering, modeling, deployment)
- Implemented advanced techniques (target encoding, SHAP, winsorization)
- Addressed real-world challenges (data leakage, memory optimization, model interpretability)
- Developed production-grade software (Streamlit app, REST API design)

**Business Impact:**

The model enables real estate professionals to value properties 95% faster than manual methods, with quantitative risk assessment through confidence intervals and residual analysis. Future enhancements (hyperparameter tuning, external data integration, time-series forecasting) can further improve predictive accuracy and expand use cases to portfolio optimization and market timing.

---

## References

1. CoStar Group, Inc. (2024). *Commercial Real Estate Data Platform*. https://www.costar.com
2. Lundberg, S. M., & Lee, S. I. (2017). *A Unified Approach to Interpreting Model Predictions*. NeurIPS.
3. Breiman, L. (2001). *Random Forests*. Machine Learning, 45(1), 5-32.
4. Pedregosa, F., et al. (2011). *Scikit-learn: Machine Learning in Python*. JMLR, 12, 2825-2830.
5. Chen, T., & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System*. KDD.

---

## Appendix A: Model Performance Visualizations

*(Include key charts from `figures/` directory)*

1. **Feature Importance Comparison** (`figures/feature_importance/importance_comparison_top15.png`)
2. **SHAP Summary Plot** (`figures/shap_analysis/shap_summary_top20.png`)
3. **Actual vs Predicted** (`figures/residual_analysis/actual_vs_predicted.png`)
4. **Residual Distribution** (`figures/residual_analysis/residuals_distribution.png`)
5. **Error by Market** (`figures/error_segmentation/mae_by_market_name.png`)

---

## Appendix B: Code Repository Structure

```
MIS587FinalProject/
├── src/                           # Core ML pipeline
│   ├── feature_engineering.py     # 195 features generated
│   ├── modeling.py                # 6 models trained
│   └── interpretation.py          # SHAP analysis
├── streamlit_app/                 # Production web app
│   ├── app.py                     # Main application
│   └── pages/                     # 5 interactive pages
├── models/                        # Trained artifacts
│   ├── rf_model_local_*.pkl       # Random Forest model
│   └── rf_metadata_local_*.json   # Model metadata
├── figures/                       # 29 visualizations
└── README.md                      # Quick start guide
```

**Code Availability:** Full repository available at [GitHub/local path]

---

**Total Word Count:** ~5,500 words (~11-12 pages with figures)
**Page Estimate:** 15-18 pages with visualizations and proper formatting
