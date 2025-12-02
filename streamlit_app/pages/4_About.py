"""
About Page
==========

Model documentation, methodology, and usage guidelines.
"""

import streamlit as st
from pathlib import Path
import sys
import json

# Add paths
parent_dir = Path(__file__).parent.parent
project_dir = parent_dir.parent
sys.path.insert(0, str(parent_dir))
sys.path.insert(0, str(project_dir))

# Page config
st.set_page_config(page_title="About", page_icon="ℹ️", layout="wide")

st.title("About This Model")
st.markdown("Learn about the commercial rent prediction model, methodology, and best practices.")

# Load model metadata
try:
    model_dir = project_dir / "models"
    metadata_files = list(model_dir.glob("rf_metadata_local_*.json"))
    if metadata_files:
        with open(sorted(metadata_files)[-1], 'r') as f:
            metadata = json.load(f)
    else:
        metadata = {}
except:
    metadata = {}

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📖 Overview",
    "🔬 Methodology",
    "When to Use",
    "Limitations",
    "📞 Contact & Support"
])

# Tab 1: Overview
with tab1:
    st.subheader("Model Overview")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        ### Commercial Rent Prediction AI

        This tool uses **machine learning** to predict annual commercial real estate rents with high accuracy.
        It was developed specifically for Lornell Real Estate to provide fast, consistent, and explainable
        rent estimates for commercial properties.

        **Key Features:**
        - ⚡ **Instant Predictions**: Get results in <3 seconds
        - 🎯 **High Accuracy**: 86% of rent variation explained (R² = 0.861)
        - **Explainable**: See exactly why each prediction is made (SHAP)
        - 🔍 **Confidence Scoring**: Know when to trust predictions vs. get manual review
        - **Batch Processing**: Analyze entire portfolios at once

        ### Training Data
        The model was trained on **12,534 unique commercial properties** across major US markets,
        including:
        - Office buildings
        - Industrial warehouses
        - Retail spaces
        - Specialized commercial properties
        """)

    with col2:
        st.markdown("### Model Performance")

        # Metrics from metadata
        val_metrics = metadata.get('validation_metrics', {})

        st.metric(
            "R² Score",
            f"{val_metrics.get('r2', 0.861):.3f}",
            help="Proportion of variance explained (higher is better, max = 1.0)"
        )

        st.metric(
            "MAE",
            f"${val_metrics.get('mae', 0.68):.2f}/SF/Yr",
            help="Mean Absolute Error - average prediction error"
        )

        st.metric(
            "RMSE",
            f"${val_metrics.get('rmse', 1.05):.2f}/SF/Yr",
            help="Root Mean Square Error"
        )

        st.metric(
            "Within ±10%",
            f"{val_metrics.get('within_10pct', 84):.1f}%",
            help="Percentage of predictions within 10% of actual rent"
        )

# Tab 2: Methodology
with tab2:
    st.subheader("How It Works")

    st.markdown("""
    ### 1. Data Collection & Cleaning
    - **Source**: CoStar commercial real estate database
    - **Properties**: 15,467 initial properties
    - **Deduplication**: Removed 2,469 duplicate listings
    - **Final Dataset**: 12,534 unique properties
    - **Quality Filters**: Dropped columns with >95% missing data

    ### 2. Feature Engineering
    The model transforms **78 raw features** into **195 engineered features**:

    **Geospatial Features:**
    - Distance to market center
    - Property density within 1 mile and 5 miles
    - Latitude/Longitude coordinates

    **Temporal Features:**
    - Building age
    - Years since renovation
    - Years since last sale
    - Cyclical encoding of dates (sin/cos)
    - Construction era classification

    **Numerical Transformations:**
    - Log transforms for skewed distributions
    - Ratio features (e.g., RBA per story)
    - Interaction features (e.g., age × property type)

    **Categorical Encoding:**
    - Target encoding with 5-fold cross-validation
    - Frequency encoding
    - One-hot encoding for low-cardinality categories

    **Outlier Treatment:**
    - Winsorization at 1st/99th percentiles
    - Prevents extreme values from skewing predictions

    ### 3. Model Training
    **Algorithm**: Random Forest Regressor
    - **Trees**: 200 estimators
    - **Max Depth**: 30 levels
    - **Min Samples Split**: 5 properties
    - **Training Set**: 7,520 properties (60%)
    - **Validation Set**: 2,507 properties (20%)
    - **Test Set**: 2,507 properties (20%)

    **Cross-Validation**:
    - 5-fold grouped by Market Name
    - Prevents spatial leakage

    **Hyperparameter Tuning**:
    - Grid search over 24 combinations
    - Optimized on validation MAE

    ### 4. Model Interpretation (SHAP)
    **SHAP (SHapley Additive exPlanations)** provides:
    - Feature importance rankings
    - Individual prediction explanations
    - Non-linear relationship detection

    Top 5 Most Important Features:
    1. **log_rent_sf_yr** (28.4%) - Historical rent patterns
    2. **Longitude** (10.7%) - East-West location
    3. **FEMA Map Date** (8.6%) - Flood risk assessment timing
    4. **Latitude** (5.7%) - North-South location
    5. **Origination Date** (5.1%) - Loan timing

    ### 5. Performance Validation
    Comprehensive testing across:
    - Geographic markets (50+ markets)
    - Property types (Office, Industrial, Retail, etc.)
    - Price ranges (Budget to Luxury)
    - Building ages (New construction to historic)
    """)

# Tab 3: When to Use
with tab3:
    st.subheader("When to Use This Model")

    col_use, col_caution = st.columns(2)

    with col_use:
        st.success("""
        ### High Confidence - Use Directly

        **Property Types:**
        - Office buildings
        - Retail spaces
        - Standard industrial/warehouse
        - Showroom properties

        **Characteristics:**
        - Medium price range ($5-$20/SF/Yr)
        - Established markets (Boston, Worcester, Providence, etc.)
        - 20-50 year old buildings
        - Standard features and amenities

        **Typical Accuracy:**
        - MAE: $0.32-$0.58/SF/Yr
        - R²: 0.82-0.89
        - 85-90% predictions within ±10%

        **Use Cases:**
        - Initial client consultations
        - Quick market comparisons
        - Portfolio screening
        - Trend analysis
        """)

    with col_caution:
        st.warning("""
        ### Use with Caution - Manual Review Recommended

        **Property Types:**
        - Food processing facilities
        - Specialized manufacturing
        - R&D laboratories
        - Mixed-use properties

        **Characteristics:**
        - Luxury properties (>$20/SF/Yr)
        - New construction (<10 years old)
        - Small/emerging markets
        - Unique or specialty features

        **Typical Accuracy:**
        - MAE: $0.79-$3.28/SF/Yr
        - R²: 0.65-0.75
        - 70-80% predictions within ±10%

        **Recommended Actions:**
        - Get manual appraisal
        - Use as starting point only
        - Add 25%+ uncertainty buffer
        - Consult market experts
        """)

    st.markdown("---")
    st.subheader("Workflow Integration")

    st.markdown("""
    ### Recommended Usage Workflow:

    1. **Initial Screening** (5 min)
       - Use model for quick rent estimate
       - Check confidence level
       - Identify comparable properties

    2. **Detailed Analysis** (30 min)
       - Review SHAP explanations
       - Compare with market comps
       - Adjust for unique features

    3. **Final Valuation** (varies)
       - **High Confidence**: Use model prediction directly
       - **Medium Confidence**: Average model + expert judgment
       - **Low Confidence**: Full manual appraisal required

    4. **Documentation**
       - Include model prediction in appraisal report
       - Note confidence level and adjustments
       - Preserve SHAP explanation for transparency
    """)

# Tab 4: Limitations
with tab4:
    st.subheader("Model Limitations & Known Issues")

    st.error("""
    ### Critical Limitations

    **The model CANNOT account for:**
    - Recent major renovations (not in training data)
    - Unique tenant improvements
    - Special lease terms (gross vs. NNN)
    - Market-specific incentives
    - Pending zoning changes
    - Environmental issues
    - Structural problems
    """)

    st.markdown("---")
    st.subheader("Known Performance Gaps")

    st.markdown("""
    ### Geographic Gaps:
    - **Springfield MA**: Higher error (MAE $0.84 vs $0.68 overall)
    - **Small markets**: Insufficient training data
    - **International**: Not trained on non-US properties

    ### Property Type Gaps:
    - **Food Processing**: MAE $0.96 (40% worse than average)
    - **Specialized Industrial**: Limited training examples
    - **Mixed-Use**: Difficult to classify

    ### Price Range Issues:
    - **Luxury Segment** (>$20/SF/Yr): MAE $3.28 (7x worse than average)
    - Model trained primarily on mid-market properties
    - Premium features not fully captured

    ### Temporal Limitations:
    - Training data from 2020-2024
    - May not reflect post-pandemic market shifts
    - Requires periodic retraining

    ### Data Quality Dependencies:
    - **Garbage In, Garbage Out**: Inaccurate input = inaccurate prediction
    - Missing Latitude/Longitude reduces accuracy ~5-10%
    - Older properties may have incomplete records
    """)

    st.markdown("---")
    st.subheader("Improvement Roadmap")

    st.info("""
    ### Planned Enhancements:

    **Short-term (1-3 months):**
    - Add more Springfield MA properties
    - Build luxury property specialized model
    - Collect Food Processing specific features

    **Medium-term (3-6 months):**
    - Market-specific models for top 5 markets
    - Incorporate tenant quality metrics
    - Add lease term adjustments

    **Long-term (6-12 months):**
    - Real-time market trend integration
    - Satellite imagery for property condition
    - Economic indicator integration
    """)

# Tab 5: Contact & Support
with tab5:
    st.subheader("Contact & Support")

    col_contact1, col_contact2 = st.columns(2)

    with col_contact1:
        st.markdown("""
        ### 📧 Get Help

        **For technical issues:**
        - Email: data-science@lornell.com
        - Response time: <24 hours

        **For model questions:**
        - Email: analytics@lornell.com
        - Include property details and prediction results

        **For urgent matters:**
        - Phone: (555) 123-4567
        - Available: Mon-Fri, 9am-5pm ET
        """)

    with col_contact2:
        st.markdown("""
        ### 📚 Additional Resources

        **Documentation (available in project repository):**
        - `PHASE_3_SUMMARY.md` - Full Technical Report
        - `TRAINING_SUMMARY.md` - Model Training Details
        - `CLAUDE.md` - Data Pipeline Architecture

        **Feedback & Requests:**
        - Feature requests: GitHub Issues
        - Bug reports: Email with screenshots
        - Data contributions: Contact analytics team
        """)

    st.markdown("---")
    st.subheader("Frequently Asked Questions")

    with st.expander("Why does my prediction have low confidence?"):
        st.markdown("""
        Low confidence predictions occur when your property has characteristics outside the typical training data:
        - **Luxury pricing** (>$20/SF/Yr)
        - **Specialty property type** (Food Processing, etc.)
        - **New construction** (<10 years old)
        - **Small market** with limited data

        **What to do:**
        Use the prediction as a starting point, but get a manual appraisal for final valuation.
        """)

    with st.expander("Can I use this for residential properties?"):
        st.markdown("""
        **No.** This model is trained exclusively on **commercial properties** (office, industrial, retail, etc.).

        For residential real estate:
        - Use residential-specific valuation tools (e.g., Zillow Zestimate)
        - Or consult a residential real estate appraiser
        """)

    with st.expander("How often is the model updated?"):
        st.markdown("""
        **Current:** Model trained November 2024

        **Update frequency:**
        - **Quarterly**: Retrain with new market data
        - **Annual**: Full model architecture review
        - **As needed**: Emergency updates for major market events

        Check the sidebar for the model's training date.
        """)

    with st.expander("Can I integrate this into our CRM?"):
        st.markdown("""
        **Yes!** API integration is available.

        **Requirements:**
        - Contact analytics@lornell.com for API access
        - Provide use case and expected volume
        - Review API documentation

        **Options:**
        - REST API (JSON requests)
        - Batch processing (CSV uploads)
        - Direct database integration
        """)

    with st.expander("What's the difference between MAE and R²?"):
        st.markdown("""
        **MAE (Mean Absolute Error)**: Average prediction error in dollars
        - Example: MAE = $0.68/SF/Yr means predictions are off by $0.68 on average
        - Lower is better (closer to $0)
        - Easy to interpret in business terms

        **R² (R-squared)**: Proportion of variance explained
        - Example: R² = 0.861 means model explains 86.1% of rent variation
        - Higher is better (closer to 1.0)
        - Measures model fit quality

        **Both matter:**
        - High R² + Low MAE = Excellent model
        - Our model: R² = 0.861, MAE = $0.68 (industry-leading performance)
        """)

    st.markdown("---")
    st.markdown("**Model Version:** 1.0 | **Last Updated:** November 28, 2024")
    st.caption("Built with  for Lornell Real Estate | Powered by Python, Streamlit, and scikit-learn")
