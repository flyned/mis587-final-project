"""
Lornell Real Estate - Commercial Rent Prediction Dashboard
===========================================================

AI-powered commercial real estate rent predictions with full explainability.

Main entry point for the Streamlit application.
"""

import streamlit as st
import sys
from pathlib import Path

# Add parent directory to path to import project modules
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

# Page configuration
st.set_page_config(
    page_title="Lornell Real Estate - Rent Predictor",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for branding
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #0066CC;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #F0F2F6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #E3F2FD;
        padding: 1rem;
        border-left: 4px solid #0066CC;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #FFF3E0;
        padding: 1rem;
        border-left: 4px solid #FF9800;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Main landing page
def main():
    """Landing page with overview and navigation."""

    # Header
    st.markdown('<div class="main-header">Commercial Rent Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">AI-Powered Rent Predictions for Commercial Real Estate</div>', unsafe_allow_html=True)

    # Introduction
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        ### Welcome to Lornell Real Estate's Rent Prediction Tool

        This dashboard provides **instant, explainable rent predictions** for commercial properties using
        state-of-the-art machine learning trained on **12,534 properties** across major markets.

        **What you can do:**
        - **Property Lookup**: Get instant rent predictions with detailed explanations
        - **Model Insights**: Understand what drives commercial rents in different markets
        - **Batch Prediction**: Analyze entire portfolios with CSV upload
        - **About**: Learn about the model, methodology, and when to use it
        - **Market Timing**: Predict days on market with bidding strategy recommendations
        - **Opportunities**: Identify underpriced properties and value-add investments
        """)

    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric(label="Model Accuracy (R²)", value="76.2%")
        st.metric(label="Average Error (MAE)", value="$0.99/SF/Yr")
        st.metric(label="Properties Analyzed", value="12,534")
        st.markdown('</div>', unsafe_allow_html=True)

    # Quick stats
    st.markdown("---")
    st.subheader("Performance at a Glance")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Predictions Within ±10%",
            value="72%",
            help="Percentage of predictions within 10% of actual rent"
        )

    with col2:
        st.metric(
            label="Markets Covered",
            value="7",
            help="Massachusetts & Rhode Island commercial markets"
        )

    with col3:
        st.metric(
            label="Feature Variables",
            value="194",
            help="Engineered features used for predictions"
        )

    with col4:
        st.metric(
            label="Prediction Speed",
            value="<3 sec",
            help="Average time to generate prediction with explanation"
        )

    # How it works
    st.markdown("---")
    st.subheader("How It Works")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        #### 1. Input Property Details
        - Location & Market
        - Building Characteristics
        - Size & Age
        - Property Type
        """)

    with col2:
        st.markdown("""
        #### 2. AI Analysis
        - Neural Network & Random Forest Models
        - 194 Feature Variables
        - Trained on 12,534 Properties
        - SHAP Explainability
        """)

    with col3:
        st.markdown("""
        #### 3. Get Prediction
        - Predicted Annual Rent/SF
        - Confidence Interval
        - Feature Contributions
        - Similar Properties
        """)

    # Confidence zones
    st.markdown("---")
    st.subheader("When to Trust Predictions")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("""
        **High Confidence - Use Directly**
        - Industrial & Warehouse properties
        - Medium price range ($5-$20/SF/Yr)
        - Major markets (Boston, Worcester, Providence, etc.)
        - 20-50 year old buildings
        - Standard property types

        *Typical error: $0.80-$1.20/SF/Yr*
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="warning-box">', unsafe_allow_html=True)
        st.markdown("""
        **Review Needed - Use with Caution**
        - Luxury properties (>$20/SF/Yr)
        - Food processing / specialty industrial
        - New construction (<10 years old)
        - Small/emerging markets
        - Properties with unique features

        *Typical error: $1.50-$2.50/SF/Yr*
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    # Navigation help
    st.markdown("---")
    st.subheader("Get Started")

    st.markdown("""
    **Use the sidebar** to navigate between different tools:

    - Start with **Property Lookup** for a single property prediction
    - Explore **Model Insights** to understand what drives rents
    - Use **Batch Prediction** to analyze multiple properties at once
    - Check **Market Timing** to predict days on market and get bidding strategies
    - Browse **Opportunities** to find undervalued investment properties
    - Read **About** to learn more about the methodology
    """)

    # Footer
    st.markdown("---")
    st.caption("Built with  for Lornell Real Estate | Powered by AI & Data Science")

if __name__ == "__main__":
    main()
