"""
Model Insights Page
===================

Interactive visualizations showing model performance and feature importance.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import sys

# Add paths
parent_dir = Path(__file__).parent.parent
project_dir = parent_dir.parent
sys.path.insert(0, str(parent_dir))
sys.path.insert(0, str(project_dir))

# Page config
st.set_page_config(page_title="Model Insights", page_icon="📊", layout="wide")

st.title("Model Insights & Performance")
st.markdown("Understand what drives commercial rent predictions and where the model performs best.")

# Add glossary in sidebar
with st.sidebar:
    st.subheader("📖 Quick Glossary")
    with st.expander("What do these metrics mean?", expanded=False):
        st.markdown("""
        **MAE (Mean Absolute Error)**
        Average difference between predicted and actual rent.
        *Lower is better.* An MAE of $1.20 means predictions are off by $1.20/SF/Yr on average.

        **R² (R-squared)**
        How much of the rent variation the model explains.
        *Higher is better.* R² of 0.76 means the model explains 76% of rent differences.

        **SHAP Values**
        Shows how each feature pushes a prediction up or down.
        *Named after Nobel laureate Lloyd Shapley's game theory work.*

        **Feature Importance**
        Ranks which property characteristics matter most for predictions.
        """)

    with st.expander("Importance Methods Explained"):
        st.markdown("""
        **MDI (Mean Decrease Impurity)**
        Measures how much each feature helps the model make decisions. Built into Random Forest.

        **Permutation Importance**
        Scrambles each feature and measures accuracy drop. More reliable but slower.

        **SHAP Importance**
        Based on game theory - fairly distributes credit among features.
        """)

# Load data
figures_dir = project_dir / "figures"

# Tabs for different analyses
tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Feature Importance",
    "🔍 SHAP Analysis",
    "Performance by Segment",
    "🗺️ Geographic Performance"
])

# Tab 1: Feature Importance
with tab1:
    st.subheader("What Drives Commercial Rent Predictions?")
    st.markdown("""
    These charts show which property characteristics have the strongest impact on rent predictions.

    **Think of it like this:** If you were a property appraiser, which factors would you look at first?
    The model learns this automatically from thousands of properties.
    """)

    st.info("💡 **Tip:** Features at the top of the chart have the most influence on predicted rent. "
            "Higher bars = more important for pricing.")

    # Load feature importance CSVs
    try:
        mdi_imp = pd.read_csv(figures_dir / "feature_importance" / "mdi_importance.csv")
        perm_imp = pd.read_csv(figures_dir / "feature_importance" / "permutation_importance.csv")
        shap_imp = pd.read_csv(figures_dir / "feature_importance" / "shap_importance.csv")

        # Method selector
        method = st.selectbox(
            "Select Importance Method",
            ["MDI (Mean Decrease Impurity)", "Permutation Importance", "SHAP Values", "Comparison"],
            help="Different methods for calculating feature importance"
        )

        if method == "Comparison":
            # Load comparison data
            comp_df = pd.read_csv(figures_dir / "feature_importance" / "importance_comparison.csv")

            # Create comparison chart
            top_n = st.slider("Number of features to display", 5, 20, 15)
            comp_df_top = comp_df.head(top_n)

            fig = go.Figure()

            if 'mdi_importance' in comp_df.columns:
                fig.add_trace(go.Bar(
                    name='MDI',
                    x=comp_df_top['feature'],
                    y=comp_df_top['mdi_importance'],
                    marker_color='#0066CC'
                ))

            if 'perm_importance' in comp_df.columns:
                fig.add_trace(go.Bar(
                    name='Permutation',
                    x=comp_df_top['feature'],
                    y=comp_df_top['perm_importance'],
                    marker_color='#FF9800'
                ))

            if 'shap_importance' in comp_df.columns:
                fig.add_trace(go.Bar(
                    name='SHAP',
                    x=comp_df_top['feature'],
                    y=comp_df_top['shap_importance'],
                    marker_color='#4CAF50'
                ))

            fig.update_layout(
                title=f"Top {top_n} Features - Method Comparison",
                xaxis_title="Feature",
                yaxis_title="Importance Score",
                barmode='group',
                height=500
            )

            st.plotly_chart(fig, use_container_width=True)

            # Show consensus features
            st.markdown("**Consensus Features** (Important across all methods):")
            consensus_features = comp_df_top.head(5)['feature'].tolist()
            for i, feat in enumerate(consensus_features, 1):
                st.markdown(f"{i}. **{feat.replace('_', ' ').title()}**")

        else:
            # Single method view
            if "MDI" in method:
                df = mdi_imp
                color = '#0066CC'
            elif "Permutation" in method:
                df = perm_imp
                color = '#FF9800'
            else:  # SHAP
                df = shap_imp
                color = '#4CAF50'

            top_n = st.slider("Number of features to display", 5, 20, 15)
            df_top = df.head(top_n)

            fig = go.Figure(go.Bar(
                x=df_top['importance'],
                y=df_top['feature'],
                orientation='h',
                marker_color=color
            ))

            fig.update_layout(
                title=f"Top {top_n} Features - {method}",
                xaxis_title="Importance Score",
                yaxis_title="",
                height=500
            )

            st.plotly_chart(fig, use_container_width=True)

            # Show dataframe
            with st.expander("View Full Data"):
                st.dataframe(df, use_container_width=True)

    except FileNotFoundError as e:
        st.error(f"Feature importance data not found. Please run Phase 3 interpretation first.")
        st.info("Run: `python test_phase3_interpretation.py`")

# Tab 2: SHAP Analysis
with tab2:
    st.subheader("SHAP Summary - How Features Affect Each Prediction")
    st.markdown("""
    **What is SHAP?** It's a way to explain *why* the model made each prediction.
    Instead of just saying "location matters," SHAP shows *how much* it raised or lowered each rent estimate.
    """)

    st.success("""
    **How to Read This Chart:**
    - Each **row** is a property feature (e.g., building size, location)
    - Each **dot** is one property in our dataset
    - **Red dots** = High values (large building, premium location)
    - **Blue dots** = Low values (small building, secondary location)
    - **Position left/right** = Did it lower or raise the rent prediction?
    """)

    # Display SHAP summary image
    shap_summary_path = figures_dir / "shap_analysis" / "shap_summary_top20.png"
    if shap_summary_path.exists():
        st.image(str(shap_summary_path), use_container_width=True)
    else:
        st.warning("SHAP summary plot not found.")

    st.markdown("---")
    st.subheader("SHAP Dependence Plots")
    st.markdown("See how individual features affect predictions across different values.")

    # List available dependence plots
    shap_dir = figures_dir / "shap_analysis"
    dependence_plots = list(shap_dir.glob("shap_dependence_*.png"))

    if dependence_plots:
        # Extract feature names from filenames
        features = [p.stem.replace('shap_dependence_', '').replace('_', ' ').title()
                   for p in dependence_plots]

        selected_feature = st.selectbox("Select Feature", features)

        # Find corresponding file
        selected_idx = features.index(selected_feature)
        plot_path = dependence_plots[selected_idx]

        st.image(str(plot_path), use_container_width=True)
    else:
        st.info("No SHAP dependence plots found.")

# Tab 3: Performance by Segment
with tab3:
    st.subheader("Model Performance Across Different Segments")
    st.markdown("""
    Not all properties are equally easy to predict. This section shows where the model
    excels and where it struggles — helping you understand when to trust its estimates more.
    """)

    col_mae_exp, col_r2_exp = st.columns(2)
    with col_mae_exp:
        st.metric(
            label="MAE (Mean Absolute Error)",
            value="Lower = Better",
            help="Average prediction error in $/SF/Yr. If MAE is $1.50, predictions are typically off by $1.50."
        )
    with col_r2_exp:
        st.metric(
            label="R² (R-squared)",
            value="Higher = Better",
            help="Percentage of rent variation explained. R² of 0.70 means the model explains 70% of price differences."
        )

    segment_type = st.radio(
        "Select Segment Type",
        ["Market", "Property Type", "Price Range", "Building Age"],
        horizontal=True
    )

    # Map to file names
    segment_files = {
        "Market": "errors_by_market_name.csv",
        "Property Type": "errors_by_property_type.csv",
        "Price Range": "errors_by_price_range.csv",
        "Building Age": "errors_by_building_age.csv"
    }

    csv_file = segment_files[segment_type]
    csv_path = figures_dir / "error_segmentation" / csv_file

    try:
        df_seg = pd.read_csv(csv_path)

        # Create dual metric chart
        col_mae, col_r2 = st.columns(2)

        with col_mae:
            fig_mae = go.Figure(go.Bar(
                x=df_seg.iloc[:, 0],  # First column is segment name
                y=df_seg['MAE'],
                marker_color='#F44336',
                text=df_seg['MAE'].round(2),
                textposition='auto'
            ))

            fig_mae.update_layout(
                title=f"MAE by {segment_type}",
                xaxis_title=segment_type,
                yaxis_title="Mean Absolute Error ($/SF/Yr)",
                height=400
            )

            st.plotly_chart(fig_mae, use_container_width=True)

        with col_r2:
            fig_r2 = go.Figure(go.Bar(
                x=df_seg.iloc[:, 0],
                y=df_seg['R2'],
                marker_color='#4CAF50',
                text=df_seg['R2'].round(3),
                textposition='auto'
            ))

            fig_r2.update_layout(
                title=f"R² by {segment_type}",
                xaxis_title=segment_type,
                yaxis_title="R² Score",
                height=400
            )

            st.plotly_chart(fig_r2, use_container_width=True)

        # Insights
        st.markdown("---")
        st.subheader("Key Insights")

        # Best performing segment
        best_idx = df_seg['MAE'].idxmin()
        best_segment = df_seg.iloc[best_idx, 0]
        best_mae = df_seg.iloc[best_idx]['MAE']

        # Worst performing segment
        worst_idx = df_seg['MAE'].idxmax()
        worst_segment = df_seg.iloc[worst_idx, 0]
        worst_mae = df_seg.iloc[worst_idx]['MAE']

        col_best, col_worst = st.columns(2)

        with col_best:
            st.success(f"""
            **Best Performance: {best_segment}**
            - MAE: ${best_mae:.2f}/SF/Yr
            - R²: {df_seg.iloc[best_idx]['R2']:.3f}
            - Sample Size: {int(df_seg.iloc[best_idx]['count'])} properties
            """)

        with col_worst:
            st.error(f"""
            **Needs Attention: {worst_segment}**
            - MAE: ${worst_mae:.2f}/SF/Yr
            - R²: {df_seg.iloc[worst_idx]['R2']:.3f}
            - Sample Size: {int(df_seg.iloc[worst_idx]['count'])} properties
            """)

        # Full data table
        with st.expander("View Full Segment Data"):
            st.dataframe(df_seg, use_container_width=True)

    except FileNotFoundError:
        st.error(f"Segment data not found: {csv_path}")

# Tab 4: Geographic Performance
with tab4:
    st.subheader("Performance Across Markets")
    st.markdown("""
    Real estate is local. This chart shows how well the model performs in different geographic markets.
    Markets with more training data typically have better predictions.
    """)

    try:
        market_df = pd.read_csv(figures_dir / "error_segmentation" / "errors_by_market_name.csv")

        # Create scatter plot: Sample Size vs MAE
        # Use count for size (can't use R2 since it can be negative)
        fig = px.scatter(
            market_df,
            x='count',
            y='MAE',
            size='count',
            color='MAE',
            hover_data=['segment', 'R2'],
            labels={
                'count': 'Number of Properties',
                'MAE': 'MAE ($/SF/Yr)',
                'R2': 'R² Score'
            },
            title="Market Performance: Sample Size vs Prediction Error",
            color_continuous_scale='RdYlGn_r'
        )

        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

        st.info("""
        **How to Read This Chart:**
        - **Bubble size** = Number of properties we have data for in that market
        - **Green bubbles** = Model predicts well here (low error)
        - **Red bubbles** = Predictions less reliable (higher error)
        - **Hover** over any bubble to see the market name and R² score

        *Generally, markets with more data (larger bubbles) have better predictions.*
        """)

        # Market recommendations
        st.markdown("---")
        st.subheader("Market Recommendations")

        high_error_markets = market_df[market_df['MAE'] > market_df['MAE'].median()]

        if len(high_error_markets) > 0:
            st.warning("**Markets Needing More Data or Specialized Models:**")
            for idx, row in high_error_markets.iterrows():
                st.markdown(f"- **{row['segment']}**: MAE ${row['MAE']:.2f}, {int(row['count'])} properties")

    except FileNotFoundError:
        st.error("Market data not found.")
