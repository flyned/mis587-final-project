"""
Market Comparison Page
======================

Compare how the same property would perform across different markets.
"""

import streamlit as st
import pandas as pd
from pathlib import Path
import sys

# Add parent and project directories to path
parent_dir = Path(__file__).parent.parent
project_dir = parent_dir.parent
sys.path.insert(0, str(parent_dir))
sys.path.insert(0, str(project_dir))

from utils.predictor_wrapper import (
    load_model_and_metadata,
    get_market_options,
    get_property_type_options
)
from utils.market_comparison import (
    compare_across_markets,
    display_comparison_chart,
    display_comparison_table,
    display_key_insights,
    display_market_shap_explanation,
    export_comparison_to_csv
)

# Page config
st.set_page_config(page_title="Market Comparison", page_icon="📍", layout="wide")

st.title("Market Comparison Tool")
st.markdown("""
Compare how the same property would perform across different markets.
This tool helps investors identify the most profitable markets for their investment strategy.
""")

# Load model (cached)
@st.cache_resource
def load_model():
    """Load model and metadata (cached)."""
    model_dir = project_dir / "models"
    # Find the latest model
    model_files = list(model_dir.glob("rf_model_local_*.pkl"))
    if not model_files:
        st.error("No trained model found. Please train a model first.")
        st.stop()

    latest_model = sorted(model_files)[-1]
    metadata_file = latest_model.with_name(latest_model.stem.replace('rf_model', 'rf_metadata') + '.json')

    if not metadata_file.exists():
        st.error(f"Metadata file not found: {metadata_file}")
        st.stop()

    return load_model_and_metadata(str(latest_model), str(metadata_file))

model, metadata = load_model()

# Sidebar info
with st.sidebar:
    st.info("""
    **How It Works**

    1. Enter property details
    2. Select markets to compare
    3. See rent predictions across markets
    4. Explore SHAP explanations
    5. Download results

    **Use Case:**
    Identify which market offers the best
    rental income for your property profile.
    """)

    val_metrics = metadata.get('validation_metrics', {})
    r2_val = val_metrics.get('r2', None)
    mae_val = val_metrics.get('mae', None)

    r2_str = f"{r2_val:.3f}" if r2_val is not None else "N/A"
    mae_str = f"${mae_val:.2f}/SF/Yr" if mae_val is not None else "N/A"

    st.markdown("---")
    st.markdown(f"""
    **Model Performance**
    - Validation R²: {r2_str}
    - MAE: {mae_str}
    """)

# Main content
st.markdown("### Property Configuration")
st.markdown("Define the property you want to compare across markets:")

with st.form("market_comparison_form"):
    # Property characteristics
    col1, col2 = st.columns(2)

    with col1:
        property_type = st.selectbox(
            "Property Type",
            options=get_property_type_options(),
            help="Type of industrial/commercial property"
        )

        rba = st.number_input(
            "RBA (Rentable Building Area)",
            min_value=1_000,
            max_value=10_000_000,
            value=50_000,
            step=1_000,
            help="Total rentable square footage"
        )

    with col2:
        from datetime import datetime
        year_built = st.number_input(
            "Year Built",
            min_value=1800,
            max_value=datetime.now().year,
            value=2000,
            step=1,
            help="Year property was constructed"
        )

        stories = st.number_input(
            "Number of Stories",
            min_value=1,
            max_value=200,
            value=3,
            step=1,
            help="Number of floors/stories"
        )

    # Market selection
    st.markdown("**Markets to Compare**")
    all_markets = get_market_options()

    # Default suggestions for New England
    default_markets = []
    for market in ["Boston, MA", "Worcester, MA", "Providence, RI"]:
        if market in all_markets:
            default_markets.append(market)

    selected_markets = st.multiselect(
        "Select Markets",
        options=all_markets,
        default=default_markets[:3] if default_markets else all_markets[:3],
        help="Select 2 or more markets to compare. Markets are represented by their geographic center coordinates."
    )

    # Submit button
    submit = st.form_submit_button("🔍 Compare Markets", use_container_width=True)

# Results section
if submit:
    # Validation
    if len(selected_markets) < 2:
        st.error("Please select at least 2 markets to compare.")
        st.stop()

    # Build property configuration
    property_config = {
        'Property Type': property_type,
        'RBA': rba,
        'Year Built': year_built,
        'Stories': stories
    }

    # Run comparison
    with st.spinner(f"Comparing across {len(selected_markets)} markets..."):
        try:
            results = compare_across_markets(
                model,
                metadata,
                property_config,
                selected_markets
            )

            if results.empty:
                st.error("No valid predictions generated. Please check your market selection.")
                st.stop()

            # Display results
            st.markdown("---")
            st.markdown("## Comparison Results")

            # Overview chart
            st.markdown("### Predicted Rent by Market")
            display_comparison_chart(results)

            # Detailed table
            st.markdown("### Detailed Comparison")
            display_comparison_table(results)

            # Key insights
            display_key_insights(results)

            # SHAP explanations
            st.markdown("---")
            st.markdown("## 🔍 Why Do Markets Differ?")
            st.markdown("""
            Expand each market below to see which features drive the rent prediction.
            This helps you understand **why** some markets command higher rents than others.
            """)

            for idx, market_row in results.iterrows():
                market_name = market_row['Market']
                predicted_rent = market_row['Predicted_Rent']
                confidence = market_row['Confidence_Level']

                # Color-code expander based on confidence
                conf_emoji = {
                    'HIGH': '🟢',
                    'MEDIUM': '🟡',
                    'LOW': '🔴'
                }
                emoji = conf_emoji.get(confidence, '⚪')

                with st.expander(f"{emoji} **{market_name}** - ${predicted_rent:.2f}/SF/Yr ({confidence} Confidence)"):
                    display_market_shap_explanation(
                        market_row,
                        model,
                        metadata
                    )

            # Export button
            st.markdown("---")
            st.markdown("### Export Results")

            csv = export_comparison_to_csv(results)
            st.download_button(
                label="📥 Download Comparison (CSV)",
                data=csv,
                file_name=f"market_comparison_{property_type.lower().replace(' ', '_')}_" \
                          f"{rba}sf.csv",
                mime="text/csv",
                use_container_width=True
            )

            # Summary insights
            st.markdown("---")
            st.markdown("### Investment Recommendation")

            best_market = results.iloc[0]
            worst_market = results.iloc[-1]
            rent_diff = best_market['Predicted_Rent'] - worst_market['Predicted_Rent']
            revenue_diff = best_market['Annual_Revenue'] - worst_market['Annual_Revenue']

            st.success(f"""
            **Best Market:** {best_market['Market']} (${best_market['Predicted_Rent']:.2f}/SF/Yr)

            Compared to the lowest market ({worst_market['Market']}), investing in
            {best_market['Market']} could generate **${revenue_diff:,.0f} more** in annual
            rental revenue (${rent_diff:.2f}/SF/Yr × {rba:,} SF).

            **Confidence:** {best_market['Confidence_Level']}

            💡 *Expand the market sections above to understand what drives these differences.*
            """)

        except Exception as e:
            st.error(f"Error during comparison: {str(e)}")
            st.exception(e)

else:
    # Placeholder when no comparison run
    st.info("""
    **Get Started:**
    1. Configure your property details (type, size, age, stories)
    2. Select 2+ markets to compare
    3. Click "Compare Markets" to see results

    **Tip:** Start with Boston, Worcester, and Providence to compare major New England markets.
    """)
