"""
Market Comparison Utilities
============================

Functions for comparing property predictions across multiple markets.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys

# Add parent directories to path
parent_dir = Path(__file__).parent.parent
project_dir = parent_dir.parent
sys.path.insert(0, str(parent_dir))
sys.path.insert(0, str(project_dir))

from utils.predictor_wrapper import predict_single_property
from utils.shap_explainer import generate_shap_force_plot


@st.cache_data
def get_market_coordinates():
    """
    Extract market center coordinates from training data.

    Returns:
        dict: {market_name: (latitude, longitude)}
    """
    try:
        # Load training data
        data_path = project_dir / "data" / "train.csv"
        df = pd.read_csv(data_path)

        # Calculate market centers (mean lat/lon for each market)
        market_coords = {}
        for market in df['Market Name'].unique():
            market_data = df[df['Market Name'] == market]
            if not market_data.empty and not market_data['Latitude'].isna().all():
                avg_lat = market_data['Latitude'].mean()
                avg_lon = market_data['Longitude'].mean()
                market_coords[market] = (avg_lat, avg_lon)

        return market_coords

    except FileNotFoundError:
        # Fallback: hardcoded coordinates for major MA/RI markets
        return {
            "Boston, MA": (42.3601, -71.0589),
            "Worcester, MA": (42.2626, -71.8023),
            "Providence, RI": (41.8240, -71.4128),
            "Springfield, MA": (42.1015, -72.5898),
            "Barnstable Town, MA": (41.7003, -70.3002),
            "Pittsfield, MA": (42.4501, -73.2453)
        }


def compare_across_markets(model, metadata, property_config, markets):
    """
    Predict rent for the same property across multiple markets.

    Args:
        model: Trained ML model
        metadata: Model metadata
        property_config: Dict with property attributes
            - Property Type (str)
            - RBA (int)
            - Year Built (int)
            - Stories (int)
        markets: List of market names to compare

    Returns:
        pd.DataFrame with columns:
            - Market
            - Predicted_Rent
            - CI_Lower
            - CI_Upper
            - Confidence_Level
            - Annual_Revenue
            - Market_Premium_Pct
            - X (feature matrix for SHAP)
            - feature_names (for SHAP)
    """
    results = []
    market_coords = get_market_coordinates()

    for market in markets:
        # Skip if market coordinates not available
        if market not in market_coords:
            st.warning(f"Market '{market}' not found in training data. Skipping...")
            continue

        # Build property data with market-specific coordinates
        property_data = {
            'Market Name': market,
            'Latitude': market_coords[market][0],
            'Longitude': market_coords[market][1],
            'Property Type': property_config.get('Property Type', 'Warehouse'),
            'RBA': property_config.get('RBA', 50000),
            'Year Built': property_config.get('Year Built', 2000),
            'Stories': property_config.get('Stories', 3)
        }

        # Make prediction
        try:
            pred_result = predict_single_property(model, metadata, property_data)

            # Calculate annual revenue
            annual_revenue = pred_result['prediction'] * property_config['RBA']

            results.append({
                'Market': market,
                'Predicted_Rent': pred_result['prediction'],
                'CI_Lower': pred_result['ci_lower'],
                'CI_Upper': pred_result['ci_upper'],
                'Confidence_Level': pred_result['confidence_level'],
                'Annual_Revenue': annual_revenue,
                'X': pred_result['X'],
                'feature_names': pred_result['feature_names']
            })

        except Exception as e:
            st.error(f"Error predicting for {market}: {str(e)}")
            continue

    # Convert to DataFrame
    if not results:
        st.error("No valid predictions generated. Please check market selection.")
        return pd.DataFrame()

    df = pd.DataFrame(results)

    # Calculate market premium/discount vs average
    avg_rent = df['Predicted_Rent'].mean()
    df['Market_Premium_Pct'] = ((df['Predicted_Rent'] - avg_rent) / avg_rent * 100)

    # Sort by predicted rent (descending)
    df = df.sort_values('Predicted_Rent', ascending=False).reset_index(drop=True)

    return df


def display_comparison_chart(results_df):
    """
    Create interactive bar chart of market comparison.

    Args:
        results_df: DataFrame from compare_across_markets()
    """
    # Prepare data for plotting
    plot_df = results_df.copy()

    # Create bar chart with Plotly
    fig = px.bar(
        plot_df,
        x='Market',
        y='Predicted_Rent',
        color='Confidence_Level',
        color_discrete_map={
            'HIGH': '#4CAF50',
            'MEDIUM': '#FF9800',
            'LOW': '#F44336'
        },
        title="Predicted Rent by Market",
        labels={
            'Predicted_Rent': 'Predicted Rent ($/SF/Yr)',
            'Market': 'Market',
            'Confidence_Level': 'Confidence'
        },
        text='Predicted_Rent'
    )

    # Update traces for better display
    fig.update_traces(
        texttemplate='$%{text:.2f}',
        textposition='outside',
        textfont_size=12
    )

    # Update layout
    fig.update_layout(
        xaxis_title="Market",
        yaxis_title="Predicted Rent ($/SF/Yr)",
        height=500,
        showlegend=True,
        legend_title_text="Confidence Level",
        hovermode='x unified'
    )

    st.plotly_chart(fig, use_container_width=True)


def display_comparison_table(results_df):
    """
    Display detailed comparison table with formatting.

    Args:
        results_df: DataFrame from compare_across_markets()
    """
    # Select and format columns for display
    display_df = results_df[['Market', 'Predicted_Rent', 'CI_Lower', 'CI_Upper',
                              'Confidence_Level', 'Annual_Revenue', 'Market_Premium_Pct']].copy()

    # Format currency and percentage columns
    display_df['Predicted_Rent'] = display_df['Predicted_Rent'].apply(lambda x: f'${x:.2f}')
    display_df['CI_Range'] = results_df.apply(
        lambda row: f"${row['CI_Lower']:.2f} - ${row['CI_Upper']:.2f}", axis=1
    )
    display_df['Annual_Revenue'] = display_df['Annual_Revenue'].apply(lambda x: f'${x:,.0f}')
    display_df['Market_Premium_Pct'] = display_df['Market_Premium_Pct'].apply(
        lambda x: f'{x:+.1f}%' if pd.notna(x) else 'N/A'
    )

    # Reorder and rename columns
    display_df = display_df[['Market', 'Predicted_Rent', 'CI_Range', 'Confidence_Level',
                             'Annual_Revenue', 'Market_Premium_Pct']]
    display_df.columns = ['Market', 'Predicted Rent', '95% CI Range', 'Confidence',
                          'Annual Revenue', 'Premium']

    # Apply styling based on confidence level
    def color_confidence(row):
        colors = {
            'HIGH': ['background-color: #E8F5E9'] * len(row),
            'MEDIUM': ['background-color: #FFF3E0'] * len(row),
            'LOW': ['background-color: #FFEBEE'] * len(row)
        }
        return colors.get(row['Confidence'], [''] * len(row))

    styled_df = display_df.style.apply(color_confidence, axis=1)

    st.dataframe(styled_df, use_container_width=True)


def display_key_insights(results_df):
    """
    Display key insights panel with metrics.

    Args:
        results_df: DataFrame from compare_across_markets()
    """
    st.markdown("### Key Insights")

    col1, col2, col3 = st.columns(3)

    # Best market (highest rent)
    best_market = results_df.iloc[0]
    with col1:
        st.metric(
            label="🏆 Best Market",
            value=best_market['Market'],
            delta=f"${best_market['Predicted_Rent']:.2f}/SF/Yr",
            help="Market with highest predicted rent"
        )

    # Most reliable (highest confidence among top markets)
    high_conf = results_df[results_df['Confidence_Level'] == 'HIGH']
    if not high_conf.empty:
        most_reliable = high_conf.iloc[0]
        with col2:
            st.metric(
                label="Most Reliable",
                value=most_reliable['Market'],
                delta="High Confidence",
                help="Highest rent market with HIGH confidence rating"
            )
    else:
        with col2:
            st.metric(
                label="Most Reliable",
                value="None",
                delta="No HIGH confidence",
                help="No markets have HIGH confidence"
            )

    # Rent spread (max - min)
    rent_spread = results_df['Predicted_Rent'].max() - results_df['Predicted_Rent'].min()
    spread_pct = (rent_spread / results_df['Predicted_Rent'].mean()) * 100
    with col3:
        st.metric(
            label="Rent Spread",
            value=f"${rent_spread:.2f}/SF/Yr",
            delta=f"{spread_pct:.1f}% of average",
            help="Difference between highest and lowest rent predictions"
        )


def display_market_shap_explanation(market_result, model, metadata):
    """
    Display SHAP explanation for a specific market.

    Args:
        market_result: Row from results DataFrame with X and feature_names
        model: Trained ML model
        metadata: Model metadata
    """
    try:
        # Generate SHAP values
        feature_impacts, base_value = generate_shap_force_plot(
            model,
            market_result['X'],
            market_result['feature_names']
        )

        # Display in two columns
        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("**Top Feature Contributions:**")
            for i, feature in enumerate(feature_impacts[:5], 1):
                impact = feature['shap_value']
                direction = "increases" if impact > 0 else "decreases"
                feature_name = feature['feature'].replace('_', ' ').title()

                # Color code based on direction
                color = "green" if impact > 0 else "red"
                st.markdown(
                    f"{i}. **{feature_name}**: "
                    f":{color}[${abs(impact):.2f}/SF/Yr] ({direction} rent)"
                )

        with col2:
            st.markdown("**Feature Impact Chart:**")

            # Create horizontal bar chart
            features = [f['feature'].replace('_', ' ').title()[:30] for f in feature_impacts[:5]]
            values = [f['shap_value'] for f in feature_impacts[:5]]
            colors = ['#4CAF50' if v > 0 else '#F44336' for v in values]

            fig = go.Figure(go.Bar(
                x=values,
                y=features,
                orientation='h',
                marker=dict(color=colors),
                text=[f"+${abs(v):.2f}" if v > 0 else f"-${abs(v):.2f}" for v in values],
                textposition='auto',
            ))

            fig.update_layout(
                xaxis_title="Impact ($/SF/Yr)",
                yaxis_title="",
                height=250,
                margin=dict(l=0, r=0, t=0, b=0),
                showlegend=False
            )

            st.plotly_chart(fig, use_container_width=True)

        # Summary explanation
        st.markdown(f"""
        **Explanation:** The base prediction for all properties is **${base_value:.2f}/SF/Yr**.
        In this market, the final prediction of **${market_result['Predicted_Rent']:.2f}/SF/Yr**
        is influenced by the factors shown above.
        """)

    except Exception as e:
        st.error(f"Error generating SHAP explanation: {str(e)}")


def export_comparison_to_csv(results_df):
    """
    Convert comparison results to CSV format for download.

    Args:
        results_df: DataFrame from compare_across_markets()

    Returns:
        str: CSV string
    """
    # Select columns for export
    export_df = results_df[['Market', 'Predicted_Rent', 'CI_Lower', 'CI_Upper',
                            'Confidence_Level', 'Annual_Revenue', 'Market_Premium_Pct']].copy()

    # Round numeric columns
    numeric_cols = ['Predicted_Rent', 'CI_Lower', 'CI_Upper', 'Annual_Revenue', 'Market_Premium_Pct']
    for col in numeric_cols:
        if col in export_df.columns:
            export_df[col] = export_df[col].round(2)

    # Convert to CSV
    return export_df.to_csv(index=False)
