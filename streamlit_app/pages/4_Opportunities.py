"""
Opportunities Page
==================

Investment opportunity identification and analysis.
Implements Proposal Objective #4: Opportunity Identification.
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import plotly.express as px
import plotly.graph_objects as go

# Add parent and project directories to path
parent_dir = Path(__file__).parent.parent
project_dir = parent_dir.parent
sys.path.insert(0, str(parent_dir))
sys.path.insert(0, str(project_dir))

# Page config
st.set_page_config(page_title="Opportunities", page_icon="💎", layout="wide")

st.title("Investment Opportunities")
st.markdown("Identify underpriced properties and value-add investment opportunities.")


@st.cache_data
def load_opportunity_report():
    """Load the latest opportunity report."""
    model_dir = project_dir / "models"

    # Find the latest opportunity report
    report_files = list(model_dir.glob("opportunity_report_*.csv"))
    if not report_files:
        return None

    latest_report = sorted(report_files)[-1]
    df = pd.read_csv(latest_report)

    return df, latest_report.stem


# Load data
report_data = load_opportunity_report()

if report_data is None:
    st.error("No opportunity report found. Please run `python train_gap_models.py` first.")
    st.stop()

df, report_name = report_data

# Sidebar filters
with st.sidebar:
    st.subheader("Filters")

    # Category filter
    categories = ['All'] + list(df['category'].unique())
    selected_category = st.selectbox("Category", categories)

    # Score range
    min_score, max_score = st.slider(
        "Opportunity Score Range",
        min_value=-100,
        max_value=100,
        value=(-100, 100)
    )

    # Market filter if available
    if 'market' in df.columns:
        markets = ['All'] + sorted(df['market'].dropna().unique().tolist())
        selected_market = st.selectbox("Market", markets)
    else:
        selected_market = 'All'

    # Show only value-add
    show_value_add = st.checkbox("Value-Add Only", value=False)

    # Show only distressed
    show_distressed = st.checkbox("Distressed Only", value=False)

    st.markdown("---")
    st.caption(f"Report: {report_name}")

# Apply filters
filtered_df = df.copy()

if selected_category != 'All':
    filtered_df = filtered_df[filtered_df['category'] == selected_category]

filtered_df = filtered_df[
    (filtered_df['opportunity_score'] >= min_score) &
    (filtered_df['opportunity_score'] <= max_score)
]

if selected_market != 'All' and 'market' in filtered_df.columns:
    filtered_df = filtered_df[filtered_df['market'] == selected_market]

if show_value_add and 'is_value_add' in filtered_df.columns:
    filtered_df = filtered_df[filtered_df['is_value_add'] == True]

if show_distressed and 'is_distressed' in filtered_df.columns:
    filtered_df = filtered_df[filtered_df['is_distressed'] == True]

# Summary metrics
st.subheader("Portfolio Summary")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    strong_buy = len(df[df['category'] == 'Strong Buy'])
    st.metric("Strong Buy", strong_buy, help="Properties with >10% upside potential")

with col2:
    buy = len(df[df['category'] == 'Buy'])
    st.metric("Buy", buy, help="Properties with 5-10% upside potential")

with col3:
    fair_value = len(df[df['category'] == 'Fair Value'])
    st.metric("Fair Value", fair_value, help="Properties priced within 5% of fair value")

with col4:
    if 'is_distressed' in df.columns:
        distressed = df['is_distressed'].sum()
    else:
        distressed = 0
    st.metric("Distressed", int(distressed), help="Properties showing distress signals")

with col5:
    if 'is_value_add' in df.columns:
        value_add = df['is_value_add'].sum()
    else:
        value_add = 0
    st.metric("Value-Add", int(value_add), help="Properties with improvement potential")

st.markdown("---")

# Tabs for different views
tab1, tab2, tab3, tab4 = st.tabs(["Top Opportunities", "Distribution", "Market Analysis", "Data Explorer"])

with tab1:
    st.subheader("Top Investment Opportunities")

    # Get top opportunities
    top_n = st.slider("Number of opportunities to show", 10, 50, 20)
    top_opps = filtered_df.nlargest(top_n, 'opportunity_score')

    if len(top_opps) == 0:
        st.warning("No opportunities match your filters.")
    else:
        # Create display dataframe
        display_cols = ['opportunity_score', 'category', 'actual_rent', 'predicted_rent', 'rent_diff_pct']

        if 'property_address' in top_opps.columns:
            display_cols = ['property_address'] + display_cols
        if 'market' in top_opps.columns:
            display_cols.append('market')
        if 'is_value_add' in top_opps.columns:
            display_cols.append('is_value_add')
        if 'is_distressed' in top_opps.columns:
            display_cols.append('is_distressed')

        display_df = top_opps[display_cols].copy()

        # Format columns
        display_df['opportunity_score'] = display_df['opportunity_score'].round(1)
        display_df['actual_rent'] = display_df['actual_rent'].apply(lambda x: f"${x:.2f}")
        display_df['predicted_rent'] = display_df['predicted_rent'].apply(lambda x: f"${x:.2f}")
        display_df['rent_diff_pct'] = display_df['rent_diff_pct'].apply(lambda x: f"{x:+.1f}%")

        # Rename columns for display
        display_df.columns = [c.replace('_', ' ').title() for c in display_df.columns]

        st.dataframe(display_df, use_container_width=True, hide_index=True)

        # Bar chart of top opportunities
        fig = go.Figure()

        colors = ['#4CAF50' if score > 0 else '#F44336' for score in top_opps['opportunity_score']]

        fig.add_trace(go.Bar(
            x=list(range(1, len(top_opps) + 1)),
            y=top_opps['opportunity_score'],
            marker_color=colors,
            text=top_opps['opportunity_score'].round(1),
            textposition='auto',
        ))

        fig.update_layout(
            title="Opportunity Scores",
            xaxis_title="Property Rank",
            yaxis_title="Opportunity Score",
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Score Distribution")

    col1, col2 = st.columns(2)

    with col1:
        # Category distribution pie chart
        category_counts = filtered_df['category'].value_counts()

        colors_map = {
            'Strong Buy': '#2E7D32',
            'Buy': '#4CAF50',
            'Fair Value': '#FF9800',
            'Avoid': '#F44336',
            'Strong Avoid': '#B71C1C'
        }

        fig = go.Figure(data=[go.Pie(
            labels=category_counts.index,
            values=category_counts.values,
            marker_colors=[colors_map.get(cat, '#666') for cat in category_counts.index],
            hole=0.4
        )])

        fig.update_layout(
            title="Category Distribution",
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Score histogram
        fig = px.histogram(
            filtered_df,
            x='opportunity_score',
            nbins=50,
            color_discrete_sequence=['#0066CC'],
            title="Opportunity Score Distribution"
        )

        fig.add_vline(x=0, line_dash="dash", line_color="red")
        fig.add_vline(x=20, line_dash="dash", line_color="green", annotation_text="Strong Buy threshold")
        fig.add_vline(x=-20, line_dash="dash", line_color="red", annotation_text="Strong Avoid threshold")

        fig.update_layout(
            xaxis_title="Opportunity Score",
            yaxis_title="Count",
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)

    # Actual vs Predicted scatter
    st.subheader("Actual vs Predicted Rent")

    fig = px.scatter(
        filtered_df,
        x='actual_rent',
        y='predicted_rent',
        color='category',
        color_discrete_map=colors_map,
        hover_data=['opportunity_score', 'rent_diff_pct'],
        title="Actual vs Predicted Rent ($/SF/Yr)"
    )

    # Add perfect prediction line
    max_val = max(filtered_df['actual_rent'].max(), filtered_df['predicted_rent'].max())
    fig.add_trace(go.Scatter(
        x=[0, max_val],
        y=[0, max_val],
        mode='lines',
        line=dict(color='gray', dash='dash'),
        name='Perfect Prediction'
    ))

    fig.update_layout(
        xaxis_title="Actual Rent ($/SF/Yr)",
        yaxis_title="Predicted Rent ($/SF/Yr)",
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    **Interpretation:**
    - Points **above** the diagonal line are **underpriced** (model predicts higher rent)
    - Points **below** the diagonal line are **overpriced** (model predicts lower rent)
    """)

with tab3:
    st.subheader("Market Analysis")

    if 'market' in filtered_df.columns:
        # Group by market
        market_stats = filtered_df.groupby('market').agg({
            'opportunity_score': ['mean', 'count'],
            'rent_diff_pct': 'mean',
            'actual_rent': 'mean',
            'predicted_rent': 'mean'
        }).round(2)

        market_stats.columns = ['Avg Score', 'Count', 'Avg Upside %', 'Avg Actual Rent', 'Avg Predicted Rent']
        market_stats = market_stats.sort_values('Avg Score', ascending=False)

        st.dataframe(market_stats, use_container_width=True)

        # Market comparison chart
        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=market_stats.index,
            y=market_stats['Avg Score'],
            marker_color=['#4CAF50' if s > 0 else '#F44336' for s in market_stats['Avg Score']],
            text=market_stats['Avg Score'].round(1),
            textposition='auto',
        ))

        fig.update_layout(
            title="Average Opportunity Score by Market",
            xaxis_title="Market",
            yaxis_title="Average Opportunity Score",
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)

        # Best opportunities by market
        st.subheader("Top Opportunity by Market")

        best_by_market = filtered_df.loc[filtered_df.groupby('market')['opportunity_score'].idxmax()]
        cols_to_show = ['market', 'opportunity_score', 'category', 'actual_rent', 'predicted_rent', 'rent_diff_pct']
        cols_to_show = [c for c in cols_to_show if c in best_by_market.columns]

        st.dataframe(
            best_by_market[cols_to_show].sort_values('opportunity_score', ascending=False),
            use_container_width=True,
            hide_index=True
        )

    else:
        st.info("Market data not available in the opportunity report.")

with tab4:
    st.subheader("Data Explorer")

    st.markdown(f"Showing **{len(filtered_df)}** of **{len(df)}** properties based on filters.")

    # Column selector
    available_cols = filtered_df.columns.tolist()
    default_cols = ['opportunity_score', 'category', 'actual_rent', 'predicted_rent', 'rent_diff_pct']
    default_cols = [c for c in default_cols if c in available_cols]

    selected_cols = st.multiselect(
        "Select columns to display",
        available_cols,
        default=default_cols
    )

    if selected_cols:
        # Sort options
        sort_col = st.selectbox("Sort by", selected_cols)
        sort_order = st.radio("Order", ["Descending", "Ascending"], horizontal=True)

        sorted_df = filtered_df[selected_cols].sort_values(
            sort_col,
            ascending=(sort_order == "Ascending")
        )

        st.dataframe(sorted_df, use_container_width=True, hide_index=True)

        # Download button
        csv = sorted_df.to_csv(index=False)
        st.download_button(
            label="Download Filtered Data (CSV)",
            data=csv,
            file_name="filtered_opportunities.csv",
            mime="text/csv"
        )

# Investment Recommendations Section
st.markdown("---")
st.subheader("Investment Recommendation Guide")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### Category Definitions

    | Category | Score Range | Interpretation |
    |----------|-------------|----------------|
    | **Strong Buy** | > 20 | >10% underpriced - High conviction opportunity |
    | **Buy** | 10 to 20 | 5-10% underpriced - Good opportunity |
    | **Fair Value** | -10 to 10 | Within 5% of fair value - Fairly priced |
    | **Avoid** | -20 to -10 | 5-10% overpriced - Poor value |
    | **Strong Avoid** | < -20 | >10% overpriced - Significantly overpriced |
    """)

with col2:
    st.markdown("""
    ### Additional Flags

    **Distressed Properties** indicate:
    - High vacancy (>20%)
    - Extended time on market (>90 days)
    - Low occupancy (<70%)
    - Potential deferred maintenance

    **Value-Add Opportunities** indicate:
    - Rent upside potential (>10%)
    - Renovation candidate (>25 years old)
    - Lease-up opportunity
    - Never renovated
    """)

st.markdown("""
---
**Disclaimer:** These recommendations are based on machine learning model predictions and should not be the sole basis for investment decisions.
Always conduct thorough due diligence and consult with real estate professionals before making investment decisions.
""")
