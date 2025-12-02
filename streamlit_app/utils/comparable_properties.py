"""
Comparable Properties Utilities
================================

Functions for finding and displaying comparable properties from training data.
Uses BallTree for efficient spatial indexing and neighbor searches.
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.neighbors import BallTree
try:
    from geopy.distance import geodesic
    GEOPY_AVAILABLE = True
except ImportError:
    GEOPY_AVAILABLE = False
    print("Warning: geopy not installed. Distance calculations will be disabled.")


# Property type similarity mapping (same category = 1.0, related = 0.5, different = 0.0)
PROPERTY_TYPE_GROUPS = {
    'Warehouse': 'industrial',
    'Distribution': 'industrial',
    'Manufacturing': 'industrial',
    'Truck Terminal': 'industrial',
    'Service': 'service',
    'Showroom': 'retail',
    'Food Processing': 'specialty',
    'Refrigeration/Cold Storage': 'specialty',
    'other': 'other',
    'unknown': 'other'
}


def calculate_similarity_score(input_prop, comp_prop):
    """
    Calculate similarity percentage between input and comp property.

    Weights:
    - Market match: 30%
    - Property Type match: 20%
    - RBA similarity: 20%
    - Year Built similarity: 15%
    - Stories similarity: 10%
    - Location proximity: 5%

    Returns:
        float: Similarity score 0-100%
    """
    score = 0

    # Market match (30%)
    if input_prop.get('Market Name') == comp_prop.get('Market Name'):
        score += 30

    # Property Type match (20%)
    # Handle both 'Property Type' (from UI) and 'Secondary Type' (from training data)
    input_type = input_prop.get('Property Type') or input_prop.get('Secondary Type')
    comp_type = comp_prop.get('Property Type') or comp_prop.get('Secondary Type')
    if input_type and comp_type and input_type == comp_type:
        score += 20

    # RBA similarity (20%) - inverse of percentage difference
    if input_prop.get('RBA', 0) > 0:
        rba_diff = abs(input_prop['RBA'] - comp_prop.get('RBA', 0)) / input_prop['RBA']
        rba_score = max(0, 20 * (1 - rba_diff))
        score += rba_score

    # Year Built similarity (15%) - inverse of year difference
    year_diff = abs(input_prop.get('Year Built', 2000) - comp_prop.get('Year Built', 2000))
    year_score = max(0, 15 * (1 - year_diff / 50))  # 50 year max diff
    score += year_score

    # Stories similarity (10%)
    # Handle both 'Stories' (from UI) and 'Number Of Stories' (from training data)
    input_stories = input_prop.get('Stories') or input_prop.get('Number Of Stories', 3)
    comp_stories = comp_prop.get('Stories') or comp_prop.get('Number Of Stories', 3)
    stories_diff = abs(input_stories - comp_stories)
    stories_score = max(0, 10 * (1 - stories_diff / 5))  # 5 story max diff
    score += stories_score

    # Location proximity (5%) - based on distance
    if GEOPY_AVAILABLE and 'Latitude' in input_prop and 'Latitude' in comp_prop:
        try:
            dist_miles = geodesic(
                (input_prop['Latitude'], input_prop['Longitude']),
                (comp_prop.get('Latitude', 0), comp_prop.get('Longitude', 0))
            ).miles
            dist_score = max(0, 5 * (1 - dist_miles / 50))  # 50 mile max
            score += dist_score
        except:
            pass

    return round(score, 1)


@st.cache_resource
def build_spatial_index(train_data_path, market_name):
    """
    Build BallTree spatial index for a specific market.

    Args:
        train_data_path: Path to training CSV
        market_name: Market to build index for

    Returns:
        tuple: (BallTree, filtered DataFrame, feature matrix)
    """
    df = pd.read_csv(train_data_path)
    market_df = df[df['Market Name'] == market_name].copy()

    if market_df.empty:
        return None, None, None

    # Create feature matrix for similarity search
    # Features: lat, lon (for spatial), log(RBA), year_built_normalized, stories_normalized
    market_df = market_df.reset_index(drop=True)

    # Normalize features for balanced similarity
    lat = market_df['Latitude'].fillna(0).values
    lon = market_df['Longitude'].fillna(0).values
    rba = np.log1p(market_df['RBA'].fillna(1).values)  # Log transform RBA
    year = (market_df['Year Built'].fillna(1980).values - 1900) / 125  # Normalize to ~[0, 1]
    stories = market_df['Number Of Stories'].fillna(1).values / 20  # Normalize assuming max ~20 stories

    # Create feature matrix (lat/lon in radians for haversine)
    X = np.column_stack([
        np.radians(lat),
        np.radians(lon),
        rba / 5,  # Scale down log RBA
        year,
        stories
    ])

    # Build BallTree with haversine metric for lat/lon (first 2 cols)
    # Note: We use euclidean for the combined features
    tree = BallTree(X, metric='euclidean')

    return tree, market_df, X


@st.cache_data
def find_market_comparables(input_property, train_data_path, n_comps=10):
    """
    Find comparable properties in the same market using spatial indexing.

    Args:
        input_property: Dict with property attributes
        train_data_path: Path to training CSV
        n_comps: Number of comparables to return

    Returns:
        pd.DataFrame with comparable properties and similarity scores
    """
    market_name = input_property.get('Market Name')
    if not market_name:
        return pd.DataFrame()

    # Get or build spatial index for this market
    tree, market_df, X = build_spatial_index(train_data_path, market_name)

    if tree is None or market_df is None:
        return pd.DataFrame()

    # Create query point from input property
    lat = input_property.get('Latitude', 0)
    lon = input_property.get('Longitude', 0)
    rba = np.log1p(input_property.get('RBA', 1)) / 5
    year = (input_property.get('Year Built', 1980) - 1900) / 125
    stories = (input_property.get('Stories') or input_property.get('Number Of Stories', 1)) / 20

    query = np.array([[np.radians(lat), np.radians(lon), rba, year, stories]])

    # Find k nearest neighbors (get more than needed to filter by similarity)
    k = min(n_comps * 3, len(market_df))  # Get 3x candidates
    distances, indices = tree.query(query, k=k)

    # Calculate detailed similarity scores for candidates
    candidates = []
    for i, idx in enumerate(indices[0]):
        row = market_df.iloc[idx]
        sim_score = calculate_similarity_score(input_property, row)

        candidates.append({
            'index': idx,
            'similarity': sim_score,
            'tree_distance': distances[0][i],
            'Property Address': row.get('Property Address', 'N/A'),
            'Market Name': row.get('Market Name', 'N/A'),
            'Property Type': row.get('Secondary Type', 'N/A'),
            'RBA': row.get('RBA', 0),
            'Year Built': int(row.get('Year Built', 0)) if pd.notna(row.get('Year Built')) else 0,
            'Stories': int(row.get('Number Of Stories', 0)) if pd.notna(row.get('Number Of Stories')) else 0,
            'Rent/SF/Yr': row.get('Rent/SF/Yr', 0),
            'Latitude': row.get('Latitude', 0),
            'Longitude': row.get('Longitude', 0)
        })

    # Sort by similarity score and take top n
    candidates.sort(key=lambda x: x['similarity'], reverse=True)
    comps_df = pd.DataFrame(candidates[:n_comps])

    if comps_df.empty:
        return pd.DataFrame()

    # Calculate distance from input
    if GEOPY_AVAILABLE and 'Latitude' in input_property:
        try:
            comps_df['distance_miles'] = comps_df.apply(
                lambda row: geodesic(
                    (input_property['Latitude'], input_property['Longitude']),
                    (row['Latitude'], row['Longitude'])
                ).miles,
                axis=1
            )
        except:
            comps_df['distance_miles'] = 0
    else:
        comps_df['distance_miles'] = 0

    # Calculate annual revenue
    comps_df['Annual Revenue'] = comps_df['Rent/SF/Yr'] * comps_df['RBA']

    # Drop tree_distance column (internal use only)
    comps_df = comps_df.drop(columns=['tree_distance'], errors='ignore')

    return comps_df


@st.cache_data
def get_market_statistics(market_name, train_data_path):
    """
    Get aggregated statistics for a market.

    Args:
        market_name: Name of the market
        train_data_path: Path to training CSV

    Returns:
        dict: Market statistics
    """
    df = pd.read_csv(train_data_path)
    market_df = df[df['Market Name'] == market_name]

    if market_df.empty:
        return None

    stats = {
        'total_properties': len(market_df),
        'avg_rent': market_df['Rent/SF/Yr'].mean(),
        'median_rent': market_df['Rent/SF/Yr'].median(),
        'min_rent': market_df['Rent/SF/Yr'].min(),
        'max_rent': market_df['Rent/SF/Yr'].max(),
        'avg_rba': market_df['RBA'].mean(),
        'property_types': market_df['Secondary Type'].value_counts().to_dict() if 'Secondary Type' in market_df.columns else {},
        'center_lat': market_df['Latitude'].mean(),
        'center_lon': market_df['Longitude'].mean()
    }

    return stats


def display_market_comparables(comps_df):
    """Display market comparables table in Streamlit."""
    if comps_df.empty:
        st.warning("No comparable properties found in this market.")
        return

    st.markdown(f"Found **{len(comps_df)}** similar properties in the same market:")

    # Format for display
    display_df = comps_df[[
        'Property Address', 'Property Type', 'RBA', 'Year Built',
        'Rent/SF/Yr', 'Annual Revenue', 'similarity', 'distance_miles'
    ]].copy()

    # Format columns
    display_df['RBA'] = display_df['RBA'].apply(lambda x: f'{int(x):,}')
    display_df['Year Built'] = display_df['Year Built'].apply(lambda x: str(int(x)) if x > 0 else 'N/A')
    display_df['Rent/SF/Yr'] = display_df['Rent/SF/Yr'].apply(lambda x: f'${x:.2f}')
    display_df['Annual Revenue'] = display_df['Annual Revenue'].apply(lambda x: f'${x:,.0f}')
    display_df['similarity'] = display_df['similarity'].apply(lambda x: f'{x:.1f}%')
    display_df['distance_miles'] = display_df['distance_miles'].apply(lambda x: f'{x:.1f} mi')

    # Rename columns
    display_df.columns = ['Address', 'Type', 'RBA', 'Built', 'Rent', 'Revenue', 'Match', 'Distance']

    # Color-code by similarity
    def color_similarity(row):
        match_str = row['Match']
        match_val = float(match_str.strip('%'))
        if match_val >= 80:
            return ['background-color: #E8F5E9'] * len(row)  # Green
        elif match_val >= 60:
            return ['background-color: #FFF9C4'] * len(row)  # Yellow
        else:
            return ['background-color: #FFEBEE'] * len(row)  # Red

    styled_df = display_df.style.apply(color_similarity, axis=1)
    st.dataframe(styled_df, use_container_width=True, hide_index=True)


def display_market_statistics(stats):
    """Display market statistics in Streamlit."""
    if not stats:
        st.warning("No market statistics available.")
        return

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Properties in Market",
            f"{stats['total_properties']:,}",
            help="Total properties in training data for this market"
        )

    with col2:
        st.metric(
            "Avg Market Rent",
            f"${stats['avg_rent']:.2f}/SF/Yr",
            help="Average rent across all properties in market"
        )

    with col3:
        st.metric(
            "Rent Range",
            f"${stats['min_rent']:.2f} - ${stats['max_rent']:.2f}",
            help="Minimum to maximum rent in market"
        )

    with col4:
        st.metric(
            "Avg Property Size",
            f"{stats['avg_rba']:,.0f} SF",
            help="Average RBA in market"
        )

    # Property type distribution
    st.markdown("**Property Type Distribution:**")
    type_df = pd.DataFrame(list(stats['property_types'].items()),
                           columns=['Type', 'Count'])
    type_df = type_df.sort_values('Count', ascending=False)

    import plotly.express as px
    fig = px.bar(type_df, x='Type', y='Count',
                 title="Properties by Type in Market")
    st.plotly_chart(fig, use_container_width=True)
