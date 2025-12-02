"""
Address Search Utilities
========================

Fuzzy matching for property address lookup from training data.
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from difflib import SequenceMatcher


def normalize_address(address):
    """
    Normalize address string for better matching.

    Args:
        address: Raw address string

    Returns:
        str: Normalized address
    """
    if not address or pd.isna(address):
        return ""

    # Convert to lowercase
    addr = str(address).lower().strip()

    # Common abbreviation standardizations
    replacements = {
        ' street': ' st',
        ' avenue': ' ave',
        ' boulevard': ' blvd',
        ' drive': ' dr',
        ' road': ' rd',
        ' lane': ' ln',
        ' court': ' ct',
        ' place': ' pl',
        ' circle': ' cir',
        ' highway': ' hwy',
        ' parkway': ' pkwy',
        ' north': ' n',
        ' south': ' s',
        ' east': ' e',
        ' west': ' w',
        ' northeast': ' ne',
        ' northwest': ' nw',
        ' southeast': ' se',
        ' southwest': ' sw',
    }

    for full, abbr in replacements.items():
        addr = addr.replace(full, abbr)

    # Remove extra whitespace
    addr = ' '.join(addr.split())

    return addr


def fuzzy_score(query, candidate):
    """
    Calculate fuzzy match score between query and candidate.

    Args:
        query: Search query
        candidate: Candidate address

    Returns:
        float: Match score 0-100
    """
    query_norm = normalize_address(query)
    candidate_norm = normalize_address(candidate)

    if not query_norm or not candidate_norm:
        return 0.0

    # Use SequenceMatcher for fuzzy matching
    ratio = SequenceMatcher(None, query_norm, candidate_norm).ratio()

    # Bonus for substring match
    if query_norm in candidate_norm:
        ratio = min(1.0, ratio + 0.2)

    return round(ratio * 100, 1)


@st.cache_data
def load_property_index(train_data_path):
    """
    Load and index properties from training data.

    Args:
        train_data_path: Path to training CSV

    Returns:
        pd.DataFrame: Properties with address index
    """
    try:
        df = pd.read_csv(train_data_path)

        # Create normalized address column for faster searching
        df['_normalized_address'] = df['Property Address'].apply(normalize_address)

        return df
    except FileNotFoundError:
        return pd.DataFrame()


def search_properties_by_address(query, train_data_path, max_results=10, min_score=30):
    """
    Search for properties by address using fuzzy matching.

    Args:
        query: Search query string
        train_data_path: Path to training CSV
        max_results: Maximum number of results to return
        min_score: Minimum match score to include (0-100)

    Returns:
        list: List of dicts with property data and match scores
    """
    if not query or len(query) < 2:
        return []

    # Load indexed data
    df = load_property_index(train_data_path)

    if df.empty:
        return []

    query_norm = normalize_address(query)

    # Calculate match scores for all properties
    results = []
    for idx, row in df.iterrows():
        score = fuzzy_score(query, row['Property Address'])

        if score >= min_score:
            results.append({
                'index': idx,
                'score': score,
                'Property Address': row.get('Property Address', 'N/A'),
                'Market Name': row.get('Market Name', 'N/A'),
                'Secondary Type': row.get('Secondary Type', 'N/A'),
                'RBA': row.get('RBA', 0),
                'Year Built': int(row.get('Year Built', 0)) if pd.notna(row.get('Year Built')) else 0,
                'Number Of Stories': int(row.get('Number Of Stories', 0)) if pd.notna(row.get('Number Of Stories')) else 0,
                'Latitude': row.get('Latitude', 0),
                'Longitude': row.get('Longitude', 0),
                'Rent/SF/Yr': row.get('Rent/SF/Yr', 0)
            })

    # Sort by score descending and limit results
    results.sort(key=lambda x: x['score'], reverse=True)

    return results[:max_results]


def get_property_by_address(address, train_data_path):
    """
    Get exact property data by address.

    Args:
        address: Property address
        train_data_path: Path to training CSV

    Returns:
        dict or None: Property data if found
    """
    df = load_property_index(train_data_path)

    if df.empty:
        return None

    # Try exact match first
    exact_match = df[df['Property Address'] == address]

    if not exact_match.empty:
        row = exact_match.iloc[0]
        return {
            'Property Address': row.get('Property Address', 'N/A'),
            'Market Name': row.get('Market Name', 'N/A'),
            'Secondary Type': row.get('Secondary Type', 'N/A'),
            'RBA': row.get('RBA', 0),
            'Year Built': int(row.get('Year Built', 0)) if pd.notna(row.get('Year Built')) else 0,
            'Stories': int(row.get('Number Of Stories', 0)) if pd.notna(row.get('Number Of Stories')) else 0,
            'Latitude': row.get('Latitude', 0),
            'Longitude': row.get('Longitude', 0),
            'Rent/SF/Yr': row.get('Rent/SF/Yr', 0)
        }

    # Try normalized match
    query_norm = normalize_address(address)
    norm_match = df[df['_normalized_address'] == query_norm]

    if not norm_match.empty:
        row = norm_match.iloc[0]
        return {
            'Property Address': row.get('Property Address', 'N/A'),
            'Market Name': row.get('Market Name', 'N/A'),
            'Secondary Type': row.get('Secondary Type', 'N/A'),
            'RBA': row.get('RBA', 0),
            'Year Built': int(row.get('Year Built', 0)) if pd.notna(row.get('Year Built')) else 0,
            'Stories': int(row.get('Number Of Stories', 0)) if pd.notna(row.get('Number Of Stories')) else 0,
            'Latitude': row.get('Latitude', 0),
            'Longitude': row.get('Longitude', 0),
            'Rent/SF/Yr': row.get('Rent/SF/Yr', 0)
        }

    return None


def display_search_results(results, on_select_callback=None):
    """
    Display search results in Streamlit with selection option.

    Args:
        results: List of property results from search
        on_select_callback: Optional callback when property is selected

    Returns:
        dict or None: Selected property data
    """
    if not results:
        st.info("No properties found matching your search.")
        return None

    st.markdown(f"Found **{len(results)}** matching properties:")

    # Create formatted options
    options = ["-- Select a property --"]
    for r in results:
        label = f"{r['Property Address']} ({r['Market Name']}) - {r['score']:.0f}% match"
        options.append(label)

    selection = st.selectbox(
        "Select property to auto-fill form:",
        options=options,
        key="property_search_selection"
    )

    if selection != "-- Select a property --":
        # Find selected property
        idx = options.index(selection) - 1  # Adjust for header option
        selected = results[idx]

        # Show selected property details
        with st.expander("Selected Property Details", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Address:** {selected['Property Address']}")
                st.write(f"**Market:** {selected['Market Name']}")
                st.write(f"**Type:** {selected['Secondary Type']}")
            with col2:
                st.write(f"**RBA:** {selected['RBA']:,} SF")
                st.write(f"**Year Built:** {selected['Year Built']}")
                st.write(f"**Actual Rent:** ${selected['Rent/SF/Yr']:.2f}/SF/Yr")

        return selected

    return None
