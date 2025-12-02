"""
Data Validation Utilities
==========================

Validate user input for property predictions.
"""

import pandas as pd
import streamlit as st
from pathlib import Path


# Cache valid options from training data
@st.cache_data
def get_valid_options(train_data_path=None):
    """
    Load valid options for categorical fields from training data.

    Args:
        train_data_path: Path to training CSV (optional)

    Returns:
        dict: Valid options for each categorical field
    """
    if train_data_path is None:
        # Try default path
        train_data_path = Path(__file__).parent.parent.parent / "data" / "train.csv"

    try:
        df = pd.read_csv(train_data_path)
        return {
            'markets': set(df['Market Name'].dropna().unique()),
            'property_types': set(df['Secondary Type'].dropna().unique())
        }
    except FileNotFoundError:
        # Fallback defaults
        return {
            'markets': {
                "Boston, MA", "Worcester, MA", "Providence, RI",
                "Springfield , MA", "Barnstable Town, MA",
                "Pittsfield, MA", "Unknown Market"
            },
            'property_types': {
                "Warehouse", "Manufacturing", "Service", "Distribution",
                "Showroom", "Truck Terminal", "Food Processing",
                "Refrigeration/Cold Storage", "other", "unknown"
            }
        }


def validate_property_input(property_data, train_data_path=None):
    """
    Validate property input data.

    Args:
        property_data: Dict with property attributes
        train_data_path: Optional path to training data for validation

    Returns:
        tuple: (is_valid, error_messages)
    """
    errors = []
    warnings = []

    # Get valid options
    valid_options = get_valid_options(train_data_path)

    # Required fields
    required_fields = ['Market Name', 'Secondary Type', 'RBA', 'Year Built']

    for field in required_fields:
        if field not in property_data or property_data[field] is None:
            errors.append(f"Missing required field: {field}")

    # Validate RBA (Rentable Building Area)
    rba = property_data.get('RBA', 0)
    if rba <= 0:
        errors.append("RBA must be greater than 0")
    elif rba > 10_000_000:  # 10 million sq ft seems unreasonable
        errors.append("RBA seems unreasonably large (> 10M sq ft)")

    # Validate Year Built
    year_built = property_data.get('Year Built', 2025)
    if year_built < 1800:
        errors.append("Year Built must be after 1800")
    elif year_built > 2025:
        errors.append("Year Built cannot be in the future")

    # Validate Latitude/Longitude if provided
    lat = property_data.get('Latitude')
    lon = property_data.get('Longitude')

    if lat is not None:
        if lat < 24 or lat > 50:  # Continental US roughly
            errors.append("Latitude seems outside continental US range")

    if lon is not None:
        if lon < -125 or lon > -65:  # Continental US roughly
            errors.append("Longitude seems outside continental US range")

    # Validate Stories if provided
    stories = property_data.get('Stories')
    if stories is not None and stories <= 0:
        errors.append("Stories must be greater than 0")

    # Validate Market Name against known markets
    market_name = property_data.get('Market Name')
    if market_name and market_name not in valid_options['markets']:
        available_markets = ', '.join(sorted(valid_options['markets']))
        errors.append(f"Unknown market: '{market_name}'. Valid markets: {available_markets}")

    # Validate Secondary Type against known property types
    secondary_type = property_data.get('Secondary Type')
    if secondary_type and secondary_type not in valid_options['property_types']:
        available_types = ', '.join(sorted(valid_options['property_types']))
        errors.append(f"Unknown property type: '{secondary_type}'. Valid types: {available_types}")

    is_valid = len(errors) == 0

    return is_valid, errors


def validate_batch_csv(df):
    """
    Validate uploaded CSV for batch prediction.

    Args:
        df: Uploaded DataFrame

    Returns:
        tuple: (is_valid, error_messages, warnings)
    """
    errors = []
    warnings = []

    # Check required columns
    required_cols = ['Market Name', 'Secondary Type', 'RBA', 'Year Built']
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        errors.append(f"Missing required columns: {', '.join(missing_cols)}")

    # Check for empty dataframe
    if len(df) == 0:
        errors.append("CSV file is empty")

    # Check for too many rows
    if len(df) > 500:
        warnings.append(f"Large file ({len(df)} rows). Consider splitting into smaller batches for faster processing.")

    # Check for missing values in required columns
    if not errors:  # Only if required columns exist
        for col in required_cols:
            null_count = df[col].isnull().sum()
            if null_count > 0:
                warnings.append(f"Column '{col}' has {null_count} missing values. These rows may have reduced accuracy.")

    # Check data types
    if 'RBA' in df.columns:
        if not pd.api.types.is_numeric_dtype(df['RBA']):
            errors.append("RBA column must contain numeric values")

    if 'Year Built' in df.columns:
        if not pd.api.types.is_numeric_dtype(df['Year Built']):
            errors.append("Year Built column must contain numeric values")

    is_valid = len(errors) == 0

    return is_valid, errors, warnings


def show_validation_results(is_valid, errors, warnings=None):
    """
    Display validation results in Streamlit.

    Args:
        is_valid: Boolean indicating if validation passed
        errors: List of error messages
        warnings: Optional list of warning messages
    """
    if not is_valid:
        st.error("**Validation Failed:**")
        for error in errors:
            st.error(f"• {error}")
        return False

    if warnings:
        st.warning("**Warnings:**")
        for warning in warnings:
            st.warning(f"• {warning}")

    return True
