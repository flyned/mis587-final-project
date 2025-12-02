"""
Predictor Wrapper for Streamlit
================================

Wraps the prediction functionality with Streamlit caching for performance.
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
import sys

# Add parent directory to import project modules
parent_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(parent_dir))

from src.feature_engineering import engineer_features
from src.modeling import prepare_data_for_modeling


@st.cache_resource
def load_model_and_metadata(model_path, metadata_path):
    """
    Load trained model and metadata (cached).

    Supports both:
    - Random Forest models (.pkl files)
    - Neural Network models (.keras files)

    Args:
        model_path: Path to model file (.pkl or .keras)
        metadata_path: Path to .json metadata file

    Returns:
        tuple: (model, metadata_dict)
    """
    model_path = str(model_path)

    # Load metadata first
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    # Check model type from extension
    if model_path.endswith('.keras') or model_path.endswith('.h5'):
        # Load Neural Network model
        try:
            from tensorflow import keras
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            raise ImportError("TensorFlow required for neural network models. "
                            "Install with: pip install tensorflow>=2.13.0")

        model = keras.models.load_model(model_path)

        # Store model type in metadata for downstream use
        metadata['_model_type'] = 'neural_network'

        # Reconstruct scaler from metadata
        if 'scaler_mean' in metadata:
            scaler = StandardScaler()
            scaler.mean_ = np.array(metadata['scaler_mean'])
            scaler.scale_ = np.array(metadata['scaler_scale'])
            scaler.var_ = scaler.scale_ ** 2
            scaler.n_features_in_ = len(scaler.mean_)
            metadata['_nn_scaler'] = scaler
    else:
        # Load Random Forest model from pickle
        with open(model_path, 'rb') as f:
            model_dict = pickle.load(f)

        # Extract model from dict (pickle contains {'model': ..., 'scaler': ...})
        if isinstance(model_dict, dict) and 'model' in model_dict:
            model = model_dict['model']
        else:
            model = model_dict

        metadata['_model_type'] = 'random_forest'

    return model, metadata


@st.cache_data
def get_market_options():
    """Get list of available markets from actual training data."""
    try:
        data_path = Path(__file__).parent.parent.parent / "data" / "train.csv"
        df = pd.read_csv(data_path, usecols=['Market Name'])
        markets = sorted(df['Market Name'].dropna().unique().tolist())
        return markets
    except FileNotFoundError:
        # Fallback if train.csv doesn't exist
        return ["Boston, MA", "Worcester, MA", "Providence, RI",
                "Springfield , MA", "Barnstable Town, MA",
                "Pittsfield, MA", "Unknown Market"]


@st.cache_data
def get_property_type_options():
    """Get list of property types from actual training data."""
    try:
        data_path = Path(__file__).parent.parent.parent / "data" / "train.csv"
        df = pd.read_csv(data_path, usecols=['Secondary Type'])
        prop_types = sorted(df['Secondary Type'].dropna().unique().tolist())
        return prop_types
    except FileNotFoundError:
        # Fallback with common types from MA/RI data
        return ["Warehouse", "Manufacturing", "Service", "Distribution",
                "Showroom", "Truck Terminal", "Food Processing",
                "Refrigeration/Cold Storage", "other", "unknown"]


def predict_single_property(model, metadata, property_data):
    """
    Make prediction for a single property.

    Args:
        model: Trained scikit-learn model
        metadata: Model metadata dict
        property_data: Dict with property attributes

    Returns:
        dict: Prediction results with confidence interval
    """
    # Convert to DataFrame
    df = pd.DataFrame([property_data])

    # Add missing columns with default values for feature engineering
    # These columns are expected by the feature engineering pipeline
    missing_columns = {
        'Last Sale Date': None,
        'Year Renovated': None,
        'Land Area (Acres)': None,
        'Number of Buildings': 1,
        'Parking Spaces': None,
        'FEMA Map Date': None,
        'Origination Date': None,
        'Building Park': None,
        'Zoning': None,
        'Land Use': None,
        'Total Available Space (SF)': None,
        'Coworking Available Space': None
    }

    for col, default_val in missing_columns.items():
        if col not in df.columns:
            df[col] = default_val

    # Add target column as placeholder (required by prepare_data_for_modeling)
    # We're predicting this, so set to 0 (will be ignored during prediction)
    if 'Rent/SF/Yr' not in df.columns:
        df['Rent/SF/Yr'] = 0.0

    # Engineer features
    df_eng = engineer_features(df.copy())

    # Manually create dummy variables for categorical features to avoid drop_first issue
    # This is needed because pd.get_dummies with drop_first=True fails on single rows
    feature_names = metadata.get('feature_names', [])

    # Find which Secondary Type columns the model expects
    secondary_type_features = [f for f in feature_names if f.startswith('Secondary Type_')]

    # Manually create Secondary Type dummy variables
    if 'Secondary Type' in df_eng.columns:
        secondary_type_value = df_eng['Secondary Type'].iloc[0]

        # Create all expected dummy columns as 0
        for feat in secondary_type_features:
            df_eng[feat] = 0

        # Set the correct one to 1
        expected_col = f'Secondary Type_{secondary_type_value}'
        if expected_col in df_eng.columns:
            df_eng[expected_col] = 1

        # Drop the original Secondary Type column
        df_eng = df_eng.drop(columns=['Secondary Type'])

    # Prepare for modeling
    X, y, actual_features, categorical_cols = prepare_data_for_modeling(
        df_eng,
        target_column='Rent/SF/Yr',
        ref_columns=feature_names
    )

    # Check if this is a neural network model
    is_nn = metadata.get('_model_type') == 'neural_network'

    # Apply scaling for neural network models
    if is_nn and '_nn_scaler' in metadata:
        X_scaled = metadata['_nn_scaler'].transform(X)
        pred = model.predict(X_scaled, verbose=0).flatten()[0]
    else:
        pred = model.predict(X)[0]

    # Calculate confidence interval (using model's estimator variance)
    # For Random Forest, use std of tree predictions
    if hasattr(model, 'estimators_'):
        tree_preds = np.array([tree.predict(X)[0] for tree in model.estimators_])
        std = np.std(tree_preds)
        ci_lower = pred - 1.96 * std  # 95% confidence interval
        ci_upper = pred + 1.96 * std
    else:
        # Fallback for neural networks and other models: use overall MAE as approximation
        mae = metadata.get('validation_metrics', {}).get('mae', 1.20)
        ci_lower = pred - 1.96 * mae
        ci_upper = pred + 1.96 * mae

    # Determine confidence level
    confidence_level = assign_confidence_level(property_data, pred)

    return {
        'prediction': pred,
        'ci_lower': max(0, ci_lower),  # Rent can't be negative
        'ci_upper': ci_upper,
        'confidence_level': confidence_level,
        'X': X,
        'feature_names': actual_features
    }


def assign_confidence_level(property_data, predicted_rent):
    """
    Assign confidence level based on property characteristics.

    Args:
        property_data: Dict with property attributes
        predicted_rent: Predicted rent value

    Returns:
        str: 'HIGH', 'MEDIUM', or 'LOW'
    """
    # Start with HIGH confidence
    confidence = 'HIGH'

    # Check property type
    risky_types = ['Food Processing', 'Manufacturing', 'Refrigeration/Cold Storage']
    if property_data.get('Secondary Type') in risky_types:
        confidence = 'MEDIUM'

    # Check price range
    if predicted_rent > 20:
        confidence = 'LOW'  # Luxury properties
    elif predicted_rent > 15:
        if confidence == 'HIGH':
            confidence = 'MEDIUM'

    # Check building age
    from datetime import datetime
    current_year = datetime.now().year
    year_built = property_data.get('Year Built', current_year)
    building_age = current_year - year_built

    if building_age < 10:  # New construction
        if confidence == 'HIGH':
            confidence = 'MEDIUM'

    # Check market (based on actual model performance - R² < 0.65)
    small_markets = ['Unknown Market', 'Springfield , MA',
                     'Pittsfield, MA', 'Barnstable Town, MA']
    if property_data.get('Market Name') in small_markets:
        if confidence == 'HIGH':
            confidence = 'MEDIUM'

    return confidence


def predict_batch(model, metadata, df):
    """
    Make predictions for multiple properties.

    Args:
        model: Trained model (scikit-learn or Keras)
        metadata: Model metadata dict
        df: DataFrame with property attributes

    Returns:
        DataFrame: Original data plus predictions and confidence levels
    """
    # Engineer features
    df_eng = engineer_features(df.copy())

    # Prepare for modeling
    feature_names = metadata.get('feature_names', [])
    X, y, actual_features, categorical_cols = prepare_data_for_modeling(
        df_eng,
        target_column='Rent/SF/Yr',
        ref_columns=feature_names
    )

    # Check if this is a neural network model
    is_nn = metadata.get('_model_type') == 'neural_network'

    # Make predictions
    if is_nn and '_nn_scaler' in metadata:
        X_scaled = metadata['_nn_scaler'].transform(X)
        predictions = model.predict(X_scaled, verbose=0).flatten()
    else:
        predictions = model.predict(X)

    # Add to original dataframe
    result = df.copy()
    result['Predicted_Rent_SF_Yr'] = predictions

    # Assign confidence levels
    confidence_levels = []
    for idx, row in df.iterrows():
        pred = predictions[idx]
        conf = assign_confidence_level(row.to_dict(), pred)
        confidence_levels.append(conf)

    result['Confidence_Level'] = confidence_levels

    return result
