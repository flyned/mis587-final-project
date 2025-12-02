"""
SHAP Explainer Utilities
=========================

Generate SHAP explanations for predictions with Streamlit-friendly formatting.
"""

import streamlit as st
import shap
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO


@st.cache_resource
def create_shap_explainer(_model):
    """
    Create SHAP TreeExplainer (cached).

    Args:
        _model: Trained tree-based model (Random Forest, XGBoost, etc.)

    Returns:
        shap.TreeExplainer
    """
    return shap.TreeExplainer(_model)


def generate_shap_force_plot(model, X, feature_names, base_value=None):
    """
    Generate SHAP force plot for a single prediction.

    Args:
        model: Trained model
        X: Feature matrix (single row)
        feature_names: List of feature names
        base_value: Base prediction value (optional)

    Returns:
        matplotlib.figure.Figure
    """
    # Create explainer
    explainer = create_shap_explainer(model)

    # Calculate SHAP values
    shap_values = explainer.shap_values(X)

    if base_value is None:
        base_value = explainer.expected_value

    # Convert base_value to scalar if it's an array (Random Forest returns array)
    if hasattr(base_value, '__len__'):
        base_value = float(base_value[0])
    else:
        base_value = float(base_value)

    # Get actual SHAP values for the single prediction
    if len(shap_values.shape) > 1:
        shap_vals = shap_values[0]
    else:
        shap_vals = shap_values

    # Create force plot data
    # Sort by absolute impact
    feature_impacts = []
    for i, (feat, val) in enumerate(zip(feature_names, shap_vals)):
        if abs(val) > 0.01:  # Only include significant features
            feature_impacts.append({
                'feature': feat,
                'shap_value': val,
                'feature_value': X[0, i] if hasattr(X, 'shape') else X.iloc[0, i]
            })

    feature_impacts.sort(key=lambda x: abs(x['shap_value']), reverse=True)

    return feature_impacts[:10], base_value  # Top 10 features


def create_shap_waterfall_text(feature_impacts, base_value, prediction):
    """
    Create text-based waterfall explanation.

    Args:
        feature_impacts: List of feature impact dicts
        base_value: Base prediction value
        prediction: Final prediction value

    Returns:
        str: Formatted explanation text
    """
    lines = []
    lines.append(f"**Base Rate:** ${base_value:.2f}/SF/Yr")
    lines.append("")

    cumulative = base_value

    for impact in feature_impacts:
        feat = impact['feature']
        shap_val = impact['shap_value']
        feat_val = impact['feature_value']

        # Make feature names more readable
        feat_display = feat.replace('_', ' ').title()

        if shap_val > 0:
            lines.append(f"+ **{feat_display}**: +${abs(shap_val):.2f}")
        else:
            lines.append(f"- **{feat_display}**: -${abs(shap_val):.2f}")

        cumulative += shap_val

    lines.append("")
    lines.append(f"**= Final Prediction:** ${prediction:.2f}/SF/Yr")

    return "\n".join(lines)


def generate_shap_summary_plot_data(model, X_sample, feature_names, max_display=20):
    """
    Generate data for SHAP summary plot.

    Args:
        model: Trained model
        X_sample: Sample of feature matrix (multiple rows)
        feature_names: List of feature names
        max_display: Number of top features to show

    Returns:
        dict: SHAP summary data
    """
    # Create explainer
    explainer = create_shap_explainer(model)

    # Calculate SHAP values
    shap_values = explainer.shap_values(X_sample)

    # Calculate mean absolute SHAP values
    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    # Get top features
    top_indices = np.argsort(mean_abs_shap)[::-1][:max_display]

    summary_data = []
    for idx in top_indices:
        feat = feature_names[idx]
        importance = mean_abs_shap[idx]

        summary_data.append({
            'feature': feat,
            'importance': importance,
            'shap_values': shap_values[:, idx],
            'feature_values': X_sample[:, idx] if hasattr(X_sample, 'shape') else X_sample.iloc[:, idx].values
        })

    return summary_data
