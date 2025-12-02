"""
Model interpretation utilities for Phase 3

This module provides comprehensive model interpretation capabilities:
- Feature importance (MDI, Permutation, SHAP)
- Residual analysis and diagnostics
- Error segmentation by property characteristics
- SHAP value calculations and explanations

All functions designed for business stakeholder communication with
clear, actionable insights.

Author: MIS587 Final Project
Date: November 2024
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from scipy import stats
import shap
import matplotlib.pyplot as plt

# Import visualization functions
from .visualization import (
    plot_residuals_vs_predicted,
    plot_residuals_distribution,
    plot_qq_plot,
    plot_prediction_scatter,
    save_figure
)


# ========================================================================
# SECTION 1: FEATURE IMPORTANCE ANALYSIS
# ========================================================================

def get_mdi_importance(model, feature_names, top_n=20):
    """
    Mean Decrease in Impurity (MDI) importance from tree model.

    Wraps the built-in feature_importances_ attribute for consistency.
    MDI measures how much each feature decreases node impurity (weighted
    by probability of reaching that node).

    Args:
        model: Trained tree model with feature_importances_ attribute
        feature_names: List of feature names (must match model features)
        top_n: Number of top features to return (default 20)

    Returns:
        pd.DataFrame with columns:
            - feature: Feature name
            - importance: MDI importance score
            - importance_pct: Percentage of total importance

    Business value:
        Fast, built-in method showing which features the model splits on most.
        Top features are primary drivers of predictions.

    Limitations:
        - Biased toward high-cardinality features
        - Doesn't measure actual predictive impact
        - Use permutation importance for more reliable ranking
    """
    # Get importances from model
    importances = model.feature_importances_

    # Create DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    })

    # Calculate percentage
    total_importance = importance_df['importance'].sum()
    importance_df['importance_pct'] = (importance_df['importance'] / total_importance) * 100

    # Sort by importance descending
    importance_df = importance_df.sort_values('importance', ascending=False).reset_index(drop=True)

    # Return top N
    return importance_df.head(top_n)


def get_permutation_importance(model, X, y, feature_names, n_repeats=10,
                                random_state=42, top_n=20):
    """
    Permutation importance: shuffle feature, measure performance drop.

    Permutation importance measures the decrease in model score when a feature's
    values are randomly shuffled. More reliable than MDI as it measures actual
    predictive impact.

    Args:
        model: Trained model (any sklearn-compatible model)
        X: Feature matrix (validation set recommended)
        y: Target values (validation set recommended)
        feature_names: List of feature names
        n_repeats: Number of permutation rounds (10 = good balance)
        random_state: Random seed for reproducibility
        top_n: Number of top features to return

    Returns:
        pd.DataFrame with columns:
            - feature: Feature name
            - importance: Mean decrease in score across repeats
            - importance_std: Standard deviation of importance

    Runtime: ~30-60 seconds for 195 features × 10 repeats (memory-safe: n_jobs=1)

    Business value:
        Model-agnostic measure of actual predictive impact.
        Features with high permutation importance are essential for accuracy.

    Usage:
        >>> perm_imp = get_permutation_importance(model, X_val, y_val, feature_names)
        >>> print(perm_imp.head())
    """
    # Calculate permutation importance (n_jobs=1 for memory safety)
    perm_result = permutation_importance(
        model, X, y,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=1,  # No parallelization to avoid memory issues
        scoring='neg_mean_absolute_error'
    )

    # Create DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': perm_result.importances_mean,
        'importance_std': perm_result.importances_std
    })

    # Sort by importance descending
    importance_df = importance_df.sort_values('importance', ascending=False).reset_index(drop=True)

    # Return top N
    return importance_df.head(top_n)


def compare_importance_methods(mdi_df, perm_df, top_n=15):
    """
    Compare MDI vs Permutation importance to identify consensus features.

    Features important by both methods are most reliable for stakeholder
    communication and decision-making.

    Args:
        mdi_df: DataFrame from get_mdi_importance()
        perm_df: DataFrame from get_permutation_importance()
        top_n: Number of features to compare

    Returns:
        pd.DataFrame with columns:
            - feature: Feature name
            - mdi_rank: Rank in MDI (1=most important)
            - mdi_importance: MDI importance score
            - perm_rank: Rank in permutation
            - perm_importance: Permutation importance score
            - rank_diff: Absolute difference in ranks
            - consensus: 'High' if both in top N, 'Medium' if one, 'Low' otherwise

    Business value:
        Identifies reliable features for presentations. Consensus features
        are robust to methodology and safe for business recommendations.

    Example:
        >>> comparison = compare_importance_methods(mdi_df, perm_df, top_n=10)
        >>> consensus_features = comparison[comparison['consensus'] == 'High']
    """
    # Create ranking dictionaries
    mdi_ranks = {row['feature']: (idx + 1, row['importance'])
                 for idx, row in mdi_df.iterrows()}
    perm_ranks = {row['feature']: (idx + 1, row['importance'])
                  for idx, row in perm_df.iterrows()}

    # Get all unique features from both methods
    all_features = set(mdi_df['feature'].tolist() + perm_df['feature'].tolist())

    # Build comparison DataFrame
    comparison_data = []
    for feature in all_features:
        mdi_rank, mdi_imp = mdi_ranks.get(feature, (999, 0))
        perm_rank, perm_imp = perm_ranks.get(feature, (999, 0))

        rank_diff = abs(mdi_rank - perm_rank)

        # Determine consensus level
        if mdi_rank <= top_n and perm_rank <= top_n:
            consensus = 'High'
        elif mdi_rank <= top_n or perm_rank <= top_n:
            consensus = 'Medium'
        else:
            consensus = 'Low'

        comparison_data.append({
            'feature': feature,
            'mdi_rank': mdi_rank,
            'mdi_importance': mdi_imp,
            'perm_rank': perm_rank,
            'perm_importance': perm_imp,
            'rank_diff': rank_diff,
            'consensus': consensus
        })

    comparison_df = pd.DataFrame(comparison_data)

    # Sort by consensus (High first), then by average rank
    consensus_order = {'High': 0, 'Medium': 1, 'Low': 2}
    comparison_df['consensus_sort'] = comparison_df['consensus'].map(consensus_order)
    comparison_df['avg_rank'] = (comparison_df['mdi_rank'] + comparison_df['perm_rank']) / 2
    comparison_df = comparison_df.sort_values(['consensus_sort', 'avg_rank'])
    comparison_df = comparison_df.drop(['consensus_sort', 'avg_rank'], axis=1)

    return comparison_df.reset_index(drop=True)


# ========================================================================
# SECTION 2: SHAP VALUE ANALYSIS
# ========================================================================

def calculate_shap_values(model, X, feature_names, max_samples=1000,
                          save_path='figures/shap_values.pkl'):
    """
    Calculate SHAP values using TreeExplainer (memory-efficient).

    SHAP (SHapley Additive exPlanations) values show the contribution of each
    feature to individual predictions. Provides local (per-prediction) and
    global (averaged) interpretability.

    Args:
        model: Trained Random Forest model
        X: Feature matrix (validation set recommended)
        feature_names: List of feature names
        max_samples: Max samples to use (memory constraint, default 1000)
        save_path: Where to save SHAP values for reuse

    Returns:
        shap.Explanation object with:
            - .values: SHAP values array (n_samples × n_features)
            - .base_values: Expected value (model's average prediction)
            - .data: Original feature values
            - .feature_names: Feature names

    Memory: ~500MB for 1000 samples × 195 features
    Runtime: ~60-90 seconds

    Business value:
        Explains individual predictions: "For this property, location adds $5/SF,
        but age subtracts $2/SF, resulting in $18/SF prediction."

    Note: Samples data if >max_samples to limit memory. Use validation set only.
    """
    # Sample if needed
    if len(X) > max_samples:
        print(f"Sampling {max_samples} from {len(X)} samples for SHAP calculation (memory optimization)")
        sample_idx = np.random.choice(len(X), max_samples, replace=False)
        X_shap = X[sample_idx] if isinstance(X, np.ndarray) else X.iloc[sample_idx]
    else:
        X_shap = X

    print(f"Calculating SHAP values for {len(X_shap)} samples × {len(feature_names)} features...")

    # Create TreeExplainer (exact algorithm for tree models)
    explainer = shap.TreeExplainer(model)

    # Calculate SHAP values
    shap_values = explainer(X_shap)

    # Add feature names
    shap_values.feature_names = feature_names

    # Save for reuse
    save_dir = Path(save_path).parent
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'wb') as f:
        pickle.dump(shap_values, f)
    print(f"SHAP values saved to {save_path} (reusable)")

    return shap_values


def get_shap_feature_importance(shap_values, feature_names, top_n=20):
    """
    Global feature importance from mean absolute SHAP values.

    Unlike MDI, SHAP importance is scale-aware and measures actual impact
    on predictions (in dollars for our case).

    Args:
        shap_values: SHAP Explanation object from calculate_shap_values()
        feature_names: List of feature names
        top_n: Number of top features to return

    Returns:
        pd.DataFrame with columns:
            - feature: Feature name
            - importance: mean(|SHAP value|) across all samples
            - importance_pct: Percentage of total importance

    Business value:
        Shows average dollar impact of each feature on predictions.
        "Building age has an average impact of $2.50/SF on rent predictions."
    """
    # Calculate mean absolute SHAP values
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)

    # Create DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': mean_abs_shap
    })

    # Calculate percentage
    total_importance = importance_df['importance'].sum()
    importance_df['importance_pct'] = (importance_df['importance'] / total_importance) * 100

    # Sort by importance descending
    importance_df = importance_df.sort_values('importance', ascending=False).reset_index(drop=True)

    # Return top N
    return importance_df.head(top_n)


def plot_shap_summary(shap_values, X, feature_names, top_n=20, save_path=None):
    """
    SHAP summary plot (beeswarm): feature importance + value distribution.

    Visualization shows:
    - X-axis: SHAP value (impact on prediction in $/SF/Yr)
    - Y-axis: Feature (sorted by importance)
    - Color: Feature value (red=high, blue=low)
    - Each dot: One prediction

    Args:
        shap_values: SHAP Explanation object
        X: Feature matrix (for coloring by feature values)
        feature_names: List of feature names
        top_n: Number of features to show (default 20)
        save_path: Optional path to save figure

    Returns:
        matplotlib.Figure

    Business value:
        "When building_age is high (red dots), SHAP values are negative (left),
        meaning older buildings → lower rent. When building_age is low (blue dots),
        SHAP values are positive (right), meaning newer buildings → higher rent."

    Usage:
        >>> fig = plot_shap_summary(shap_values, X_val, feature_names)
        >>> plt.show()
    """
    # SHAP library handles plotting
    fig, ax = plt.subplots(figsize=(10, max(8, top_n * 0.4)))

    shap.summary_plot(
        shap_values.values,
        X,
        feature_names=feature_names,
        max_display=top_n,
        show=False,
        plot_type='dot'  # Beeswarm plot
    )

    plt.title(f'SHAP Summary Plot (Top {top_n} Features)', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('SHAP Value (Impact on Prediction, $/SF/Yr)', fontsize=12)
    plt.tight_layout()

    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')

    return fig


def plot_shap_dependence(shap_values, X, feature_names, feature,
                          interaction_feature=None, save_path=None):
    """
    SHAP dependence plot: how feature value affects prediction.

    Shows the relationship between feature value and its impact on predictions.
    Optionally colored by interaction feature to reveal interactions.

    Args:
        shap_values: SHAP Explanation object
        X: Feature matrix
        feature_names: List of feature names
        feature: Feature to analyze (str or int index)
        interaction_feature: Feature to color by (auto-selected if None)
        save_path: Optional path to save figure

    Returns:
        matplotlib.Figure

    Visualization shows:
        - X-axis: Feature value
        - Y-axis: SHAP value (impact on prediction)
        - Color: Interaction feature value
        - Trend: Smoothed relationship

    Business value:
        "As building size increases, rent impact increases linearly (positive SHAP).
        However, this effect is stronger in urban markets (red dots higher)."

    Usage:
        >>> fig = plot_shap_dependence(shap_values, X_val, feature_names, 'building_age')
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Get feature index
    if isinstance(feature, str):
        feature_idx = feature_names.index(feature)
        feature_name = feature
    else:
        feature_idx = feature
        feature_name = feature_names[feature_idx]

    # SHAP dependence plot
    shap.dependence_plot(
        feature_idx,
        shap_values.values,
        X,
        feature_names=feature_names,
        interaction_index=interaction_feature,
        show=False,
        ax=ax
    )

    plt.title(f'SHAP Dependence Plot: {feature_name}', fontsize=14, fontweight='bold')
    plt.xlabel(feature_name, fontsize=12)
    plt.ylabel(f'SHAP Value (Impact on Rent Prediction)', fontsize=12)
    plt.tight_layout()

    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')

    return fig


# ========================================================================
# SECTION 3: RESIDUAL ANALYSIS
# ========================================================================

def analyze_residuals(y_true, y_pred):
    """
    Comprehensive residual analysis with statistical tests.

    Performs diagnostic checks on model residuals to validate assumptions:
    - Normality (Shapiro-Wilk test)
    - Zero mean (t-test)
    - Homoscedasticity (constant variance)

    Args:
        y_true: Actual target values (numpy array or pandas Series)
        y_pred: Predicted values (numpy array or pandas Series)

    Returns:
        dict with:
            - 'residuals': Array of residuals (predicted - actual)
            - 'abs_residuals': Array of absolute residuals
            - 'percent_errors': Array of percent errors
            - 'residual_stats': Dict with mean, std, skew, kurtosis
            - 'normality_test': Dict with statistic, p_value, is_normal
            - 'outlier_count': Number of outliers (|z-score| > 3)

    Business value:
        Validates model assumptions:
        - Normal residuals → prediction intervals reliable
        - Zero mean → no systematic bias
        - Constant variance → equal confidence across price ranges

    Usage:
        >>> residual_analysis = analyze_residuals(y_val, y_pred_val)
        >>> print(f"Mean residual: ${residual_analysis['residual_stats']['mean']:.2f}")
        >>> if residual_analysis['normality_test']['is_normal']:
        >>>     print("Residuals are normally distributed ✓")
    """
    # Convert to numpy arrays
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    # Calculate residuals
    residuals = y_pred - y_true
    abs_residuals = np.abs(residuals)
    percent_errors = (abs_residuals / np.abs(y_true)) * 100

    # Residual statistics
    residual_stats = {
        'mean': np.mean(residuals),
        'std': np.std(residuals),
        'median': np.median(residuals),
        'skew': stats.skew(residuals),
        'kurtosis': stats.kurtosis(residuals),
        'min': np.min(residuals),
        'max': np.max(residuals)
    }

    # Normality test (Shapiro-Wilk, max 5000 samples)
    sample_size = min(5000, len(residuals))
    sample_idx = np.random.choice(len(residuals), sample_size, replace=False)
    stat, p_value = stats.shapiro(residuals[sample_idx])
    is_normal = p_value > 0.05

    normality_test = {
        'statistic': stat,
        'p_value': p_value,
        'is_normal': is_normal,
        'test_name': 'Shapiro-Wilk',
        'sample_size': sample_size
    }

    # Outlier detection (z-score > 3)
    z_scores = np.abs(stats.zscore(residuals))
    outlier_mask = z_scores > 3
    outlier_count = np.sum(outlier_mask)
    outlier_pct = (outlier_count / len(residuals)) * 100

    return {
        'residuals': residuals,
        'abs_residuals': abs_residuals,
        'percent_errors': percent_errors,
        'residual_stats': residual_stats,
        'normality_test': normality_test,
        'outlier_count': int(outlier_count),
        'outlier_pct': outlier_pct
    }


def identify_outlier_predictions(y_true, y_pred, threshold_std=3.0):
    """
    Identify predictions with unusually large errors.

    Uses standardized residuals (z-scores): outlier if |residual| > threshold * std(residuals)

    Args:
        y_true: Actual target values
        y_pred: Predicted values
        threshold_std: Standard deviation threshold (3.0 = 99.7% coverage)

    Returns:
        dict with:
            - 'outlier_indices': Array of row indices with outlier predictions
            - 'outlier_residuals': Array of residual values for outliers
            - 'outlier_y_true': Actual values for outliers
            - 'outlier_y_pred': Predicted values for outliers
            - 'outlier_count': Number of outliers
            - 'outlier_pct': Percentage of outliers

    Business value:
        Flags properties requiring manual appraisal review.
        Large errors may indicate:
        - Data quality issues (missing/incorrect features)
        - Unique property characteristics not captured by model
        - Market anomalies (distressed sales, special deals)

    Usage:
        >>> outliers = identify_outlier_predictions(y_val, y_pred_val)
        >>> print(f"Flag {outliers['outlier_count']} properties for manual review")
        >>> outlier_properties = df_val.iloc[outliers['outlier_indices']]
    """
    # Convert to numpy arrays
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    # Calculate residuals and z-scores
    residuals = y_pred - y_true
    z_scores = np.abs(stats.zscore(residuals))

    # Identify outliers
    outlier_mask = z_scores > threshold_std
    outlier_indices = np.where(outlier_mask)[0]

    return {
        'outlier_indices': outlier_indices,
        'outlier_residuals': residuals[outlier_mask],
        'outlier_y_true': y_true[outlier_mask],
        'outlier_y_pred': y_pred[outlier_mask],
        'outlier_count': len(outlier_indices),
        'outlier_pct': (len(outlier_indices) / len(y_true)) * 100
    }


def create_residual_plots(y_true, y_pred, save_dir='figures/residual_analysis'):
    """
    Generate comprehensive residual diagnostic plots.

    Creates 4 diagnostic plots:
    1. Residuals vs Predicted (check heteroscedasticity)
    2. Residual histogram + KDE (check normality)
    3. Q-Q plot (check normality visually)
    4. Actual vs Predicted scatter (check bias)

    Args:
        y_true: Actual target values
        y_pred: Predicted values
        save_dir: Directory to save plots (creates if needed)

    Returns:
        dict mapping plot_name -> matplotlib.Figure

    Business value:
        Visual validation of model quality for stakeholder presentations.

    Red flags:
        - Funnel shape in residuals → heteroscedasticity (error variance not constant)
        - Skewed residual distribution → systematic bias
        - Points far from diagonal in Q-Q plot → non-normality
        - Systematic pattern in actual vs predicted → model missing key relationships

    Usage:
        >>> plots = create_residual_plots(y_val, y_pred_val)
        >>> # Plots auto-saved to figures/residual_analysis/
    """
    # Convert to numpy arrays
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    residuals = y_pred - y_true

    # Create save directory
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    plots = {}

    # Plot 1: Residuals vs Predicted
    fig1 = plot_residuals_vs_predicted(
        y_true, y_pred,
        title='Residuals vs Predicted Values',
        save_path=save_path / 'residuals_vs_predicted.png'
    )
    plots['residuals_vs_predicted'] = fig1
    plt.close(fig1)

    # Plot 2: Residual Distribution
    fig2 = plot_residuals_distribution(
        residuals,
        title='Residual Distribution',
        save_path=save_path / 'residuals_distribution.png'
    )
    plots['residuals_distribution'] = fig2
    plt.close(fig2)

    # Plot 3: Q-Q Plot
    fig3 = plot_qq_plot(
        residuals,
        title='Q-Q Plot (Normality Check)',
        save_path=save_path / 'qq_plot.png'
    )
    plots['qq_plot'] = fig3
    plt.close(fig3)

    # Plot 4: Actual vs Predicted
    fig4 = plot_prediction_scatter(
        y_true, y_pred,
        title='Predicted vs Actual Rent',
        save_path=save_path / 'actual_vs_predicted.png'
    )
    plots['actual_vs_predicted'] = fig4
    plt.close(fig4)

    print(f"Saved 4 residual diagnostic plots to {save_dir}/")

    return plots


# ========================================================================
# SECTION 4: ERROR SEGMENTATION
# ========================================================================

def segment_errors_by_feature(y_true, y_pred, feature_data, feature_name,
                               bins=None, labels=None):
    """
    Calculate error metrics for segments of a feature.

    For categorical features: Groups by category
    For numerical features: Bins into ranges (e.g., price quartiles)

    Args:
        y_true: Actual target values
        y_pred: Predicted values
        feature_data: Array or Series of feature values (aligned with y_true)
        feature_name: Name of feature (for display)
        bins: For numerical features, bin edges (int or list)
              If int, creates that many equal-width bins
        labels: For numerical features, bin labels (must match bins-1)

    Returns:
        pd.DataFrame with columns:
            - segment: Category or bin label
            - count: Number of samples in segment
            - MAE: Mean absolute error
            - RMSE: Root mean squared error
            - R2: R² score
            - MAPE: Mean absolute percentage error
            - Within_10pct: Percentage within ±10% band

    Business value:
        "Model underperforms in Chicago market (MAE=$1.20 vs $0.68 overall)"
        "Model excels at predicting high-priced properties (R²=0.92 for top quartile)"

    Usage:
        >>> # Categorical segmentation
        >>> seg_market = segment_errors_by_feature(y_val, y_pred, df_val['Market Name'], 'Market')
        >>>
        >>> # Numerical segmentation (price quintiles)
        >>> seg_price = segment_errors_by_feature(y_val, y_pred, y_val, 'Price Range',
        >>>                                       bins=5, labels=['Very Low', 'Low', 'Med', 'High', 'Very High'])
    """
    # Convert to numpy/pandas
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    # Handle binning for numerical features
    if bins is not None:
        feature_segments = pd.cut(feature_data, bins=bins, labels=labels, include_lowest=True)
    else:
        feature_segments = feature_data

    # Create DataFrame for grouping
    df = pd.DataFrame({
        'segment': feature_segments,
        'y_true': y_true,
        'y_pred': y_pred,
        'error': y_pred - y_true,
        'abs_error': np.abs(y_pred - y_true),
        'pct_error': np.abs((y_pred - y_true) / y_true) * 100,
        'within_10pct': (np.abs((y_pred - y_true) / y_true) < 0.10).astype(int)
    })

    # Calculate metrics by segment
    segment_metrics = []
    for segment, group in df.groupby('segment'):
        if len(group) < 5:  # Skip segments with too few samples
            continue

        mae = mean_absolute_error(group['y_true'], group['y_pred'])
        rmse = mean_squared_error(group['y_true'], group['y_pred'], squared=False)
        r2 = r2_score(group['y_true'], group['y_pred'])
        mape = mean_absolute_percentage_error(group['y_true'], group['y_pred']) * 100
        within_10pct = group['within_10pct'].mean() * 100

        segment_metrics.append({
            'segment': segment,
            'count': len(group),
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2,
            'MAPE': mape,
            'Within_10pct': within_10pct
        })

    result_df = pd.DataFrame(segment_metrics)

    # Sort by MAE descending (worst first)
    result_df = result_df.sort_values('MAE', ascending=False).reset_index(drop=True)

    return result_df


def analyze_errors_by_market(y_true, y_pred, market_data):
    """
    Error analysis segmented by Market Name.

    Identifies markets where model underperforms, requiring:
    - More training data
    - Market-specific models
    - Additional market-level features

    Args:
        y_true: Actual target values
        y_pred: Predicted values
        market_data: Array/Series of market names (aligned with y_true)

    Returns:
        pd.DataFrame sorted by MAE descending (worst markets first)

    Business value:
        "Focus data collection on Chicago and Miami markets (MAE >$1.00)"
        "Consider market-specific models for underperforming regions"
    """
    return segment_errors_by_feature(y_true, y_pred, market_data, 'Market Name')


def analyze_errors_by_property_type(y_true, y_pred, property_type_data):
    """
    Error analysis segmented by Property Type (Office, Industrial, Retail, etc.).

    Identifies property types needing:
    - More training data
    - Type-specific features
    - Separate models

    Args:
        y_true: Actual target values
        y_pred: Predicted values
        property_type_data: Array/Series of property types

    Returns:
        pd.DataFrame sorted by MAE descending

    Business value:
        "Model excels at Office (R²=0.90) but struggles with Industrial (R²=0.75)"
        "Retail properties have high variance - need more type-specific features"
    """
    return segment_errors_by_feature(y_true, y_pred, property_type_data, 'Property Type')


def analyze_errors_by_price_range(y_true, y_pred, price_data=None, n_bins=5):
    """
    Error analysis segmented by price/rent ranges (quintiles by default).

    Checks if model has heteroscedastic errors (worse at high/low prices).

    Args:
        y_true: Actual target values
        y_pred: Predicted values
        price_data: Array/Series of price values (uses y_true if None)
        n_bins: Number of bins (5 = quintiles, 4 = quartiles, etc.)

    Returns:
        pd.DataFrame sorted by MAE descending

    Business value:
        "Model is consistent across price ranges (MAE within ±10%)"
        "Model underperforms at luxury properties (top quintile: MAE=$1.50)"
    """
    if price_data is None:
        price_data = y_true

    # Create bin labels
    if n_bins == 5:
        labels = ['Very Low', 'Low', 'Medium', 'High', 'Very High']
    elif n_bins == 4:
        labels = ['Low', 'Medium-Low', 'Medium-High', 'High']
    else:
        labels = [f'Bin {i+1}' for i in range(n_bins)]

    return segment_errors_by_feature(y_true, y_pred, price_data, 'Price Range',
                                     bins=n_bins, labels=labels)


def analyze_errors_by_building_age(y_true, y_pred, age_data, bins=[0, 10, 20, 50, 150]):
    """
    Error analysis segmented by building age ranges.

    Checks if model handles new construction vs. historic buildings differently.

    Args:
        y_true: Actual target values
        y_pred: Predicted values
        age_data: Array/Series of building ages (in years)
        bins: Age bin edges (default: [0, 10, 20, 50, 150] years)

    Returns:
        pd.DataFrame sorted by MAE descending

    Business value:
        "Model slightly underperforms on new construction (<5 years)"
        "Predictions are most reliable for 20-50 year old buildings"
    """
    labels = ['0-10 yrs', '10-20 yrs', '20-50 yrs', '50+ yrs']

    return segment_errors_by_feature(y_true, y_pred, age_data, 'Building Age',
                                     bins=bins, labels=labels)


def create_error_segmentation_report(y_true, y_pred, feature_dict, save_dir='figures/error_segmentation'):
    """
    Comprehensive error segmentation across all relevant features.

    Generates:
    - CSV reports for each segmentation
    - Bar charts comparing segments
    - Summary report with key insights

    Args:
        y_true: Actual target values
        y_pred: Predicted values
        feature_dict: Dict mapping feature_name -> feature_data
            Example: {
                'Market Name': market_array,
                'Property Type': type_array,
                'Price Range': price_array,
                'Building Age': age_array
            }
        save_dir: Directory to save reports and plots

    Returns:
        dict mapping feature_name -> segmentation DataFrame

    Business value:
        Executive summary of model strengths/weaknesses.
        Identifies specific improvement opportunities.

    Usage:
        >>> feature_dict = {
        >>>     'Market Name': df_val['Market Name'],
        >>>     'Property Type': df_val['Secondary Type'],
        >>>     'Price Range': y_val,
        >>>     'Building Age': df_val['building_age']
        >>> }
        >>> reports = create_error_segmentation_report(y_val, y_pred_val, feature_dict)
    """
    from .visualization import plot_error_by_segment

    # Create save directory
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    all_reports = {}

    for feature_name, feature_data in feature_dict.items():
        print(f"\nAnalyzing errors by {feature_name}...")

        # Determine if numerical feature needing binning
        if feature_name == 'Price Range':
            seg_df = analyze_errors_by_price_range(y_true, y_pred, feature_data, n_bins=5)
        elif feature_name == 'Building Age':
            seg_df = analyze_errors_by_building_age(y_true, y_pred, feature_data)
        else:
            # Categorical segmentation
            seg_df = segment_errors_by_feature(y_true, y_pred, feature_data, feature_name)

        # Save CSV
        csv_filename = f"errors_by_{feature_name.lower().replace(' ', '_')}.csv"
        seg_df.to_csv(save_path / csv_filename, index=False)
        print(f"  Saved: {csv_filename}")

        # Create visualizations
        # MAE bar chart
        fig_mae = plot_error_by_segment(
            seg_df, segment_col='segment', metric='MAE',
            title=f'Mean Absolute Error by {feature_name}',
            save_path=save_path / f"mae_by_{feature_name.lower().replace(' ', '_')}.png"
        )
        plt.close(fig_mae)

        # R² bar chart
        fig_r2 = plot_error_by_segment(
            seg_df, segment_col='segment', metric='R2',
            title=f'R² Score by {feature_name}',
            save_path=save_path / f"r2_by_{feature_name.lower().replace(' ', '_')}.png"
        )
        plt.close(fig_r2)

        all_reports[feature_name] = seg_df

    print(f"\nSaved all segmentation reports to {save_dir}/")

    return all_reports
