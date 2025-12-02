"""
Visualization utilities for Phase 3: Model Interpretation

This module provides publication-quality plotting functions for:
- Feature importance visualizations
- Residual diagnostic plots
- Error segmentation charts
- SHAP value visualizations (integrated with shap library)

All plots use business-friendly styling with clear labels and annotations.
Designed for stakeholder presentations and technical reports.

Author: MIS587 Final Project
Date: November 2024
"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server environments

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats


def setup_plot_style():
    """
    Configure matplotlib and seaborn for business-ready visualizations.

    Sets publication-quality defaults:
    - Font sizes appropriate for presentations
    - Colorblind-friendly palette
    - Clean, professional styling
    - High-DPI rendering

    Call this once at the start of your analysis script.
    """
    # Set seaborn style
    sns.set_style("whitegrid")
    sns.set_palette("deep")  # Colorblind-friendly

    # Configure matplotlib
    plt.rcParams['figure.figsize'] = (10, 6)  # Standard size
    plt.rcParams['figure.dpi'] = 100  # Screen resolution
    plt.rcParams['savefig.dpi'] = 300  # Publication quality
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['xtick.labelsize'] = 11
    plt.rcParams['ytick.labelsize'] = 11
    plt.rcParams['legend.fontsize'] = 11
    plt.rcParams['figure.titlesize'] = 16


def save_figure(fig, filename, subdirectory='', dpi=300):
    """
    Save figure to figures/ directory with auto-created subdirectories.

    Args:
        fig: matplotlib.Figure object
        filename: File name (e.g., 'importance_plot.png')
        subdirectory: Optional subdirectory (e.g., 'feature_importance')
        dpi: Resolution for saved image (default 300 for publication)

    Returns:
        Path: Full path to saved file

    Example:
        >>> fig, ax = plt.subplots()
        >>> ax.plot([1, 2, 3])
        >>> path = save_figure(fig, 'my_plot.png', 'analysis')
        >>> print(path)
        figures/analysis/my_plot.png
    """
    # Create base directory
    base_dir = Path('figures')
    base_dir.mkdir(exist_ok=True)

    # Create subdirectory if specified
    if subdirectory:
        save_dir = base_dir / subdirectory
        save_dir.mkdir(exist_ok=True)
    else:
        save_dir = base_dir

    # Full path
    save_path = save_dir / filename

    # Save figure
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')

    return save_path


def plot_feature_importance_bar(importance_df, top_n=20, title='Feature Importance',
                                 xlabel='Importance', save_path=None):
    """
    Horizontal bar chart of feature importance.

    Args:
        importance_df: DataFrame with columns ['feature', 'importance']
        top_n: Number of top features to display (default 20)
        title: Plot title
        xlabel: X-axis label
        save_path: Optional path to save figure

    Returns:
        matplotlib.Figure

    Business value:
        Shows which features drive model predictions. Top features are
        candidates for focused data collection and stakeholder presentations.
    """
    # Take top N and sort ascending for bottom-to-top display
    plot_data = importance_df.nlargest(top_n, 'importance').sort_values('importance')

    # Create figure
    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.35)))

    # Horizontal bar chart
    bars = ax.barh(range(len(plot_data)), plot_data['importance'], color='steelblue')

    # Add value labels at end of bars
    for i, (idx, row) in enumerate(plot_data.iterrows()):
        ax.text(row['importance'], i, f" {row['importance']:.4f}",
                va='center', fontsize=10)

    # Formatting
    ax.set_yticks(range(len(plot_data)))
    ax.set_yticklabels(plot_data['feature'])
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()

    # Save if path provided
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')

    return fig


def plot_feature_importance_comparison(mdi_df, perm_df, shap_df=None, top_n=15,
                                        save_path=None):
    """
    Side-by-side comparison of feature importance methods.

    Args:
        mdi_df: DataFrame from get_mdi_importance()
        perm_df: DataFrame from get_permutation_importance()
        shap_df: Optional DataFrame from get_shap_feature_importance()
        top_n: Number of features to show
        save_path: Optional path to save figure

    Returns:
        matplotlib.Figure (3-panel horizontal layout if shap_df provided, else 2-panel)

    Business value:
        Identifies consensus features important across multiple methods.
        Features important by all methods are most reliable for decision-making.
    """
    n_panels = 3 if shap_df is not None else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 8))

    if n_panels == 2:
        axes = [axes[0], axes[1]]

    # Panel 1: MDI Importance
    mdi_top = mdi_df.nlargest(top_n, 'importance').sort_values('importance')
    axes[0].barh(range(len(mdi_top)), mdi_top['importance'], color='steelblue')
    axes[0].set_yticks(range(len(mdi_top)))
    axes[0].set_yticklabels(mdi_top['feature'], fontsize=10)
    axes[0].set_xlabel('MDI Importance', fontsize=11)
    axes[0].set_title('Mean Decrease Impurity', fontsize=12, fontweight='bold')
    axes[0].grid(axis='x', alpha=0.3)

    # Panel 2: Permutation Importance
    perm_top = perm_df.nlargest(top_n, 'importance').sort_values('importance')
    axes[1].barh(range(len(perm_top)), perm_top['importance'], color='darkorange')
    axes[1].set_yticks(range(len(perm_top)))
    axes[1].set_yticklabels(perm_top['feature'], fontsize=10)
    axes[1].set_xlabel('Permutation Importance', fontsize=11)
    axes[1].set_title('Permutation Feature Importance', fontsize=12, fontweight='bold')
    axes[1].grid(axis='x', alpha=0.3)

    # Panel 3: SHAP Importance (if provided)
    if shap_df is not None:
        shap_top = shap_df.nlargest(top_n, 'importance').sort_values('importance')
        axes[2].barh(range(len(shap_top)), shap_top['importance'], color='forestgreen')
        axes[2].set_yticks(range(len(shap_top)))
        axes[2].set_yticklabels(shap_top['feature'], fontsize=10)
        axes[2].set_xlabel('Mean |SHAP Value|', fontsize=11)
        axes[2].set_title('SHAP Feature Importance', fontsize=12, fontweight='bold')
        axes[2].grid(axis='x', alpha=0.3)

    fig.suptitle(f'Feature Importance Comparison (Top {top_n} Features)',
                 fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()

    # Save if path provided
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')

    return fig


def plot_residuals_vs_predicted(y_true, y_pred, title='Residual Analysis', save_path=None):
    """
    Scatter plot of residuals vs predicted values with trend line.

    Args:
        y_true: Actual target values
        y_pred: Predicted values
        title: Plot title
        save_path: Optional path to save figure

    Returns:
        matplotlib.Figure

    Diagnostic checks:
        - Funnel shape → heteroscedasticity (error variance increases with prediction)
        - Curved pattern → non-linearity in data
        - Random scatter around zero → good model fit

    Business value:
        Identifies if model has systematic biases at certain price ranges.
        Flag for manual review if residuals show patterns.
    """
    residuals = y_pred - y_true

    fig, ax = plt.subplots(figsize=(10, 6))

    # Scatter plot
    ax.scatter(y_pred, residuals, alpha=0.5, s=20, color='steelblue')

    # Zero line (perfect predictions)
    ax.axhline(y=0, color='red', linestyle='--', linewidth=2, label='Zero Residual')

    # Add LOESS smoothing trend line
    from scipy.signal import savgol_filter
    sorted_idx = np.argsort(y_pred)
    y_pred_sorted = y_pred[sorted_idx]
    residuals_sorted = residuals[sorted_idx]

    # Smooth with Savitzky-Golay filter (if enough points)
    if len(y_pred) > 51:
        window = min(51, len(y_pred) // 10 * 2 + 1)  # Odd window
        smoothed = savgol_filter(residuals_sorted, window_length=window, polyorder=3)
        ax.plot(y_pred_sorted, smoothed, color='orange', linewidth=2, label='Trend')

    # Formatting
    ax.set_xlabel('Predicted Rent ($/SF/Yr)', fontsize=12)
    ax.set_ylabel('Residual (Predicted - Actual)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)

    # Add reference lines at ±2 std
    std_residuals = np.std(residuals)
    ax.axhline(y=2*std_residuals, color='gray', linestyle=':', alpha=0.7, label='±2σ')
    ax.axhline(y=-2*std_residuals, color='gray', linestyle=':', alpha=0.7)

    plt.tight_layout()

    # Save if path provided
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')

    return fig


def plot_residuals_distribution(residuals, title='Residual Distribution', save_path=None):
    """
    Histogram + KDE of residuals with normality overlay.

    Args:
        residuals: Array of residuals (predicted - actual)
        title: Plot title
        save_path: Optional path to save figure

    Returns:
        matplotlib.Figure

    Diagnostic checks:
        - Bell curve → normally distributed residuals (good)
        - Skewed → systematic bias in predictions
        - Heavy tails → outliers present

    Business value:
        Normal residuals mean prediction intervals are reliable.
        Skewed distribution indicates model may need recalibration.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Histogram
    ax.hist(residuals, bins=50, density=True, alpha=0.7, color='steelblue',
            edgecolor='black', label='Residuals')

    # KDE (kernel density estimate)
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(residuals)
    x_range = np.linspace(residuals.min(), residuals.max(), 200)
    ax.plot(x_range, kde(x_range), color='darkblue', linewidth=2, label='KDE')

    # Normal distribution overlay
    mu, sigma = residuals.mean(), residuals.std()
    normal_curve = stats.norm.pdf(x_range, mu, sigma)
    ax.plot(x_range, normal_curve, color='red', linestyle='--', linewidth=2,
            label=f'Normal(μ={mu:.3f}, σ={sigma:.3f})')

    # Normality test
    stat, p_value = stats.shapiro(residuals[:5000])  # Shapiro test (max 5000 samples)
    is_normal = p_value > 0.05
    normality_text = f'Shapiro-Wilk: p={p_value:.4f} {"(Normal)" if is_normal else "(Not Normal)"}'
    ax.text(0.02, 0.98, normality_text, transform=ax.transAxes,
            fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Formatting
    ax.set_xlabel('Residual ($/SF/Yr)', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)

    plt.tight_layout()

    # Save if path provided
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')

    return fig


def plot_qq_plot(residuals, title='Q-Q Plot', save_path=None):
    """
    Quantile-Quantile plot to check normality assumption.

    Args:
        residuals: Array of residuals (predicted - actual)
        title: Plot title
        save_path: Optional path to save figure

    Returns:
        matplotlib.Figure

    Interpretation:
        - Points along diagonal line → normally distributed
        - S-curve → heavy tails (outliers)
        - Deviations at ends → skewness

    Business value:
        Validates regression assumption of normal residuals.
        Significant deviations may require transformation or robust methods.
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    # Q-Q plot
    stats.probplot(residuals, dist="norm", plot=ax)

    # Formatting
    ax.set_xlabel('Theoretical Quantiles', fontsize=12)
    ax.set_ylabel('Sample Quantiles', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)

    # Add interpretation text
    interpretation = "Points along red line → Normal distribution\nDeviations indicate non-normality"
    ax.text(0.05, 0.95, interpretation, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()

    # Save if path provided
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')

    return fig


def plot_prediction_scatter(y_true, y_pred, title='Predicted vs Actual', save_path=None):
    """
    Scatter plot of actual vs predicted values with diagonal line.

    Args:
        y_true: Actual target values
        y_pred: Predicted values
        title: Plot title
        save_path: Optional path to save figure

    Returns:
        matplotlib.Figure

    Interpretation:
        - Points on diagonal → perfect predictions
        - Points above diagonal → over-predictions
        - Points below diagonal → under-predictions

    Business value:
        Quick visual check of model accuracy.
        R² value shows percentage of variance explained.
    """
    from sklearn.metrics import r2_score, mean_absolute_error

    fig, ax = plt.subplots(figsize=(8, 8))

    # Scatter plot
    ax.scatter(y_true, y_pred, alpha=0.5, s=20, color='steelblue')

    # Perfect prediction line (diagonal)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2,
            label='Perfect Prediction')

    # Calculate metrics
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)

    # Add metrics text box
    metrics_text = f'R² = {r2:.3f}\nMAE = ${mae:.2f}/SF/Yr'
    ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes,
            fontsize=12, verticalalignment='top', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Formatting
    ax.set_xlabel('Actual Rent ($/SF/Yr)', fontsize=12)
    ax.set_ylabel('Predicted Rent ($/SF/Yr)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)

    # Equal aspect ratio
    ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()

    # Save if path provided
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')

    return fig


def plot_error_by_segment(segment_df, segment_col='segment', metric='MAE',
                          title=None, ylabel=None, save_path=None):
    """
    Bar chart of error metrics by category (market, property type, etc.).

    Args:
        segment_df: DataFrame with columns [segment_col, metric]
        segment_col: Name of segment column (default 'segment')
        metric: Metric to plot ('MAE', 'RMSE', 'R2', 'MAPE', 'Within_10pct')
        title: Plot title (auto-generated if None)
        ylabel: Y-axis label (auto-generated if None)
        save_path: Optional path to save figure

    Returns:
        matplotlib.Figure

    Business value:
        Identifies segments where model underperforms.
        Guides data collection and model improvement efforts.
    """
    # Sort by metric (descending for most metrics, ascending for R2)
    ascending = (metric == 'R2')
    plot_data = segment_df.sort_values(metric, ascending=ascending)

    # Auto-generate labels if not provided
    if title is None:
        title = f'Model Performance by {segment_col.replace("_", " ").title()}'
    if ylabel is None:
        ylabel_map = {
            'MAE': 'Mean Absolute Error ($/SF/Yr)',
            'RMSE': 'Root Mean Squared Error ($/SF/Yr)',
            'R2': 'R² Score',
            'MAPE': 'Mean Absolute Percentage Error (%)',
            'Within_10pct': '% Predictions Within ±10%'
        }
        ylabel = ylabel_map.get(metric, metric)

    # Create figure
    n_segments = len(plot_data)
    fig, ax = plt.subplots(figsize=(max(10, n_segments * 0.6), 6))

    # Color bars by performance (green=good, red=bad)
    if metric in ['MAE', 'RMSE', 'MAPE']:
        # Lower is better
        colors = plt.cm.RdYlGn_r(np.linspace(0.3, 0.9, n_segments))
    else:
        # Higher is better (R2, Within_10pct)
        colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, n_segments))

    # Bar chart
    bars = ax.bar(range(n_segments), plot_data[metric], color=colors, edgecolor='black')

    # Add value labels on bars
    for i, (idx, row) in enumerate(plot_data.iterrows()):
        value = row[metric]
        if metric in ['R2', 'Within_10pct']:
            label = f'{value:.3f}'
        else:
            label = f'${value:.2f}' if metric in ['MAE', 'RMSE'] else f'{value:.1f}%'
        ax.text(i, value, label, ha='center', va='bottom', fontsize=10)

    # Formatting
    ax.set_xticks(range(n_segments))
    ax.set_xticklabels(plot_data[segment_col], rotation=45, ha='right')
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    # Save if path provided
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')

    return fig


def plot_error_heatmap(error_matrix, row_labels, col_labels, title='Error Heatmap',
                       metric='MAE', save_path=None):
    """
    Heatmap for 2D error segmentation (e.g., market × property type).

    Args:
        error_matrix: 2D numpy array or DataFrame with error values
        row_labels: List of row labels (e.g., markets)
        col_labels: List of column labels (e.g., property types)
        title: Plot title
        metric: Metric name for colorbar label
        save_path: Optional path to save figure

    Returns:
        matplotlib.Figure

    Business value:
        Identifies specific combinations (e.g., "Chicago + Office") where
        model struggles. Enables targeted improvements.
    """
    fig, ax = plt.subplots(figsize=(max(8, len(col_labels) * 0.8),
                                     max(6, len(row_labels) * 0.5)))

    # Colormap (red=high error, green=low error for MAE/RMSE)
    if metric in ['MAE', 'RMSE', 'MAPE']:
        cmap = 'RdYlGn_r'  # Reversed (red=bad)
    else:
        cmap = 'RdYlGn'  # Normal (green=good)

    # Heatmap
    im = ax.imshow(error_matrix, cmap=cmap, aspect='auto')

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar_label_map = {
        'MAE': 'Mean Absolute Error ($/SF/Yr)',
        'RMSE': 'Root Mean Squared Error ($/SF/Yr)',
        'R2': 'R² Score',
        'MAPE': 'Mean Absolute Percentage Error (%)'
    }
    cbar.set_label(cbar_label_map.get(metric, metric), rotation=270, labelpad=20, fontsize=11)

    # Axis labels
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha='right')
    ax.set_yticklabels(row_labels)

    # Add text annotations
    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            value = error_matrix[i, j]
            if not np.isnan(value):
                if metric in ['R2', 'Within_10pct']:
                    text = f'{value:.2f}'
                else:
                    text = f'${value:.2f}' if metric in ['MAE', 'RMSE'] else f'{value:.1f}'
                ax.text(j, i, text, ha='center', va='center', fontsize=9,
                       color='white' if value > np.nanmedian(error_matrix) else 'black')

    # Title
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)

    plt.tight_layout()

    # Save if path provided
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')

    return fig


# Initialize plot style when module is imported
setup_plot_style()
