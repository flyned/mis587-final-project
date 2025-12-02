"""
Days on Market (DOM) Prediction Module

This module implements Market Timing Forecast (Proposal Objective #2):
- Predict how long a property will stay on market
- Inform bidding strategies and negotiation tactics
- Identify properties likely to sell quickly vs. slowly

Target: Days On Market (continuous, right-skewed)
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, median_absolute_error
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')


def prepare_dom_data(df, ref_columns=None):
    """
    Prepare data for Days on Market prediction.

    Args:
        df: DataFrame with features
        ref_columns: Reference column list for aligning test/val sets

    Returns:
        Tuple of (X, y, feature_names)
    """
    target_column = 'Days On Market'

    # Filter to rows with target
    df_filtered = df[df[target_column].notna()].copy()

    # Separate target
    y = df_filtered[target_column].values

    # Drop target and non-feature columns (similar to rent model)
    # Also drop rent-related columns to avoid leakage (DOM affects rent negotiations)
    drop_cols = [
        target_column,
        'Rent/SF/Yr', 'log_rent_sf_yr', 'Average Weighted Rent',
        'Avg Rent-Direct (Industrial)', 'Avg Rent-Direct (Office)',
        'Avg Rent-Direct (Retail)', 'Avg Rent-Sublet (Industrial)',
        'Avg Rent-Sublet (Office)', 'Avg Rent-Sublet (Retail)',
        'Property Address', 'Market Name', 'Submarket Name', 'City', 'State',
        'Zip', 'County Name', 'Building Park', 'Submarket Cluster',
        'Continent', 'Country', 'Subcontinent', 'Cross Street',
        'Last Sale Date', 'FEMA Map Date', 'Construction Begin',
        'Maturity Date', 'Origination Date'
    ]

    # Keep only columns that exist
    drop_cols = [col for col in drop_cols if col in df_filtered.columns]
    X_df = df_filtered.drop(columns=drop_cols)

    # Identify categorical columns
    categorical_cols = X_df.select_dtypes(include=['object', 'category']).columns.tolist()

    # One-hot encode categoricals
    if categorical_cols:
        X_df = pd.get_dummies(X_df, columns=categorical_cols, drop_first=True)

    # Fill any remaining nulls with 0
    X_df = X_df.fillna(0)

    # Align with reference columns if provided
    if ref_columns is not None:
        for col in ref_columns:
            if col not in X_df.columns:
                X_df[col] = 0
        extra_cols = [col for col in X_df.columns if col not in ref_columns]
        if extra_cols:
            X_df = X_df.drop(columns=extra_cols)
        X_df = X_df[ref_columns]

    feature_names = X_df.columns.tolist()
    X = X_df.values

    return X, y, feature_names


def evaluate_dom_model(y_true, y_pred, name="Model"):
    """
    Evaluate Days on Market prediction model.

    Uses different metrics since DOM is count data with heavy right skew:
    - Median Absolute Error (robust to outliers)
    - MAE, RMSE, R²
    - Accuracy within N days bands
    """
    mae = mean_absolute_error(y_true, y_pred)
    med_ae = median_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)

    # Prediction accuracy bands (for DOM, days are the unit)
    within_7_days = np.mean(np.abs(y_pred - y_true) <= 7)
    within_30_days = np.mean(np.abs(y_pred - y_true) <= 30)
    within_90_days = np.mean(np.abs(y_pred - y_true) <= 90)

    metrics = {
        'name': name,
        'MAE': mae,
        'Median_AE': med_ae,
        'RMSE': rmse,
        'R²': r2,
        'Within_7_days': within_7_days,
        'Within_30_days': within_30_days,
        'Within_90_days': within_90_days
    }

    return metrics


def print_dom_metrics(metrics):
    """Print DOM evaluation metrics."""
    print(f"\n{metrics['name']} Performance:")
    print(f"  MAE:              {metrics['MAE']:>10.1f} days")
    print(f"  Median AE:        {metrics['Median_AE']:>10.1f} days")
    print(f"  RMSE:             {metrics['RMSE']:>10.1f} days")
    print(f"  R²:               {metrics['R²']:>11.3f}")
    print(f"  Within ±7 days:   {metrics['Within_7_days']:>10.1%}")
    print(f"  Within ±30 days:  {metrics['Within_30_days']:>10.1%}")
    print(f"  Within ±90 days:  {metrics['Within_90_days']:>10.1%}")


def train_dom_random_forest(X_train, y_train, X_val, y_val, cv_folds=3):
    """
    Train Random Forest for Days on Market prediction.

    Note: DOM data is heavily right-skewed (median=0, mean=9.3, max=4889).
    We use log-transform of target for better predictions.
    """
    print("\n" + "="*70)
    print("TRAINING DAYS ON MARKET MODEL (Random Forest)")
    print("="*70)

    # Log-transform target (add 1 to handle zeros)
    y_train_log = np.log1p(y_train)
    y_val_log = np.log1p(y_val)

    print(f"\nTarget statistics (original):")
    print(f"  Train mean: {y_train.mean():.1f} days, median: {np.median(y_train):.1f}")
    print(f"  Train range: {y_train.min():.0f} - {y_train.max():.0f} days")

    # Train model
    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features='sqrt',
        random_state=42,
        n_jobs=1  # Memory-efficient
    )

    print("\nTraining Random Forest on log-transformed target...")
    rf.fit(X_train, y_train_log)

    # Cross-validation
    print(f"\nPerforming {cv_folds}-fold cross-validation...")
    cv_scores = cross_val_score(
        rf, X_train, y_train_log,
        cv=cv_folds,
        scoring='neg_mean_absolute_error',
        n_jobs=1
    )
    cv_mae_log = -cv_scores.mean()
    cv_std = cv_scores.std()
    print(f"CV MAE (log scale): {cv_mae_log:.3f} ± {cv_std:.3f}")

    # Predictions (transform back from log)
    y_pred_log = rf.predict(X_val)
    y_pred = np.expm1(y_pred_log)  # Inverse of log1p
    y_pred = np.maximum(y_pred, 0)  # Ensure non-negative

    # Evaluate
    metrics = evaluate_dom_model(y_val, y_pred, name="Random Forest (DOM)")
    metrics['cv_mae_log'] = cv_mae_log
    metrics['cv_std'] = cv_std

    print_dom_metrics(metrics)

    return {'model': rf, 'transform': 'log1p'}, metrics


def train_dom_gradient_boosting(X_train, y_train, X_val, y_val):
    """
    Train Gradient Boosting for Days on Market prediction.

    GB often handles skewed data well with proper loss functions.
    """
    print("\n" + "="*70)
    print("TRAINING DAYS ON MARKET MODEL (Gradient Boosting)")
    print("="*70)

    # Log-transform target
    y_train_log = np.log1p(y_train)

    # Train model with Huber loss (robust to outliers)
    gb = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        min_samples_split=10,
        min_samples_leaf=5,
        loss='huber',  # Robust to outliers
        random_state=42
    )

    print("\nTraining Gradient Boosting with Huber loss...")
    gb.fit(X_train, y_train_log)

    # Predictions
    y_pred_log = gb.predict(X_val)
    y_pred = np.expm1(y_pred_log)
    y_pred = np.maximum(y_pred, 0)

    # Evaluate
    metrics = evaluate_dom_model(y_val, y_pred, name="Gradient Boosting (DOM)")
    print_dom_metrics(metrics)

    return {'model': gb, 'transform': 'log1p'}, metrics


def classify_market_timing(predicted_dom):
    """
    Classify properties into market timing categories.

    Categories:
    - Quick Sale (0-7 days): High demand, competitive pricing
    - Normal (8-30 days): Average market absorption
    - Slow (31-90 days): May need price adjustment
    - Extended (90+ days): Potential issues, negotiation opportunity

    Args:
        predicted_dom: Array of predicted days on market

    Returns:
        Array of category labels
    """
    categories = np.empty(len(predicted_dom), dtype=object)

    categories[predicted_dom <= 7] = 'Quick Sale'
    categories[(predicted_dom > 7) & (predicted_dom <= 30)] = 'Normal'
    categories[(predicted_dom > 30) & (predicted_dom <= 90)] = 'Slow'
    categories[predicted_dom > 90] = 'Extended'

    return categories


def get_bidding_strategy(predicted_dom, market_timing_class):
    """
    Generate bidding strategy recommendations based on DOM prediction.

    Args:
        predicted_dom: Predicted days on market
        market_timing_class: Category from classify_market_timing()

    Returns:
        Dictionary with strategy recommendations
    """
    strategies = {
        'Quick Sale': {
            'urgency': 'High',
            'bid_premium': '+2-5%',
            'recommendation': 'Act quickly. Property likely to receive multiple offers. '
                            'Consider offering above asking price or waiving contingencies.',
            'negotiation_power': 'Low'
        },
        'Normal': {
            'urgency': 'Medium',
            'bid_premium': '0%',
            'recommendation': 'Standard market conditions. Offer at or slightly below asking. '
                            'Normal contingencies acceptable.',
            'negotiation_power': 'Medium'
        },
        'Slow': {
            'urgency': 'Low',
            'bid_premium': '-5-10%',
            'recommendation': 'Property has been on market longer than average. '
                            'Seller may be motivated. Consider below-asking offer.',
            'negotiation_power': 'High'
        },
        'Extended': {
            'urgency': 'Very Low',
            'bid_premium': '-10-20%',
            'recommendation': 'Property has significant time on market. '
                            'Strong negotiation position. Investigate reasons for slow sale.',
            'negotiation_power': 'Very High'
        }
    }

    return strategies.get(market_timing_class, strategies['Normal'])


def predict_dom(model_dict, X, feature_names=None):
    """
    Make Days on Market predictions.

    Args:
        model_dict: Dictionary with 'model' and 'transform' keys
        X: Features array
        feature_names: Optional feature names

    Returns:
        DataFrame with predictions and market timing analysis
    """
    model = model_dict['model']
    transform = model_dict.get('transform', None)

    # Predict
    y_pred_raw = model.predict(X)

    # Inverse transform if needed
    if transform == 'log1p':
        y_pred = np.expm1(y_pred_raw)
        y_pred = np.maximum(y_pred, 0)
    else:
        y_pred = y_pred_raw

    # Classify and get strategies
    timing_class = classify_market_timing(y_pred)

    # Build results DataFrame
    results = pd.DataFrame({
        'predicted_dom': y_pred.round(1),
        'market_timing': timing_class
    })

    # Add strategy recommendations
    strategies = [get_bidding_strategy(dom, cls) for dom, cls in zip(y_pred, timing_class)]
    results['urgency'] = [s['urgency'] for s in strategies]
    results['bid_premium'] = [s['bid_premium'] for s in strategies]
    results['negotiation_power'] = [s['negotiation_power'] for s in strategies]

    return results


def get_dom_feature_importance(model, feature_names, top_n=20):
    """
    Extract feature importance for DOM model.

    Args:
        model: Trained model with feature_importances_
        feature_names: List of feature names
        top_n: Number of top features

    Returns:
        DataFrame with feature importance
    """
    if hasattr(model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

        return importance_df.head(top_n)
    return None
