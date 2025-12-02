"""
Train Gap-Filling Models

This script trains the models that fill the gaps from the original proposal:
1. Days on Market (DOM) prediction - Market Timing Forecast (Objective #2)
2. Opportunity Identification framework - (Objective #4)

Usage:
    python train_gap_models.py
"""

import os
import sys
import json
import pickle
import gc
import numpy as np
import pandas as pd
from datetime import datetime

# Limit parallel jobs to prevent memory exhaustion
# Set before importing sklearn-based modules
os.environ['LOKY_MAX_CPU_COUNT'] = '2'

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from utils import load_data_splits
from feature_engineering import (
    impute_missing_values,
    add_geospatial_features,
    add_temporal_features,
    add_numerical_transformations,
    add_categorical_encoding,
    apply_outlier_treatment
)
from dom_model import (
    prepare_dom_data,
    train_dom_random_forest,
    train_dom_gradient_boosting,
    evaluate_dom_model,
    print_dom_metrics,
    predict_dom,
    get_dom_feature_importance
)
from opportunity import (
    generate_opportunity_report,
    get_investment_recommendation
)


def apply_feature_engineering(train_df, val_df, test_df):
    """Apply feature engineering pipeline to all splits."""
    print("\n" + "="*70)
    print("APPLYING FEATURE ENGINEERING")
    print("="*70)

    # Imputation
    print("\n1. Imputing missing values...")
    impute_missing_values(train_df)
    impute_missing_values(val_df)
    impute_missing_values(test_df)

    # Geospatial features
    print("2. Creating geospatial features...")
    add_geospatial_features(train_df)
    add_geospatial_features(val_df)
    add_geospatial_features(test_df)

    # Temporal features
    print("3. Creating temporal features...")
    add_temporal_features(train_df)
    add_temporal_features(val_df)
    add_temporal_features(test_df)

    # Numerical features
    print("4. Creating numerical features...")
    add_numerical_transformations(train_df)
    add_numerical_transformations(val_df)
    add_numerical_transformations(test_df)

    # Categorical encoding (train first for target encoding stats)
    print("5. Encoding categorical features...")
    add_categorical_encoding(train_df)
    add_categorical_encoding(val_df)
    add_categorical_encoding(test_df)

    # Outlier treatment
    print("6. Treating outliers...")
    apply_outlier_treatment(train_df)
    apply_outlier_treatment(val_df)
    apply_outlier_treatment(test_df)

    print("\nFeature engineering complete!")
    print(f"  Train shape: {train_df.shape}")
    print(f"  Val shape: {val_df.shape}")
    print(f"  Test shape: {test_df.shape}")

    return train_df, val_df, test_df


def train_days_on_market_model(train_df, val_df, test_df):
    """Train and evaluate Days on Market prediction model."""
    print("\n" + "="*70)
    print("DAYS ON MARKET MODEL (OBJECTIVE #2: MARKET TIMING FORECAST)")
    print("="*70)

    # Prepare data
    print("\nPreparing DOM training data...")
    X_train, y_train, feature_names = prepare_dom_data(train_df)
    X_val, y_val, _ = prepare_dom_data(val_df, ref_columns=feature_names)
    X_test, y_test, _ = prepare_dom_data(test_df, ref_columns=feature_names)

    print(f"Training samples: {len(y_train)}")
    print(f"Validation samples: {len(y_val)}")
    print(f"Test samples: {len(y_test)}")
    print(f"Features: {len(feature_names)}")

    # Train Random Forest
    rf_model, rf_metrics = train_dom_random_forest(X_train, y_train, X_val, y_val)

    # Train Gradient Boosting
    gb_model, gb_metrics = train_dom_gradient_boosting(X_train, y_train, X_val, y_val)

    # Compare models
    print("\n" + "="*70)
    print("DOM MODEL COMPARISON")
    print("="*70)

    print(f"\n{'Model':<25} {'MAE':>10} {'Med AE':>10} {'R²':>8} {'±7 days':>10} {'±30 days':>10}")
    print("-"*75)

    for name, metrics in [('Random Forest', rf_metrics), ('Gradient Boosting', gb_metrics)]:
        print(f"{name:<25} {metrics['MAE']:>9.1f}d {metrics['Median_AE']:>9.1f}d "
              f"{metrics['R²']:>7.3f} {metrics['Within_7_days']:>9.1%} {metrics['Within_30_days']:>9.1%}")

    # Select best model (by Median AE, since DOM is skewed)
    if rf_metrics['Median_AE'] <= gb_metrics['Median_AE']:
        best_model = rf_model
        best_metrics = rf_metrics
        best_name = 'Random Forest'
    else:
        best_model = gb_model
        best_metrics = gb_metrics
        best_name = 'Gradient Boosting'

    print(f"\nBest DOM model: {best_name}")

    # Evaluate on test set
    print("\n" + "="*70)
    print("DOM TEST SET EVALUATION")
    print("="*70)

    # Predict on test
    y_pred_log = best_model['model'].predict(X_test)
    y_pred_test = np.expm1(y_pred_log)
    y_pred_test = np.maximum(y_pred_test, 0)

    test_metrics = evaluate_dom_model(y_test, y_pred_test, name=f"{best_name} (Test)")
    print_dom_metrics(test_metrics)

    # Feature importance
    print("\n" + "="*70)
    print("DOM TOP FEATURES")
    print("="*70)

    importance_df = get_dom_feature_importance(best_model['model'], feature_names, top_n=15)
    if importance_df is not None:
        print("\n" + importance_df.to_string(index=False))

    return best_model, best_metrics, test_metrics, feature_names


def run_opportunity_analysis(train_df, val_df, test_df, rent_model_path=None):
    """Run opportunity identification analysis using rent model."""
    print("\n" + "="*70)
    print("OPPORTUNITY IDENTIFICATION (OBJECTIVE #4)")
    print("="*70)

    from modeling import prepare_data_for_modeling
    from sklearn.ensemble import RandomForestRegressor

    # Always train a fresh rent model on the feature-engineered data
    # to ensure feature alignment
    print("\nTraining rent model for opportunity analysis...")
    X_train, y_train, feature_names, _ = prepare_data_for_modeling(train_df)
    X_val, y_val, _, _ = prepare_data_for_modeling(val_df, ref_columns=feature_names)

    # Train a lightweight Random Forest (memory-efficient settings)
    print("Training lightweight Random Forest (n_jobs=1, 100 trees)...")
    rent_model = RandomForestRegressor(
        n_estimators=100,  # Reduced from 200
        max_depth=15,      # Limited depth
        min_samples_split=10,
        min_samples_leaf=5,
        max_features='sqrt',
        random_state=42,
        n_jobs=1  # Single core to prevent memory exhaustion
    )
    rent_model.fit(X_train, y_train)

    # Clear training data from memory
    del X_train
    gc.collect()

    # Make predictions
    print("\nGenerating rent predictions for opportunity analysis...")
    y_pred = rent_model.predict(X_val)

    # Generate opportunity report
    report = generate_opportunity_report(
        val_df[val_df['Rent/SF/Yr'].notna()].reset_index(drop=True),
        y_val,
        y_pred,
        X=X_val,
        top_n=20
    )

    # Clean up
    del rent_model, X_val
    gc.collect()

    return report


def save_models(dom_model, dom_metrics, dom_test_metrics, dom_features, output_dir='models'):
    """Save trained models and metadata."""
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Save DOM model
    dom_model_path = os.path.join(output_dir, f'dom_model_{timestamp}.pkl')
    with open(dom_model_path, 'wb') as f:
        pickle.dump(dom_model, f)
    print(f"\nDOM model saved to: {dom_model_path}")

    # Save DOM metadata
    dom_metadata = {
        'model_type': 'RandomForest' if 'RandomForest' in str(type(dom_model['model'])) else 'GradientBoosting',
        'target': 'Days On Market',
        'transform': dom_model.get('transform', 'log1p'),
        'feature_count': len(dom_features),
        'validation_metrics': {
            'mae': float(dom_metrics['MAE']),
            'median_ae': float(dom_metrics['Median_AE']),
            'rmse': float(dom_metrics['RMSE']),
            'r2': float(dom_metrics['R²']),
            'within_7_days': float(dom_metrics['Within_7_days']),
            'within_30_days': float(dom_metrics['Within_30_days']),
            'within_90_days': float(dom_metrics['Within_90_days'])
        },
        'test_metrics': {
            'mae': float(dom_test_metrics['MAE']),
            'median_ae': float(dom_test_metrics['Median_AE']),
            'rmse': float(dom_test_metrics['RMSE']),
            'r2': float(dom_test_metrics['R²']),
            'within_7_days': float(dom_test_metrics['Within_7_days']),
            'within_30_days': float(dom_test_metrics['Within_30_days']),
            'within_90_days': float(dom_test_metrics['Within_90_days'])
        },
        'timestamp': timestamp
    }

    dom_meta_path = os.path.join(output_dir, f'dom_metadata_{timestamp}.json')
    with open(dom_meta_path, 'w') as f:
        json.dump(dom_metadata, f, indent=2)
    print(f"DOM metadata saved to: {dom_meta_path}")

    return dom_model_path, dom_meta_path


def main():
    """Main training pipeline for gap-filling models."""
    print("="*70)
    print("TRAINING GAP-FILLING MODELS")
    print("Filling gaps from original project proposal")
    print("="*70)
    print("\nNote: Using memory-efficient settings (limited parallelism)")

    # Load data
    print("\nLoading data splits...")
    train_df, val_df, test_df = load_data_splits()
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # Apply feature engineering
    train_df, val_df, test_df = apply_feature_engineering(train_df, val_df, test_df)

    # Force garbage collection after feature engineering
    gc.collect()

    # 1. Train Days on Market model
    dom_model, dom_metrics, dom_test_metrics, dom_features = train_days_on_market_model(
        train_df, val_df, test_df
    )

    # Clean up before next phase
    gc.collect()

    # 2. Run opportunity analysis
    opportunity_report = run_opportunity_analysis(train_df, val_df, test_df)

    # Save models
    dom_model_path, dom_meta_path = save_models(
        dom_model, dom_metrics, dom_test_metrics, dom_features
    )

    # Save opportunity report
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_path = os.path.join('models', f'opportunity_report_{timestamp}.csv')
    opportunity_report['results'].to_csv(report_path, index=False)
    print(f"Opportunity report saved to: {report_path}")

    # Final summary
    print("\n" + "="*70)
    print("GAP-FILLING COMPLETE")
    print("="*70)

    print("\n1. MARKET TIMING FORECAST (Objective #2)")
    print(f"   Model: Days on Market Prediction")
    print(f"   Validation MAE: {dom_metrics['MAE']:.1f} days")
    print(f"   Test MAE: {dom_test_metrics['MAE']:.1f} days")
    print(f"   Within ±30 days: {dom_test_metrics['Within_30_days']:.1%}")

    print("\n2. OPPORTUNITY IDENTIFICATION (Objective #4)")
    summary = opportunity_report['summary']
    print(f"   Total properties analyzed: {summary['total_properties']}")
    print(f"   Strong Buy opportunities: {summary['strong_buy_count']}")
    print(f"   Buy opportunities: {summary['buy_count']}")
    print(f"   Distressed properties: {summary['distressed_count']}")
    print(f"   Value-add opportunities: {summary['value_add_count']}")

    print("\n" + "="*70)
    print("ALL PROPOSAL OBJECTIVES NOW ADDRESSED")
    print("="*70)


if __name__ == '__main__':
    main()
