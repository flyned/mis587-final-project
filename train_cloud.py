"""
Cloud Training Script for Google Colab
Optimized end-to-end pipeline with model persistence

This script provides a complete training pipeline for Random Forest models
that can be executed on Google Colab or other cloud environments.

Usage:
    python train_cloud.py

    Or import and use the function:
    from train_cloud import train_and_save_model
    results = train_and_save_model(tune_hyperparameters=True)
"""

import pandas as pd
import pickle
import json
from datetime import datetime
from pathlib import Path

from src.feature_engineering import engineer_features
from src.modeling import (
    prepare_data_for_modeling,
    train_random_forest,
    evaluate_model,
    print_metrics,
    get_feature_importance
)


def train_and_save_model(
    train_path='../data/train.csv',
    val_path='../data/val.csv',
    test_path='../data/test.csv',
    output_dir='./models',
    tune_hyperparameters=True,
    cv_folds=5,
    target_column='Rent/SF/Yr'
):
    """
    End-to-end training pipeline for cloud execution.

    This function orchestrates the complete ML pipeline:
    1. Load train/val/test data
    2. Apply feature engineering
    3. Prepare data for modeling
    4. Train Random Forest with hyperparameter tuning
    5. Evaluate on test set
    6. Save model and metadata

    Args:
        train_path: Path to training CSV
        val_path: Path to validation CSV
        test_path: Path to test CSV
        output_dir: Directory to save outputs
        tune_hyperparameters: Enable GridSearchCV (216 combinations, ~2-3 hours)
        cv_folds: Number of cross-validation folds
        target_column: Name of target variable

    Returns:
        Dictionary with:
            - model: Trained model dictionary {'model': rf, 'scaler': None}
            - val_metrics: Validation set metrics
            - test_metrics: Test set metrics
            - paths: File paths for saved artifacts
            - feature_names: List of feature names
            - importance_df: Top 20 features by importance
    """
    print("="*70)
    print("CLOUD TRAINING PIPELINE - RANDOM FOREST")
    print("="*70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # ========================================================================
    # STEP 1: Load Data
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 1: LOADING DATA")
    print("="*70)

    train = pd.read_csv(train_path)
    val = pd.read_csv(val_path)
    test = pd.read_csv(test_path)

    print(f"Train: {train.shape}")
    print(f"Val:   {val.shape}")
    print(f"Test:  {test.shape}")

    # ========================================================================
    # STEP 2: Feature Engineering
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 2: FEATURE ENGINEERING")
    print("="*70)

    train_eng = engineer_features(train.copy(), target_col=target_column)
    val_eng = engineer_features(val.copy(), target_col=target_column)
    test_eng = engineer_features(test.copy(), target_col=target_column)

    print(f"\nAfter feature engineering:")
    print(f"Train: {train_eng.shape}")
    print(f"Val:   {val_eng.shape}")
    print(f"Test:  {test_eng.shape}")

    # ========================================================================
    # STEP 3: Prepare Data for Modeling
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 3: PREPARING DATA FOR MODELING")
    print("="*70)

    X_train, y_train, feature_names, _ = prepare_data_for_modeling(
        train_eng, target_column=target_column
    )
    X_val, y_val, _, _ = prepare_data_for_modeling(
        val_eng, target_column=target_column, ref_columns=feature_names
    )
    X_test, y_test, _, _ = prepare_data_for_modeling(
        test_eng, target_column=target_column, ref_columns=feature_names
    )

    print(f"\nPrepared data shapes:")
    print(f"X_train: {X_train.shape}")
    print(f"X_val:   {X_val.shape}")
    print(f"X_test:  {X_test.shape}")
    print(f"Features: {len(feature_names)}")

    # ========================================================================
    # STEP 4: Train Random Forest
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 4: TRAINING RANDOM FOREST")
    print("="*70)

    if tune_hyperparameters:
        print("\nWARNING: Hyperparameter tuning enabled (216 combinations)")
        print("Expected time: 2-3 hours on CPU, 30-45 min on GPU")

    rf_model, rf_metrics = train_random_forest(
        X_train, y_train, X_val, y_val,
        tune_hyperparameters=tune_hyperparameters,
        cv_folds=cv_folds
    )

    # ========================================================================
    # STEP 5: Evaluate on Test Set
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 5: EVALUATING ON TEST SET")
    print("="*70)

    y_pred_test = rf_model['model'].predict(X_test)
    test_metrics = evaluate_model(y_test, y_pred_test, name="Test Set")
    print_metrics(test_metrics)

    # ========================================================================
    # STEP 6: Feature Importance
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 6: FEATURE IMPORTANCE")
    print("="*70)

    importance_df = get_feature_importance(rf_model['model'], feature_names, top_n=20)

    if importance_df is not None:
        print("\nTop 20 Most Important Features:")
        print(importance_df.to_string(index=False))

    # ========================================================================
    # STEP 7: Save Model and Metadata
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 7: SAVING MODEL AND METADATA")
    print("="*70)

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save model
    model_path = Path(output_dir) / f'rf_model_{timestamp}.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(rf_model, f)

    print(f"\n✓ Model saved: {model_path}")
    print(f"  Size: {model_path.stat().st_size / (1024*1024):.2f} MB")

    # Save metadata
    metadata = {
        'model_type': 'RandomForestRegressor',
        'timestamp': timestamp,
        'training_date': datetime.now().isoformat(),
        'hyperparameters': rf_model['model'].get_params(),
        'feature_count': len(feature_names),
        'feature_names': feature_names,
        'training_samples': X_train.shape[0],
        'validation_samples': X_val.shape[0],
        'test_samples': X_test.shape[0],
        'target_column': target_column,
        'tuning_enabled': tune_hyperparameters,
        'cv_folds': cv_folds,
        'validation_metrics': {
            'mae': float(rf_metrics['MAE']),
            'rmse': float(rf_metrics['RMSE']),
            'r2': float(rf_metrics['R²']),
            'mape': float(rf_metrics['MAPE']),
            'within_10pct': float(rf_metrics['Within_10pct']),
            'cv_mae': float(rf_metrics.get('cv_mae', 0)),
            'cv_std': float(rf_metrics.get('cv_std', 0))
        },
        'test_metrics': {
            'mae': float(test_metrics['MAE']),
            'rmse': float(test_metrics['RMSE']),
            'r2': float(test_metrics['R²']),
            'mape': float(test_metrics['MAPE']),
            'within_10pct': float(test_metrics['Within_10pct'])
        }
    }

    # Add feature importance if available
    if importance_df is not None:
        metadata['top_features'] = importance_df.head(20).to_dict('records')

    metadata_path = Path(output_dir) / f'rf_metadata_{timestamp}.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"✓ Metadata saved: {metadata_path}")
    print(f"  Size: {metadata_path.stat().st_size / 1024:.2f} KB")

    # ========================================================================
    # STEP 8: Summary
    # ========================================================================
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)

    print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nValidation Performance:")
    print(f"  R²:   {rf_metrics['R²']:.3f}")
    print(f"  MAE:  ${rf_metrics['MAE']:.2f}")
    print(f"  MAPE: {rf_metrics['MAPE']:.1f}%")

    print(f"\nTest Performance:")
    print(f"  R²:   {test_metrics['R²']:.3f}")
    print(f"  MAE:  ${test_metrics['MAE']:.2f}")
    print(f"  MAPE: {test_metrics['MAPE']:.1f}%")

    print(f"\nSaved artifacts:")
    print(f"  {model_path}")
    print(f"  {metadata_path}")

    print("\nNext steps:")
    print("  1. Download the .pkl and .json files")
    print("  2. Use predict.py to make predictions on new data")
    print("  3. Load metadata to inspect hyperparameters and feature importance")

    return {
        'model': rf_model,
        'val_metrics': rf_metrics,
        'test_metrics': test_metrics,
        'paths': {
            'model': str(model_path),
            'metadata': str(metadata_path)
        },
        'feature_names': feature_names,
        'importance_df': importance_df
    }


if __name__ == "__main__":
    # Run with full hyperparameter tuning
    print("\nStarting cloud training with hyperparameter tuning...")
    print("This will take approximately 2-3 hours on CPU, 30-45 min on GPU\n")

    results = train_and_save_model(
        tune_hyperparameters=True,
        cv_folds=5
    )

    print("\n" + "="*70)
    print("Script completed successfully!")
    print("="*70)
