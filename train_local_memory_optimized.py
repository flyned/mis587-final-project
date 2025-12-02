"""
Memory-Optimized Training Script for Local Execution

This script addresses the 70GB memory issue by:
1. Using n_jobs=1 (no parallelization) to avoid data copying
2. Smaller parameter grid (24 combinations instead of 216)
3. Fewer CV folds (3 instead of 5)
4. Explicit garbage collection between steps
5. Memory profiling to track actual usage

Use this for LOCAL training. For cloud (Colab), use train_cloud.py instead.
"""

import pandas as pd
import pickle
import json
import gc
from datetime import datetime
from pathlib import Path

from src.feature_engineering import engineer_features
from src.modeling import (
    prepare_data_for_modeling,
    evaluate_model,
    print_metrics,
    get_feature_importance
)

# Import specific models
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score


def get_memory_usage():
    """Get current memory usage in MB."""
    import psutil
    import os
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / (1024 * 1024)
    return mem_mb


def train_random_forest_memory_optimized(
    X_train, y_train, X_val, y_val,
    tune_hyperparameters=True,
    cv_folds=3  # Reduced from 5
):
    """
    Memory-optimized Random Forest training.

    Key differences from original:
    - n_jobs=1 (no parallelization to avoid data copying)
    - Smaller parameter grid (24 vs 216 combinations)
    - Fewer CV folds (3 vs 5)
    - Explicit memory monitoring

    Args:
        X_train: Training features
        y_train: Training target
        X_val: Validation features
        y_val: Validation target
        tune_hyperparameters: If True, use GridSearchCV
        cv_folds: Number of cross-validation folds

    Returns:
        Dictionary with trained model and evaluation metrics
    """
    print("\n" + "="*70)
    print("TRAINING RANDOM FOREST (MEMORY OPTIMIZED)")
    print("="*70)

    print(f"\nMemory before training: {get_memory_usage():.1f} MB")

    if tune_hyperparameters:
        print("\nPerforming hyperparameter tuning with REDUCED parameter grid...")
        print(f"Cross-validation folds: {cv_folds}")
        print("Note: Using n_jobs=1 to minimize memory usage")

        # REDUCED parameter grid: 24 combinations (vs 216 in original)
        param_grid = {
            'n_estimators': [100, 200],  # Reduced from [100, 200, 300]
            'max_depth': [10, 20, 30],   # Reduced from [10, 20, 30, None]
            'min_samples_split': [5, 10], # Reduced from [2, 5, 10]
            'min_samples_leaf': [2, 4],   # Reduced from [1, 2, 4]
            'max_features': ['sqrt']      # Reduced from ['sqrt', 'log2']
        }

        num_combinations = (
            len(param_grid['n_estimators']) *
            len(param_grid['max_depth']) *
            len(param_grid['min_samples_split']) *
            len(param_grid['min_samples_leaf']) *
            len(param_grid['max_features'])
        )

        print(f"Parameter grid: {num_combinations} combinations")
        print(f"Total model fits: {num_combinations * cv_folds} = {num_combinations * cv_folds}")

        # Initialize base model with n_jobs=1
        rf_base = RandomForestRegressor(
            random_state=42,
            n_jobs=1  # KEY: No parallelization to avoid memory explosion
        )

        # Grid search with n_jobs=1
        grid_search = GridSearchCV(
            rf_base,
            param_grid,
            cv=cv_folds,
            scoring='neg_mean_absolute_error',
            n_jobs=1,  # KEY: No parallelization
            verbose=2  # Show progress
        )

        print("\nStarting GridSearchCV (this will be slower but use less memory)...")
        grid_search.fit(X_train, y_train)

        print(f"\nMemory after GridSearchCV: {get_memory_usage():.1f} MB")

        print(f"\nBest parameters: {grid_search.best_params_}")
        print(f"Best CV MAE: ${-grid_search.best_score_:,.2f}")

        rf = grid_search.best_estimator_

        # Free GridSearchCV memory
        del grid_search
        gc.collect()

    else:
        print("\nTraining with default hyperparameters...")
        rf = RandomForestRegressor(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42,
            n_jobs=1  # KEY: No parallelization
        )
        rf.fit(X_train, y_train)

        print(f"\nMemory after model fit: {get_memory_usage():.1f} MB")

    # Cross-validation scores
    print(f"\nPerforming {cv_folds}-fold cross-validation...")
    cv_scores = cross_val_score(
        rf, X_train, y_train,
        cv=cv_folds,
        scoring='neg_mean_absolute_error',
        n_jobs=1  # KEY: No parallelization
    )
    cv_mae = -cv_scores.mean()
    cv_std = cv_scores.std()

    print(f"CV MAE: ${cv_mae:,.2f} ± ${cv_std:,.2f}")

    # Validation predictions
    y_pred = rf.predict(X_val)
    metrics = evaluate_model(y_val, y_pred, name="Random Forest")
    metrics['cv_mae'] = cv_mae
    metrics['cv_std'] = cv_std

    print_metrics(metrics)

    print(f"\nMemory at end of training: {get_memory_usage():.1f} MB")

    return {'model': rf, 'scaler': None}, metrics


def train_and_save_model_local(
    train_path='../data/train.csv',
    val_path='../data/val.csv',
    test_path='../data/test.csv',
    output_dir='./models',
    tune_hyperparameters=True,
    cv_folds=3,  # Reduced from 5
    target_column='Rent/SF/Yr',
    monitor_memory=True
):
    """
    Memory-optimized end-to-end training pipeline for local execution.

    Args:
        train_path: Path to training CSV
        val_path: Path to validation CSV
        test_path: Path to test CSV
        output_dir: Directory to save outputs
        tune_hyperparameters: Enable GridSearchCV (24 combinations, ~30-45 min)
        cv_folds: Number of cross-validation folds
        target_column: Name of target variable
        monitor_memory: Print memory usage at each step

    Returns:
        Dictionary with model, metrics, and paths
    """
    if monitor_memory:
        try:
            import psutil
        except ImportError:
            print("Warning: psutil not installed. Install with: pip install psutil")
            monitor_memory = False

    print("="*70)
    print("MEMORY-OPTIMIZED TRAINING - RANDOM FOREST (LOCAL)")
    print("="*70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    if monitor_memory:
        print(f"Initial memory: {get_memory_usage():.1f} MB")

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

    if monitor_memory:
        print(f"Memory after loading: {get_memory_usage():.1f} MB")

    # ========================================================================
    # STEP 2: Feature Engineering
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 2: FEATURE ENGINEERING")
    print("="*70)

    train_eng = engineer_features(train.copy(), target_col=target_column)
    del train  # Free original DataFrame
    gc.collect()

    if monitor_memory:
        print(f"Memory after train engineering: {get_memory_usage():.1f} MB")

    val_eng = engineer_features(val.copy(), target_col=target_column)
    del val
    gc.collect()

    if monitor_memory:
        print(f"Memory after val engineering: {get_memory_usage():.1f} MB")

    test_eng = engineer_features(test.copy(), target_col=target_column)
    del test
    gc.collect()

    if monitor_memory:
        print(f"Memory after test engineering: {get_memory_usage():.1f} MB")

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
    del train_eng
    gc.collect()

    if monitor_memory:
        print(f"Memory after train preparation: {get_memory_usage():.1f} MB")

    X_val, y_val, _, _ = prepare_data_for_modeling(
        val_eng, target_column=target_column, ref_columns=feature_names
    )
    del val_eng
    gc.collect()

    if monitor_memory:
        print(f"Memory after val preparation: {get_memory_usage():.1f} MB")

    X_test, y_test, _, _ = prepare_data_for_modeling(
        test_eng, target_column=target_column, ref_columns=feature_names
    )
    del test_eng
    gc.collect()

    if monitor_memory:
        print(f"Memory after test preparation: {get_memory_usage():.1f} MB")

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
        print("\nHyperparameter tuning enabled (REDUCED grid for memory optimization)")
        print("Parameter combinations: 24 (vs 216 in original)")
        print("Expected time: 30-45 minutes (slower due to n_jobs=1)")

    rf_model, rf_metrics = train_random_forest_memory_optimized(
        X_train, y_train, X_val, y_val,
        tune_hyperparameters=tune_hyperparameters,
        cv_folds=cv_folds
    )

    if monitor_memory:
        print(f"\nMemory after training: {get_memory_usage():.1f} MB")

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

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    model_path = Path(output_dir) / f'rf_model_local_{timestamp}.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(rf_model, f)

    print(f"\n✓ Model saved: {model_path}")
    print(f"  Size: {model_path.stat().st_size / (1024*1024):.2f} MB")

    metadata = {
        'model_type': 'RandomForestRegressor',
        'training_mode': 'memory_optimized_local',
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
        'parameter_grid_size': 24 if tune_hyperparameters else 1,
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

    if importance_df is not None:
        metadata['top_features'] = importance_df.head(20).to_dict('records')

    metadata_path = Path(output_dir) / f'rf_metadata_local_{timestamp}.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"✓ Metadata saved: {metadata_path}")

    # ========================================================================
    # STEP 8: Summary
    # ========================================================================
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)

    print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    if monitor_memory:
        print(f"Final memory usage: {get_memory_usage():.1f} MB")

    print(f"\nValidation Performance:")
    print(f"  R²:   {rf_metrics['R²']:.3f}")
    print(f"  MAE:  ${rf_metrics['MAE']:.2f}")

    print(f"\nTest Performance:")
    print(f"  R²:   {test_metrics['R²']:.3f}")
    print(f"  MAE:  ${test_metrics['MAE']:.2f}")

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
    print("\nMemory-Optimized Training for Local Execution")
    print("="*70)
    print("\nThis script uses:")
    print("  - n_jobs=1 (no parallelization)")
    print("  - 24 hyperparameter combinations (vs 216)")
    print("  - 3 CV folds (vs 5)")
    print("  - Explicit garbage collection")
    print("\nExpected memory usage: 2-4 GB (vs 70GB before)")
    print("Expected training time: 30-45 minutes\n")

    input("Press Enter to start training...")

    results = train_and_save_model_local(
        tune_hyperparameters=True,
        cv_folds=3,
        monitor_memory=True
    )

    print("\n" + "="*70)
    print("✓ Training completed successfully!")
    print("="*70)
