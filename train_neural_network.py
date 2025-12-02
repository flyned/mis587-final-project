"""
Neural Network Training Script

This script trains a TensorFlow/Keras neural network for rent prediction
as an alternative to the Random Forest model.

Features:
- 3-layer feedforward architecture (256->128->64->1)
- Early stopping to prevent overfitting
- Comparison with Random Forest baseline
- Memory-efficient (typically <1GB)

Usage:
    python train_neural_network.py

Output:
    - models/nn_model_YYYYMMDD_HHMMSS.keras (trained model)
    - models/nn_metadata_YYYYMMDD_HHMMSS.json (metadata + scaler params)
"""

import pandas as pd
import numpy as np
import json
import gc
import os
from datetime import datetime
from pathlib import Path

# Set TensorFlow logging level before import
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress INFO and WARNING messages

from src.feature_engineering import engineer_features
from src.modeling import prepare_data_for_modeling, evaluate_model
from src.modeling_deep import (
    check_tensorflow,
    train_neural_network,
    save_nn_model,
    evaluate_nn_model,
    print_nn_metrics
)


def get_memory_usage():
    """Get current memory usage in MB."""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        mem_mb = process.memory_info().rss / (1024 * 1024)
        return mem_mb
    except ImportError:
        return 0


def load_and_engineer_data(data_dir="../data"):
    """
    Load data splits and apply feature engineering.

    Args:
        data_dir: Path to data directory

    Returns:
        Tuple of (train_df, val_df, test_df) with engineered features
    """
    data_path = Path(data_dir)

    print("\n" + "="*70)
    print("LOADING DATA")
    print("="*70)

    # Load splits
    print("\nLoading train/val/test splits...")
    train_df = pd.read_csv(data_path / "train.csv")
    val_df = pd.read_csv(data_path / "val.csv")
    test_df = pd.read_csv(data_path / "test.csv")

    print(f"Train: {train_df.shape[0]:,} samples")
    print(f"Val:   {val_df.shape[0]:,} samples")
    print(f"Test:  {test_df.shape[0]:,} samples")
    print(f"Memory: {get_memory_usage():.1f} MB")

    # Apply feature engineering
    print("\n" + "="*70)
    print("FEATURE ENGINEERING")
    print("="*70)

    print("\nEngineering training features...")
    train_df = engineer_features(train_df.copy())
    gc.collect()

    print("\nEngineering validation features...")
    val_df = engineer_features(val_df.copy())
    gc.collect()

    print("\nEngineering test features...")
    test_df = engineer_features(test_df.copy())
    gc.collect()

    print(f"\nMemory after engineering: {get_memory_usage():.1f} MB")

    return train_df, val_df, test_df


def main():
    """Main training function."""
    print("\n" + "="*70)
    print("NEURAL NETWORK TRAINING SCRIPT")
    print("="*70)

    # Check TensorFlow installation
    print("\nChecking TensorFlow installation...")
    try:
        check_tensorflow()
    except ImportError as e:
        print(f"Error: {e}")
        print("\nPlease install TensorFlow: pip install tensorflow>=2.13.0")
        return

    # Load and prepare data
    train_df, val_df, test_df = load_and_engineer_data()

    # Prepare data for modeling
    print("\n" + "="*70)
    print("PREPARING DATA FOR MODELING")
    print("="*70)

    print("\nPreparing training data...")
    X_train, y_train, feature_names, _ = prepare_data_for_modeling(
        train_df, target_column='Rent/SF/Yr'
    )

    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"Features: {len(feature_names)}")

    # Free training DataFrame
    del train_df
    gc.collect()

    print("\nPreparing validation data...")
    X_val, y_val, _, _ = prepare_data_for_modeling(
        val_df, target_column='Rent/SF/Yr', ref_columns=feature_names
    )

    print(f"X_val shape: {X_val.shape}")
    print(f"y_val shape: {y_val.shape}")

    # Free validation DataFrame
    del val_df
    gc.collect()

    print("\nPreparing test data...")
    X_test, y_test, _, _ = prepare_data_for_modeling(
        test_df, target_column='Rent/SF/Yr', ref_columns=feature_names
    )

    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")

    # Free test DataFrame
    del test_df
    gc.collect()

    print(f"\nMemory after data prep: {get_memory_usage():.1f} MB")

    # Train neural network
    model, scaler, history, val_metrics = train_neural_network(
        X_train, y_train, X_val, y_val,
        epochs=100,
        batch_size=32,
        learning_rate=0.001,
        patience=10,
        verbose=1
    )

    print(f"\nMemory after NN training: {get_memory_usage():.1f} MB")

    # Evaluate on test set
    print("\n" + "="*70)
    print("TEST SET EVALUATION")
    print("="*70)

    X_test_scaled = scaler.transform(X_test)
    y_pred_test = model.predict(X_test_scaled, verbose=0).flatten()
    test_metrics = evaluate_nn_model(y_test, y_pred_test, name="Neural Network (Test Set)")
    print_nn_metrics(test_metrics)

    # Compare validation vs test performance
    print("\nValidation vs Test Performance:")
    print(f"  MAE:  Val=${val_metrics['mae']:.2f}  |  Test=${test_metrics['mae']:.2f}  |  Diff=${abs(val_metrics['mae']-test_metrics['mae']):.2f}")
    print(f"  RMSE: Val=${val_metrics['rmse']:.2f}  |  Test=${test_metrics['rmse']:.2f}  |  Diff=${abs(val_metrics['rmse']-test_metrics['rmse']):.2f}")
    print(f"  R2:   Val={val_metrics['r2']:.3f}  |  Test={test_metrics['r2']:.3f}  |  Diff={abs(val_metrics['r2']-test_metrics['r2']):.3f}")

    # Check for overfitting
    if test_metrics['r2'] < val_metrics['r2'] - 0.05:
        print("\nWarning: Potential overfitting detected (R2 drops >0.05 on test set)")
    elif test_metrics['mae'] > val_metrics['mae'] * 1.1:
        print("\nWarning: Test MAE is >10% higher than validation MAE")
    else:
        print("\nModel generalizes well to test set")

    # Save model
    print("\n" + "="*70)
    print("SAVING MODEL")
    print("="*70)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = Path("models")

    # Build metadata
    metadata = {
        'training_samples': int(X_train.shape[0]),
        'validation_samples': int(X_val.shape[0]),
        'test_samples': int(X_test.shape[0]),
        'feature_count': int(len(feature_names)),
        'epochs_trained': int(val_metrics.get('epochs_trained', 0)),
        'validation_metrics': {
            'mae': float(val_metrics['mae']),
            'rmse': float(val_metrics['rmse']),
            'r2': float(val_metrics['r2']),
            'mape': float(val_metrics['mape']),
            'within_10pct': float(val_metrics['within_10pct'])
        },
        'test_metrics': {
            'mae': float(test_metrics['mae']),
            'rmse': float(test_metrics['rmse']),
            'r2': float(test_metrics['r2']),
            'mape': float(test_metrics['mape']),
            'within_10pct': float(test_metrics['within_10pct'])
        },
        'training_history': {
            'final_train_loss': float(history.history['loss'][-1]),
            'final_val_loss': float(history.history['val_loss'][-1]),
            'final_train_mae': float(history.history['mae'][-1]),
            'final_val_mae': float(history.history['val_mae'][-1])
        }
    }

    model_path, metadata_path = save_nn_model(
        model, scaler, feature_names, metadata, model_dir, timestamp
    )

    # Final summary
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)

    print(f"\nModel saved to: {model_path}")
    print(f"Metadata saved to: {metadata_path}")

    print("\nFinal Performance (Validation Set):")
    print(f"  R2 Score: {val_metrics['r2']:.3f}")
    print(f"  MAE: ${val_metrics['mae']:.2f}")
    print(f"  Within +/-10%: {val_metrics['within_10pct']:.1%}")

    print(f"\nMemory at completion: {get_memory_usage():.1f} MB")
    print("\nTo use the model:")
    print(f"  from src.modeling_deep import load_nn_model, predict_with_nn")
    print(f"  model, scaler, metadata = load_nn_model('{model_path}', '{metadata_path}')")


if __name__ == "__main__":
    main()
