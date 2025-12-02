"""
Deep Learning Module for Rent Prediction

This module implements a feedforward neural network using TensorFlow/Keras
as an alternative to the Random Forest model for rent prediction.

Architecture:
- 3-layer feedforward network (256->128->64->1)
- Dropout regularization to prevent overfitting
- Early stopping for optimal training

Usage:
    from src.modeling_deep import train_neural_network, load_nn_model

    # Training
    model, scaler, history, metrics = train_neural_network(X_train, y_train, X_val, y_val)

    # Inference
    model, scaler, metadata = load_nn_model(model_path, metadata_path)
    X_scaled = scaler.transform(X_new)
    predictions = model.predict(X_scaled)
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error

# TensorFlow imports with error handling
try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("Warning: TensorFlow not installed. Run: pip install tensorflow>=2.13.0")


def check_tensorflow():
    """Check if TensorFlow is available and print version info."""
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow is not installed. Run: pip install tensorflow>=2.13.0")
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Keras version: {keras.__version__}")
    return True


def build_rent_model(input_dim, hidden_sizes=[256, 128, 64], dropout_rates=[0.3, 0.2, 0.1]):
    """
    Build a feedforward neural network for rent prediction.

    Architecture:
    - Input: input_dim features (194 after engineering)
    - Hidden 1: 256 units, ReLU, Dropout(0.3)
    - Hidden 2: 128 units, ReLU, Dropout(0.2)
    - Hidden 3: 64 units, ReLU, Dropout(0.1)
    - Output: 1 unit (rent prediction, linear activation)

    Args:
        input_dim: Number of input features
        hidden_sizes: List of hidden layer sizes
        dropout_rates: List of dropout rates for each hidden layer

    Returns:
        Compiled Keras Sequential model
    """
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow is not installed")

    # Build sequential model
    layers = [keras.layers.Input(shape=(input_dim,))]

    for i, (hidden_size, dropout_rate) in enumerate(zip(hidden_sizes, dropout_rates)):
        layers.append(keras.layers.Dense(hidden_size, activation='relu', name=f'hidden_{i+1}'))
        layers.append(keras.layers.Dropout(dropout_rate, name=f'dropout_{i+1}'))

    # Output layer (single value for regression)
    layers.append(keras.layers.Dense(1, name='output'))

    model = keras.Sequential(layers)

    return model


def train_neural_network(X_train, y_train, X_val, y_val,
                         epochs=100, batch_size=32, learning_rate=0.001,
                         patience=10, verbose=1):
    """
    Train neural network with early stopping.

    Args:
        X_train: Training features (numpy array)
        y_train: Training target (numpy array)
        X_val: Validation features (numpy array)
        y_val: Validation target (numpy array)
        epochs: Maximum number of training epochs
        batch_size: Mini-batch size
        learning_rate: Adam optimizer learning rate
        patience: Early stopping patience (epochs without improvement)
        verbose: Training verbosity (0=silent, 1=progress bar, 2=one line per epoch)

    Returns:
        model: Trained Keras model
        scaler: Fitted StandardScaler (CRITICAL for inference)
        history: Training history object
        metrics: Dictionary of evaluation metrics
    """
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow is not installed")

    print("\n" + "="*70)
    print("TRAINING NEURAL NETWORK")
    print("="*70)

    # Scale features (critical for neural networks)
    print("\nScaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    print(f"Training samples: {X_train_scaled.shape[0]}")
    print(f"Validation samples: {X_val_scaled.shape[0]}")
    print(f"Input features: {X_train_scaled.shape[1]}")

    # Build model
    print("\nBuilding model...")
    model = build_rent_model(input_dim=X_train.shape[1])

    # Compile with Adam optimizer
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae']
    )

    # Print model summary
    print("\nModel architecture:")
    model.summary()

    # Define callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
    ]

    # Train model
    print(f"\nTraining for up to {epochs} epochs with early stopping (patience={patience})...")
    history = model.fit(
        X_train_scaled, y_train,
        validation_data=(X_val_scaled, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=verbose
    )

    # Evaluate on validation set
    print("\nEvaluating on validation set...")
    y_pred = model.predict(X_val_scaled, verbose=0).flatten()
    metrics = evaluate_nn_model(y_val, y_pred, name="Neural Network")
    print_nn_metrics(metrics)

    # Add training info to metrics
    metrics['epochs_trained'] = len(history.history['loss'])
    metrics['final_train_loss'] = history.history['loss'][-1]
    metrics['final_val_loss'] = history.history['val_loss'][-1]

    return model, scaler, history, metrics


def evaluate_nn_model(y_true, y_pred, name="Neural Network"):
    """
    Evaluate neural network with same metrics as RF model.

    Args:
        y_true: True target values
        y_pred: Predicted values
        name: Model name for display

    Returns:
        Dictionary of evaluation metrics
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100

    # Prediction interval accuracy (within +/-10%)
    within_10pct = np.mean(np.abs(y_pred - y_true) / np.abs(y_true) < 0.10)

    return {
        'name': name,
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'mape': mape,
        'within_10pct': within_10pct
    }


def print_nn_metrics(metrics):
    """Print evaluation metrics in formatted output."""
    print(f"\n{metrics['name']} Performance:")
    print(f"  MAE:              ${metrics['mae']:>10,.2f}")
    print(f"  RMSE:             ${metrics['rmse']:>10,.2f}")
    print(f"  R2:               {metrics['r2']:>11.3f}")
    print(f"  MAPE:             {metrics['mape']:>10.1f}%")
    print(f"  Within +/-10%:    {metrics['within_10pct']:>10.1%}")


def save_nn_model(model, scaler, feature_names, metadata, model_dir, timestamp):
    """
    Save neural network model, scaler, and metadata.

    Args:
        model: Trained Keras model
        scaler: Fitted StandardScaler
        feature_names: List of feature names
        metadata: Dictionary with training info and metrics
        model_dir: Directory to save model files
        timestamp: Timestamp string for filename

    Returns:
        Tuple of (model_path, metadata_path)
    """
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    # Save Keras model
    model_path = model_dir / f"nn_model_{timestamp}.keras"
    model.save(model_path)
    print(f"Model saved to: {model_path}")

    # Add scaler parameters to metadata
    metadata['model_type'] = 'NeuralNetwork'
    metadata['framework'] = 'tensorflow'
    metadata['scaler_mean'] = scaler.mean_.tolist()
    metadata['scaler_scale'] = scaler.scale_.tolist()
    metadata['feature_names'] = feature_names

    # Save metadata as JSON
    metadata_path = model_dir / f"nn_metadata_{timestamp}.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to: {metadata_path}")

    return str(model_path), str(metadata_path)


def load_nn_model(model_path, metadata_path):
    """
    Load neural network model and recreate scaler.

    Args:
        model_path: Path to .keras model file
        metadata_path: Path to .json metadata file

    Returns:
        model: Loaded Keras model
        scaler: Reconstructed StandardScaler
        metadata: Metadata dictionary
    """
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow is not installed")

    # Load Keras model
    model = keras.models.load_model(model_path)

    # Load metadata
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    # Reconstruct scaler from saved parameters
    scaler = StandardScaler()
    scaler.mean_ = np.array(metadata['scaler_mean'])
    scaler.scale_ = np.array(metadata['scaler_scale'])
    scaler.var_ = scaler.scale_ ** 2
    scaler.n_features_in_ = len(scaler.mean_)

    return model, scaler, metadata


def predict_with_nn(model, scaler, X):
    """
    Make predictions with neural network model.

    Args:
        model: Trained Keras model
        scaler: Fitted StandardScaler
        X: Input features (numpy array)

    Returns:
        predictions: Predicted rent values (numpy array)
    """
    X_scaled = scaler.transform(X)
    predictions = model.predict(X_scaled, verbose=0).flatten()
    return predictions


def compare_with_baseline(nn_metrics, rf_metrics):
    """
    Compare neural network with Random Forest baseline.

    Args:
        nn_metrics: Dictionary of NN metrics
        rf_metrics: Dictionary of RF metrics

    Returns:
        Comparison DataFrame
    """
    comparison = pd.DataFrame({
        'Metric': ['MAE', 'RMSE', 'R2', 'MAPE', 'Within +/-10%'],
        'Neural Network': [
            f"${nn_metrics['mae']:.2f}",
            f"${nn_metrics['rmse']:.2f}",
            f"{nn_metrics['r2']:.3f}",
            f"{nn_metrics['mape']:.1f}%",
            f"{nn_metrics['within_10pct']:.1%}"
        ],
        'Random Forest': [
            f"${rf_metrics['mae']:.2f}",
            f"${rf_metrics['rmse']:.2f}",
            f"{rf_metrics['r2']:.3f}",
            f"{rf_metrics['mape']:.1f}%",
            f"{rf_metrics['within_10pct']:.1%}"
        ]
    })

    print("\n" + "="*70)
    print("MODEL COMPARISON: Neural Network vs Random Forest")
    print("="*70)
    print(comparison.to_string(index=False))

    # Determine winner for each metric
    print("\n" + "-"*70)
    if nn_metrics['mae'] < rf_metrics['mae']:
        print("MAE: Neural Network wins")
    else:
        print("MAE: Random Forest wins")

    if nn_metrics['r2'] > rf_metrics['r2']:
        print("R2: Neural Network wins")
    else:
        print("R2: Random Forest wins")

    return comparison
