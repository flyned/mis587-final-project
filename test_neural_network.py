"""
Test Script for Neural Network Implementation

This script verifies:
1. TensorFlow installation
2. Model building
3. Training on small sample
4. Save/load cycle
5. Inference correctness

Run with: python test_neural_network.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys
import os

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Test results tracking
tests_passed = 0
tests_failed = 0


def test_passed(name):
    global tests_passed
    tests_passed += 1
    print(f"  [PASS] {name}")


def test_failed(name, error):
    global tests_failed
    tests_failed += 1
    print(f"  [FAIL] {name}: {error}")


def print_section(title):
    print("\n" + "="*60)
    print(title)
    print("="*60)


def main():
    global tests_passed, tests_failed

    print("\n" + "="*60)
    print("NEURAL NETWORK IMPLEMENTATION TEST")
    print("="*60)

    # Test 1: TensorFlow installation
    print_section("1. Testing TensorFlow Installation")
    try:
        from src.modeling_deep import check_tensorflow, TF_AVAILABLE

        if TF_AVAILABLE:
            check_tensorflow()
            test_passed("TensorFlow imported successfully")
        else:
            test_failed("TensorFlow import", "TensorFlow not available")
            print("\nPlease install TensorFlow: pip install tensorflow>=2.13.0")
            return
    except Exception as e:
        test_failed("TensorFlow import", str(e))
        return

    # Test 2: Model building
    print_section("2. Testing Model Building")
    try:
        from src.modeling_deep import build_rent_model

        # Build model with sample input dimension
        model = build_rent_model(input_dim=194)
        test_passed("Model built successfully")

        # Check model structure
        if len(model.layers) >= 7:  # Input + 3 Dense + 3 Dropout + Output
            test_passed("Model has correct number of layers")
        else:
            test_failed("Layer count", f"Expected >= 7, got {len(model.layers)}")

        # Check input/output shapes
        if model.input_shape == (None, 194):
            test_passed("Input shape correct")
        else:
            test_failed("Input shape", f"Expected (None, 194), got {model.input_shape}")

        if model.output_shape == (None, 1):
            test_passed("Output shape correct")
        else:
            test_failed("Output shape", f"Expected (None, 1), got {model.output_shape}")

    except Exception as e:
        test_failed("Model building", str(e))
        import traceback
        traceback.print_exc()

    # Test 3: Training on synthetic data
    print_section("3. Testing Training on Synthetic Data")
    try:
        from src.modeling_deep import train_neural_network

        # Create small synthetic dataset
        np.random.seed(42)
        X_train = np.random.randn(100, 50)
        y_train = np.random.randn(100) * 10 + 12  # Mean ~12 (like rent)
        X_val = np.random.randn(20, 50)
        y_val = np.random.randn(20) * 10 + 12

        # Train with small epochs for testing
        model, scaler, history, metrics = train_neural_network(
            X_train, y_train, X_val, y_val,
            epochs=5,
            batch_size=16,
            verbose=0
        )

        test_passed("Training completed without errors")

        # Check outputs
        if scaler is not None:
            test_passed("Scaler was created")
        else:
            test_failed("Scaler", "Scaler is None")

        if 'mae' in metrics:
            test_passed(f"Metrics computed (MAE: ${metrics['mae']:.2f})")
        else:
            test_failed("Metrics", "MAE not in metrics")

    except Exception as e:
        test_failed("Training", str(e))
        import traceback
        traceback.print_exc()

    # Test 4: Save/Load cycle
    print_section("4. Testing Save/Load Cycle")
    try:
        from src.modeling_deep import save_nn_model, load_nn_model
        import tempfile

        # Create temp directory
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save model
            feature_names = [f"feature_{i}" for i in range(50)]
            metadata = {
                'training_samples': 100,
                'validation_metrics': {'mae': 1.0, 'r2': 0.5}
            }

            model_path, metadata_path = save_nn_model(
                model, scaler, feature_names, metadata,
                tmpdir, "test"
            )

            test_passed("Model saved successfully")

            # Load model
            loaded_model, loaded_scaler, loaded_metadata = load_nn_model(
                model_path, metadata_path
            )

            test_passed("Model loaded successfully")

            # Verify loaded scaler
            if np.allclose(loaded_scaler.mean_, scaler.mean_):
                test_passed("Scaler mean preserved")
            else:
                test_failed("Scaler mean", "Mean values don't match")

            if np.allclose(loaded_scaler.scale_, scaler.scale_):
                test_passed("Scaler scale preserved")
            else:
                test_failed("Scaler scale", "Scale values don't match")

            # Verify loaded metadata
            if loaded_metadata.get('training_samples') == 100:
                test_passed("Metadata preserved")
            else:
                test_failed("Metadata", "Training samples don't match")

    except Exception as e:
        test_failed("Save/Load", str(e))
        import traceback
        traceback.print_exc()

    # Test 5: Inference
    print_section("5. Testing Inference")
    try:
        from src.modeling_deep import predict_with_nn

        # Make predictions
        X_test = np.random.randn(10, 50)
        predictions = predict_with_nn(model, scaler, X_test)

        if len(predictions) == 10:
            test_passed("Prediction output shape correct")
        else:
            test_failed("Prediction shape", f"Expected 10, got {len(predictions)}")

        if predictions.ndim == 1:
            test_passed("Predictions are 1D array")
        else:
            test_failed("Prediction dims", f"Expected 1D, got {predictions.ndim}D")

        test_passed(f"Sample predictions: {predictions[:3].round(2)}")

    except Exception as e:
        test_failed("Inference", str(e))
        import traceback
        traceback.print_exc()

    # Test 6: Check predict.py integration
    print_section("6. Testing predict.py Integration")
    try:
        from predict import CoStarPredictor

        # Just check the class exists and has expected methods
        if hasattr(CoStarPredictor, '_load_neural_network'):
            test_passed("CoStarPredictor has NN loading method")
        else:
            test_failed("predict.py", "Missing _load_neural_network method")

        if hasattr(CoStarPredictor, '_load_random_forest'):
            test_passed("CoStarPredictor has RF loading method")
        else:
            test_failed("predict.py", "Missing _load_random_forest method")

    except Exception as e:
        test_failed("predict.py integration", str(e))

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"\nPassed: {tests_passed}")
    print(f"Failed: {tests_failed}")
    print(f"Total:  {tests_passed + tests_failed}")

    if tests_failed == 0:
        print("\n[SUCCESS] All tests passed!")
        print("\nNext steps:")
        print("1. Run: python train_neural_network.py")
        print("   This will train on actual data and save model to models/")
        print("\n2. Then the model will appear in the Streamlit app's model selector")
    else:
        print(f"\n[WARNING] {tests_failed} test(s) failed")
        print("Please check the errors above and fix before proceeding")

    return tests_failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
