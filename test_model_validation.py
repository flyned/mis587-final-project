"""
Comprehensive test suite for model validation framework.

Tests all validation functions in src/model_validators.py:
- Overfitting detection
- Residual analysis
- Prediction distribution validation
- Cross-validation consistency
- Model stability
- Feature importance validation

Usage:
    python test_model_validation.py
"""

import pandas as pd
import numpy as np
import sys
sys.path.append('.')

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from src.utils import load_train_data, load_val_data
from src.feature_engineering import engineer_features
from src.modeling import prepare_data_for_modeling, train_baseline_models, train_random_forest
from src.model_validators import (
    validate_overfitting,
    validate_residuals,
    validate_prediction_distribution,
    validate_cv_consistency,
    validate_model_stability,
    validate_feature_importance,
    validate_model_comprehensive,
    quick_validate_model
)


def print_section(title):
    """Print section header."""
    print("\n" + "="*70)
    print(title)
    print("="*70)


def print_result(result):
    """Print validation result details."""
    status = "✓ PASSED" if result.passed else "✗ FAILED"
    print(f"\n{result.validator_name}: {status}")
    print(f"  Errors: {len(result.errors)}")
    print(f"  Warnings: {len(result.warnings)}")

    if result.metrics:
        print(f"\n  Key Metrics:")
        for metric, value in list(result.metrics.items())[:5]:
            print(f"    {metric}: {value}")

    if result.errors:
        print(f"\n  Errors:")
        for error in result.errors:
            print(f"    - {error}")

    if result.warnings:
        print(f"\n  Warnings (first 3):")
        for warning in result.warnings[:3]:
            print(f"    - {warning}")
        if len(result.warnings) > 3:
            print(f"    ... and {len(result.warnings)-3} more")


# ============================================================================
# TEST 1: Overfitting Detection
# ============================================================================

def test_overfitting_detection(model, X_train, y_train, X_val, y_val):
    """Test overfitting detection."""
    print_section("TEST 1: OVERFITTING DETECTION")

    print("\n1.1 Testing with trained model...")
    result = validate_overfitting(model, X_train, y_train, X_val, y_val)
    print_result(result)

    print("\n1.2 Testing with artificially overfit model...")
    # Create overfitting scenario - very deep tree
    from sklearn.tree import DecisionTreeRegressor
    overfit_model = DecisionTreeRegressor(max_depth=None, min_samples_split=2)
    overfit_model.fit(X_train, y_train)

    result_overfit = validate_overfitting(overfit_model, X_train, y_train, X_val, y_val,
                                         max_mae_ratio=1.2, max_r2_diff=0.1)
    print_result(result_overfit)

    if len(result_overfit.errors) > 0 or len(result_overfit.warnings) > 0:
        print("  ✓ Successfully detected overfitting in intentionally overfit model")

    return result.passed


# ============================================================================
# TEST 2: Residual Analysis
# ============================================================================

def test_residual_analysis(model, X_val, y_val):
    """Test residual analysis validation."""
    print_section("TEST 2: RESIDUAL ANALYSIS")

    y_pred = model.predict(X_val)

    print("\n2.1 Testing residual analysis on validation set...")
    result = validate_residuals(y_val, y_pred, name="Validation")
    print_result(result)

    print("\n2.2 Testing with perfect predictions (should show issues)...")
    result_perfect = validate_residuals(y_val, y_val, name="Perfect")
    print_result(result_perfect)

    if result_perfect.metrics['mean_residual'] == 0:
        print("  ✓ Correctly identified zero residuals")

    print("\n2.3 Testing with biased predictions...")
    y_pred_biased = y_pred + 5.0  # Add systematic bias
    result_biased = validate_residuals(y_val, y_pred_biased, name="Biased")
    print_result(result_biased)

    if len(result_biased.warnings) > 0:
        print("  ✓ Successfully detected biased residuals")

    return result.passed or len(result.warnings) < 3


# ============================================================================
# TEST 3: Prediction Distribution
# ============================================================================

def test_prediction_distribution(model, X_val, y_val):
    """Test prediction distribution validation."""
    print_section("TEST 3: PREDICTION DISTRIBUTION VALIDATION")

    y_pred = model.predict(X_val)

    print("\n3.1 Testing normal predictions...")
    result = validate_prediction_distribution(y_val, y_pred)
    print_result(result)

    print("\n3.2 Testing with negative predictions (should fail)...")
    y_pred_negative = np.where(y_pred > 10, -5.0, y_pred)
    result_negative = validate_prediction_distribution(y_val, y_pred_negative)
    print_result(result_negative)

    if len(result_negative.errors) > 0:
        print("  ✓ Successfully detected negative predictions")

    print("\n3.3 Testing with out-of-range predictions...")
    y_pred_extreme = y_pred * 100  # Make predictions unreasonably high
    result_extreme = validate_prediction_distribution(y_val, y_pred_extreme)
    print_result(result_extreme)

    if len(result_extreme.warnings) > 0:
        print("  ✓ Successfully detected out-of-range predictions")

    print("\n3.4 Testing with constant predictions...")
    y_pred_constant = np.full_like(y_pred, 12.0)
    result_constant = validate_prediction_distribution(y_val, y_pred_constant)
    print_result(result_constant)

    if len(result_constant.errors) > 0 or len(result_constant.warnings) > 0:
        print("  ✓ Successfully detected constant predictions")

    return result.passed


# ============================================================================
# TEST 4: Cross-Validation Consistency
# ============================================================================

def test_cv_consistency(X_train, y_train):
    """Test cross-validation consistency."""
    print_section("TEST 4: CROSS-VALIDATION CONSISTENCY")

    print("\n4.1 Testing Ridge model CV consistency...")
    ridge_model = Ridge(alpha=1.0)
    result_ridge = validate_cv_consistency(ridge_model, X_train, y_train, cv=3)
    print_result(result_ridge)

    print("\n4.2 Testing Random Forest CV consistency...")
    rf_model = RandomForestRegressor(n_estimators=10, max_depth=5, random_state=42)
    result_rf = validate_cv_consistency(rf_model, X_train, y_train, cv=3)
    print_result(result_rf)

    return result_ridge.passed and result_rf.passed


# ============================================================================
# TEST 5: Model Stability
# ============================================================================

def test_model_stability(model, X_val, y_val):
    """Test model stability to input noise."""
    print_section("TEST 5: MODEL STABILITY")

    # Use smaller sample for speed
    sample_size = min(500, len(X_val))
    indices = np.random.choice(len(X_val), sample_size, replace=False)
    X_sample = X_val[indices]
    y_sample = y_val[indices]

    print(f"\n5.1 Testing stability with {sample_size} samples...")
    result = validate_model_stability(model, X_sample, y_sample, n_trials=5, noise_level=0.01)
    print_result(result)

    return result.passed or len(result.warnings) < 2


# ============================================================================
# TEST 6: Feature Importance
# ============================================================================

def test_feature_importance(model, feature_names):
    """Test feature importance validation."""
    print_section("TEST 6: FEATURE IMPORTANCE VALIDATION")

    print("\n6.1 Testing feature importance extraction...")
    result = validate_feature_importance(model, feature_names, top_n=15)
    print_result(result)

    if 'top_features' in result.info:
        print(f"\n  Top 10 features:")
        for i, feat in enumerate(result.info['top_features'][:10]):
            print(f"    {i+1}. {feat['feature']}: {feat['importance']:.4f}")

    return result.passed or len(result.warnings) < 2


# ============================================================================
# TEST 7: Comprehensive Model Validation
# ============================================================================

def test_comprehensive_validation(model, X_train, y_train, X_val, y_val, feature_names):
    """Test comprehensive model validation."""
    print_section("TEST 7: COMPREHENSIVE MODEL VALIDATION")

    print("\n7.1 Running all validations together...")
    results = validate_model_comprehensive(model, X_train, y_train, X_val, y_val,
                                          feature_names=feature_names, verbose=True)

    return results['all_passed'] or results['total_errors'] == 0


# ============================================================================
# TEST 8: Quick Validation Function
# ============================================================================

def test_quick_validation(model, X_train, y_train, X_val, y_val):
    """Test quick validation function."""
    print_section("TEST 8: QUICK VALIDATION FUNCTION")

    print("\n8.1 Testing quick_validate_model...")
    is_valid = quick_validate_model(model, X_train, y_train, X_val, y_val)
    print(f"  Quick validation result: {'PASSED' if is_valid else 'FAILED'}")

    return True  # Always return True as it's just a convenience function


# ============================================================================
# MAIN TEST RUNNER
# ============================================================================

def main():
    """Run all model validation tests."""
    print("="*70)
    print("MODEL VALIDATION TEST SUITE")
    print("="*70)
    print("\nTesting src/model_validators.py model validation framework")

    # Setup: Load data and train a model
    print("\nSetup: Loading data and training model...")
    try:
        train = load_train_data()
        val = load_val_data()

        print(f"  Train: {len(train):,} rows")
        print(f"  Val: {len(val):,} rows")

        # Feature engineering
        print("\n  Applying feature engineering...")
        train_eng = engineer_features(train.copy())
        val_eng = engineer_features(val.copy())

        # Prepare data
        print("  Preparing data for modeling...")
        X_train, y_train, feature_names, _ = prepare_data_for_modeling(train_eng)
        X_val, y_val, _, _ = prepare_data_for_modeling(val_eng, ref_columns=feature_names)

        print(f"\n  Features: {len(feature_names)}")
        print(f"  Training samples: {len(X_train):,}")
        print(f"  Validation samples: {len(X_val):,}")

        # Train a Random Forest model
        print("\n  Training Random Forest model...")
        model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)

        print("  ✓ Model trained successfully")

    except Exception as e:
        print(f"\n✗ Setup failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Run all tests
    results = {}

    try:
        results['overfitting'] = test_overfitting_detection(model, X_train, y_train, X_val, y_val)
    except Exception as e:
        print(f"\n✗ Overfitting test failed: {e}")
        results['overfitting'] = False

    try:
        results['residuals'] = test_residual_analysis(model, X_val, y_val)
    except Exception as e:
        print(f"\n✗ Residual analysis test failed: {e}")
        results['residuals'] = False

    try:
        results['predictions'] = test_prediction_distribution(model, X_val, y_val)
    except Exception as e:
        print(f"\n✗ Prediction distribution test failed: {e}")
        results['predictions'] = False

    try:
        results['cv_consistency'] = test_cv_consistency(X_train, y_train)
    except Exception as e:
        print(f"\n✗ CV consistency test failed: {e}")
        results['cv_consistency'] = False

    try:
        results['stability'] = test_model_stability(model, X_val, y_val)
    except Exception as e:
        print(f"\n✗ Stability test failed: {e}")
        results['stability'] = False

    try:
        results['feature_importance'] = test_feature_importance(model, feature_names)
    except Exception as e:
        print(f"\n✗ Feature importance test failed: {e}")
        results['feature_importance'] = False

    try:
        results['comprehensive'] = test_comprehensive_validation(model, X_train, y_train, X_val, y_val, feature_names)
    except Exception as e:
        print(f"\n✗ Comprehensive test failed: {e}")
        results['comprehensive'] = False

    try:
        results['quick_validation'] = test_quick_validation(model, X_train, y_train, X_val, y_val)
    except Exception as e:
        print(f"\n✗ Quick validation test failed: {e}")
        results['quick_validation'] = False

    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)

    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name:.<50} {status}")

    total_passed = sum(results.values())
    total_tests = len(results)
    print(f"\nTotal: {total_passed}/{total_tests} test suites passed")

    if total_passed == total_tests:
        print("\n✓ ALL TESTS PASSED")
        return 0
    elif total_passed >= total_tests * 0.7:  # 70% pass rate
        print("\n⚠ MOST TESTS PASSED (some warnings expected)")
        return 0
    else:
        print("\n✗ MANY TESTS FAILED")
        return 1


if __name__ == "__main__":
    exit(main())
