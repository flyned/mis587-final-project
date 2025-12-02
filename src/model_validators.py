"""
Model validation framework for ML pipeline.

This module provides comprehensive validation functions for:
1. Overfitting detection - compare train vs validation performance
2. Residual analysis - check for patterns, normality, heteroscedasticity
3. Prediction distribution validation - ensure predictions are reasonable
4. Cross-validation consistency - verify stable performance across folds
5. Model stability - test robustness to input variations

Usage:
    from src.model_validators import validate_model_performance, validate_residuals

    # Validate a trained model
    result = validate_model_performance(model, X_train, y_train, X_val, y_val)
    if not result['passed']:
        print(result['errors'])
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
from scipy import stats
import warnings


class ModelValidationResult:
    """Container for model validation results."""

    def __init__(self, validator_name: str):
        self.validator_name = validator_name
        self.passed = True
        self.warnings = []
        self.errors = []
        self.info = {}
        self.metrics = {}

    def add_error(self, message: str):
        """Add an error (causes validation to fail)."""
        self.errors.append(message)
        self.passed = False

    def add_warning(self, message: str):
        """Add a warning (doesn't cause validation to fail)."""
        self.warnings.append(message)

    def add_info(self, key: str, value: Any):
        """Add informational metadata."""
        self.info[key] = value

    def add_metric(self, key: str, value: float):
        """Add a metric value."""
        self.metrics[key] = value

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'validator': self.validator_name,
            'passed': self.passed,
            'errors': self.errors,
            'warnings': self.warnings,
            'info': self.info,
            'metrics': self.metrics
        }

    def __repr__(self):
        status = "PASSED" if self.passed else "FAILED"
        return f"ModelValidationResult({self.validator_name}: {status}, {len(self.errors)} errors, {len(self.warnings)} warnings)"


# ============================================================================
# OVERFITTING DETECTION
# ============================================================================

def validate_overfitting(model, X_train, y_train, X_val, y_val,
                        max_mae_ratio: float = 1.3,
                        max_r2_diff: float = 0.15) -> ModelValidationResult:
    """
    Detect overfitting by comparing train vs validation performance.

    Overfitting indicators:
    - Val MAE >> Train MAE (ratio > 1.3 indicates overfitting)
    - Val R² << Train R² (diff > 0.15 indicates overfitting)

    Args:
        model: Trained model with predict() method
        X_train: Training features
        y_train: Training target
        X_val: Validation features
        y_val: Validation target
        max_mae_ratio: Max allowed ratio of val_mae / train_mae
        max_r2_diff: Max allowed difference in R² scores

    Returns:
        ModelValidationResult object
    """
    result = ModelValidationResult('overfitting_detection')

    try:
        # Get predictions
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)

        # Calculate metrics
        train_mae = mean_absolute_error(y_train, y_train_pred)
        val_mae = mean_absolute_error(y_val, y_val_pred)

        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))

        train_r2 = r2_score(y_train, y_train_pred)
        val_r2 = r2_score(y_val, y_val_pred)

        # Store metrics
        result.add_metric('train_mae', round(train_mae, 3))
        result.add_metric('val_mae', round(val_mae, 3))
        result.add_metric('train_rmse', round(train_rmse, 3))
        result.add_metric('val_rmse', round(val_rmse, 3))
        result.add_metric('train_r2', round(train_r2, 4))
        result.add_metric('val_r2', round(val_r2, 4))

        # Check MAE ratio
        mae_ratio = val_mae / train_mae if train_mae > 0 else float('inf')
        result.add_metric('mae_ratio', round(mae_ratio, 3))

        if mae_ratio > max_mae_ratio:
            result.add_error(
                f"Overfitting detected: Val MAE ({val_mae:.3f}) is {mae_ratio:.2f}x Train MAE ({train_mae:.3f}). "
                f"Expected ratio < {max_mae_ratio}"
            )
        elif mae_ratio > max_mae_ratio * 0.85:  # Warning zone
            result.add_warning(
                f"Possible overfitting: MAE ratio is {mae_ratio:.2f} (threshold: {max_mae_ratio})"
            )

        # Check R² difference
        r2_diff = train_r2 - val_r2
        result.add_metric('r2_diff', round(r2_diff, 4))

        if r2_diff > max_r2_diff:
            result.add_error(
                f"Overfitting detected: Train R² ({train_r2:.4f}) >> Val R² ({val_r2:.4f}). "
                f"Difference {r2_diff:.4f} exceeds threshold {max_r2_diff}"
            )
        elif r2_diff > max_r2_diff * 0.75:  # Warning zone
            result.add_warning(
                f"Possible overfitting: R² difference is {r2_diff:.4f} (threshold: {max_r2_diff})"
            )

        # Additional checks
        if train_r2 > 0.99:
            result.add_warning(f"Train R² is very high ({train_r2:.4f}) - possible overfitting or data leakage")

        if val_r2 < 0:
            result.add_error(f"Validation R² is negative ({val_r2:.4f}) - model performs worse than mean baseline")

        result.add_info('message', 'No significant overfitting detected' if result.passed else 'Overfitting detected')

    except Exception as e:
        result.add_error(f"Overfitting validation failed: {str(e)}")

    return result


# ============================================================================
# RESIDUAL ANALYSIS
# ============================================================================

def validate_residuals(y_true, y_pred, name: str = "Model") -> ModelValidationResult:
    """
    Validate residuals for patterns, normality, and heteroscedasticity.

    Good residuals should:
    - Be centered around zero (mean ≈ 0)
    - Be approximately normally distributed
    - Have constant variance (homoscedastic)
    - Show no systematic patterns

    Args:
        y_true: True target values
        y_pred: Predicted values
        name: Model name for display

    Returns:
        ModelValidationResult object
    """
    result = ModelValidationResult(f'residual_analysis_{name}')

    try:
        residuals = y_true - y_pred

        # 1. Check mean (should be close to 0)
        mean_residual = np.mean(residuals)
        std_residual = np.std(residuals)

        result.add_metric('mean_residual', round(mean_residual, 4))
        result.add_metric('std_residual', round(std_residual, 4))

        if abs(mean_residual) > 0.1:  # For rent data
            result.add_warning(f"Residuals not centered at zero: mean = {mean_residual:.4f}")

        # 2. Normality test (Shapiro-Wilk for small samples, Anderson-Darling for large)
        if len(residuals) < 5000:
            # Shapiro-Wilk test
            statistic, p_value = stats.shapiro(residuals)
            result.add_metric('shapiro_p_value', round(p_value, 4))

            if p_value < 0.05:
                result.add_warning(f"Residuals may not be normally distributed (Shapiro p={p_value:.4f})")
        else:
            # Anderson-Darling test for large samples
            result_anderson = stats.anderson(residuals, dist='norm')
            result.add_metric('anderson_statistic', round(result_anderson.statistic, 4))

            if result_anderson.statistic > result_anderson.critical_values[2]:  # 5% significance
                result.add_warning("Residuals may not be normally distributed (Anderson-Darling test)")

        # 3. Skewness and kurtosis
        skewness = stats.skew(residuals)
        kurtosis = stats.kurtosis(residuals)

        result.add_metric('skewness', round(skewness, 4))
        result.add_metric('kurtosis', round(kurtosis, 4))

        if abs(skewness) > 1:
            result.add_warning(f"Residuals are skewed: skewness = {skewness:.4f}")

        if abs(kurtosis) > 3:
            result.add_warning(f"Residuals have heavy tails: kurtosis = {kurtosis:.4f}")

        # 4. Heteroscedasticity check (Breusch-Pagan-like test)
        # Check if variance increases with predicted values
        if len(y_pred) > 30:
            # Divide predictions into quartiles
            quartiles = pd.qcut(y_pred, q=4, labels=False, duplicates='drop')
            variances = []

            for q in range(int(quartiles.max()) + 1):
                mask = quartiles == q
                if mask.sum() > 1:
                    variances.append(np.var(residuals[mask]))

            if len(variances) >= 2:
                variance_ratio = max(variances) / min(variances) if min(variances) > 0 else float('inf')
                result.add_metric('variance_ratio', round(variance_ratio, 3))

                if variance_ratio > 5:
                    result.add_warning(
                        f"Heteroscedasticity detected: variance ratio = {variance_ratio:.2f} "
                        "(variance increases with predictions)"
                    )

        # 5. Check for outliers in residuals
        residual_abs = np.abs(residuals)
        outlier_threshold = std_residual * 3
        outliers = (residual_abs > outlier_threshold).sum()
        outlier_pct = outliers / len(residuals) * 100

        result.add_metric('outlier_count', outliers)
        result.add_metric('outlier_pct', round(outlier_pct, 2))

        if outlier_pct > 5:
            result.add_warning(f"{outliers} residual outliers ({outlier_pct:.1f}%) exceed 5%")

        result.add_info('message', 'Residuals look reasonable' if result.passed and len(result.warnings) < 2 else 'Some residual issues detected')

    except Exception as e:
        result.add_error(f"Residual validation failed: {str(e)}")

    return result


# ============================================================================
# PREDICTION DISTRIBUTION VALIDATION
# ============================================================================

def validate_prediction_distribution(y_true, y_pred,
                                     min_value: float = 0.5,
                                     max_value: float = 200.0) -> ModelValidationResult:
    """
    Validate that predictions are in reasonable range and distribution.

    Args:
        y_true: True target values
        y_pred: Predicted values
        min_value: Minimum reasonable value (default $0.50/SF/Yr)
        max_value: Maximum reasonable value (default $200/SF/Yr)

    Returns:
        ModelValidationResult object
    """
    result = ModelValidationResult('prediction_distribution')

    try:
        # 1. Check for NaN or infinite predictions
        nan_count = np.isnan(y_pred).sum()
        inf_count = np.isinf(y_pred).sum()

        if nan_count > 0:
            result.add_error(f"{nan_count} predictions are NaN")

        if inf_count > 0:
            result.add_error(f"{inf_count} predictions are infinite")

        # 2. Check prediction range
        pred_min = np.min(y_pred[~np.isnan(y_pred)])
        pred_max = np.max(y_pred[~np.isnan(y_pred)])

        result.add_metric('pred_min', round(pred_min, 3))
        result.add_metric('pred_max', round(pred_max, 3))

        out_of_range = ((y_pred < min_value) | (y_pred > max_value)).sum()
        out_of_range_pct = out_of_range / len(y_pred) * 100

        if out_of_range > 0:
            result.add_warning(
                f"{out_of_range} predictions ({out_of_range_pct:.1f}%) "
                f"outside reasonable range [{min_value}, {max_value}]"
            )

        # 3. Check for negative predictions (should never happen for rent)
        negative_count = (y_pred < 0).sum()
        if negative_count > 0:
            result.add_error(f"{negative_count} negative predictions (rent cannot be negative)")

        # 4. Compare prediction vs actual distributions
        pred_mean = np.mean(y_pred)
        true_mean = np.mean(y_true)
        mean_diff_pct = abs(pred_mean - true_mean) / true_mean * 100

        result.add_metric('pred_mean', round(pred_mean, 3))
        result.add_metric('true_mean', round(true_mean, 3))
        result.add_metric('mean_diff_pct', round(mean_diff_pct, 2))

        if mean_diff_pct > 20:
            result.add_warning(
                f"Prediction mean ({pred_mean:.2f}) differs significantly from "
                f"actual mean ({true_mean:.2f}) by {mean_diff_pct:.1f}%"
            )

        # 5. Check prediction variance
        pred_std = np.std(y_pred)
        true_std = np.std(y_true)
        std_ratio = pred_std / true_std if true_std > 0 else 0

        result.add_metric('pred_std', round(pred_std, 3))
        result.add_metric('true_std', round(true_std, 3))
        result.add_metric('std_ratio', round(std_ratio, 3))

        if std_ratio < 0.5:
            result.add_warning(
                f"Predictions have low variance (std={pred_std:.2f}) compared to actuals (std={true_std:.2f}). "
                "Model may be underconfident or averaging too much."
            )
        elif std_ratio > 2.0:
            result.add_warning(
                f"Predictions have high variance (std={pred_std:.2f}) compared to actuals (std={true_std:.2f}). "
                "Model may be overconfident or unstable."
            )

        # 6. Check for constant predictions (no variance)
        unique_preds = np.unique(y_pred)
        if len(unique_preds) == 1:
            result.add_error(f"All predictions are constant: {unique_preds[0]:.3f}")
        elif len(unique_preds) < 10:
            result.add_warning(f"Very few unique predictions: {len(unique_preds)}")

        result.add_info('message', 'Predictions in reasonable range' if result.passed else 'Prediction issues detected')

    except Exception as e:
        result.add_error(f"Prediction distribution validation failed: {str(e)}")

    return result


# ============================================================================
# CROSS-VALIDATION CONSISTENCY
# ============================================================================

def validate_cv_consistency(model, X, y, cv=5, scoring='neg_mean_absolute_error',
                           max_cv_std: float = 0.5) -> ModelValidationResult:
    """
    Validate model stability across cross-validation folds.

    Args:
        model: Model to validate
        X: Features
        y: Target
        cv: Number of CV folds
        scoring: Scoring metric for CV
        max_cv_std: Max allowed std deviation across folds

    Returns:
        ModelValidationResult object
    """
    result = ModelValidationResult('cv_consistency')

    try:
        # Run cross-validation
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)

        # Convert to positive MAE
        if 'neg' in scoring:
            cv_scores = -cv_scores

        cv_mean = np.mean(cv_scores)
        cv_std = np.std(cv_scores)
        cv_min = np.min(cv_scores)
        cv_max = np.max(cv_scores)

        result.add_metric('cv_mean', round(cv_mean, 3))
        result.add_metric('cv_std', round(cv_std, 3))
        result.add_metric('cv_min', round(cv_min, 3))
        result.add_metric('cv_max', round(cv_max, 3))

        # Check consistency
        cv_range = cv_max - cv_min
        result.add_metric('cv_range', round(cv_range, 3))

        if cv_std > max_cv_std:
            result.add_warning(
                f"High variance across CV folds: std={cv_std:.3f} (threshold: {max_cv_std}). "
                f"Scores range from {cv_min:.3f} to {cv_max:.3f}"
            )

        # Check coefficient of variation
        cv_coef = cv_std / cv_mean if cv_mean > 0 else 0
        result.add_metric('cv_coefficient', round(cv_coef, 4))

        if cv_coef > 0.15:
            result.add_warning(
                f"Coefficient of variation is high ({cv_coef:.4f}). "
                "Model performance is inconsistent across folds."
            )

        result.add_info('message', f'CV scores consistent across {cv} folds' if result.passed else 'CV consistency issues detected')
        result.add_info('cv_scores', [round(s, 3) for s in cv_scores])

    except Exception as e:
        result.add_error(f"CV consistency validation failed: {str(e)}")

    return result


# ============================================================================
# MODEL STABILITY
# ============================================================================

def validate_model_stability(model, X, y, n_trials: int = 5, noise_level: float = 0.01) -> ModelValidationResult:
    """
    Test model stability by adding small noise to inputs and checking prediction variance.

    A stable model should produce similar predictions even with small input perturbations.

    Args:
        model: Trained model
        X: Features
        y: Target
        n_trials: Number of noise trials
        noise_level: Std dev of Gaussian noise to add (as fraction of feature std)

    Returns:
        ModelValidationResult object
    """
    result = ModelValidationResult('model_stability')

    try:
        # Get baseline predictions
        baseline_preds = model.predict(X)

        # Run trials with noise
        trial_preds = []

        for trial in range(n_trials):
            # Add small Gaussian noise to features
            noise = np.random.normal(0, noise_level * X.std(axis=0), X.shape)
            X_noisy = X + noise

            # Get predictions
            preds = model.predict(X_noisy)
            trial_preds.append(preds)

        trial_preds = np.array(trial_preds)

        # Calculate prediction variance across trials for each sample
        pred_std_per_sample = np.std(trial_preds, axis=0)
        mean_pred_std = np.mean(pred_std_per_sample)
        max_pred_std = np.max(pred_std_per_sample)

        result.add_metric('mean_pred_std', round(mean_pred_std, 4))
        result.add_metric('max_pred_std', round(max_pred_std, 4))

        # Compare to target scale
        target_std = np.std(y)
        stability_ratio = mean_pred_std / target_std

        result.add_metric('stability_ratio', round(stability_ratio, 4))

        # High stability ratio means predictions change a lot with small input changes
        if stability_ratio > 0.05:
            result.add_warning(
                f"Model shows instability: predictions vary by {mean_pred_std:.3f} "
                f"({stability_ratio:.1%} of target std) with {noise_level:.1%} input noise"
            )
        elif stability_ratio > 0.02:
            result.add_warning(
                f"Model shows some instability: stability ratio = {stability_ratio:.4f}"
            )

        # Check for samples with very high variance
        unstable_samples = (pred_std_per_sample > target_std * 0.1).sum()
        unstable_pct = unstable_samples / len(X) * 100

        if unstable_pct > 5:
            result.add_warning(
                f"{unstable_samples} samples ({unstable_pct:.1f}%) show high prediction variance"
            )

        result.add_info('message', 'Model predictions are stable' if result.passed else 'Some stability issues detected')
        result.add_info('n_trials', n_trials)
        result.add_info('noise_level', noise_level)

    except Exception as e:
        result.add_error(f"Model stability validation failed: {str(e)}")

    return result


# ============================================================================
# FEATURE IMPORTANCE VALIDATION
# ============================================================================

def validate_feature_importance(model, feature_names: List[str],
                                top_n: int = 20) -> ModelValidationResult:
    """
    Validate feature importance and check for suspicious patterns.

    Args:
        model: Trained model with feature_importances_ or coef_ attribute
        feature_names: List of feature names
        top_n: Number of top features to analyze

    Returns:
        ModelValidationResult object
    """
    result = ModelValidationResult('feature_importance')

    try:
        # Get feature importances
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_)
        else:
            result.add_warning("Model does not have feature_importances_ or coef_ attribute")
            return result

        # Sort by importance
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)

        # Get top features
        top_features = importance_df.head(top_n)
        result.add_info('top_features', top_features.to_dict('records'))

        # Check for dominance of single feature
        total_importance = importances.sum()
        if total_importance > 0:
            max_importance = importances.max()
            max_importance_ratio = max_importance / total_importance

            result.add_metric('max_importance_ratio', round(max_importance_ratio, 4))

            if max_importance_ratio > 0.5:
                top_feature = importance_df.iloc[0]['feature']
                result.add_warning(
                    f"Single feature dominates: '{top_feature}' accounts for {max_importance_ratio:.1%} "
                    "of total importance - possible feature leakage or overly dominant feature"
                )

        # Check for many zero-importance features
        zero_importance = (importances == 0).sum()
        zero_importance_pct = zero_importance / len(importances) * 100

        result.add_metric('zero_importance_count', zero_importance)
        result.add_metric('zero_importance_pct', round(zero_importance_pct, 2))

        if zero_importance_pct > 50:
            result.add_warning(
                f"{zero_importance} features ({zero_importance_pct:.1f}%) have zero importance - "
                "consider removing uninformative features"
            )

        # Check for uniform importance (all features equally important - suspicious)
        importance_std = np.std(importances)
        importance_mean = np.mean(importances)
        cv = importance_std / importance_mean if importance_mean > 0 else 0

        result.add_metric('importance_cv', round(cv, 4))

        if cv < 0.1 and len(importances) > 10:
            result.add_warning(
                f"Feature importances are very uniform (CV={cv:.4f}) - "
                "this is unusual and may indicate issues"
            )

        result.add_info('message', 'Feature importances look reasonable' if result.passed else 'Feature importance issues detected')

    except Exception as e:
        result.add_error(f"Feature importance validation failed: {str(e)}")

    return result


# ============================================================================
# COMPREHENSIVE MODEL VALIDATION
# ============================================================================

def validate_model_comprehensive(model, X_train, y_train, X_val, y_val,
                                 feature_names: Optional[List[str]] = None,
                                 verbose: bool = True) -> Dict:
    """
    Run all model validation checks and return comprehensive results.

    Args:
        model: Trained model
        X_train: Training features
        y_train: Training target
        X_val: Validation features
        y_val: Validation target
        feature_names: Feature names for importance analysis
        verbose: Print results to console

    Returns:
        Dictionary with all validation results
    """
    results = {
        'timestamp': datetime.now().isoformat(),
        'validations': {},
        'all_passed': True,
        'total_errors': 0,
        'total_warnings': 0
    }

    # 1. Overfitting detection
    overfitting_result = validate_overfitting(model, X_train, y_train, X_val, y_val)
    results['validations']['overfitting'] = overfitting_result.to_dict()
    if not overfitting_result.passed:
        results['all_passed'] = False
    results['total_errors'] += len(overfitting_result.errors)
    results['total_warnings'] += len(overfitting_result.warnings)

    # 2. Residual analysis (on validation set)
    y_val_pred = model.predict(X_val)
    residual_result = validate_residuals(y_val, y_val_pred, name="Validation")
    results['validations']['residuals'] = residual_result.to_dict()
    if not residual_result.passed:
        results['all_passed'] = False
    results['total_errors'] += len(residual_result.errors)
    results['total_warnings'] += len(residual_result.warnings)

    # 3. Prediction distribution
    pred_dist_result = validate_prediction_distribution(y_val, y_val_pred)
    results['validations']['prediction_distribution'] = pred_dist_result.to_dict()
    if not pred_dist_result.passed:
        results['all_passed'] = False
    results['total_errors'] += len(pred_dist_result.errors)
    results['total_warnings'] += len(pred_dist_result.warnings)

    # 4. Feature importance (if available)
    if feature_names is not None:
        feature_imp_result = validate_feature_importance(model, feature_names)
        results['validations']['feature_importance'] = feature_imp_result.to_dict()
        if not feature_imp_result.passed:
            results['all_passed'] = False
        results['total_errors'] += len(feature_imp_result.errors)
        results['total_warnings'] += len(feature_imp_result.warnings)

    # Print summary if verbose
    if verbose:
        print("\n" + "="*70)
        print("MODEL VALIDATION SUMMARY")
        print("="*70)
        print(f"Status: {'PASSED' if results['all_passed'] else 'FAILED'}")
        print(f"Total Errors: {results['total_errors']}")
        print(f"Total Warnings: {results['total_warnings']}")

        for val_name, val_result in results['validations'].items():
            status = "✓ PASSED" if val_result['passed'] else "✗ FAILED"
            print(f"\n{val_name}: {status}")

            if val_result['metrics']:
                print(f"  Metrics:")
                for metric, value in list(val_result['metrics'].items())[:5]:
                    print(f"    {metric}: {value}")

            if val_result['errors']:
                print(f"  Errors:")
                for error in val_result['errors']:
                    print(f"    - {error}")

            if val_result['warnings']:
                print(f"  Warnings:")
                for warning in val_result['warnings'][:3]:
                    print(f"    - {warning}")
                if len(val_result['warnings']) > 3:
                    print(f"    ... and {len(val_result['warnings'])-3} more warnings")

    return results


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def quick_validate_model(model, X_train, y_train, X_val, y_val) -> bool:
    """
    Quick model validation (returns True/False).

    Args:
        model: Trained model
        X_train: Training features
        y_train: Training target
        X_val: Validation features
        y_val: Validation target

    Returns:
        True if all validations pass, False otherwise
    """
    results = validate_model_comprehensive(model, X_train, y_train, X_val, y_val, verbose=False)
    return results['all_passed']
