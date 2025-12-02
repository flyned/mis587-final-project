"""
Test script for Complete Phase 2: Model Training & Selection

Tests all Phase 2 components:
- Phase 2.1: Baseline models (Ridge, Lasso, Decision Tree) ✓
- Phase 2.2: Evaluation metrics ✓
- Phase 2.3: Random Forest with cross-validation ✓
- Phase 2.4: XGBoost/LightGBM with hyperparameter tuning ✓
- Phase 2.5: Model comparison and selection ✓
"""

import pandas as pd
import numpy as np
import sys
sys.path.append('.')

from src.utils import load_train_data, load_val_data, load_test_data
from src.feature_engineering import engineer_features
from src.modeling import (
    prepare_data_for_modeling,
    train_all_models,
    compare_models,
    get_feature_importance
)


def test_phase2_complete():
    """Test complete Phase 2 model training and selection pipeline."""
    print("="*70)
    print("PHASE 2 COMPLETE: MODEL TRAINING & SELECTION")
    print("="*70)

    # 1. Load data
    print("\n1. Loading pre-made train/val/test splits...")
    train = load_train_data()
    val = load_val_data()
    test = load_test_data()

    print(f"   Train: {len(train):,} rows")
    print(f"   Val:   {len(val):,} rows")
    print(f"   Test:  {len(test):,} rows")

    # 2. Apply feature engineering
    print("\n2. Applying feature engineering pipeline...")
    train_engineered = engineer_features(train.copy())
    val_engineered = engineer_features(val.copy())
    test_engineered = engineer_features(test.copy())

    print(f"\n   Engineered features: {len(train_engineered.columns)}")

    # 3. Prepare data for modeling
    print("\n3. Preparing data for modeling...")
    X_train, y_train, feature_names, categorical_cols = prepare_data_for_modeling(
        train_engineered, target_column='Rent/SF/Yr'
    )
    # Use train feature names as reference for val/test to ensure same columns
    X_val, y_val, _, _ = prepare_data_for_modeling(
        val_engineered, target_column='Rent/SF/Yr', ref_columns=feature_names
    )
    X_test, y_test, _, _ = prepare_data_for_modeling(
        test_engineered, target_column='Rent/SF/Yr', ref_columns=feature_names
    )

    print(f"   Training samples: {len(X_train):,}")
    print(f"   Validation samples: {len(X_val):,}")
    print(f"   Test samples: {len(X_test):,}")
    print(f"   Final feature count: {X_train.shape[1]}")
    print(f"   Categorical columns encoded: {len(categorical_cols)}")

    # 4. Train all models (without hyperparameter tuning for speed)
    print("\n" + "="*70)
    print("4. TRAINING ALL MODELS")
    print("="*70)
    print("\nNote: Hyperparameter tuning disabled for faster execution.")
    print("      Set tune_rf=True, tune_xgb=True to enable full tuning.")

    all_models, all_results = train_all_models(
        X_train, y_train, X_val, y_val, feature_names,
        tune_rf=False,  # Set to True for full tuning (slower)
        tune_xgb=False,  # Set to True for full tuning (slower)
        tune_lgb=False   # Set to True for full tuning (slower)
    )

    # 5. Compare models
    print("\n" + "="*70)
    print("5. MODEL COMPARISON & SELECTION")
    print("="*70)

    comparison_df, best_model_name, test_metrics = compare_models(
        all_results,
        X_test=X_test,
        y_test=y_test,
        models_dict=all_models
    )

    # 6. Feature importance analysis
    print("\n" + "="*70)
    print("6. FEATURE IMPORTANCE ANALYSIS")
    print("="*70)

    best_model = all_models[best_model_name]['model']

    print(f"\nTop 20 Most Important Features ({best_model_name}):")
    print("-"*70)

    importance_df = get_feature_importance(best_model, feature_names, top_n=20)

    if importance_df is not None:
        print(f"\n{'Rank':<6} {'Feature':<50} {'Importance':>12}")
        print("-"*70)
        for idx, (i, row) in enumerate(importance_df.iterrows(), 1):
            print(f"{idx:<6} {row['feature']:<50} {row['importance']:>12.4f}")

        # Analyze feature importance patterns
        print("\n" + "="*70)
        print("FEATURE IMPORTANCE INSIGHTS")
        print("="*70)

        # Group features by type
        temporal_features = importance_df[importance_df['feature'].str.contains('age|year|sale|era', case=False, na=False)]
        geospatial_features = importance_df[importance_df['feature'].str.contains('dist|properties_within', case=False, na=False)]
        engineered_features = importance_df[importance_df['feature'].str.contains('log_|ratio|interaction|per_', case=False, na=False)]

        print(f"\nTop 20 features breakdown:")
        print(f"  Temporal features: {len(temporal_features)} ({len(temporal_features)/20*100:.0f}%)")
        if len(temporal_features) > 0:
            print(f"    Example: {temporal_features.iloc[0]['feature']}")

        print(f"  Geospatial features: {len(geospatial_features)} ({len(geospatial_features)/20*100:.0f}%)")
        if len(geospatial_features) > 0:
            print(f"    Example: {geospatial_features.iloc[0]['feature']}")

        print(f"  Engineered features: {len(engineered_features)} ({len(engineered_features)/20*100:.0f}%)")
        if len(engineered_features) > 0:
            print(f"    Example: {engineered_features.iloc[0]['feature']}")

    # 7. Final summary
    print("\n" + "="*70)
    print("7. FINAL SUMMARY")
    print("="*70)

    print(f"\nBest Model: {best_model_name}")
    print(f"\nValidation Performance:")
    best_val_metrics = all_results[best_model_name]
    print(f"  MAE:  ${best_val_metrics['MAE']:.2f}")
    print(f"  RMSE: ${best_val_metrics['RMSE']:.2f}")
    print(f"  R²:   {best_val_metrics['R²']:.3f}")
    print(f"  MAPE: {best_val_metrics['MAPE']:.1f}%")

    if test_metrics:
        print(f"\nTest Performance:")
        print(f"  MAE:  ${test_metrics['MAE']:.2f}")
        print(f"  RMSE: ${test_metrics['RMSE']:.2f}")
        print(f"  R²:   {test_metrics['R²']:.3f}")
        print(f"  MAPE: {test_metrics['MAPE']:.1f}%")

        # Generalization assessment
        mae_diff = abs(best_val_metrics['MAE'] - test_metrics['MAE'])
        r2_diff = abs(best_val_metrics['R²'] - test_metrics['R²'])

        print(f"\nGeneralization Analysis:")
        print(f"  MAE difference: ${mae_diff:.2f}")
        print(f"  R² difference:  {r2_diff:.3f}")

        if r2_diff < 0.05 and mae_diff < best_val_metrics['MAE'] * 0.1:
            print(f"  ✓ Excellent generalization")
        elif r2_diff < 0.10 and mae_diff < best_val_metrics['MAE'] * 0.2:
            print(f"  ✓ Good generalization")
        else:
            print(f"  ⚠ Check for overfitting")

    # Model comparison summary
    print(f"\nAll Models Ranking (by R² on validation set):")
    print("-"*70)
    for idx, (model_name, row) in enumerate(comparison_df.iterrows(), 1):
        print(f"  {idx}. {row['name']:<30} R²={row['R²']:.3f}  MAE=${row['MAE']:.2f}")

    # Business interpretation
    print("\n" + "="*70)
    print("BUSINESS INTERPRETATION")
    print("="*70)

    avg_rent = y_test.mean()
    mae_pct = (test_metrics['MAE'] / avg_rent) * 100 if test_metrics else (best_val_metrics['MAE'] / avg_rent) * 100

    print(f"\nAverage Rent/SF/Yr: ${avg_rent:.2f}")
    print(f"Prediction Error (MAE): ${test_metrics['MAE'] if test_metrics else best_val_metrics['MAE']:.2f} ({mae_pct:.1f}% of average)")

    print(f"\nPractical Implications:")
    if mae_pct < 10:
        print(f"  ✓ Excellent: Model predictions within 10% of actual values")
        print(f"  ✓ Suitable for commercial real estate valuation")
    elif mae_pct < 15:
        print(f"  ✓ Good: Model provides reliable estimates")
        print(f"  ✓ Useful for portfolio analysis and market trends")
    elif mae_pct < 20:
        print(f"  ⚠ Moderate: Model provides directional guidance")
        print(f"  ⚠ Should be combined with expert judgment")
    else:
        print(f"  ✗ Needs improvement: High prediction error")
        print(f"  ✗ Consider additional features or data quality issues")

    # 8. Save results option
    print("\n" + "="*70)
    print("✓ PHASE 2 COMPLETE: ALL MODELS TRAINED & EVALUATED")
    print("="*70)

    print("\nNext steps:")
    print("  - Phase 3: Model interpretation (SHAP, residual analysis)")
    print("  - Phase 4: Production pipeline (sklearn Pipeline, persistence)")
    print("  - Optional: Hyperparameter tuning (set tune_rf=True, etc.)")

    # Save best model option
    save_option = input("\n\nSave best model to disk? (y/n): ").lower()
    if save_option == 'y':
        import pickle
        from pathlib import Path

        models_dir = Path(__file__).parent / "models"
        models_dir.mkdir(exist_ok=True)

        model_path = models_dir / f"best_model_{best_model_name.lower()}.pkl"

        with open(model_path, 'wb') as f:
            pickle.dump(all_models[best_model_name], f)

        print(f"✓ Saved {best_model_name} to {model_path}")

        # Save comparison results
        results_path = models_dir / "model_comparison.csv"
        comparison_df.to_csv(results_path)
        print(f"✓ Saved comparison results to {results_path}")


if __name__ == "__main__":
    test_phase2_complete()
