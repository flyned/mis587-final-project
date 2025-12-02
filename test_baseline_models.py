"""
Test script for Phase 2: Baseline Model Training
"""

import pandas as pd
import numpy as np
import sys
sys.path.append('.')

from sklearn.model_selection import train_test_split
from src.utils import load_clean_data
from src.feature_engineering import engineer_features
from src.modeling import prepare_data_for_modeling, train_baseline_models


def test_baseline_models():
    """Test baseline model training pipeline."""
    print("="*70)
    print("PHASE 2: BASELINE MODEL TRAINING")
    print("="*70)

    # Load and engineer features
    print("\n1. Loading data and applying feature engineering...")
    df = load_clean_data()
    df = engineer_features(df)
    print(f"   Total rows: {len(df):,}")
    print(f"   Total features: {len(df.columns)}")

    # Prepare data
    print("\n2. Preparing data for modeling...")
    X, y, feature_names, categorical_cols = prepare_data_for_modeling(df, target_column='Rent/SF/Yr')
    print(f"   Target: Rent/SF/Yr")
    print(f"   Samples: {len(y):,}")
    print(f"   Features: {len(feature_names)}")
    print(f"   Categorical cols encoded: {len(categorical_cols)}")

    # Create train/val/test splits
    print("\n3. Creating train/validation/test splits...")
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42  # 0.25 x 0.8 = 0.2
    )

    print(f"   Train:      {len(X_train):,} samples ({len(X_train)/len(y)*100:.1f}%)")
    print(f"   Validation: {len(X_val):,} samples ({len(X_val)/len(y)*100:.1f}%)")
    print(f"   Test:       {len(X_test):,} samples ({len(X_test)/len(y)*100:.1f}%)")

    # Train baseline models
    print("\n" + "="*70)
    print("4. TRAINING BASELINE MODELS")
    print("="*70)

    models, results = train_baseline_models(X_train, y_train, X_val, y_val)

    # Compare models
    print("\n" + "="*70)
    print("5. MODEL COMPARISON (VALIDATION SET)")
    print("="*70)

    results_df = pd.DataFrame(results).T
    results_df = results_df.sort_values('R²', ascending=False)

    print(f"\n{'Model':<25} {'MAE':>10} {'RMSE':>10} {'R²':>8} {'MAPE':>8} {'±10%':>8}")
    print("-"*70)

    for idx, row in results_df.iterrows():
        print(f"{row['name']:<25} ${row['MAE']:>9,.2f} ${row['RMSE']:>9,.2f} "
              f"{row['R²']:>7.3f} {row['MAPE']:>7.1f}% {row['Within_10pct']:>7.1%}")

    # Identify best model
    print("\n" + "="*70)
    print("6. BEST MODEL")
    print("="*70)

    best_model_name = results_df.index[0]
    best_metrics = results_df.iloc[0]

    print(f"\nBest model by R²: {best_metrics['name']}")
    print(f"  R² Score: {best_metrics['R²']:.3f}")
    print(f"  MAE: ${best_metrics['MAE']:.2f}")
    print(f"  MAPE: {best_metrics['MAPE']:.1f}%")

    # Interpret results
    print("\n" + "="*70)
    print("7. INTERPRETATION")
    print("="*70)

    avg_rent = y_val.mean()
    print(f"\nAverage Rent/SF/Yr: ${avg_rent:.2f}")
    print(f"Best MAE: ${best_metrics['MAE']:.2f} ({best_metrics['MAE']/avg_rent*100:.1f}% of avg)")

    if best_metrics['R²'] > 0.5:
        print(f"\n✓ Strong baseline performance (R² = {best_metrics['R²']:.3f})")
    elif best_metrics['R²'] > 0.3:
        print(f"\n⚠ Moderate baseline performance (R² = {best_metrics['R²']:.3f})")
    else:
        print(f"\n✗ Weak baseline performance (R² = {best_metrics['R²']:.3f})")

    print("\n" + "="*70)
    print("✓ PHASE 2 BASELINE MODELS COMPLETE")
    print("="*70)
    print("\nNext steps:")
    print("  - Analyze feature importance")
    print("  - Try advanced models (Random Forest, XGBoost)")
    print("  - Consider additional feature engineering")


if __name__ == "__main__":
    test_baseline_models()
