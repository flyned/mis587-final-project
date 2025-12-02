"""
Test script for Complete Phase 1: Feature Engineering Pipeline

Tests all Phase 1 components:
- Phase 1.3: Advanced missing data imputation
- Phase 1.2: Geospatial features
- Phase 1.3: Temporal features
- Phase 1.4: Numerical transformations
- Phase 1.5: Categorical encoding
- Phase 1.6: Outlier treatment
"""

import pandas as pd
import numpy as np
import sys
sys.path.append('.')

from src.utils import load_train_data, load_val_data, load_test_data
from src.feature_engineering import engineer_features


def analyze_missing_data(df, name="Dataset"):
    """Analyze missing data patterns."""
    print(f"\n{name} Missing Data:")
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if len(missing) > 0:
        print(f"  Columns with nulls: {len(missing)}")
        for col in missing.head(5).index:
            pct = missing[col] / len(df) * 100
            print(f"    - {col}: {missing[col]:,} ({pct:.1f}%)")
        if len(missing) > 5:
            print(f"    ... and {len(missing) - 5} more")
    else:
        print("  ✓ No missing values!")


def analyze_new_features(df_before, df_after):
    """Analyze features added during engineering."""
    print("\nNew Features Added:")

    new_cols = set(df_after.columns) - set(df_before.columns)
    print(f"  Total new features: {len(new_cols)}")

    # Categorize new features
    categories = {
        'Temporal': [c for c in new_cols if any(x in c.lower() for x in ['age', 'sale', 'year', 'era'])],
        'Geospatial': [c for c in new_cols if any(x in c for x in ['dist_', 'properties_within'])],
        'Log transforms': [c for c in new_cols if c.startswith('log_')],
        'Ratios': [c for c in new_cols if any(x in c for x in ['ratio', 'per_', 'avg_floor'])],
        'Interactions': [c for c in new_cols if 'interaction' in c],
        'Missing indicators': [c for c in new_cols if 'was_missing' in c],
        'Outlier indicators': [c for c in new_cols if 'is_outlier' in c],
        'Encodings': [c for c in new_cols if any(x in c for x in ['_target_encoded', '_frequency'])]
    }

    for category, cols in categories.items():
        if cols:
            print(f"\n  {category}: {len(cols)}")
            for col in sorted(cols)[:3]:  # Show first 3
                print(f"    - {col}")
            if len(cols) > 3:
                print(f"    ... and {len(cols) - 3} more")


def validate_transformations(df):
    """Validate specific transformations worked correctly."""
    print("\nValidation Checks:")

    checks_passed = 0
    checks_total = 0

    # Check 1: Log transforms are non-negative
    log_cols = [c for c in df.columns if c.startswith('log_')]
    if log_cols:
        checks_total += 1
        all_non_negative = all(df[col].min() >= 0 for col in log_cols)
        if all_non_negative:
            print("  ✓ All log transforms are non-negative")
            checks_passed += 1
        else:
            print("  ✗ Some log transforms have negative values")

    # Check 2: Ratios are reasonable (not infinite)
    ratio_cols = [c for c in df.columns if any(x in c for x in ['ratio', 'per_'])]
    if ratio_cols:
        checks_total += 1
        all_finite = all(np.isfinite(df[col]).all() for col in ratio_cols)
        if all_finite:
            print("  ✓ All ratio features are finite")
            checks_passed += 1
        else:
            print("  ✗ Some ratio features have inf/nan values")

    # Check 3: Outlier indicators are binary
    outlier_cols = [c for c in df.columns if 'is_outlier' in c]
    if outlier_cols:
        checks_total += 1
        all_binary = all(df[col].isin([0, 1]).all() for col in outlier_cols)
        if all_binary:
            print("  ✓ All outlier indicators are binary (0/1)")
            checks_passed += 1
        else:
            print("  ✗ Some outlier indicators have non-binary values")

    # Check 4: Missing indicators are binary
    missing_cols = [c for c in df.columns if 'was_missing' in c]
    if missing_cols:
        checks_total += 1
        all_binary = all(df[col].isin([0, 1]).all() for col in missing_cols)
        if all_binary:
            print("  ✓ All missing indicators are binary (0/1)")
            checks_passed += 1
        else:
            print("  ✗ Some missing indicators have non-binary values")

    # Check 5: Target encoding values are reasonable
    target_encoded_cols = [c for c in df.columns if '_target_encoded' in c]
    if target_encoded_cols:
        checks_total += 1
        # Should be roughly in range of target variable
        target_min = df['Rent/SF/Yr'].min() if 'Rent/SF/Yr' in df.columns else 0
        target_max = df['Rent/SF/Yr'].max() if 'Rent/SF/Yr' in df.columns else 100
        all_reasonable = all(
            df[col].min() >= target_min * 0.5 and df[col].max() <= target_max * 1.5
            for col in target_encoded_cols
        )
        if all_reasonable:
            print("  ✓ Target encoded features are in reasonable range")
            checks_passed += 1
        else:
            print("  ⚠ Some target encoded features outside expected range")

    print(f"\n  Total: {checks_passed}/{checks_total} checks passed")
    return checks_passed == checks_total


def test_phase1_complete():
    """Test complete Phase 1 feature engineering pipeline."""
    print("="*70)
    print("PHASE 1 COMPLETE: FEATURE ENGINEERING VALIDATION")
    print("="*70)

    # Load pre-made splits (from Phase 1.2)
    print("\n1. Loading pre-made train/val/test splits...")
    train = load_train_data()
    val = load_val_data()
    test = load_test_data()

    print(f"   Train: {len(train):,} rows, {len(train.columns)} columns")
    print(f"   Val:   {len(val):,} rows, {len(val.columns)} columns")
    print(f"   Test:  {len(test):,} rows, {len(test.columns)} columns")

    # Analyze missing data before
    analyze_missing_data(train, "Train (before)")

    # Apply feature engineering
    print("\n" + "="*70)
    print("2. APPLYING FEATURE ENGINEERING PIPELINE")
    print("="*70)

    train_engineered = engineer_features(train.copy())

    # Analyze results
    print("\n" + "="*70)
    print("3. ANALYZING RESULTS")
    print("="*70)

    analyze_new_features(train, train_engineered)
    analyze_missing_data(train_engineered, "Train (after)")

    # Validate transformations
    print("\n" + "="*70)
    print("4. VALIDATING TRANSFORMATIONS")
    print("="*70)

    all_valid = validate_transformations(train_engineered)

    # Summary statistics
    print("\n" + "="*70)
    print("5. SUMMARY")
    print("="*70)

    print(f"\nOriginal features: {len(train.columns)}")
    print(f"Engineered features: {len(train_engineered.columns)}")
    print(f"Features added: {len(train_engineered.columns) - len(train.columns)}")
    print(f"Percentage increase: {(len(train_engineered.columns) - len(train.columns)) / len(train.columns) * 100:.1f}%")

    # Feature breakdown
    print("\nFeature type breakdown:")
    numeric_cols = train_engineered.select_dtypes(include=[np.number]).columns
    categorical_cols = train_engineered.select_dtypes(include=['object']).columns
    print(f"  Numeric: {len(numeric_cols)}")
    print(f"  Categorical: {len(categorical_cols)}")

    # Target variable stats
    if 'Rent/SF/Yr' in train_engineered.columns:
        target = train_engineered['Rent/SF/Yr']
        print(f"\nTarget variable (Rent/SF/Yr):")
        print(f"  Mean: ${target.mean():.2f}")
        print(f"  Median: ${target.median():.2f}")
        print(f"  Std: ${target.std():.2f}")
        print(f"  Range: ${target.min():.2f} - ${target.max():.2f}")
        print(f"  Missing: {target.isnull().sum()} ({target.isnull().sum()/len(target)*100:.1f}%)")

    # Final verdict
    print("\n" + "="*70)
    if all_valid:
        print("✓ PHASE 1 COMPLETE: ALL VALIDATIONS PASSED")
    else:
        print("⚠ PHASE 1 COMPLETE: SOME VALIDATIONS FAILED (see above)")
    print("="*70)

    print("\nNext steps:")
    print("  - Apply same transformations to val/test sets")
    print("  - Begin Phase 2: Model training")
    print("  - Try Random Forest and XGBoost models")

    # Save engineered dataset option
    save_option = input("\n\nSave engineered train data to CSV? (y/n): ").lower()
    if save_option == 'y':
        from pathlib import Path
        data_path = Path(__file__).parent.parent / "data"
        output_path = data_path / "train_engineered.csv"
        train_engineered.to_csv(output_path, index=False)
        print(f"✓ Saved to {output_path}")


if __name__ == "__main__":
    test_phase1_complete()
