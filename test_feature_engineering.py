"""
Test script for Phase 1.1 (Imputation) and Phase 1.2 (Geospatial) feature engineering
"""

import pandas as pd
import sys
sys.path.append('.')

from src.utils import load_clean_data
from src.feature_engineering import engineer_features


def test_feature_engineering():
    """Test complete feature engineering pipeline."""
    print("="*70)
    print("TESTING PHASE 1.1 (IMPUTATION) + 1.2 (GEOSPATIAL) + 1.3 (TEMPORAL)")
    print("="*70)

    # Load data
    print("\n1. Loading data...")
    df = load_clean_data()
    print(f"   Loaded {len(df):,} rows, {len(df.columns)} columns")

    # Count nulls before
    print("\n2. Null counts BEFORE feature engineering...")
    key_cols = ['Rent/SF/Yr', 'building_age', 'Parking Ratio', 'Number Of Cranes',
                'Ceiling Ht', 'Latitude', 'Longitude']

    nulls_before = {}
    for col in key_cols:
        if col in df.columns:
            nulls_before[col] = df[col].isna().sum()

    # Apply feature engineering
    print("\n3. Applying feature engineering pipeline...")
    df_before_cols = len(df.columns)
    df = engineer_features(df)
    df_after_cols = len(df.columns)
    print(f"   Columns before: {df_before_cols}")
    print(f"   Columns after:  {df_after_cols}")
    print(f"   New features:   {df_after_cols - df_before_cols}")

    # Verify Phase 1.1: Imputation
    print("\n4. PHASE 1.1 - IMPUTATION RESULTS:")
    print("   " + "-"*66)
    print(f"   {'Column':<30} {'Before':>10} {'After':>10} {'Reduction':>10}")
    print("   " + "-"*66)

    for col in key_cols:
        if col in df.columns:
            before = nulls_before.get(col, 0)
            after = df[col].isna().sum()
            reduction = before - after
            status = "✓" if reduction > 0 else "-"
            print(f"   {status} {col:<28} {before:>10,} {after:>10,} {reduction:>10,}")

    # Verify Phase 1.2: Geospatial features
    print("\n5. PHASE 1.2 - GEOSPATIAL FEATURES:")
    geo_features = ['dist_to_market_center', 'properties_within_1mi', 'properties_within_5mi']

    for feat in geo_features:
        if feat in df.columns:
            non_null = df[feat].notna().sum()
            mean_val = df[feat].mean()
            print(f"   ✓ {feat:<30} non-null={non_null:>6,} mean={mean_val:>10.2f}")
        else:
            print(f"   ✗ {feat} MISSING")

    # Verify Phase 1.3: Temporal features
    print("\n6. PHASE 1.3 - TEMPORAL FEATURES:")
    temporal_features = ['building_age', 'years_since_sale', 'has_sale_data',
                         'sale_month_sin', 'sale_month_cos', 'construction_era']

    for feat in temporal_features:
        if feat in df.columns:
            print(f"   ✓ {feat}")
        else:
            print(f"   ✗ {feat} MISSING")

    # Overall null reduction
    print("\n7. OVERALL NULL REDUCTION:")
    total_nulls_after = df.isnull().sum().sum()
    total_cells = df.shape[0] * df.shape[1]
    null_pct = (total_nulls_after / total_cells) * 100
    print(f"   Total nulls remaining: {total_nulls_after:,} ({null_pct:.1f}% of all cells)")

    # Sample geospatial statistics
    print("\n8. GEOSPATIAL STATISTICS (sample):")
    if 'dist_to_market_center' in df.columns:
        print(f"   Distance to market center:")
        print(f"     Min:    {df['dist_to_market_center'].min():.2f} miles")
        print(f"     Max:    {df['dist_to_market_center'].max():.2f} miles")
        print(f"     Mean:   {df['dist_to_market_center'].mean():.2f} miles")
        print(f"     Median: {df['dist_to_market_center'].median():.2f} miles")

    if 'properties_within_1mi' in df.columns:
        print(f"\n   Properties within 1 mile:")
        print(f"     Min:    {df['properties_within_1mi'].min():.0f}")
        print(f"     Max:    {df['properties_within_1mi'].max():.0f}")
        print(f"     Mean:   {df['properties_within_1mi'].mean():.1f}")

    print("\n" + "="*70)
    print("✓ FEATURE ENGINEERING PHASES 1.1, 1.2, 1.3 COMPLETE")
    print("="*70)


if __name__ == "__main__":
    test_feature_engineering()
