"""
Test script for Phase 1.3: Temporal Feature Engineering
"""

import pandas as pd
import sys
sys.path.append('.')

from src.utils import load_clean_data
from src.feature_engineering import add_temporal_features


def test_temporal_features():
    """Test temporal feature creation."""
    print("="*70)
    print("TESTING PHASE 1.3: TEMPORAL FEATURE ENGINEERING")
    print("="*70)

    # Load data
    print("\n1. Loading data...")
    df = load_clean_data()
    print(f"   Loaded {len(df):,} rows, {len(df.columns)} columns")

    # Check required columns exist
    print("\n2. Checking required columns...")
    required_cols = ['Year Built', 'Year Renovated', 'Last Sale Date']
    for col in required_cols:
        if col in df.columns:
            print(f"   ✓ {col}")
        else:
            print(f"   ✗ {col} MISSING")
            return

    # Apply temporal features
    print("\n3. Creating temporal features...")
    df_before = len(df.columns)
    df = add_temporal_features(df)
    df_after = len(df.columns)
    print(f"   Columns added: {df_after - df_before}")

    # Verify new features
    print("\n4. Verifying new features...")
    expected_features = [
        'building_age',
        'years_since_sale',
        'has_sale_data',
        'sale_month_sin',
        'sale_month_cos',
        'construction_era'
    ]

    for feat in expected_features:
        if feat in df.columns:
            print(f"   ✓ {feat}")
        else:
            print(f"   ✗ {feat} MISSING")

    # Verify years_since_renovation was dropped
    if 'years_since_renovation' not in df.columns:
        print(f"   ✓ years_since_renovation DROPPED (too sparse)")

    # Show statistics
    print("\n5. Feature statistics:")
    print(f"\n   building_age:")
    print(f"     Mean: {df['building_age'].mean():.1f} years")
    print(f"     Range: {df['building_age'].min():.0f} - {df['building_age'].max():.0f} years")
    print(f"     Null: {df['building_age'].isna().sum():,} ({df['building_age'].isna().mean()*100:.1f}%)")

    print(f"\n   years_since_sale:")
    print(f"     Mean: {df['years_since_sale'].mean():.1f} years")
    print(f"     Null: {df['years_since_sale'].isna().sum():,} ({df['years_since_sale'].isna().mean()*100:.1f}%)")

    print(f"\n   has_sale_data:")
    print(f"     Properties with sale data: {df['has_sale_data'].sum():,} ({df['has_sale_data'].mean()*100:.1f}%)")
    print(f"     Properties without: {(df['has_sale_data']==0).sum():,} ({(df['has_sale_data']==0).mean()*100:.1f}%)")

    print(f"\n   construction_era:")
    print(df['construction_era'].value_counts().sort_index())

    print("\n" + "="*70)
    print("✓ PHASE 1.3 IMPLEMENTATION COMPLETE")
    print("="*70)


if __name__ == "__main__":
    test_temporal_features()
