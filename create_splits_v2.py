"""
Create train/validation/test data splits (FIXED VERSION)

Fixes applied:
1. Deduplicate properties by address to prevent data leakage
2. Drop columns with single value or >95% missing data
3. Add comprehensive data quality checks
4. Use Property Address to ensure no leakage across splits
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from src.utils import load_clean_data, save_data_splits


def analyze_data_quality(df):
    """Analyze data quality and identify columns to drop."""
    print("\n" + "="*70)
    print("DATA QUALITY ANALYSIS")
    print("="*70)

    columns_to_drop = []

    # 1. Columns with only one unique value
    print("\n1. Columns with single value (zero variance):")
    single_value_cols = []
    for col in df.columns:
        if df[col].nunique() == 1:
            single_value_cols.append(col)
            print(f"   - {col}: {df[col].iloc[0]}")

    if single_value_cols:
        columns_to_drop.extend(single_value_cols)
        print(f"   → Dropping {len(single_value_cols)} columns")
    else:
        print("   ✓ No single-value columns found")

    # 2. Columns with >95% missing data
    print("\n2. Columns with >95% missing data:")
    missing_pct = df.isnull().sum() / len(df) * 100
    high_missing_cols = missing_pct[missing_pct > 95].index.tolist()

    if high_missing_cols:
        for col in high_missing_cols:
            print(f"   - {col}: {missing_pct[col]:.1f}% missing")
        columns_to_drop.extend(high_missing_cols)
        print(f"   → Dropping {len(high_missing_cols)} columns")
    else:
        print("   ✓ No columns with >95% missing")

    # 3. High cardinality identifier columns
    print("\n3. High cardinality identifier columns:")
    high_cardinality_cols = []
    for col in df.select_dtypes(include=['object']).columns:
        nunique = df[col].nunique()
        pct_unique = (nunique / len(df)) * 100
        # If >80% of rows have unique value, likely an identifier
        if pct_unique > 80:
            high_cardinality_cols.append(col)
            print(f"   - {col}: {nunique} unique values ({pct_unique:.1f}%)")

    if high_cardinality_cols:
        # Keep Property Address for deduplication, drop after split
        identifiers_to_drop = [c for c in high_cardinality_cols if c != 'Property Address']
        if identifiers_to_drop:
            columns_to_drop.extend(identifiers_to_drop)
            print(f"   → Dropping {len(identifiers_to_drop)} identifier columns")
            print(f"   → Keeping 'Property Address' temporarily for deduplication")
    else:
        print("   ✓ No high cardinality identifiers")

    # Remove duplicates from list
    columns_to_drop = list(set(columns_to_drop))

    print(f"\n→ Total columns to drop: {len(columns_to_drop)}")

    return columns_to_drop


def deduplicate_properties(df):
    """
    Deduplicate properties by address to prevent data leakage.

    Strategy: Keep the most recent listing for each property address.
    """
    print("\n" + "="*70)
    print("DEDUPLICATION")
    print("="*70)

    initial_count = len(df)
    print(f"\nInitial properties: {initial_count:,}")

    # Check for duplicates
    duplicates = df['Property Address'].duplicated(keep=False).sum()
    print(f"Properties with duplicate addresses: {duplicates:,}")

    if duplicates == 0:
        print("✓ No duplicates found")
        return df

    # Show duplicate examples
    dup_addresses = df[df['Property Address'].duplicated(keep=False)]['Property Address'].value_counts()
    print(f"\nTop 5 most duplicated addresses:")
    for addr, count in dup_addresses.head(5).items():
        print(f"  {addr}: {count} listings")

    # Deduplication strategy: Keep most recent based on Last Sale Date, or first if no date
    print("\nDeduplication strategy: Keep most recent listing per address")
    print("  - Primary: Use Last Sale Date (most recent)")
    print("  - Fallback: Keep first occurrence if no date available")

    # Sort by Last Sale Date (nulls last), then drop duplicates
    df_sorted = df.sort_values('Last Sale Date', ascending=False, na_position='last')
    df_dedup = df_sorted.drop_duplicates(subset='Property Address', keep='first')

    final_count = len(df_dedup)
    removed = initial_count - final_count

    print(f"\nAfter deduplication:")
    print(f"  Unique properties: {final_count:,}")
    print(f"  Duplicates removed: {removed:,} ({removed/initial_count*100:.1f}%)")

    return df_dedup.reset_index(drop=True)


def create_data_splits_v2(target_column='Rent/SF/Yr', test_size=0.2, val_size=0.2, random_state=42):
    """
    Create stratified train/validation/test splits with proper data quality controls.

    Args:
        target_column: Name of target variable
        test_size: Proportion for test set (0.2 = 20%)
        val_size: Proportion for validation set (0.2 = 20%)
        random_state: Random seed for reproducibility

    Returns:
        tuple: (train_df, val_df, test_df)
    """
    print("="*70)
    print("CREATING TRAIN/VALIDATION/TEST SPLITS (V2 - FIXED)")
    print("="*70)

    # Load data
    print("\n1. Loading cleaned data...")
    df = load_clean_data()
    print(f"   Total rows: {len(df):,}")
    print(f"   Total columns: {len(df.columns)}")

    # Data quality analysis
    columns_to_drop = analyze_data_quality(df)

    # Drop low-quality columns
    print("\n" + "="*70)
    print("DROPPING LOW-QUALITY COLUMNS")
    print("="*70)
    df_cleaned = df.drop(columns=columns_to_drop)
    print(f"\nColumns after dropping: {len(df_cleaned.columns)}")
    print(f"Columns dropped: {len(columns_to_drop)}")

    # Filter to properties with target variable
    print("\n" + "="*70)
    print("FILTERING BY TARGET VARIABLE")
    print("="*70)
    print(f"\nTarget: {target_column}")
    df_filtered = df_cleaned[df_cleaned[target_column].notna()].copy()
    print(f"Properties with target: {len(df_filtered):,}")
    print(f"Properties dropped: {len(df_cleaned) - len(df_filtered):,}")

    # Deduplicate properties
    df_dedup = deduplicate_properties(df_filtered)

    # Prepare for stratified split
    print("\n" + "="*70)
    print("PREPARING FOR STRATIFIED SPLIT")
    print("="*70)

    # Fill missing Market Names for stratification
    missing_markets = df_dedup['Market Name'].isna().sum()
    if missing_markets > 0:
        print(f"\nFilling {missing_markets} missing Market Names with 'Unknown'")
        df_dedup['Market Name'] = df_dedup['Market Name'].fillna('Unknown Market')

    # Check market distribution
    market_counts = df_dedup['Market Name'].value_counts()
    print(f"\nMarket distribution ({len(market_counts)} markets):")
    for market, count in market_counts.items():
        pct = (count / len(df_dedup)) * 100
        print(f"  {market}: {count:,} ({pct:.1f}%)")

    # Create splits with stratification
    print("\n" + "="*70)
    print("CREATING SPLITS")
    print("="*70)

    # First split: separate test set (20%)
    print(f"\n1. Splitting train+val / test ({100-test_size*100:.0f}% / {test_size*100:.0f}%)...")
    train_val, test = train_test_split(
        df_dedup,
        test_size=test_size,
        random_state=random_state,
        stratify=df_dedup['Market Name']
    )
    print(f"   Train+Val: {len(train_val):,}")
    print(f"   Test:      {len(test):,}")

    # Second split: separate validation from training
    val_proportion = val_size / (1 - test_size)
    print(f"\n2. Splitting train / validation ({(1-val_proportion)*100:.0f}% / {val_proportion*100:.0f}%)...")
    train, val = train_test_split(
        train_val,
        test_size=val_proportion,
        random_state=random_state,
        stratify=train_val['Market Name']
    )
    print(f"   Train:      {len(train):,}")
    print(f"   Validation: {len(val):,}")

    # Verify no data leakage
    print("\n" + "="*70)
    print("VERIFYING NO DATA LEAKAGE")
    print("="*70)

    train_addr = set(train['Property Address'])
    val_addr = set(val['Property Address'])
    test_addr = set(test['Property Address'])

    train_val_overlap = train_addr & val_addr
    train_test_overlap = train_addr & test_addr
    val_test_overlap = val_addr & test_addr

    print(f"\nAddress overlap check:")
    print(f"  Train-Val overlap: {len(train_val_overlap)}")
    print(f"  Train-Test overlap: {len(train_test_overlap)}")
    print(f"  Val-Test overlap: {len(val_test_overlap)}")

    if len(train_val_overlap) == 0 and len(train_test_overlap) == 0 and len(val_test_overlap) == 0:
        print("\n  ✓ NO DATA LEAKAGE DETECTED!")
    else:
        print("\n  ⚠️  WARNING: Data leakage still present!")
        raise ValueError("Data leakage detected after deduplication - investigate!")

    # Verify split proportions
    print("\n" + "="*70)
    print("SPLIT VERIFICATION")
    print("="*70)

    total = len(train) + len(val) + len(test)
    print(f"\nSplit sizes:")
    print(f"  Total:      {total:,} properties")
    print(f"  Train:      {len(train):,} ({len(train)/total*100:.1f}%)")
    print(f"  Validation: {len(val):,} ({len(val)/total*100:.1f}%)")
    print(f"  Test:       {len(test):,} ({len(test)/total*100:.1f}%)")

    # Verify market stratification
    print(f"\nMarket stratification:")
    print(f"  {'Market':<25} {'Train %':>10} {'Val %':>10} {'Test %':>10}")
    print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*10}")
    for market in market_counts.index:
        train_pct = (train['Market Name'] == market).sum() / len(train) * 100
        val_pct = (val['Market Name'] == market).sum() / len(val) * 100
        test_pct = (test['Market Name'] == market).sum() / len(test) * 100
        print(f"  {market:<25} {train_pct:>9.1f}% {val_pct:>9.1f}% {test_pct:>9.1f}%")

    # Verify target distribution
    print(f"\nTarget variable ('{target_column}') distribution:")
    print(f"  {'Split':<15} {'Mean':>12} {'Median':>12} {'Std':>12}")
    print(f"  {'-'*15} {'-'*12} {'-'*12} {'-'*12}")
    for name, split_df in [('Train', train), ('Validation', val), ('Test', test)]:
        mean_val = split_df[target_column].mean()
        median_val = split_df[target_column].median()
        std_val = split_df[target_column].std()
        print(f"  {name:<15} ${mean_val:>11.2f} ${median_val:>11.2f} ${std_val:>11.2f}")

    # Save splits
    print("\n" + "="*70)
    print("SAVING SPLITS")
    print("="*70)
    save_data_splits(train, val, test)

    print("\n" + "="*70)
    print("✓ SPLITS CREATED SUCCESSFULLY (NO DATA LEAKAGE)")
    print("="*70)

    # Summary
    print(f"\nKey changes from v1:")
    print(f"  - Removed {len(columns_to_drop)} low-quality columns")
    print(f"  - Deduplicated {len(df_filtered) - len(df_dedup):,} duplicate properties")
    print(f"  - Verified zero address overlap between splits")
    print(f"  - Final dataset: {total:,} unique properties, {len(train.columns)} features")

    return train, val, test


if __name__ == "__main__":
    train, val, test = create_data_splits_v2()
