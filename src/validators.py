"""
Data validation framework for CoStar property data pipeline.

This module provides comprehensive validation functions for:
1. Schema validation - verify structure, types, required columns
2. Data quality checks - missing values, outliers, duplicates, variance
3. Business rules validation - domain-specific constraints for commercial real estate
4. Pipeline integrity checks - prevent data leakage, verify stratification

Usage:
    from src.validators import validate_schema, validate_data_quality, validate_business_rules

    # Run all validations
    results = validate_all(df, stage='raw')
    if not results['all_passed']:
        print(results['errors'])
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import warnings


class ValidationResult:
    """Container for validation results."""

    def __init__(self, validator_name: str):
        self.validator_name = validator_name
        self.passed = True
        self.warnings = []
        self.errors = []
        self.info = {}

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

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'validator': self.validator_name,
            'passed': self.passed,
            'errors': self.errors,
            'warnings': self.warnings,
            'info': self.info
        }

    def __repr__(self):
        status = "PASSED" if self.passed else "FAILED"
        return f"ValidationResult({self.validator_name}: {status}, {len(self.errors)} errors, {len(self.warnings)} warnings)"


# ============================================================================
# SCHEMA VALIDATION
# ============================================================================

def validate_schema(df: pd.DataFrame, stage: str = 'raw') -> ValidationResult:
    """
    Validate DataFrame schema (columns, types, structure).

    Args:
        df: DataFrame to validate
        stage: Pipeline stage ('raw', 'clean', 'split', 'engineered')

    Returns:
        ValidationResult object
    """
    from src.column_types import (
        location_columns, categorical_columns, numerical_columns,
        date_columns, year_columns
    )

    result = ValidationResult('schema_validation')

    # Basic structure checks
    if df.empty:
        result.add_error("DataFrame is empty")
        return result

    result.add_info('num_rows', len(df))
    result.add_info('num_columns', len(df.columns))

    # Check for duplicate column names
    dup_cols = df.columns[df.columns.duplicated()].tolist()
    if dup_cols:
        result.add_error(f"Duplicate column names found: {dup_cols}")

    # Stage-specific validation
    if stage == 'raw':
        # Raw data should have all expected columns
        expected_cols = set(location_columns + categorical_columns +
                          numerical_columns + date_columns + year_columns)
        missing_cols = expected_cols - set(df.columns)
        if missing_cols:
            result.add_warning(f"Expected columns missing: {list(missing_cols)[:5]}... ({len(missing_cols)} total)")

    elif stage == 'clean':
        # Clean data should have required columns
        required_cols = ['Property Address', 'Market Name', 'Latitude', 'Longitude', 'Rent/SF/Yr']
        missing_required = set(required_cols) - set(df.columns)
        if missing_required:
            result.add_error(f"Required columns missing: {missing_required}")

    elif stage == 'split':
        # Split data must have target variable
        if 'Rent/SF/Yr' not in df.columns:
            result.add_error("Target variable 'Rent/SF/Yr' not found")

        # Should NOT have Property Address (removed to prevent leakage)
        if 'Property Address' in df.columns:
            result.add_warning("Property Address still present - should be removed after deduplication")

    # Data type validation
    for col in df.columns:
        dtype = df[col].dtype

        # Check for object types (should be categorical or string)
        if dtype == 'object':
            unique_ratio = df[col].nunique() / len(df)
            if unique_ratio > 0.95:
                result.add_warning(f"Column '{col}' has very high cardinality ({unique_ratio:.1%}) - likely an identifier")

        # Check for numeric columns with wrong types
        if col in numerical_columns and dtype not in ['int64', 'float64']:
            result.add_warning(f"Numerical column '{col}' has non-numeric type: {dtype}")

        # Check for date columns
        if col in date_columns and not pd.api.types.is_datetime64_any_dtype(df[col]):
            result.add_info(f'non_datetime_date_col', col)

    return result


# ============================================================================
# DATA QUALITY VALIDATION
# ============================================================================

def validate_data_quality(df: pd.DataFrame,
                          missing_threshold: float = 0.95,
                          variance_threshold: float = 0.0) -> ValidationResult:
    """
    Validate data quality (missing values, variance, outliers).

    Args:
        df: DataFrame to validate
        missing_threshold: Max allowed missing ratio (default 95%)
        variance_threshold: Min required variance (default >0)

    Returns:
        ValidationResult object
    """
    result = ValidationResult('data_quality_validation')

    # 1. Missing value analysis
    missing_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
    high_missing = missing_pct[missing_pct > missing_threshold * 100]

    if len(high_missing) > 0:
        result.add_warning(f"{len(high_missing)} columns with >{missing_threshold*100:.0f}% missing: {high_missing.index.tolist()[:5]}...")
        result.add_info('high_missing_columns', high_missing.to_dict())

    # Overall missing rate
    total_missing_pct = df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100
    result.add_info('overall_missing_pct', round(total_missing_pct, 2))

    if total_missing_pct > 50:
        result.add_warning(f"Overall missing rate is high: {total_missing_pct:.1f}%")

    # 2. Zero variance columns
    zero_var_cols = []
    for col in df.columns:
        if df[col].nunique() == 1:
            zero_var_cols.append(col)

    if zero_var_cols:
        result.add_warning(f"{len(zero_var_cols)} columns with zero variance: {zero_var_cols}")
        result.add_info('zero_variance_columns', zero_var_cols)

    # 3. Duplicate rows
    num_duplicates = df.duplicated().sum()
    if num_duplicates > 0:
        dup_pct = num_duplicates / len(df) * 100
        result.add_warning(f"{num_duplicates} duplicate rows ({dup_pct:.2f}%)")
        result.add_info('num_duplicates', num_duplicates)

    # 4. Check for all-null columns
    all_null_cols = df.columns[df.isnull().all()].tolist()
    if all_null_cols:
        result.add_error(f"Columns with all null values: {all_null_cols}")

    # 5. Check numeric columns for outliers
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    outlier_info = {}

    for col in numeric_cols:
        if df[col].notna().sum() < 10:  # Skip if too few values
            continue

        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        if IQR > 0:  # Only if there's variance
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR

            outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
            outlier_pct = outliers / df[col].notna().sum() * 100

            if outlier_pct > 10:  # More than 10% outliers
                outlier_info[col] = {
                    'count': outliers,
                    'percentage': round(outlier_pct, 2),
                    'bounds': (round(lower_bound, 2), round(upper_bound, 2))
                }

    if outlier_info:
        result.add_info('columns_with_high_outliers', outlier_info)
        result.add_warning(f"{len(outlier_info)} columns with >10% outliers")

    # 6. Check for negative values in columns that should be positive
    positive_columns = ['RBA', 'Land Area (AC)', 'Land Area (SF)', 'Rent/SF/Yr',
                       'For Sale Price', 'Last Sale Price', 'Number Of Stories']

    for col in positive_columns:
        if col in df.columns and df[col].dtype in ['int64', 'float64']:
            negative_count = (df[col] < 0).sum()
            if negative_count > 0:
                result.add_warning(f"Column '{col}' has {negative_count} negative values")

    return result


# ============================================================================
# BUSINESS RULES VALIDATION
# ============================================================================

def validate_business_rules(df: pd.DataFrame) -> ValidationResult:
    """
    Validate domain-specific business rules for commercial real estate data.

    Business rules:
    - Rent/SF/Yr should be between $0.50 and $200/SF/Yr
    - Building age should be reasonable (0-200 years)
    - Percent Leased should be 0-100%
    - Latitude/Longitude should be valid US coordinates
    - RBA (Rentable Building Area) should be positive
    - Number of Stories should be >= 1

    Args:
        df: DataFrame to validate

    Returns:
        ValidationResult object
    """
    result = ValidationResult('business_rules_validation')

    # Rule 1: Rent/SF/Yr should be in reasonable range
    if 'Rent/SF/Yr' in df.columns:
        rent_col = df['Rent/SF/Yr'].dropna()
        if len(rent_col) > 0:
            invalid_rent = ((rent_col < 0.5) | (rent_col > 200)).sum()
            if invalid_rent > 0:
                result.add_warning(f"Rent/SF/Yr: {invalid_rent} values outside reasonable range [$0.50-$200]")
                result.add_info('rent_min', round(rent_col.min(), 2))
                result.add_info('rent_max', round(rent_col.max(), 2))

    # Rule 2: Year Built should be reasonable
    if 'Year Built' in df.columns:
        current_year = datetime.now().year
        year_built = df['Year Built'].dropna()
        if len(year_built) > 0:
            invalid_years = ((year_built < 1800) | (year_built > current_year + 5)).sum()
            if invalid_years > 0:
                result.add_warning(f"Year Built: {invalid_years} values outside range [1800-{current_year+5}]")

    # Rule 3: Percent Leased should be 0-100%
    if 'Percent Leased' in df.columns:
        pct_leased = df['Percent Leased'].dropna()
        if len(pct_leased) > 0:
            invalid_pct = ((pct_leased < 0) | (pct_leased > 100)).sum()
            if invalid_pct > 0:
                result.add_error(f"Percent Leased: {invalid_pct} values outside valid range [0-100]")

    # Rule 4: Latitude/Longitude validation (US coordinates)
    if 'Latitude' in df.columns and 'Longitude' in df.columns:
        lat = df['Latitude'].dropna()
        lon = df['Longitude'].dropna()

        if len(lat) > 0 and len(lon) > 0:
            # US mainland approximately: lat 24-50, lon -125 to -65
            invalid_coords = ((lat < 20) | (lat > 55) | (lon < -130) | (lon > -60)).sum()
            if invalid_coords > 0:
                result.add_warning(f"{invalid_coords} coordinates outside typical US range")

    # Rule 5: RBA should be positive
    if 'RBA' in df.columns:
        rba = df['RBA'].dropna()
        if len(rba) > 0:
            invalid_rba = (rba <= 0).sum()
            if invalid_rba > 0:
                result.add_error(f"RBA: {invalid_rba} values <= 0 (should be positive)")

    # Rule 6: Number of Stories should be >= 1
    if 'Number Of Stories' in df.columns:
        stories = df['Number Of Stories'].dropna()
        if len(stories) > 0:
            invalid_stories = (stories < 1).sum()
            if invalid_stories > 0:
                result.add_warning(f"Number Of Stories: {invalid_stories} values < 1")

    # Rule 7: Property Type should be valid commercial types
    if 'Property Type' in df.columns:
        valid_types = ['Office', 'Industrial', 'Retail', 'Flex', 'Land', 'Specialty', 'Multifamily']
        prop_types = df['Property Type'].dropna()
        if len(prop_types) > 0:
            invalid_types = ~prop_types.isin(valid_types)
            if invalid_types.sum() > 0:
                unique_invalid = prop_types[invalid_types].unique()
                result.add_warning(f"Property Type: {invalid_types.sum()} values not in standard types: {unique_invalid.tolist()[:5]}")

    # Rule 8: Market Name should not be missing
    if 'Market Name' in df.columns:
        missing_market = df['Market Name'].isna().sum()
        if missing_market > 0:
            result.add_warning(f"Market Name: {missing_market} missing values ({missing_market/len(df)*100:.1f}%)")

    # Rule 9: Building Class validation
    if 'Building Class' in df.columns:
        valid_classes = ['A', 'B', 'C', 'unknown']
        bldg_class = df['Building Class'].dropna()
        if len(bldg_class) > 0:
            invalid_class = ~bldg_class.isin(valid_classes)
            if invalid_class.sum() > 0:
                result.add_warning(f"Building Class: {invalid_class.sum()} values not in [A, B, C, unknown]")

    # Rule 10: Logical consistency - Total Available Space <= RBA
    if all(col in df.columns for col in ['Total Available Space (SF)', 'RBA']):
        mask = df[['Total Available Space (SF)', 'RBA']].notna().all(axis=1)
        if mask.sum() > 0:
            inconsistent = (df.loc[mask, 'Total Available Space (SF)'] > df.loc[mask, 'RBA']).sum()
            if inconsistent > 0:
                result.add_warning(f"Logical inconsistency: {inconsistent} properties with Available Space > RBA")

    return result


# ============================================================================
# PIPELINE INTEGRITY VALIDATION
# ============================================================================

def validate_no_data_leakage(train_df: pd.DataFrame,
                            val_df: pd.DataFrame,
                            test_df: pd.DataFrame,
                            key_column: str = 'Property Address') -> ValidationResult:
    """
    Validate that there is no data leakage between train/val/test splits.

    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame
        test_df: Test DataFrame
        key_column: Column to check for overlap (e.g., 'Property Address')

    Returns:
        ValidationResult object
    """
    result = ValidationResult('data_leakage_validation')

    # Check if key column exists
    if key_column not in train_df.columns:
        result.add_error(f"Key column '{key_column}' not found in train set")
        return result

    if key_column not in val_df.columns:
        result.add_error(f"Key column '{key_column}' not found in validation set")
        return result

    if key_column not in test_df.columns:
        result.add_error(f"Key column '{key_column}' not found in test set")
        return result

    # Extract unique identifiers
    train_keys = set(train_df[key_column].dropna())
    val_keys = set(val_df[key_column].dropna())
    test_keys = set(test_df[key_column].dropna())

    result.add_info('train_unique_keys', len(train_keys))
    result.add_info('val_unique_keys', len(val_keys))
    result.add_info('test_unique_keys', len(test_keys))

    # Check for overlaps
    train_val_overlap = train_keys & val_keys
    train_test_overlap = train_keys & test_keys
    val_test_overlap = val_keys & test_keys

    if len(train_val_overlap) > 0:
        result.add_error(f"DATA LEAKAGE: {len(train_val_overlap)} overlapping keys between train and validation")
        result.add_info('train_val_overlap_examples', list(train_val_overlap)[:5])

    if len(train_test_overlap) > 0:
        result.add_error(f"DATA LEAKAGE: {len(train_test_overlap)} overlapping keys between train and test")
        result.add_info('train_test_overlap_examples', list(train_test_overlap)[:5])

    if len(val_test_overlap) > 0:
        result.add_error(f"DATA LEAKAGE: {len(val_test_overlap)} overlapping keys between validation and test")
        result.add_info('val_test_overlap_examples', list(val_test_overlap)[:5])

    if result.passed:
        result.add_info('message', 'No data leakage detected - all splits have unique keys')

    return result


def validate_split_stratification(train_df: pd.DataFrame,
                                  val_df: pd.DataFrame,
                                  test_df: pd.DataFrame,
                                  stratify_column: str = 'Market Name',
                                  max_deviation: float = 0.05) -> ValidationResult:
    """
    Validate that splits maintain proper stratification.

    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame
        test_df: Test DataFrame
        stratify_column: Column used for stratification
        max_deviation: Maximum allowed deviation from overall distribution (default 5%)

    Returns:
        ValidationResult object
    """
    result = ValidationResult('stratification_validation')

    # Check if stratify column exists
    for name, df in [('train', train_df), ('val', val_df), ('test', test_df)]:
        if stratify_column not in df.columns:
            result.add_error(f"Stratification column '{stratify_column}' not found in {name} set")
            return result

    # Combine all data to get overall distribution
    all_data = pd.concat([train_df, val_df, test_df], axis=0)
    overall_dist = all_data[stratify_column].value_counts(normalize=True)

    # Check each split's distribution
    train_dist = train_df[stratify_column].value_counts(normalize=True)
    val_dist = val_df[stratify_column].value_counts(normalize=True)
    test_dist = test_df[stratify_column].value_counts(normalize=True)

    result.add_info('stratify_column', stratify_column)
    result.add_info('num_categories', len(overall_dist))

    # Compare distributions
    max_train_dev = 0
    max_val_dev = 0
    max_test_dev = 0

    for category in overall_dist.index:
        overall_pct = overall_dist[category]
        train_pct = train_dist.get(category, 0)
        val_pct = val_dist.get(category, 0)
        test_pct = test_dist.get(category, 0)

        train_dev = abs(train_pct - overall_pct)
        val_dev = abs(val_pct - overall_pct)
        test_dev = abs(test_pct - overall_pct)

        max_train_dev = max(max_train_dev, train_dev)
        max_val_dev = max(max_val_dev, val_dev)
        max_test_dev = max(max_test_dev, test_dev)

        if train_dev > max_deviation or val_dev > max_deviation or test_dev > max_deviation:
            result.add_warning(
                f"Category '{category}': Overall={overall_pct:.1%}, "
                f"Train={train_pct:.1%}, Val={val_pct:.1%}, Test={test_pct:.1%}"
            )

    result.add_info('max_train_deviation', round(max_train_dev, 4))
    result.add_info('max_val_deviation', round(max_val_dev, 4))
    result.add_info('max_test_deviation', round(max_test_dev, 4))

    if max_train_dev > max_deviation:
        result.add_error(f"Train set stratification deviation too high: {max_train_dev:.2%} > {max_deviation:.2%}")
    if max_val_dev > max_deviation:
        result.add_error(f"Validation set stratification deviation too high: {max_val_dev:.2%} > {max_deviation:.2%}")
    if max_test_dev > max_deviation:
        result.add_error(f"Test set stratification deviation too high: {max_test_dev:.2%} > {max_deviation:.2%}")

    return result


def validate_split_proportions(train_df: pd.DataFrame,
                               val_df: pd.DataFrame,
                               test_df: pd.DataFrame,
                               expected_ratios: Tuple[float, float, float] = (0.6, 0.2, 0.2),
                               tolerance: float = 0.02) -> ValidationResult:
    """
    Validate that split proportions match expected ratios.

    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame
        test_df: Test DataFrame
        expected_ratios: Expected (train, val, test) proportions (default 60/20/20)
        tolerance: Allowed deviation from expected (default 2%)

    Returns:
        ValidationResult object
    """
    result = ValidationResult('split_proportion_validation')

    total = len(train_df) + len(val_df) + len(test_df)

    actual_train = len(train_df) / total
    actual_val = len(val_df) / total
    actual_test = len(test_df) / total

    expected_train, expected_val, expected_test = expected_ratios

    result.add_info('total_samples', total)
    result.add_info('train_size', len(train_df))
    result.add_info('val_size', len(val_df))
    result.add_info('test_size', len(test_df))
    result.add_info('actual_ratios', (round(actual_train, 3), round(actual_val, 3), round(actual_test, 3)))
    result.add_info('expected_ratios', expected_ratios)

    # Check deviations
    train_dev = abs(actual_train - expected_train)
    val_dev = abs(actual_val - expected_val)
    test_dev = abs(actual_test - expected_test)

    if train_dev > tolerance:
        result.add_error(f"Train proportion {actual_train:.1%} deviates from expected {expected_train:.1%} by {train_dev:.1%}")

    if val_dev > tolerance:
        result.add_error(f"Validation proportion {actual_val:.1%} deviates from expected {expected_val:.1%} by {val_dev:.1%}")

    if test_dev > tolerance:
        result.add_error(f"Test proportion {actual_test:.1%} deviates from expected {expected_test:.1%} by {test_dev:.1%}")

    return result


def validate_target_distribution(train_df: pd.DataFrame,
                                 val_df: pd.DataFrame,
                                 test_df: pd.DataFrame,
                                 target_column: str = 'Rent/SF/Yr',
                                 max_mean_deviation: float = 0.10) -> ValidationResult:
    """
    Validate that target variable distribution is similar across splits.

    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame
        test_df: Test DataFrame
        target_column: Target variable column name
        max_mean_deviation: Max allowed deviation in mean (default 10%)

    Returns:
        ValidationResult object
    """
    result = ValidationResult('target_distribution_validation')

    # Check if target exists
    for name, df in [('train', train_df), ('val', val_df), ('test', test_df)]:
        if target_column not in df.columns:
            result.add_error(f"Target column '{target_column}' not found in {name} set")
            return result

    # Calculate statistics
    train_target = train_df[target_column].dropna()
    val_target = val_df[target_column].dropna()
    test_target = test_df[target_column].dropna()

    train_mean = train_target.mean()
    val_mean = val_target.mean()
    test_mean = test_target.mean()

    overall_mean = pd.concat([train_target, val_target, test_target]).mean()

    result.add_info('target_column', target_column)
    result.add_info('train_mean', round(train_mean, 2))
    result.add_info('val_mean', round(val_mean, 2))
    result.add_info('test_mean', round(test_mean, 2))
    result.add_info('overall_mean', round(overall_mean, 2))

    # Check deviations
    train_dev = abs(train_mean - overall_mean) / overall_mean
    val_dev = abs(val_mean - overall_mean) / overall_mean
    test_dev = abs(test_mean - overall_mean) / overall_mean

    result.add_info('train_deviation', round(train_dev, 3))
    result.add_info('val_deviation', round(val_dev, 3))
    result.add_info('test_deviation', round(test_dev, 3))

    if train_dev > max_mean_deviation:
        result.add_warning(f"Train target mean deviates {train_dev:.1%} from overall mean")

    if val_dev > max_mean_deviation:
        result.add_warning(f"Validation target mean deviates {val_dev:.1%} from overall mean")

    if test_dev > max_mean_deviation:
        result.add_warning(f"Test target mean deviates {test_dev:.1%} from overall mean")

    # Check for missing values
    train_missing = train_df[target_column].isna().sum()
    val_missing = val_df[target_column].isna().sum()
    test_missing = test_df[target_column].isna().sum()

    if train_missing > 0:
        result.add_error(f"Train set has {train_missing} missing target values")
    if val_missing > 0:
        result.add_error(f"Validation set has {val_missing} missing target values")
    if test_missing > 0:
        result.add_error(f"Test set has {test_missing} missing target values")

    return result


# ============================================================================
# COMPREHENSIVE VALIDATION
# ============================================================================

def validate_all(df: pd.DataFrame = None,
                train_df: pd.DataFrame = None,
                val_df: pd.DataFrame = None,
                test_df: pd.DataFrame = None,
                stage: str = 'clean',
                verbose: bool = True) -> Dict:
    """
    Run all validation checks and return comprehensive results.

    Args:
        df: Single DataFrame to validate (for schema/quality checks)
        train_df: Training DataFrame (for split validation)
        val_df: Validation DataFrame (for split validation)
        test_df: Test DataFrame (for split validation)
        stage: Pipeline stage ('raw', 'clean', 'split', 'engineered')
        verbose: Print results to console

    Returns:
        Dictionary with all validation results
    """
    results = {
        'timestamp': datetime.now().isoformat(),
        'stage': stage,
        'validations': {},
        'all_passed': True,
        'total_errors': 0,
        'total_warnings': 0
    }

    # Single DataFrame validations
    if df is not None:
        # Schema validation
        schema_result = validate_schema(df, stage=stage)
        results['validations']['schema'] = schema_result.to_dict()
        if not schema_result.passed:
            results['all_passed'] = False
        results['total_errors'] += len(schema_result.errors)
        results['total_warnings'] += len(schema_result.warnings)

        # Data quality validation
        quality_result = validate_data_quality(df)
        results['validations']['data_quality'] = quality_result.to_dict()
        if not quality_result.passed:
            results['all_passed'] = False
        results['total_errors'] += len(quality_result.errors)
        results['total_warnings'] += len(quality_result.warnings)

        # Business rules validation
        business_result = validate_business_rules(df)
        results['validations']['business_rules'] = business_result.to_dict()
        if not business_result.passed:
            results['all_passed'] = False
        results['total_errors'] += len(business_result.errors)
        results['total_warnings'] += len(business_result.warnings)

    # Split validations
    if all(x is not None for x in [train_df, val_df, test_df]):
        # Data leakage validation
        leakage_result = validate_no_data_leakage(train_df, val_df, test_df)
        results['validations']['data_leakage'] = leakage_result.to_dict()
        if not leakage_result.passed:
            results['all_passed'] = False
        results['total_errors'] += len(leakage_result.errors)
        results['total_warnings'] += len(leakage_result.warnings)

        # Stratification validation
        strat_result = validate_split_stratification(train_df, val_df, test_df)
        results['validations']['stratification'] = strat_result.to_dict()
        if not strat_result.passed:
            results['all_passed'] = False
        results['total_errors'] += len(strat_result.errors)
        results['total_warnings'] += len(strat_result.warnings)

        # Proportion validation
        prop_result = validate_split_proportions(train_df, val_df, test_df)
        results['validations']['proportions'] = prop_result.to_dict()
        if not prop_result.passed:
            results['all_passed'] = False
        results['total_errors'] += len(prop_result.errors)
        results['total_warnings'] += len(prop_result.warnings)

        # Target distribution validation
        target_result = validate_target_distribution(train_df, val_df, test_df)
        results['validations']['target_distribution'] = target_result.to_dict()
        if not target_result.passed:
            results['all_passed'] = False
        results['total_errors'] += len(target_result.errors)
        results['total_warnings'] += len(target_result.warnings)

    # Print summary if verbose
    if verbose:
        print("\n" + "="*70)
        print("VALIDATION SUMMARY")
        print("="*70)
        print(f"Stage: {stage}")
        print(f"Status: {'PASSED' if results['all_passed'] else 'FAILED'}")
        print(f"Total Errors: {results['total_errors']}")
        print(f"Total Warnings: {results['total_warnings']}")

        for val_name, val_result in results['validations'].items():
            status = "✓ PASSED" if val_result['passed'] else "✗ FAILED"
            print(f"\n{val_name}: {status}")

            if val_result['errors']:
                print(f"  Errors:")
                for error in val_result['errors']:
                    print(f"    - {error}")

            if val_result['warnings']:
                print(f"  Warnings:")
                for warning in val_result['warnings'][:5]:  # Limit to 5 warnings
                    print(f"    - {warning}")
                if len(val_result['warnings']) > 5:
                    print(f"    ... and {len(val_result['warnings'])-5} more warnings")

    return results


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def quick_validate_clean_data(df: pd.DataFrame) -> bool:
    """
    Quick validation for cleaned data (returns True/False).

    Args:
        df: Cleaned DataFrame

    Returns:
        True if all validations pass, False otherwise
    """
    results = validate_all(df, stage='clean', verbose=False)
    return results['all_passed']


def quick_validate_splits(train_df: pd.DataFrame,
                          val_df: pd.DataFrame,
                          test_df: pd.DataFrame) -> bool:
    """
    Quick validation for data splits (returns True/False).

    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame
        test_df: Test DataFrame

    Returns:
        True if all validations pass, False otherwise
    """
    results = validate_all(train_df=train_df, val_df=val_df, test_df=test_df,
                          stage='split', verbose=False)
    return results['all_passed']
