"""
Comprehensive test suite for data validation framework.

Tests all validation functions in src/validators.py:
- Schema validation
- Data quality validation
- Business rules validation
- Pipeline integrity validation

Usage:
    python test_data_validation.py
"""

import pandas as pd
import numpy as np
from src.validators import (
    validate_schema,
    validate_data_quality,
    validate_business_rules,
    validate_all,
    quick_validate_clean_data,
    ValidationResult
)
from src.utils import load_clean_data, load_data_splits


def print_section(title):
    """Print section header."""
    print("\n" + "="*70)
    print(title)
    print("="*70)


def print_result(result: ValidationResult):
    """Print validation result details."""
    status = "✓ PASSED" if result.passed else "✗ FAILED"
    print(f"\n{result.validator_name}: {status}")
    print(f"  Errors: {len(result.errors)}")
    print(f"  Warnings: {len(result.warnings)}")

    if result.errors:
        print("\n  Errors:")
        for error in result.errors:
            print(f"    - {error}")

    if result.warnings:
        print("\n  Warnings (first 5):")
        for warning in result.warnings[:5]:
            print(f"    - {warning}")
        if len(result.warnings) > 5:
            print(f"    ... and {len(result.warnings)-5} more")

    if result.info:
        print("\n  Info:")
        for key, value in result.info.items():
            if isinstance(value, dict) and len(value) > 5:
                print(f"    {key}: {dict(list(value.items())[:3])}... ({len(value)} items)")
            elif isinstance(value, (int, float, str)):
                print(f"    {key}: {value}")


# ============================================================================
# TEST 1: Schema Validation
# ============================================================================

def test_schema_validation():
    """Test schema validation with clean data."""
    print_section("TEST 1: SCHEMA VALIDATION")

    print("\n1.1 Testing with clean data...")
    df = load_clean_data()
    result = validate_schema(df, stage='clean')
    print_result(result)

    # Test with missing required columns
    print("\n1.2 Testing with missing required columns...")
    df_missing = df.drop(columns=['Rent/SF/Yr'])
    result_missing = validate_schema(df_missing, stage='clean')
    print_result(result_missing)
    assert not result_missing.passed, "Should fail when required column missing"

    # Test with empty DataFrame
    print("\n1.3 Testing with empty DataFrame...")
    df_empty = pd.DataFrame()
    result_empty = validate_schema(df_empty, stage='clean')
    print_result(result_empty)
    assert not result_empty.passed, "Should fail for empty DataFrame"

    print("\n✓ Schema validation tests completed")
    return result.passed


# ============================================================================
# TEST 2: Data Quality Validation
# ============================================================================

def test_data_quality_validation():
    """Test data quality validation."""
    print_section("TEST 2: DATA QUALITY VALIDATION")

    print("\n2.1 Testing with clean data...")
    df = load_clean_data()
    result = validate_data_quality(df)
    print_result(result)

    # Test with high missing values
    print("\n2.2 Testing with synthetic high missing data...")
    df_missing = df.copy()
    # Create columns with >95% missing
    df_missing['all_null_col'] = np.nan
    df_missing['mostly_null_col'] = [np.nan] * int(0.96 * len(df_missing)) + list(range(int(0.04 * len(df_missing))))

    result_missing = validate_data_quality(df_missing, missing_threshold=0.95)
    print_result(result_missing)
    assert len(result_missing.errors) > 0 or len(result_missing.warnings) > 0, "Should detect high missing values"

    # Test with zero variance
    print("\n2.3 Testing with zero variance columns...")
    df_zero_var = df.head(100).copy()
    df_zero_var['constant_col'] = 42
    df_zero_var['all_same_str'] = 'same_value'

    result_zero_var = validate_data_quality(df_zero_var)
    print_result(result_zero_var)
    assert len(result_zero_var.warnings) > 0, "Should detect zero variance columns"

    # Test with duplicates
    print("\n2.4 Testing with duplicate rows...")
    df_duplicates = pd.concat([df.head(50), df.head(10)], ignore_index=True)
    result_duplicates = validate_data_quality(df_duplicates)
    print_result(result_duplicates)
    assert len(result_duplicates.warnings) > 0, "Should detect duplicate rows"

    print("\n✓ Data quality validation tests completed")
    return True


# ============================================================================
# TEST 3: Business Rules Validation
# ============================================================================

def test_business_rules_validation():
    """Test business rules validation."""
    print_section("TEST 3: BUSINESS RULES VALIDATION")

    print("\n3.1 Testing with clean data...")
    df = load_clean_data()
    result = validate_business_rules(df)
    print_result(result)

    # Test with invalid rent values
    print("\n3.2 Testing with invalid rent values...")
    df_bad_rent = df.head(100).copy()
    df_bad_rent.loc[0:4, 'Rent/SF/Yr'] = -10  # Negative rents
    df_bad_rent.loc[5:9, 'Rent/SF/Yr'] = 500  # Unreasonably high

    result_bad_rent = validate_business_rules(df_bad_rent)
    print_result(result_bad_rent)
    assert len(result_bad_rent.warnings) > 0, "Should detect invalid rent values"

    # Test with invalid coordinates
    print("\n3.3 Testing with invalid coordinates...")
    df_bad_coords = df.head(100).copy()
    df_bad_coords.loc[0:4, 'Latitude'] = 90  # Out of US range
    df_bad_coords.loc[5:9, 'Longitude'] = 0  # Out of US range

    result_bad_coords = validate_business_rules(df_bad_coords)
    print_result(result_bad_coords)
    assert len(result_bad_coords.warnings) > 0, "Should detect invalid coordinates"

    # Test with invalid percent leased
    print("\n3.4 Testing with invalid percent leased...")
    df_bad_pct = df.head(100).copy()
    if 'Percent Leased' in df_bad_pct.columns:
        df_bad_pct.loc[0:4, 'Percent Leased'] = 150  # >100%
        df_bad_pct.loc[5:9, 'Percent Leased'] = -10  # <0%

        result_bad_pct = validate_business_rules(df_bad_pct)
        print_result(result_bad_pct)
        assert len(result_bad_pct.errors) > 0, "Should detect invalid percent leased"
    else:
        print("  'Percent Leased' column not found - skipping test")

    # Test with invalid RBA
    print("\n3.5 Testing with invalid RBA...")
    df_bad_rba = df.head(100).copy()
    if 'RBA' in df_bad_rba.columns:
        df_bad_rba.loc[0:9, 'RBA'] = -100  # Negative RBA

        result_bad_rba = validate_business_rules(df_bad_rba)
        print_result(result_bad_rba)
        assert len(result_bad_rba.errors) > 0, "Should detect negative RBA"
    else:
        print("  'RBA' column not found - skipping test")

    print("\n✓ Business rules validation tests completed")
    return True


# ============================================================================
# TEST 4: Comprehensive Validation
# ============================================================================

def test_comprehensive_validation():
    """Test comprehensive validation function."""
    print_section("TEST 4: COMPREHENSIVE VALIDATION")

    print("\n4.1 Running validate_all on clean data...")
    df = load_clean_data()
    results = validate_all(df, stage='clean', verbose=True)

    print(f"\n  All passed: {results['all_passed']}")
    print(f"  Total errors: {results['total_errors']}")
    print(f"  Total warnings: {results['total_warnings']}")

    # Test quick validation
    print("\n4.2 Testing quick_validate_clean_data...")
    is_valid = quick_validate_clean_data(df)
    print(f"  Quick validation result: {'PASSED' if is_valid else 'FAILED'}")

    print("\n✓ Comprehensive validation tests completed")
    return True


# ============================================================================
# TEST 5: Edge Cases
# ============================================================================

def test_edge_cases():
    """Test edge cases and boundary conditions."""
    print_section("TEST 5: EDGE CASES")

    # Test with minimal DataFrame
    print("\n5.1 Testing with minimal DataFrame...")
    df_minimal = pd.DataFrame({
        'Property Address': ['123 Main St', '456 Oak Ave'],
        'Rent/SF/Yr': [10.5, 12.0],
        'Market Name': ['Market A', 'Market B'],
        'Latitude': [40.0, 41.0],
        'Longitude': [-74.0, -73.0]
    })
    result_minimal = validate_schema(df_minimal, stage='clean')
    print_result(result_minimal)

    # Test with all numeric columns
    print("\n5.2 Testing with all numeric DataFrame...")
    df_numeric = pd.DataFrame(np.random.randn(100, 5), columns=[f'col_{i}' for i in range(5)])
    result_numeric = validate_data_quality(df_numeric)
    print_result(result_numeric)

    # Test with all categorical columns
    print("\n5.3 Testing with all categorical DataFrame...")
    df_categorical = pd.DataFrame({
        f'cat_{i}': np.random.choice(['A', 'B', 'C'], 100) for i in range(5)
    })
    result_categorical = validate_data_quality(df_categorical)
    print_result(result_categorical)

    # Test with single row
    print("\n5.4 Testing with single-row DataFrame...")
    df_single = df_minimal.head(1)
    result_single = validate_data_quality(df_single)
    print_result(result_single)

    print("\n✓ Edge case tests completed")
    return True


# ============================================================================
# TEST 6: Integration Test with Real Data
# ============================================================================

def test_real_data_integration():
    """Integration test with actual clean data."""
    print_section("TEST 6: INTEGRATION TEST WITH REAL DATA")

    try:
        print("\n6.1 Loading clean data...")
        df = load_clean_data()
        print(f"  Loaded {len(df):,} rows, {len(df.columns)} columns")

        print("\n6.2 Running all validations...")
        results = validate_all(df, stage='clean', verbose=False)

        print("\n  Results:")
        print(f"    Overall status: {'PASSED' if results['all_passed'] else 'FAILED'}")
        print(f"    Total validations: {len(results['validations'])}")
        print(f"    Total errors: {results['total_errors']}")
        print(f"    Total warnings: {results['total_warnings']}")

        print("\n  Validation breakdown:")
        for val_name, val_result in results['validations'].items():
            status = "✓" if val_result['passed'] else "✗"
            print(f"    {status} {val_name}: {len(val_result['errors'])} errors, {len(val_result['warnings'])} warnings")

        print("\n6.3 Testing data quality metrics...")
        result = validate_data_quality(df)
        if 'overall_missing_pct' in result.info:
            print(f"  Overall missing rate: {result.info['overall_missing_pct']:.2f}%")
        if 'zero_variance_columns' in result.info:
            print(f"  Zero variance columns: {len(result.info['zero_variance_columns'])}")

        print("\n6.4 Testing business rules...")
        result = validate_business_rules(df)
        if 'rent_min' in result.info:
            print(f"  Rent/SF/Yr range: ${result.info['rent_min']:.2f} - ${result.info['rent_max']:.2f}")

        print("\n✓ Integration test completed")
        return True

    except Exception as e:
        print(f"\n✗ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# MAIN TEST RUNNER
# ============================================================================

def main():
    """Run all validation tests."""
    print("="*70)
    print("DATA VALIDATION TEST SUITE")
    print("="*70)
    print("\nTesting src/validators.py validation framework")

    results = {}

    # Run all tests
    try:
        results['schema'] = test_schema_validation()
    except Exception as e:
        print(f"\n✗ Schema validation tests failed: {e}")
        results['schema'] = False

    try:
        results['quality'] = test_data_quality_validation()
    except Exception as e:
        print(f"\n✗ Quality validation tests failed: {e}")
        results['quality'] = False

    try:
        results['business'] = test_business_rules_validation()
    except Exception as e:
        print(f"\n✗ Business rules tests failed: {e}")
        results['business'] = False

    try:
        results['comprehensive'] = test_comprehensive_validation()
    except Exception as e:
        print(f"\n✗ Comprehensive validation tests failed: {e}")
        results['comprehensive'] = False

    try:
        results['edge_cases'] = test_edge_cases()
    except Exception as e:
        print(f"\n✗ Edge case tests failed: {e}")
        results['edge_cases'] = False

    try:
        results['integration'] = test_real_data_integration()
    except Exception as e:
        print(f"\n✗ Integration test failed: {e}")
        results['integration'] = False

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
    else:
        print("\n✗ SOME TESTS FAILED")
        return 1


if __name__ == "__main__":
    exit(main())
