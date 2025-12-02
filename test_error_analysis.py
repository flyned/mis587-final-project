"""
Comprehensive error analysis test suite for model predictions.

Analyzes model errors by:
1. Market and property type segmentation
2. Price range analysis
3. Systematic bias detection
4. Error correlation with features
5. Worst/best predictions identification

Usage:
    python test_error_analysis.py
"""

import pandas as pd
import numpy as np
import sys
sys.path.append('.')

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.utils import load_train_data, load_val_data, load_test_data
from src.feature_engineering import engineer_features
from src.modeling import prepare_data_for_modeling


def print_section(title):
    """Print section header."""
    print("\n" + "="*70)
    print(title)
    print("="*70)


# ============================================================================
# TEST 1: Error Analysis by Market
# ============================================================================

def test_error_by_market(df, y_true, y_pred):
    """Analyze prediction errors by market."""
    print_section("TEST 1: ERROR ANALYSIS BY MARKET")

    df_analysis = df.copy()
    df_analysis['y_true'] = y_true
    df_analysis['y_pred'] = y_pred
    df_analysis['error'] = y_true - y_pred
    df_analysis['abs_error'] = np.abs(df_analysis['error'])
    df_analysis['pct_error'] = (df_analysis['abs_error'] / df_analysis['y_true'] * 100)

    if 'Market Name' not in df_analysis.columns:
        print("  Market Name not available - skipping market analysis")
        return True

    print("\n1.1 Error metrics by market:")
    market_stats = df_analysis.groupby('Market Name').agg({
        'abs_error': ['mean', 'median', 'std'],
        'pct_error': 'mean',
        'error': 'mean',  # Systematic bias
        'y_true': 'count'
    }).round(3)

    market_stats.columns = ['MAE', 'Median_AE', 'Std_AE', 'MAPE', 'Bias', 'Count']
    market_stats = market_stats.sort_values('MAE', ascending=False)

    print("\n  Top 5 markets with highest errors:")
    print(market_stats.head(5).to_string())

    print("\n  Top 5 markets with lowest errors:")
    print(market_stats.tail(5).to_string())

    # Check for markets with systematic bias
    high_bias = market_stats[np.abs(market_stats['Bias']) > 0.5]
    if len(high_bias) > 0:
        print(f"\n  ⚠️  {len(high_bias)} markets with systematic bias (|bias| > $0.50):")
        for market, row in high_bias.iterrows():
            direction = "overestimating" if row['Bias'] < 0 else "underestimating"
            print(f"    {market}: {direction} by ${abs(row['Bias']):.2f}")

    return True


# ============================================================================
# TEST 2: Error Analysis by Property Type
# ============================================================================

def test_error_by_property_type(df, y_true, y_pred):
    """Analyze prediction errors by property type."""
    print_section("TEST 2: ERROR ANALYSIS BY PROPERTY TYPE")

    df_analysis = df.copy()
    df_analysis['y_true'] = y_true
    df_analysis['y_pred'] = y_pred
    df_analysis['error'] = y_true - y_pred
    df_analysis['abs_error'] = np.abs(df_analysis['error'])
    df_analysis['pct_error'] = (df_analysis['abs_error'] / df_analysis['y_true'] * 100)

    if 'Property Type' not in df_analysis.columns:
        print("  Property Type not available - skipping property type analysis")
        return True

    print("\n2.1 Error metrics by property type:")
    type_stats = df_analysis.groupby('Property Type').agg({
        'abs_error': ['mean', 'median', 'std'],
        'pct_error': 'mean',
        'error': 'mean',
        'y_true': 'count'
    }).round(3)

    type_stats.columns = ['MAE', 'Median_AE', 'Std_AE', 'MAPE', 'Bias', 'Count']
    type_stats = type_stats.sort_values('MAE', ascending=False)

    print(type_stats.to_string())

    # Identify problematic property types
    worst_type = type_stats['MAE'].idxmax()
    best_type = type_stats['MAE'].idxmin()

    print(f"\n  Worst performing: {worst_type} (MAE: ${type_stats.loc[worst_type, 'MAE']:.3f})")
    print(f"  Best performing: {best_type} (MAE: ${type_stats.loc[best_type, 'MAE']:.3f})")

    return True


# ============================================================================
# TEST 3: Error Analysis by Price Range
# ============================================================================

def test_error_by_price_range(y_true, y_pred):
    """Analyze prediction errors by actual price range."""
    print_section("TEST 3: ERROR ANALYSIS BY PRICE RANGE")

    df_analysis = pd.DataFrame({
        'y_true': y_true,
        'y_pred': y_pred,
        'error': y_true - y_pred,
        'abs_error': np.abs(y_true - y_pred),
        'pct_error': np.abs(y_true - y_pred) / y_true * 100
    })

    # Create price bins
    bins = [0, 8, 10, 12, 15, 20, 200]
    labels = ['<$8', '$8-10', '$10-12', '$12-15', '$15-20', '>$20']
    df_analysis['price_range'] = pd.cut(df_analysis['y_true'], bins=bins, labels=labels)

    print("\n3.1 Error metrics by price range:")
    range_stats = df_analysis.groupby('price_range').agg({
        'abs_error': ['mean', 'median', 'std'],
        'pct_error': 'mean',
        'error': 'mean',
        'y_true': 'count'
    }).round(3)

    range_stats.columns = ['MAE', 'Median_AE', 'Std_AE', 'MAPE', 'Bias', 'Count']

    print(range_stats.to_string())

    # Check for systematic patterns
    print("\n3.2 Bias analysis by price range:")
    for price_range, row in range_stats.iterrows():
        if abs(row['Bias']) > 0.3:
            direction = "overestimating" if row['Bias'] < 0 else "underestimating"
            print(f"  {price_range}: {direction} by ${abs(row['Bias']):.2f} ({abs(row['Bias']/row['MAE'])*100:.1f}% of MAE)")

    # Check if errors increase with price
    correlation = df_analysis['y_true'].corr(df_analysis['abs_error'])
    print(f"\n  Correlation between price and absolute error: {correlation:.3f}")
    if abs(correlation) > 0.3:
        print(f"  ⚠️  {'Strong positive' if correlation > 0 else 'Strong negative'} correlation detected")

    return True


# ============================================================================
# TEST 4: Systematic Bias Detection
# ============================================================================

def test_systematic_bias(y_true, y_pred):
    """Detect systematic biases in predictions."""
    print_section("TEST 4: SYSTEMATIC BIAS DETECTION")

    errors = y_true - y_pred
    abs_errors = np.abs(errors)

    # Overall bias
    mean_error = np.mean(errors)
    median_error = np.median(errors)

    print(f"\n4.1 Overall bias metrics:")
    print(f"  Mean error: ${mean_error:.3f}")
    print(f"  Median error: ${median_error:.3f}")

    if abs(mean_error) > 0.1:
        direction = "overestimating" if mean_error < 0 else "underestimating"
        print(f"  ⚠️  Systematic bias detected: {direction} by ${abs(mean_error):.3f}")
    else:
        print(f"  ✓ No significant systematic bias")

    # Overestimation vs underestimation
    overestimated = (errors < 0).sum()
    underestimated = (errors > 0).sum()
    total = len(errors)

    print(f"\n4.2 Direction of errors:")
    print(f"  Overestimated: {overestimated} ({overestimated/total*100:.1f}%)")
    print(f"  Underestimated: {underestimated} ({underestimated/total*100:.1f}%)")

    if abs(overestimated - underestimated) / total > 0.1:
        print(f"  ⚠️  Imbalanced error direction")

    # Error magnitude distribution
    print(f"\n4.3 Error magnitude distribution:")
    print(f"  Mean absolute error: ${np.mean(abs_errors):.3f}")
    print(f"  Median absolute error: ${np.median(abs_errors):.3f}")
    print(f"  Std dev of errors: ${np.std(errors):.3f}")
    print(f"  25th percentile: ${np.percentile(abs_errors, 25):.3f}")
    print(f"  75th percentile: ${np.percentile(abs_errors, 75):.3f}")
    print(f"  95th percentile: ${np.percentile(abs_errors, 95):.3f}")

    # Large errors
    large_errors = (abs_errors > 3).sum()
    print(f"\n  Properties with error >$3.00: {large_errors} ({large_errors/total*100:.1f}%)")

    very_large_errors = (abs_errors > 5).sum()
    print(f"  Properties with error >$5.00: {very_large_errors} ({very_large_errors/total*100:.1f}%)")

    return True


# ============================================================================
# TEST 5: Worst Predictions Analysis
# ============================================================================

def test_worst_predictions(df, y_true, y_pred, n=10):
    """Identify and analyze worst predictions."""
    print_section("TEST 5: WORST PREDICTIONS ANALYSIS")

    df_analysis = df.copy()
    df_analysis['y_true'] = y_true
    df_analysis['y_pred'] = y_pred
    df_analysis['error'] = y_true - y_pred
    df_analysis['abs_error'] = np.abs(df_analysis['error'])
    df_analysis['pct_error'] = (df_analysis['abs_error'] / df_analysis['y_true'] * 100)

    # Get worst predictions
    worst = df_analysis.nlargest(n, 'abs_error')

    print(f"\n5.1 Top {n} worst predictions:")
    cols_to_show = ['y_true', 'y_pred', 'error', 'abs_error', 'pct_error']

    # Add market and property type if available
    if 'Market Name' in df_analysis.columns:
        cols_to_show.insert(0, 'Market Name')
    if 'Property Type' in df_analysis.columns:
        cols_to_show.insert(0, 'Property Type')

    print(worst[cols_to_show].to_string())

    # Analyze common characteristics
    print(f"\n5.2 Common characteristics of worst predictions:")

    if 'Market Name' in df_analysis.columns:
        worst_markets = worst['Market Name'].value_counts()
        if len(worst_markets) > 0:
            print(f"  Markets: {worst_markets.to_dict()}")

    if 'Property Type' in df_analysis.columns:
        worst_types = worst['Property Type'].value_counts()
        if len(worst_types) > 0:
            print(f"  Property types: {worst_types.to_dict()}")

    # Price range of worst predictions
    avg_price_worst = worst['y_true'].mean()
    avg_price_all = y_true.mean()
    print(f"  Avg actual price (worst): ${avg_price_worst:.2f}")
    print(f"  Avg actual price (all): ${avg_price_all:.2f}")

    if avg_price_worst > avg_price_all * 1.2:
        print(f"  ⚠️  Worst predictions tend to be for higher-priced properties")
    elif avg_price_worst < avg_price_all * 0.8:
        print(f"  ⚠️  Worst predictions tend to be for lower-priced properties")

    return True


# ============================================================================
# TEST 6: Best Predictions Analysis
# ============================================================================

def test_best_predictions(df, y_true, y_pred, n=10):
    """Identify and analyze best predictions."""
    print_section("TEST 6: BEST PREDICTIONS ANALYSIS")

    df_analysis = df.copy()
    df_analysis['y_true'] = y_true
    df_analysis['y_pred'] = y_pred
    df_analysis['error'] = y_true - y_pred
    df_analysis['abs_error'] = np.abs(df_analysis['error'])
    df_analysis['pct_error'] = (df_analysis['abs_error'] / df_analysis['y_true'] * 100)

    # Get best predictions
    best = df_analysis.nsmallest(n, 'abs_error')

    print(f"\n6.1 Top {n} best predictions:")
    cols_to_show = ['y_true', 'y_pred', 'error', 'abs_error', 'pct_error']

    if 'Market Name' in df_analysis.columns:
        cols_to_show.insert(0, 'Market Name')
    if 'Property Type' in df_analysis.columns:
        cols_to_show.insert(0, 'Property Type')

    print(best[cols_to_show].to_string())

    # Within-10% accuracy
    within_10pct = (df_analysis['pct_error'] < 10).sum()
    total = len(df_analysis)

    print(f"\n6.2 Accuracy thresholds:")
    print(f"  Within ±10%: {within_10pct} ({within_10pct/total*100:.1f}%)")

    within_5pct = (df_analysis['pct_error'] < 5).sum()
    print(f"  Within ±5%: {within_5pct} ({within_5pct/total*100:.1f}%)")

    within_1dollar = (df_analysis['abs_error'] < 1).sum()
    print(f"  Within ±$1.00: {within_1dollar} ({within_1dollar/total*100:.1f}%)")

    return True


# ============================================================================
# TEST 7: Error Correlation Analysis
# ============================================================================

def test_error_correlations(df, y_true, y_pred):
    """Analyze correlation between errors and features."""
    print_section("TEST 7: ERROR CORRELATION ANALYSIS")

    df_analysis = df.copy()
    df_analysis['abs_error'] = np.abs(y_true - y_pred)

    # Get numeric columns only
    numeric_cols = df_analysis.select_dtypes(include=[np.number]).columns.tolist()

    # Remove target-related columns
    numeric_cols = [col for col in numeric_cols if col not in ['abs_error', 'Rent/SF/Yr']]

    if len(numeric_cols) == 0:
        print("  No numeric features available for correlation analysis")
        return True

    # Calculate correlations
    correlations = []
    for col in numeric_cols:
        if df_analysis[col].notna().sum() > 100:  # Need enough non-null values
            corr = df_analysis[['abs_error', col]].corr().iloc[0, 1]
            if not np.isnan(corr):
                correlations.append({'feature': col, 'correlation': corr})

    if len(correlations) == 0:
        print("  Unable to calculate correlations")
        return True

    corr_df = pd.DataFrame(correlations).sort_values('correlation', key=abs, ascending=False)

    print("\n7.1 Top 15 features correlated with absolute error:")
    print(corr_df.head(15).to_string(index=False))

    # Highlight strong correlations
    strong_corr = corr_df[abs(corr_df['correlation']) > 0.3]
    if len(strong_corr) > 0:
        print(f"\n  ⚠️  {len(strong_corr)} features with strong correlation (|r| > 0.3):")
        for _, row in strong_corr.iterrows():
            print(f"    {row['feature']}: {row['correlation']:.3f}")
        print("  These features may indicate where the model struggles")

    return True


# ============================================================================
# MAIN TEST RUNNER
# ============================================================================

def main():
    """Run all error analysis tests."""
    print("="*70)
    print("ERROR ANALYSIS TEST SUITE")
    print("="*70)
    print("\nAnalyzing model prediction errors")

    # Setup: Load data and train model
    print("\nSetup: Loading data and training model...")
    try:
        val = load_val_data()
        print(f"  Validation set: {len(val):,} rows")

        # Feature engineering
        print("\n  Applying feature engineering...")
        val_eng = engineer_features(val.copy())

        # For error analysis, we need the original dataframe with market/property type
        # So we'll keep a copy before prepare_data_for_modeling
        val_original = val_eng.copy()

        # Train a quick model
        train = load_train_data()
        train_eng = engineer_features(train.copy())

        X_train, y_train, feature_names, _ = prepare_data_for_modeling(train_eng)
        X_val, y_val, _, _ = prepare_data_for_modeling(val_eng, ref_columns=feature_names)

        print(f"  Training Random Forest...")
        model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)

        # Get predictions
        y_pred = model.predict(X_val)

        # Basic metrics
        mae = mean_absolute_error(y_val, y_pred)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        r2 = r2_score(y_val, y_pred)

        print(f"\n  Model performance:")
        print(f"    MAE: ${mae:.3f}")
        print(f"    RMSE: ${rmse:.3f}")
        print(f"    R²: {r2:.4f}")

    except Exception as e:
        print(f"\n✗ Setup failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Run all tests
    results = {}

    try:
        results['market'] = test_error_by_market(val_original, y_val, y_pred)
    except Exception as e:
        print(f"\n✗ Market analysis failed: {e}")
        results['market'] = False

    try:
        results['property_type'] = test_error_by_property_type(val_original, y_val, y_pred)
    except Exception as e:
        print(f"\n✗ Property type analysis failed: {e}")
        results['property_type'] = False

    try:
        results['price_range'] = test_error_by_price_range(y_val, y_pred)
    except Exception as e:
        print(f"\n✗ Price range analysis failed: {e}")
        results['price_range'] = False

    try:
        results['bias'] = test_systematic_bias(y_val, y_pred)
    except Exception as e:
        print(f"\n✗ Bias detection failed: {e}")
        results['bias'] = False

    try:
        results['worst'] = test_worst_predictions(val_original, y_val, y_pred, n=10)
    except Exception as e:
        print(f"\n✗ Worst predictions analysis failed: {e}")
        results['worst'] = False

    try:
        results['best'] = test_best_predictions(val_original, y_val, y_pred, n=10)
    except Exception as e:
        print(f"\n✗ Best predictions analysis failed: {e}")
        results['best'] = False

    try:
        results['correlations'] = test_error_correlations(val_original, y_val, y_pred)
    except Exception as e:
        print(f"\n✗ Correlation analysis failed: {e}")
        results['correlations'] = False

    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)

    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name:.<50} {status}")

    total_passed = sum(results.values())
    total_tests = len(results)
    print(f"\nTotal: {total_passed}/{total_tests} analyses completed")

    if total_passed == total_tests:
        print("\n✓ ALL ANALYSES COMPLETED")
        return 0
    else:
        print("\n⚠ SOME ANALYSES FAILED")
        return 1


if __name__ == "__main__":
    exit(main())
