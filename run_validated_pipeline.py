"""
Validated ML Pipeline - Integrates all validation checks

This script runs the complete ML pipeline with validation gates at each stage:
1. Data cleaning → validate schema & quality
2. Train/val/test split → validate no data leakage
3. Feature engineering → validate engineered features
4. Model training → validate model performance
5. Error analysis → identify improvement areas

Usage:
    python run_validated_pipeline.py [--skip-validation]
"""

import pandas as pd
import numpy as np
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path

from src.utils import load_clean_data, load_data_splits, save_data_splits
from src.feature_engineering import engineer_features
from src.modeling import prepare_data_for_modeling
from sklearn.ensemble import RandomForestRegressor

# Import validators
from src.validators import validate_all as validate_data
from src.model_validators import validate_model_comprehensive


def print_stage(title):
    """Print pipeline stage header."""
    print("\n" + "="*70)
    print(f"STAGE: {title}")
    print("="*70)


def save_validation_report(report: dict, stage: str, output_dir: str = "validation_reports"):
    """Save validation report to JSON file."""
    Path(output_dir).mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{output_dir}/{stage}_validation_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\n  ✓ Validation report saved: {filename}")
    return filename


def stage1_validate_clean_data(skip_validation=False):
    """Stage 1: Load and validate clean data."""
    print_stage("1. VALIDATE CLEAN DATA")
    
    print("\n1.1 Loading clean data...")
    df = load_clean_data()
    print(f"  Loaded {len(df):,} rows, {len(df.columns)} columns")
    
    if skip_validation:
        print("\n  ⚠️  Skipping validation (--skip-validation flag)")
        return df, None
    
    print("\n1.2 Running data validation...")
    results = validate_data(df, stage='clean', verbose=False)
    
    # Print summary
    print(f"\n  Validation Status: {'✓ PASSED' if results['all_passed'] else '✗ FAILED'}")
    print(f"  Total Errors: {results['total_errors']}")
    print(f"  Total Warnings: {results['total_warnings']}")
    
    # Show critical errors
    if results['total_errors'] > 0:
        print("\n  Critical Errors:")
        for val_name, val_result in results['validations'].items():
            if val_result['errors']:
                for error in val_result['errors']:
                    print(f"    - [{val_name}] {error}")
    
    # Save report
    report_file = save_validation_report(results, 'clean_data')
    
    # Decision gate
    if results['total_errors'] > 0:
        print("\n  ⚠️  VALIDATION GATE: Errors detected but continuing (warnings only)")
    else:
        print("\n  ✓ VALIDATION GATE PASSED")
    
    return df, results


def stage2_validate_splits(skip_validation=False):
    """Stage 2: Load and validate train/val/test splits."""
    print_stage("2. VALIDATE DATA SPLITS")
    
    print("\n2.1 Loading data splits...")
    train, val, test = load_data_splits()
    print(f"  Train: {len(train):,} rows")
    print(f"  Val: {len(val):,} rows")
    print(f"  Test: {len(test):,} rows")
    
    if skip_validation:
        print("\n  ⚠️  Skipping validation (--skip-validation flag)")
        return train, val, test, None
    
    print("\n2.2 Running split validation (data leakage detection)...")
    results = validate_data(train_df=train, val_df=val, test_df=test, 
                           stage='split', verbose=False)
    
    # Print summary
    print(f"\n  Validation Status: {'✓ PASSED' if results['all_passed'] else '✗ FAILED'}")
    print(f"  Total Errors: {results['total_errors']}")
    print(f"  Total Warnings: {results['total_warnings']}")
    
    # Check data leakage specifically
    if 'data_leakage' in results['validations']:
        leakage_result = results['validations']['data_leakage']
        if leakage_result['passed']:
            print("\n  ✓ NO DATA LEAKAGE DETECTED")
        else:
            print("\n  ✗ DATA LEAKAGE DETECTED!")
            for error in leakage_result['errors']:
                print(f"    - {error}")
    
    # Save report
    report_file = save_validation_report(results, 'data_splits')
    
    # Decision gate - data leakage is CRITICAL
    if results['total_errors'] > 0:
        if any('leakage' in str(e).lower() for v in results['validations'].values() for e in v['errors']):
            print("\n  ✗ VALIDATION GATE FAILED: Data leakage detected - CANNOT CONTINUE")
            sys.exit(1)
        else:
            print("\n  ⚠️  VALIDATION GATE: Errors detected but continuing")
    else:
        print("\n  ✓ VALIDATION GATE PASSED")
    
    return train, val, test, results


def stage3_feature_engineering(train, val, test, skip_validation=False):
    """Stage 3: Apply feature engineering and validate."""
    print_stage("3. FEATURE ENGINEERING & VALIDATION")
    
    print("\n3.1 Applying feature engineering to train set...")
    train_eng = engineer_features(train.copy())
    print(f"  Features: {len(train_eng.columns)}")
    
    print("\n3.2 Applying feature engineering to val set...")
    val_eng = engineer_features(val.copy())
    
    print("\n3.3 Applying feature engineering to test set...")
    test_eng = engineer_features(test.copy())
    
    if skip_validation:
        print("\n  ⚠️  Skipping validation (--skip-validation flag)")
        return train_eng, val_eng, test_eng, None
    
    print("\n3.4 Validating engineered features...")
    results = validate_data(train_eng, stage='engineered', verbose=False)
    
    print(f"\n  Validation Status: {'✓ PASSED' if results['all_passed'] else '✗ FAILED'}")
    print(f"  Total Errors: {results['total_errors']}")
    print(f"  Total Warnings: {results['total_warnings']}")
    
    # Save report
    report_file = save_validation_report(results, 'engineered_features')
    
    # Decision gate
    if results['total_errors'] > 0:
        print("\n  ⚠️  VALIDATION GATE: Errors detected but continuing")
    else:
        print("\n  ✓ VALIDATION GATE PASSED")
    
    return train_eng, val_eng, test_eng, results


def stage4_train_and_validate_model(train_eng, val_eng, skip_validation=False):
    """Stage 4: Train model and validate performance."""
    print_stage("4. MODEL TRAINING & VALIDATION")
    
    print("\n4.1 Preparing data for modeling...")
    X_train, y_train, feature_names, _ = prepare_data_for_modeling(train_eng)
    X_val, y_val, _, _ = prepare_data_for_modeling(val_eng, ref_columns=feature_names)
    
    print(f"  Training samples: {len(X_train):,}")
    print(f"  Features: {len(feature_names)}")
    
    print("\n4.2 Training Random Forest model...")
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_split=10,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    print("  ✓ Model trained")
    
    if skip_validation:
        print("\n  ⚠️  Skipping validation (--skip-validation flag)")
        return model, X_train, y_train, X_val, y_val, feature_names, None
    
    print("\n4.3 Running model validation...")
    results = validate_model_comprehensive(
        model, X_train, y_train, X_val, y_val,
        feature_names=feature_names,
        verbose=False
    )
    
    # Print summary
    print(f"\n  Validation Status: {'✓ PASSED' if results['all_passed'] else '✗ FAILED'}")
    print(f"  Total Errors: {results['total_errors']}")
    print(f"  Total Warnings: {results['total_warnings']}")
    
    # Show key metrics
    if 'overfitting' in results['validations']:
        metrics = results['validations']['overfitting']['metrics']
        print(f"\n  Model Performance:")
        print(f"    Train MAE: ${metrics.get('train_mae', 0):.3f}")
        print(f"    Val MAE: ${metrics.get('val_mae', 0):.3f}")
        print(f"    MAE Ratio: {metrics.get('mae_ratio', 0):.3f}")
        print(f"    Train R²: {metrics.get('train_r2', 0):.4f}")
        print(f"    Val R²: {metrics.get('val_r2', 0):.4f}")
    
    # Check for overfitting
    if 'overfitting' in results['validations']:
        overfitting = results['validations']['overfitting']
        if not overfitting['passed']:
            print("\n  ⚠️  OVERFITTING DETECTED:")
            for error in overfitting['errors']:
                print(f"    - {error}")
    
    # Save report
    report_file = save_validation_report(results, 'model_validation')
    
    # Decision gate - overfitting is a warning, not blocker
    if results['total_errors'] > 5:
        print("\n  ⚠️  VALIDATION GATE: Multiple errors detected - review recommended")
    elif results['total_errors'] > 0:
        print("\n  ⚠️  VALIDATION GATE: Errors detected but acceptable")
    else:
        print("\n  ✓ VALIDATION GATE PASSED")
    
    return model, X_train, y_train, X_val, y_val, feature_names, results


def stage5_error_analysis(model, val_eng, X_val, y_val):
    """Stage 5: Comprehensive error analysis."""
    print_stage("5. ERROR ANALYSIS")
    
    print("\n5.1 Generating predictions...")
    y_pred = model.predict(X_val)
    
    print("\n5.2 Computing error metrics...")
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    
    mae = mean_absolute_error(y_val, y_pred)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    r2 = r2_score(y_val, y_pred)
    mape = np.mean(np.abs((y_val - y_pred) / y_val)) * 100
    
    print(f"  MAE: ${mae:.3f}")
    print(f"  RMSE: ${rmse:.3f}")
    print(f"  R²: {r2:.4f}")
    print(f"  MAPE: {mape:.2f}%")
    
    # Within-threshold accuracy
    within_1dollar = (np.abs(y_val - y_pred) < 1.0).sum() / len(y_val) * 100
    within_10pct = (np.abs((y_val - y_pred) / y_val) < 0.10).sum() / len(y_val) * 100
    
    print(f"\n  Accuracy:")
    print(f"    Within ±$1.00: {within_1dollar:.1f}%")
    print(f"    Within ±10%: {within_10pct:.1f}%")
    
    # Price range analysis
    print("\n5.3 Error by price range:")
    df_analysis = pd.DataFrame({
        'y_true': y_val,
        'y_pred': y_pred,
        'abs_error': np.abs(y_val - y_pred)
    })
    
    bins = [0, 8, 10, 12, 15, 20, 200]
    labels = ['<$8', '$8-10', '$10-12', '$12-15', '$15-20', '>$20']
    df_analysis['price_range'] = pd.cut(df_analysis['y_true'], bins=bins, labels=labels)
    
    range_stats = df_analysis.groupby('price_range')['abs_error'].agg(['mean', 'count'])
    for idx, row in range_stats.iterrows():
        if row['count'] > 0:
            print(f"    {idx}: MAE ${row['mean']:.3f} (n={int(row['count'])})")
    
    # Market analysis if available
    if 'Market Name' in val_eng.columns:
        print("\n5.4 Error by market (top 5):")
        val_analysis = val_eng.copy()
        val_analysis['abs_error'] = np.abs(y_val - y_pred)
        
        market_errors = val_analysis.groupby('Market Name')['abs_error'].agg(['mean', 'count'])
        market_errors = market_errors.sort_values('mean', ascending=False).head(5)
        
        for idx, row in market_errors.iterrows():
            print(f"    {idx}: MAE ${row['mean']:.3f} (n={int(row['count'])})")
    
    # Save error analysis
    error_report = {
        'timestamp': datetime.now().isoformat(),
        'metrics': {
            'mae': float(mae),
            'rmse': float(rmse),
            'r2': float(r2),
            'mape': float(mape),
            'within_1dollar_pct': float(within_1dollar),
            'within_10pct_pct': float(within_10pct)
        },
        'price_range_analysis': range_stats.to_dict()
    }
    
    report_file = save_validation_report(error_report, 'error_analysis')
    
    print("\n  ✓ ERROR ANALYSIS COMPLETE")
    
    return error_report


def generate_summary_report(all_reports):
    """Generate final summary report."""
    print_stage("6. VALIDATION SUMMARY")
    
    print("\n6.1 Pipeline Validation Summary:")
    
    stages = ['clean_data', 'data_splits', 'engineered_features', 'model_validation', 'error_analysis']
    
    for stage in stages:
        if stage in all_reports and all_reports[stage] is not None:
            if 'all_passed' in all_reports[stage]:
                status = "✓ PASSED" if all_reports[stage]['all_passed'] else "⚠️  WARNINGS"
                errors = all_reports[stage].get('total_errors', 0)
                warnings = all_reports[stage].get('total_warnings', 0)
                print(f"  {stage:.<30} {status} ({errors} errors, {warnings} warnings)")
            else:
                print(f"  {stage:.<30} ✓ COMPLETED")
        else:
            print(f"  {stage:.<30} ⊘ SKIPPED")
    
    # Critical issues
    critical_issues = []
    for stage, report in all_reports.items():
        if report and 'total_errors' in report and report['total_errors'] > 0:
            critical_issues.append(f"{stage}: {report['total_errors']} errors")
    
    if critical_issues:
        print(f"\n6.2 Critical Issues:")
        for issue in critical_issues:
            print(f"  ⚠️  {issue}")
    else:
        print(f"\n6.2 ✓ No critical issues detected")
    
    # Save consolidated report
    summary = {
        'timestamp': datetime.now().isoformat(),
        'pipeline_status': 'completed',
        'stages': all_reports,
        'critical_issues': critical_issues
    }
    
    report_file = save_validation_report(summary, 'pipeline_summary')
    
    print("\n" + "="*70)
    print("✓ VALIDATED PIPELINE COMPLETE")
    print("="*70)
    print(f"\nAll validation reports saved in: validation_reports/")


def main():
    """Run complete validated ML pipeline."""
    parser = argparse.ArgumentParser(description='Run validated ML pipeline')
    parser.add_argument('--skip-validation', action='store_true',
                       help='Skip all validation steps (faster but not recommended)')
    args = parser.parse_args()
    
    print("="*70)
    print("VALIDATED ML PIPELINE")
    print("="*70)
    print(f"Validation: {'DISABLED' if args.skip_validation else 'ENABLED'}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    all_reports = {}
    
    try:
        # Stage 1: Validate clean data
        df, report = stage1_validate_clean_data(args.skip_validation)
        all_reports['clean_data'] = report
        
        # Stage 2: Validate splits
        train, val, test, report = stage2_validate_splits(args.skip_validation)
        all_reports['data_splits'] = report
        
        # Stage 3: Feature engineering
        train_eng, val_eng, test_eng, report = stage3_feature_engineering(
            train, val, test, args.skip_validation
        )
        all_reports['engineered_features'] = report
        
        # Stage 4: Model training and validation
        model, X_train, y_train, X_val, y_val, feature_names, report = \
            stage4_train_and_validate_model(train_eng, val_eng, args.skip_validation)
        all_reports['model_validation'] = report
        
        # Stage 5: Error analysis
        error_report = stage5_error_analysis(model, val_eng, X_val, y_val)
        all_reports['error_analysis'] = error_report
        
        # Stage 6: Summary
        generate_summary_report(all_reports)
        
        return 0
        
    except Exception as e:
        print(f"\n✗ PIPELINE FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
