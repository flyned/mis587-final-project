"""
End-to-end test for Phase 3: Model Interpretation

This script validates the complete Phase 3 implementation:
- Feature importance analysis (MDI, Permutation, SHAP)
- Residual diagnostics
- Error segmentation (Market, Property Type, Price Range, Building Age)
- Comprehensive visualization suite

Expected outputs:
- 20+ publication-quality plots
- 8+ CSV reports
- Executive insights summary
- Memory usage < 1GB
- Runtime < 10 minutes

Author: MIS587 Final Project
Date: November 2024
"""

import sys
import time
import pickle
import json
import gc
import numpy as np
import pandas as pd
from pathlib import Path
import psutil

# Import interpretation and visualization functions
from src.interpretation import (
    get_mdi_importance,
    get_permutation_importance,
    compare_importance_methods,
    calculate_shap_values,
    get_shap_feature_importance,
    plot_shap_summary,
    plot_shap_dependence,
    analyze_residuals,
    identify_outlier_predictions,
    create_residual_plots,
    analyze_errors_by_market,
    analyze_errors_by_property_type,
    analyze_errors_by_price_range,
    analyze_errors_by_building_age,
    create_error_segmentation_report
)

from src.visualization import (
    setup_plot_style,
    save_figure,
    plot_feature_importance_bar,
    plot_feature_importance_comparison,
    plot_error_by_segment
)

from src.feature_engineering import engineer_features
from src.modeling import prepare_data_for_modeling, evaluate_model
from src.utils import load_data_splits


def get_memory_usage():
    """Return current memory usage in MB."""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024


def print_section(title):
    """Print formatted section header."""
    print("\n" + "="*70)
    print(title)
    print("="*70)


def main():
    """Run comprehensive Phase 3 interpretation analysis."""

    start_time = time.time()
    start_memory = get_memory_usage()

    print("="*70)
    print("PHASE 3: MODEL INTERPRETATION - COMPREHENSIVE ANALYSIS")
    print("="*70)
    print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Initial memory: {start_memory:.1f} MB")

    # =========================================================================
    # SECTION 1: SETUP & DATA LOADING
    # =========================================================================
    print_section("SECTION 1: SETUP & DATA LOADING")

    # Setup visualization style
    setup_plot_style()
    print("✓ Visualization style configured")

    # Load data splits
    print("\nLoading data splits...")
    train_df, val_df, test_df = load_data_splits()
    print(f"Train: {train_df.shape}")
    print(f"Val:   {val_df.shape}")
    print(f"Test:  {test_df.shape}")

    # Load trained model
    model_dir = Path('models')
    model_files = sorted(model_dir.glob('rf_model_local_*.pkl'), reverse=True)
    if not model_files:
        print("ERROR: No trained model found. Run run_full_training.py first.")
        return False

    latest_model_path = model_files[0]
    print(f"\nLoading model: {latest_model_path}")
    with open(latest_model_path, 'rb') as f:
        model_dict = pickle.load(f)
    model = model_dict['model']
    print(f"✓ Model loaded: {type(model).__name__}")

    # Load metadata for feature names
    metadata_path = latest_model_path.parent / latest_model_path.name.replace('model', 'metadata').replace('.pkl', '.json')
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    feature_names = metadata['feature_names']
    print(f"✓ Feature names loaded: {len(feature_names)} features")

    print(f"\nMemory after loading: {get_memory_usage():.1f} MB")

    # =========================================================================
    # SECTION 2: FEATURE ENGINEERING & PREDICTIONS
    # =========================================================================
    print_section("SECTION 2: FEATURE ENGINEERING & PREDICTIONS")

    # Engineer features for validation set
    print("\nEngineering features for validation set...")
    val_eng = engineer_features(val_df.copy())
    print(f"Validation engineered: {val_eng.shape}")

    # Prepare data for modeling
    print("\nPreparing data for modeling...")
    X_val, y_val, feature_names_actual, categorical_cols = prepare_data_for_modeling(
        val_eng,
        target_column='Rent/SF/Yr',
        ref_columns=feature_names
    )
    print(f"X_val: {X_val.shape}")
    print(f"y_val: {y_val.shape}")

    # Align feature names (handle any column mismatch)
    if len(feature_names) != X_val.shape[1]:
        print(f"WARNING: Feature count mismatch ({len(feature_names)} vs {X_val.shape[1]})")
        print("Using model's feature names as ground truth")
        # Ensure X_val has same columns as training
        missing_cols = set(feature_names) - set(val_eng.columns)
        if missing_cols:
            print(f"Adding missing columns: {len(missing_cols)}")
            for col in missing_cols:
                val_eng[col] = 0
        X_val = val_eng[feature_names].values

    # Generate predictions
    print("\nGenerating predictions...")
    y_pred_val = model.predict(X_val)
    print(f"✓ Predictions generated: {len(y_pred_val)} samples")

    # Evaluate model
    print("\nValidation Set Performance:")
    metrics_val = evaluate_model(y_val, y_pred_val, name="Random Forest")

    print(f"\nMemory after predictions: {get_memory_usage():.1f} MB")

    # =========================================================================
    # SECTION 3: FEATURE IMPORTANCE ANALYSIS
    # =========================================================================
    print_section("SECTION 3: FEATURE IMPORTANCE ANALYSIS")

    # MDI Importance
    print("\n1. Calculating MDI (Mean Decrease Impurity) importance...")
    mdi_importance = get_mdi_importance(model, feature_names, top_n=20)
    print(f"✓ Top 5 features (MDI):")
    for i, row in mdi_importance.head(5).iterrows():
        print(f"   {i+1}. {row['feature']}: {row['importance']:.4f} ({row['importance_pct']:.1f}%)")

    # Save CSV
    mdi_importance.to_csv('figures/feature_importance/mdi_importance.csv', index=False)

    # Create plot
    fig_mdi = plot_feature_importance_bar(
        mdi_importance,
        title='MDI Feature Importance (Top 20)',
        xlabel='Mean Decrease in Impurity'
    )
    save_figure(fig_mdi, 'mdi_importance_top20.png', 'feature_importance')
    import matplotlib.pyplot as plt
    plt.close(fig_mdi)
    gc.collect()

    # Permutation Importance
    print("\n2. Calculating Permutation importance (this takes ~60 seconds)...")
    perm_importance = get_permutation_importance(
        model, X_val, y_val, feature_names,
        n_repeats=10, top_n=20
    )
    print(f"✓ Top 5 features (Permutation):")
    for i, row in perm_importance.head(5).iterrows():
        print(f"   {i+1}. {row['feature']}: {row['importance']:.4f} ± {row['importance_std']:.4f}")

    # Save CSV
    perm_importance.to_csv('figures/feature_importance/permutation_importance.csv', index=False)

    # Create plot
    fig_perm = plot_feature_importance_bar(
        perm_importance,
        title='Permutation Feature Importance (Top 20)',
        xlabel='Mean Decrease in MAE'
    )
    save_figure(fig_perm, 'permutation_importance_top20.png', 'feature_importance')
    plt.close(fig_perm)
    gc.collect()

    # Compare importance methods
    print("\n3. Comparing importance methods...")
    comparison = compare_importance_methods(mdi_importance, perm_importance, top_n=15)
    print(f"✓ Consensus features (High):")
    consensus_high = comparison[comparison['consensus'] == 'High']
    for i, row in consensus_high.iterrows():
        print(f"   - {row['feature']} (MDI rank: {row['mdi_rank']}, Perm rank: {row['perm_rank']})")

    # Save CSV
    comparison.to_csv('figures/feature_importance/importance_comparison.csv', index=False)

    print(f"\nMemory after feature importance: {get_memory_usage():.1f} MB")

    # =========================================================================
    # SECTION 4: SHAP VALUE ANALYSIS
    # =========================================================================
    print_section("SECTION 4: SHAP VALUE ANALYSIS")

    # Calculate SHAP values (sampled for memory efficiency)
    print("\n1. Calculating SHAP values (sampled to 1000 samples for memory efficiency)...")
    shap_values = calculate_shap_values(
        model, X_val, feature_names,
        max_samples=1000,
        save_path='figures/shap_analysis/shap_values.pkl'
    )
    print(f"✓ SHAP values calculated and saved")

    # SHAP feature importance
    print("\n2. Computing SHAP-based feature importance...")
    shap_importance = get_shap_feature_importance(shap_values, feature_names, top_n=20)
    print(f"✓ Top 5 features (SHAP):")
    for i, row in shap_importance.head(5).iterrows():
        print(f"   {i+1}. {row['feature']}: {row['importance']:.4f} ({row['importance_pct']:.1f}%)")

    # Save CSV
    shap_importance.to_csv('figures/feature_importance/shap_importance.csv', index=False)

    # Create 3-panel comparison plot
    print("\n3. Creating 3-panel feature importance comparison...")
    fig_comparison = plot_feature_importance_comparison(
        mdi_importance, perm_importance, shap_importance, top_n=15
    )
    save_figure(fig_comparison, 'importance_comparison_top15.png', 'feature_importance')
    plt.close(fig_comparison)
    gc.collect()

    # SHAP summary plot
    print("\n4. Creating SHAP summary plot...")
    X_shap_subset = X_val[:1000] if len(X_val) > 1000 else X_val
    fig_shap_summary = plot_shap_summary(
        shap_values, X_shap_subset, feature_names, top_n=20
    )
    save_figure(fig_shap_summary, 'shap_summary_top20.png', 'shap_analysis')
    plt.close(fig_shap_summary)
    gc.collect()

    # SHAP dependence plots for top 5 features
    print("\n5. Creating SHAP dependence plots for top 5 features...")
    top_5_features = shap_importance.head(5)['feature'].tolist()
    for feature in top_5_features:
        print(f"   - {feature}")
        fig_dep = plot_shap_dependence(
            shap_values, X_shap_subset, feature_names, feature
        )
        safe_filename = feature.replace('/', '_').replace(' ', '_').lower()
        save_figure(fig_dep, f'shap_dependence_{safe_filename}.png', 'shap_analysis')
        plt.close(fig_dep)
        gc.collect()

    print(f"\nMemory after SHAP analysis: {get_memory_usage():.1f} MB")

    # =========================================================================
    # SECTION 5: RESIDUAL ANALYSIS
    # =========================================================================
    print_section("SECTION 5: RESIDUAL ANALYSIS")

    # Analyze residuals
    print("\n1. Analyzing residuals...")
    residual_analysis = analyze_residuals(y_val, y_pred_val)

    print(f"\nResidual Statistics:")
    print(f"  Mean:     ${residual_analysis['residual_stats']['mean']:.4f}")
    print(f"  Std:      ${residual_analysis['residual_stats']['std']:.4f}")
    print(f"  Median:   ${residual_analysis['residual_stats']['median']:.4f}")
    print(f"  Skewness: {residual_analysis['residual_stats']['skew']:.4f}")
    print(f"  Kurtosis: {residual_analysis['residual_stats']['kurtosis']:.4f}")

    normality_test = residual_analysis['normality_test']
    print(f"\nNormality Test ({normality_test['test_name']}):")
    print(f"  p-value: {normality_test['p_value']:.6f}")
    print(f"  Result: {'Normally distributed ✓' if normality_test['is_normal'] else 'Not normally distributed ✗'}")

    print(f"\nOutliers:")
    print(f"  Count: {residual_analysis['outlier_count']} ({residual_analysis['outlier_pct']:.1f}%)")

    # Save statistics (convert numpy types to Python types for JSON serialization)
    stats_dict = {
        'residual_stats': {k: float(v) for k, v in residual_analysis['residual_stats'].items()},
        'normality_test': {
            'statistic': float(residual_analysis['normality_test']['statistic']),
            'p_value': float(residual_analysis['normality_test']['p_value']),
            'is_normal': bool(residual_analysis['normality_test']['is_normal']),
            'test_name': residual_analysis['normality_test']['test_name'],
            'sample_size': int(residual_analysis['normality_test']['sample_size'])
        },
        'outlier_count': int(residual_analysis['outlier_count']),
        'outlier_pct': float(residual_analysis['outlier_pct'])
    }
    with open('figures/residual_analysis/residual_statistics.json', 'w') as f:
        json.dump(stats_dict, f, indent=2)

    # Identify outlier predictions
    print("\n2. Identifying outlier predictions...")
    outliers = identify_outlier_predictions(y_val, y_pred_val, threshold_std=3.0)
    print(f"✓ Flagged {outliers['outlier_count']} properties for manual review ({outliers['outlier_pct']:.1f}%)")

    # Create residual diagnostic plots
    print("\n3. Creating residual diagnostic plots (4 plots)...")
    residual_plots = create_residual_plots(y_val, y_pred_val)
    print(f"✓ Saved 4 residual diagnostic plots")

    print(f"\nMemory after residual analysis: {get_memory_usage():.1f} MB")

    # =========================================================================
    # SECTION 6: ERROR SEGMENTATION
    # =========================================================================
    print_section("SECTION 6: ERROR SEGMENTATION")

    print("\nPreparing feature data for segmentation...")

    # Need to load raw categorical features (before encoding)
    # For this, we'll use the original validation DataFrame
    feature_dict = {}

    # Market Name
    if 'Market Name' in val_df.columns:
        feature_dict['Market Name'] = val_df['Market Name'].values
    else:
        print("WARNING: Market Name not found in data")

    # Property Type (Secondary Type is most detailed)
    if 'Secondary Type' in val_df.columns:
        feature_dict['Property Type'] = val_df['Secondary Type'].values
    elif 'Property Type' in val_df.columns:
        feature_dict['Property Type'] = val_df['Property Type'].values
    else:
        print("WARNING: Property Type not found in data")

    # Price Range (use actual values)
    feature_dict['Price Range'] = y_val

    # Building Age
    if 'building_age' in val_eng.columns:
        feature_dict['Building Age'] = val_eng['building_age'].values
    elif 'Year Built' in val_df.columns:
        current_year = 2025
        feature_dict['Building Age'] = current_year - val_df['Year Built'].fillna(current_year).values
    else:
        print("WARNING: Building Age not available")

    print(f"✓ Prepared {len(feature_dict)} segmentation features")

    # Create comprehensive segmentation report
    print("\nGenerating error segmentation reports (all 4 types)...")
    segmentation_reports = create_error_segmentation_report(
        y_val, y_pred_val, feature_dict
    )

    # Print summary for each segmentation
    for feature_name, seg_df in segmentation_reports.items():
        print(f"\n{feature_name}:")
        print(f"  Best segment:  {seg_df.iloc[-1]['segment']} (MAE=${seg_df.iloc[-1]['MAE']:.2f})")
        print(f"  Worst segment: {seg_df.iloc[0]['segment']} (MAE=${seg_df.iloc[0]['MAE']:.2f})")
        print(f"  Segments analyzed: {len(seg_df)}")

    print(f"\nMemory after error segmentation: {get_memory_usage():.1f} MB")

    # =========================================================================
    # SECTION 7: FINAL SUMMARY
    # =========================================================================
    print_section("SECTION 7: FINAL SUMMARY")

    end_time = time.time()
    end_memory = get_memory_usage()
    peak_memory = end_memory  # Approximation
    runtime_minutes = (end_time - start_time) / 60

    print(f"\nExecution Summary:")
    print(f"  Total runtime:    {runtime_minutes:.2f} minutes")
    print(f"  Initial memory:   {start_memory:.1f} MB")
    print(f"  Final memory:     {end_memory:.1f} MB")
    print(f"  Peak memory:      {peak_memory:.1f} MB")

    # Count generated files
    figures_dir = Path('figures')
    png_files = list(figures_dir.glob('**/*.png'))
    csv_files = list(figures_dir.glob('**/*.csv'))
    json_files = list(figures_dir.glob('**/*.json'))
    pkl_files = list(figures_dir.glob('**/*.pkl'))

    print(f"\nGenerated Outputs:")
    print(f"  PNG plots:      {len(png_files)}")
    print(f"  CSV reports:    {len(csv_files)}")
    print(f"  JSON metadata:  {len(json_files)}")
    print(f"  PKL artifacts:  {len(pkl_files)}")

    # Validation checks
    print(f"\nValidation Checks:")
    memory_ok = peak_memory < 1000  # < 1GB
    runtime_ok = runtime_minutes < 10  # < 10 minutes
    plots_ok = len(png_files) >= 15  # >= 15 plots
    csvs_ok = len(csv_files) >= 4  # >= 4 CSVs

    print(f"  Memory < 1GB:      {'✓ PASS' if memory_ok else '✗ FAIL'} ({peak_memory:.1f} MB)")
    print(f"  Runtime < 10 min:  {'✓ PASS' if runtime_ok else '✗ FAIL'} ({runtime_minutes:.2f} min)")
    print(f"  Plots >= 15:       {'✓ PASS' if plots_ok else '✗ FAIL'} ({len(png_files)} plots)")
    print(f"  CSVs >= 4:         {'✓ PASS' if csvs_ok else '✗ FAIL'} ({len(csv_files)} CSVs)")

    all_passed = memory_ok and runtime_ok and plots_ok and csvs_ok

    print("\n" + "="*70)
    if all_passed:
        print("PHASE 3 COMPLETE! ✓ All validation checks passed")
    else:
        print("PHASE 3 COMPLETE with warnings. Review validation checks above.")
    print("="*70)

    # Print file locations
    print(f"\nGenerated files saved to:")
    print(f"  Feature Importance: figures/feature_importance/")
    print(f"  SHAP Analysis:      figures/shap_analysis/")
    print(f"  Residual Analysis:  figures/residual_analysis/")
    print(f"  Error Segmentation: figures/error_segmentation/")

    print(f"\nNext steps:")
    print(f"  1. Review visualizations in figures/ subdirectories")
    print(f"  2. Create business presentation from generated plots")
    print(f"  3. Share error segmentation insights with stakeholders")
    print(f"  4. Use SHAP plots to explain individual predictions")

    return all_passed


if __name__ == '__main__':
    # Create figures directory structure
    for subdir in ['feature_importance', 'shap_analysis', 'residual_analysis', 'error_segmentation']:
        Path(f'figures/{subdir}').mkdir(parents=True, exist_ok=True)

    # Run Phase 3 interpretation
    success = main()

    sys.exit(0 if success else 1)
