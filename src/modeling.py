"""
Modeling Module

This module contains model training, evaluation, and selection:
- Phase 2.1: Baseline models (Ridge, Lasso, Decision Tree) ✓
- Phase 2.2: Evaluation metrics suite ✓
- Phase 2.3: Random Forest with cross-validation ✓
- Phase 2.4: XGBoost/LightGBM with hyperparameter tuning ✓
- Phase 2.5: Model comparison and selection ✓
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.model_selection import cross_val_score, GridSearchCV
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')


def prepare_data_for_modeling(df, target_column='Rent/SF/Yr', ref_columns=None):
    """
    Prepare data for modeling: separate features/target, handle categoricals, scale.

    Args:
        df: DataFrame with engineered features
        target_column: Name of target variable
        ref_columns: Reference column list for aligning test/val sets (use train columns)

    Returns:
        Tuple of (X, y, feature_names, categorical_cols)
    """
    # Filter to rows with target
    df_filtered = df[df[target_column].notna()].copy()

    # Separate target
    y = df_filtered[target_column].values

    # Drop target and non-feature columns
    drop_cols = [
        target_column,
        'log_rent_sf_yr',  # CRITICAL: Remove target variable transformation to prevent data leakage
        'Property Address', 'Market Name', 'Submarket Name', 'City', 'State',
        'Zip', 'County Name', 'Building Park', 'Submarket Cluster',
        'Continent', 'Country', 'Subcontinent', 'Cross Street',
        'Last Sale Date', 'FEMA Map Date', 'Construction Begin',
        'Maturity Date', 'Origination Date'
    ]

    # Keep only columns that exist
    drop_cols = [col for col in drop_cols if col in df_filtered.columns]
    X_df = df_filtered.drop(columns=drop_cols)

    # Identify categorical columns
    categorical_cols = X_df.select_dtypes(include=['object', 'category']).columns.tolist()

    # One-hot encode categoricals
    if categorical_cols:
        X_df = pd.get_dummies(X_df, columns=categorical_cols, drop_first=True)

    # Fill any remaining nulls with 0
    X_df = X_df.fillna(0)

    # Align with reference columns if provided (for val/test sets)
    if ref_columns is not None:
        # Add missing columns with 0s
        for col in ref_columns:
            if col not in X_df.columns:
                X_df[col] = 0

        # Remove extra columns not in reference
        extra_cols = [col for col in X_df.columns if col not in ref_columns]
        if extra_cols:
            X_df = X_df.drop(columns=extra_cols)

        # Reorder to match reference
        X_df = X_df[ref_columns]

    feature_names = X_df.columns.tolist()
    X = X_df.values

    return X, y, feature_names, categorical_cols


def evaluate_model(y_true, y_pred, name="Model"):
    """
    Phase 2.2: Evaluate model performance with comprehensive metrics.

    Args:
        y_true: True target values
        y_pred: Predicted values
        name: Model name for display

    Returns:
        Dictionary of evaluation metrics
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100  # Convert to percentage

    # Prediction interval accuracy (±10%)
    within_10pct = np.mean(np.abs(y_pred - y_true) / np.abs(y_true) < 0.10)

    metrics = {
        'name': name,
        'MAE': mae,
        'RMSE': rmse,
        'R²': r2,
        'MAPE': mape,
        'Within_10pct': within_10pct
    }

    return metrics


def print_metrics(metrics):
    """Print evaluation metrics in formatted output."""
    print(f"\n{metrics['name']} Performance:")
    print(f"  MAE:              ${metrics['MAE']:>10,.2f}")
    print(f"  RMSE:             ${metrics['RMSE']:>10,.2f}")
    print(f"  R²:               {metrics['R²']:>11.3f}")
    print(f"  MAPE:             {metrics['MAPE']:>10.1f}%")
    print(f"  Within ±10% band: {metrics['Within_10pct']:>10.1%}")


def train_baseline_models(X_train, y_train, X_val, y_val):
    """
    Phase 2.1: Train baseline regression models.

    Models:
    - Ridge Regression (L2 regularization)
    - Lasso Regression (L1 regularization, feature selection)
    - Decision Tree (max_depth=5)

    Args:
        X_train: Training features
        y_train: Training target
        X_val: Validation features
        y_val: Validation target

    Returns:
        Dictionary of trained models with evaluation metrics
    """
    # Scale features (important for linear models)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    models = {}
    results = {}

    # Ridge Regression
    print("\nTraining Ridge Regression...")
    ridge = Ridge(alpha=1.0, random_state=42)
    ridge.fit(X_train_scaled, y_train)
    y_pred = ridge.predict(X_val_scaled)
    models['Ridge'] = {'model': ridge, 'scaler': scaler}
    results['Ridge'] = evaluate_model(y_val, y_pred, name="Ridge Regression")
    print_metrics(results['Ridge'])

    # Lasso Regression
    print("\nTraining Lasso Regression...")
    lasso = Lasso(alpha=0.1, random_state=42, max_iter=10000)
    lasso.fit(X_train_scaled, y_train)
    y_pred = lasso.predict(X_val_scaled)
    models['Lasso'] = {'model': lasso, 'scaler': scaler}
    results['Lasso'] = evaluate_model(y_val, y_pred, name="Lasso Regression")
    print_metrics(results['Lasso'])

    # Decision Tree (no scaling needed)
    print("\nTraining Decision Tree...")
    dt = DecisionTreeRegressor(max_depth=5, random_state=42)
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_val)
    models['DecisionTree'] = {'model': dt, 'scaler': None}
    results['DecisionTree'] = evaluate_model(y_val, y_pred, name="Decision Tree (depth=5)")
    print_metrics(results['DecisionTree'])

    return models, results


def train_random_forest(X_train, y_train, X_val, y_val, tune_hyperparameters=True, cv_folds=5):
    """
    Phase 2.3: Train Random Forest with optional hyperparameter tuning.

    Random Forest advantages:
    - Handles non-linear relationships
    - Robust to outliers
    - Provides feature importance
    - No scaling required

    Args:
        X_train: Training features
        y_train: Training target
        X_val: Validation features
        y_val: Validation target
        tune_hyperparameters: If True, use GridSearchCV
        cv_folds: Number of cross-validation folds

    Returns:
        Dictionary with trained model and evaluation metrics
    """
    print("\n" + "="*70)
    print("TRAINING RANDOM FOREST")
    print("="*70)

    if tune_hyperparameters:
        print("\nPerforming hyperparameter tuning with GridSearchCV...")
        print(f"Cross-validation folds: {cv_folds}")

        # Define parameter grid
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2']
        }

        print(f"Parameter grid: {len(param_grid['n_estimators']) * len(param_grid['max_depth']) * len(param_grid['min_samples_split']) * len(param_grid['min_samples_leaf']) * len(param_grid['max_features'])} combinations")

        # Initialize base model
        rf_base = RandomForestRegressor(random_state=42, n_jobs=-1)

        # Grid search
        grid_search = GridSearchCV(
            rf_base,
            param_grid,
            cv=cv_folds,
            scoring='neg_mean_absolute_error',
            n_jobs=-1,
            verbose=1
        )

        grid_search.fit(X_train, y_train)

        print(f"\nBest parameters: {grid_search.best_params_}")
        print(f"Best CV MAE: ${-grid_search.best_score_:,.2f}")

        rf = grid_search.best_estimator_

    else:
        print("\nTraining with default hyperparameters...")
        rf = RandomForestRegressor(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        )
        rf.fit(X_train, y_train)

    # Cross-validation scores
    print(f"\nPerforming {cv_folds}-fold cross-validation...")
    cv_scores = cross_val_score(
        rf, X_train, y_train,
        cv=cv_folds,
        scoring='neg_mean_absolute_error',
        n_jobs=-1
    )
    cv_mae = -cv_scores.mean()
    cv_std = cv_scores.std()

    print(f"CV MAE: ${cv_mae:,.2f} ± ${cv_std:,.2f}")

    # Validation predictions
    y_pred = rf.predict(X_val)
    metrics = evaluate_model(y_val, y_pred, name="Random Forest")
    metrics['cv_mae'] = cv_mae
    metrics['cv_std'] = cv_std

    print_metrics(metrics)

    return {'model': rf, 'scaler': None}, metrics


def train_xgboost(X_train, y_train, X_val, y_val, tune_hyperparameters=True):
    """
    Phase 2.4: Train XGBoost with optional hyperparameter tuning.

    XGBoost advantages:
    - State-of-the-art gradient boosting
    - Handles missing values natively
    - Regularization to prevent overfitting
    - Fast training with GPU support

    Args:
        X_train: Training features
        y_train: Training target
        X_val: Validation features
        y_val: Validation target
        tune_hyperparameters: If True, use GridSearchCV

    Returns:
        Dictionary with trained model and evaluation metrics
    """
    print("\n" + "="*70)
    print("TRAINING XGBOOST")
    print("="*70)

    if tune_hyperparameters:
        print("\nPerforming hyperparameter tuning with GridSearchCV...")

        # Define parameter grid
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
            'reg_alpha': [0, 0.1, 1.0],  # L1 regularization
            'reg_lambda': [1, 5, 10]  # L2 regularization
        }

        print(f"Parameter grid: {len(param_grid['n_estimators']) * len(param_grid['max_depth']) * len(param_grid['learning_rate']) * len(param_grid['subsample']) * len(param_grid['colsample_bytree']) * len(param_grid['reg_alpha']) * len(param_grid['reg_lambda'])} combinations")

        # Initialize base model
        xgb_base = xgb.XGBRegressor(
            random_state=42,
            n_jobs=-1,
            tree_method='hist'  # Faster
        )

        # Grid search
        grid_search = GridSearchCV(
            xgb_base,
            param_grid,
            cv=5,
            scoring='neg_mean_absolute_error',
            n_jobs=-1,
            verbose=1
        )

        grid_search.fit(X_train, y_train)

        print(f"\nBest parameters: {grid_search.best_params_}")
        print(f"Best CV MAE: ${-grid_search.best_score_:,.2f}")

        model = grid_search.best_estimator_

    else:
        print("\nTraining with tuned hyperparameters...")
        model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_alpha=0.1,
            reg_lambda=5,
            random_state=42,
            n_jobs=-1,
            tree_method='hist'
        )
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    # Validation predictions
    y_pred = model.predict(X_val)
    metrics = evaluate_model(y_val, y_pred, name="XGBoost")

    print_metrics(metrics)

    return {'model': model, 'scaler': None}, metrics


def train_lightgbm(X_train, y_train, X_val, y_val, tune_hyperparameters=False):
    """
    Phase 2.4: Train LightGBM as an alternative to XGBoost.

    LightGBM advantages:
    - Faster training than XGBoost
    - Lower memory usage
    - Better with large datasets
    - Leaf-wise tree growth

    Args:
        X_train: Training features
        y_train: Training target
        X_val: Validation features
        y_val: Validation target
        tune_hyperparameters: If True, use GridSearchCV

    Returns:
        Dictionary with trained model and evaluation metrics
    """
    print("\n" + "="*70)
    print("TRAINING LIGHTGBM")
    print("="*70)

    if tune_hyperparameters:
        print("\nPerforming hyperparameter tuning with GridSearchCV...")

        # Define parameter grid
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [5, 10, 15],
            'learning_rate': [0.01, 0.05, 0.1],
            'num_leaves': [31, 50, 100],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
            'reg_alpha': [0, 0.1, 1.0],
            'reg_lambda': [1, 5, 10]
        }

        # Initialize base model
        lgb_base = lgb.LGBMRegressor(
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )

        # Grid search
        grid_search = GridSearchCV(
            lgb_base,
            param_grid,
            cv=5,
            scoring='neg_mean_absolute_error',
            n_jobs=-1,
            verbose=1
        )

        grid_search.fit(X_train, y_train)

        print(f"\nBest parameters: {grid_search.best_params_}")
        print(f"Best CV MAE: ${-grid_search.best_score_:,.2f}")

        model = grid_search.best_estimator_

    else:
        print("\nTraining with default hyperparameters...")
        model = lgb.LGBMRegressor(
            n_estimators=200,
            max_depth=10,
            learning_rate=0.05,
            num_leaves=50,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_alpha=0.1,
            reg_lambda=5,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        model.fit(X_train, y_train)

    # Validation predictions
    y_pred = model.predict(X_val)
    metrics = evaluate_model(y_val, y_pred, name="LightGBM")

    print_metrics(metrics)

    return {'model': model, 'scaler': None}, metrics


def get_feature_importance(model, feature_names, top_n=20):
    """
    Extract feature importance from tree-based models.

    Args:
        model: Trained model with feature_importances_ attribute
        feature_names: List of feature names
        top_n: Number of top features to return

    Returns:
        DataFrame with feature importance sorted descending
    """
    if hasattr(model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

        return importance_df.head(top_n)
    else:
        print("Model does not have feature_importances_ attribute")
        return None


def compare_models(results_dict, X_test=None, y_test=None, models_dict=None):
    """
    Phase 2.5: Compare all trained models and select the best.

    Args:
        results_dict: Dictionary of {model_name: metrics_dict}
        X_test: Optional test features for final evaluation
        y_test: Optional test target for final evaluation
        models_dict: Optional dictionary of {model_name: model_object}

    Returns:
        DataFrame with model comparison, best model name
    """
    print("\n" + "="*70)
    print("MODEL COMPARISON")
    print("="*70)

    # Create comparison DataFrame
    comparison_df = pd.DataFrame(results_dict).T

    # Sort by R² (descending)
    comparison_df = comparison_df.sort_values('R²', ascending=False)

    # Display comparison
    print(f"\n{'Model':<25} {'MAE':>10} {'RMSE':>10} {'R²':>8} {'MAPE':>8} {'±10%':>8}")
    print("-"*70)

    for idx, row in comparison_df.iterrows():
        print(f"{row['name']:<25} ${row['MAE']:>9,.2f} ${row['RMSE']:>9,.2f} "
              f"{row['R²']:>7.3f} {row['MAPE']:>7.1f}% {row['Within_10pct']:>7.1%}")

    # Identify best model
    best_model_name = comparison_df.index[0]
    best_metrics = comparison_df.iloc[0]

    print("\n" + "="*70)
    print("BEST MODEL")
    print("="*70)

    print(f"\nModel: {best_metrics['name']}")
    print(f"  R² Score: {best_metrics['R²']:.3f}")
    print(f"  MAE: ${best_metrics['MAE']:.2f}")
    print(f"  RMSE: ${best_metrics['RMSE']:.2f}")
    print(f"  MAPE: {best_metrics['MAPE']:.1f}%")
    print(f"  Within ±10%: {best_metrics['Within_10pct']:.1%}")

    # Optional test set evaluation
    if X_test is not None and y_test is not None and models_dict is not None:
        print("\n" + "="*70)
        print("FINAL TEST SET EVALUATION")
        print("="*70)

        best_model_obj = models_dict[best_model_name]
        model = best_model_obj['model']
        scaler = best_model_obj['scaler']

        # Scale if needed
        if scaler is not None:
            X_test_scaled = scaler.transform(X_test)
            y_pred_test = model.predict(X_test_scaled)
        else:
            y_pred_test = model.predict(X_test)

        test_metrics = evaluate_model(y_test, y_pred_test, name=f"{best_metrics['name']} (Test Set)")
        print_metrics(test_metrics)

        # Performance comparison
        print("\nValidation vs Test Performance:")
        print(f"  MAE:  Val=${best_metrics['MAE']:.2f}  |  Test=${test_metrics['MAE']:.2f}  |  Diff=${abs(best_metrics['MAE']-test_metrics['MAE']):.2f}")
        print(f"  RMSE: Val=${best_metrics['RMSE']:.2f}  |  Test=${test_metrics['RMSE']:.2f}  |  Diff=${abs(best_metrics['RMSE']-test_metrics['RMSE']):.2f}")
        print(f"  R²:   Val={best_metrics['R²']:.3f}  |  Test={test_metrics['R²']:.3f}  |  Diff={abs(best_metrics['R²']-test_metrics['R²']):.3f}")

        # Check for overfitting
        if test_metrics['R²'] < best_metrics['R²'] - 0.05:
            print("\n⚠ Warning: Potential overfitting detected (R² drops >0.05 on test set)")
        elif test_metrics['MAE'] > best_metrics['MAE'] * 1.1:
            print("\n⚠ Warning: Test MAE is >10% higher than validation MAE")
        else:
            print("\n✓ Model generalizes well to test set")

        return comparison_df, best_model_name, test_metrics

    return comparison_df, best_model_name, None


def train_all_models(X_train, y_train, X_val, y_val, feature_names,
                     tune_rf=False, tune_xgb=False, tune_lgb=False):
    """
    Convenience function to train all models at once.

    Args:
        X_train: Training features
        y_train: Training target
        X_val: Validation features
        y_val: Validation target
        feature_names: List of feature names
        tune_rf: If True, tune Random Forest hyperparameters
        tune_xgb: If True, tune XGBoost hyperparameters
        tune_lgb: If True, tune LightGBM hyperparameters

    Returns:
        Dictionary of all trained models and results
    """
    all_models = {}
    all_results = {}

    print("="*70)
    print("TRAINING ALL MODELS")
    print("="*70)

    # Baseline models
    print("\n1. Training baseline models...")
    baseline_models, baseline_results = train_baseline_models(X_train, y_train, X_val, y_val)
    all_models.update(baseline_models)
    all_results.update(baseline_results)

    # Random Forest
    print("\n2. Training Random Forest...")
    rf_model, rf_results = train_random_forest(X_train, y_train, X_val, y_val,
                                                 tune_hyperparameters=tune_rf)
    all_models['RandomForest'] = rf_model
    all_results['RandomForest'] = rf_results

    # XGBoost
    print("\n3. Training XGBoost...")
    xgb_model, xgb_results = train_xgboost(X_train, y_train, X_val, y_val,
                                             tune_hyperparameters=tune_xgb)
    all_models['XGBoost'] = xgb_model
    all_results['XGBoost'] = xgb_results

    # LightGBM
    print("\n4. Training LightGBM...")
    lgb_model, lgb_results = train_lightgbm(X_train, y_train, X_val, y_val,
                                              tune_hyperparameters=tune_lgb)
    all_models['LightGBM'] = lgb_model
    all_results['LightGBM'] = lgb_results

    print("\n" + "="*70)
    print("ALL MODELS TRAINED")
    print("="*70)

    return all_models, all_results
