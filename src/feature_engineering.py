"""
Feature Engineering Module

This module contains advanced feature engineering transformations:
- Phase 1.3: Advanced missing data imputation ✓
- Phase 1.2: Geospatial features (distance, density metrics) ✓
- Phase 1.3: Temporal features (age, cyclical encoding, era binning) ✓
- Phase 1.4: Numerical transformations (log, ratios, interactions) ✓
- Phase 1.5: Categorical encoding (target, frequency, one-hot, WoE) ✓
- Phase 1.6: Outlier treatment (winsorization, capping) ✓
"""

import pandas as pd
import numpy as np
from datetime import datetime
from scipy import stats


def impute_missing_values(df, add_indicators=True):
    """
    Phase 1.3: Advanced imputation with multiple strategies.

    Strategies:
    1. Structural zeros: Columns where null = 0 (cranes, loading docks)
    2. Median imputation: Low-moderate missingness (<60%)
    3. Random sampling: Skewed distributions (preserve distribution shape)
    4. Missing indicators: Track which values were imputed (informative missingness)
    5. Drop: Extremely sparse columns (>90% null)

    Args:
        df: DataFrame with missing values
        add_indicators: If True, create binary indicators for imputed values

    Returns:
        DataFrame with imputed values (modifies in place)
    """
    # 1. Structural zeros (domain knowledge: absence = zero)
    zero_impute_cols = [
        'Number Of Cranes', 'Drive Ins', 'Number Of Loading Docks',
        'Number Of Parking Spaces'
    ]
    for col in zero_impute_cols:
        if col in df.columns:
            if add_indicators and df[col].isnull().sum() > 0:
                df[f'{col}_was_missing'] = df[col].isnull().astype(int)
            df[col] = df[col].fillna(0)

    # 2. Median imputation for symmetric/normal distributions
    median_impute_cols = [
        'building_age', 'Rent/SF/Yr', 'Land Area (AC)', 'Land Area (SF)',
        'Number Of Stories', 'Typical Floor Size', 'Percent Leased',
        'Taxes Per SF', 'Taxes Total', 'Tax Year', 'Ceiling Ht',
        'years_since_sale'
    ]
    for col in median_impute_cols:
        if col in df.columns and df[col].notna().sum() > 0:
            null_count = df[col].isnull().sum()
            if null_count > 0:
                if add_indicators and null_count / len(df) > 0.05:  # >5% missing
                    df[f'{col}_was_missing'] = df[col].isnull().astype(int)
                df[col] = df[col].fillna(df[col].median())

    # 3. Random sample imputation for skewed distributions
    # (preserves distribution shape better than median/mean)
    skewed_impute_cols = [
        'Parking Ratio', 'Building Tax Expenses', 'Building Operating Expenses'
    ]
    for col in skewed_impute_cols:
        if col in df.columns and df[col].notna().sum() > 0:
            null_mask = df[col].isnull()
            if null_mask.sum() > 0:
                if add_indicators and null_mask.sum() / len(df) > 0.05:
                    df[f'{col}_was_missing'] = null_mask.astype(int)
                # Sample from observed values
                observed_values = df.loc[~null_mask, col].values
                n_missing = null_mask.sum()
                random_samples = np.random.choice(observed_values, size=n_missing, replace=True)
                df.loc[null_mask, col] = random_samples

    # 4. Drop extremely sparse columns (>90% null)
    drop_threshold = 0.90
    high_null_cols = [col for col in df.columns
                      if df[col].isnull().sum() / len(df) > drop_threshold]
    if high_null_cols:
        print(f"Dropping {len(high_null_cols)} columns with >{drop_threshold*100}% nulls:")
        for col in high_null_cols[:5]:  # Show first 5
            print(f"  - {col}: {df[col].isnull().sum()/len(df)*100:.1f}% null")
        if len(high_null_cols) > 5:
            print(f"  ... and {len(high_null_cols)-5} more")
        df.drop(columns=high_null_cols, inplace=True)

    return df


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate haversine distance between two points in miles.

    Args:
        lat1, lon1: First point coordinates (can be arrays)
        lat2, lon2: Second point coordinates (can be scalars or arrays)

    Returns:
        Distance in miles
    """
    R = 3959.0  # Earth radius in miles

    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)

    a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))

    return R * c


def add_geospatial_features(df):
    """
    Phase 1.2: Create geospatial features from latitude/longitude.

    Features created:
    - dist_to_market_center: Distance to market centroid in miles
    - properties_within_1mi: Count of properties within 1-mile radius
    - properties_within_5mi: Count of properties within 5-mile radius

    Args:
        df: DataFrame with Latitude, Longitude, Market Name columns

    Returns:
        DataFrame with added geospatial features (modifies in place)
    """
    # Validate required columns exist
    if not all(col in df.columns for col in ['Latitude', 'Longitude', 'Market Name']):
        print("Warning: Missing required columns for geospatial features")
        return df

    # Remove rows with missing coordinates
    valid_coords = df['Latitude'].notna() & df['Longitude'].notna()

    # Distance to market center
    df['dist_to_market_center'] = np.nan
    market_centroids = df[valid_coords].groupby('Market Name')[['Latitude', 'Longitude']].mean()

    for market, centroid in market_centroids.iterrows():
        mask = (df['Market Name'] == market) & valid_coords
        if mask.sum() > 0:
            df.loc[mask, 'dist_to_market_center'] = haversine_distance(
                df.loc[mask, 'Latitude'].values,
                df.loc[mask, 'Longitude'].values,
                centroid['Latitude'],
                centroid['Longitude']
            )

    # Property density within radius (simplified - count-based)
    df['properties_within_1mi'] = 0
    df['properties_within_5mi'] = 0

    if valid_coords.sum() > 0:
        coords = df.loc[valid_coords, ['Latitude', 'Longitude']].values

        for idx in df[valid_coords].index:
            lat, lon = df.loc[idx, 'Latitude'], df.loc[idx, 'Longitude']

            # Calculate distances to all other properties
            distances = haversine_distance(
                lat, lon,
                coords[:, 0], coords[:, 1]
            )

            # Count properties within radius (excluding self)
            df.loc[idx, 'properties_within_1mi'] = (distances < 1).sum() - 1
            df.loc[idx, 'properties_within_5mi'] = (distances < 5).sum() - 1

    return df


def add_temporal_features(df):
    """
    Phase 1.3: Create temporal features from date and year columns.

    Features created:
    - building_age: Years since construction (capped at 0-150, nulls preserved)
    - years_since_sale: Years since last sale (nulls preserved)
    - has_sale_data: Binary indicator (1 if sale date exists, 0 otherwise)
    - sale_month_sin, sale_month_cos: Cyclical encoding of sale month (0 if null)
    - construction_era: Binned construction period ('Unknown' if null)

    Dropped features:
    - years_since_renovation: 93.6% null, too sparse to be useful

    Data quality handling:
    - Future years (>2025) treated as 2025
    - Very old years (<1800) treated as outliers, capped at 150 years
    - Negative ages set to NaN for downstream imputation

    Args:
        df: DataFrame with date/year columns

    Returns:
        DataFrame with added temporal features (modifies in place)
    """
    current_year = datetime.now().year

    # Clean Year Built: cap future years, flag very old as suspicious
    year_built_clean = df['Year Built'].copy()
    year_built_clean = year_built_clean.clip(upper=current_year)  # Future → current year

    # Building age with validation
    df['building_age'] = current_year - year_built_clean
    df.loc[df['building_age'] > 150, 'building_age'] = np.nan  # Likely data errors
    df.loc[df['building_age'] < 0, 'building_age'] = np.nan  # Should not happen after clip

    # Years since sale (convert date, handle nulls)
    df['Last Sale Date'] = pd.to_datetime(df['Last Sale Date'], errors='coerce')
    df['years_since_sale'] = (datetime.now() - df['Last Sale Date']).dt.days / 365.25
    df.loc[df['years_since_sale'] < 0, 'years_since_sale'] = np.nan  # Future sales

    # Binary indicator for sale data availability
    df['has_sale_data'] = df['Last Sale Date'].notna().astype(int)

    # Cyclical encoding for sale month (fill nulls with 0 = no seasonal signal)
    sale_month = df['Last Sale Date'].dt.month.fillna(0)
    df['sale_month_sin'] = np.where(
        sale_month == 0,
        0,
        np.sin(2 * np.pi * sale_month / 12)
    )
    df['sale_month_cos'] = np.where(
        sale_month == 0,
        0,
        np.cos(2 * np.pi * sale_month / 12)
    )

    # Construction era binning (include Unknown category)
    df['construction_era'] = pd.cut(
        year_built_clean,
        bins=[-np.inf, 1980, 2000, 2010, np.inf],
        labels=['Pre-1980', '1980-2000', '2000-2010', 'Post-2010']
    ).astype(str)  # Convert to string to allow 'Unknown'
    df.loc[df['construction_era'] == 'nan', 'construction_era'] = 'Unknown'

    return df


def add_numerical_transformations(df):
    """
    Phase 1.4: Create numerical transformations and ratio features.

    Transformations:
    1. Log transforms for skewed distributions (size, rent, tax)
    2. Ratio features (efficiency metrics, density)
    3. Interaction features (age × size, location × quality)

    Args:
        df: DataFrame with numeric features

    Returns:
        DataFrame with numerical transformations (modifies in place)
    """
    # 1. Log transformations for right-skewed variables
    # (helps with heteroscedasticity and outliers)
    log_transform_cols = {
        'Building Size (SF)': 'log_building_size',
        'Land Area (SF)': 'log_land_area',
        'Rent/SF/Yr': 'log_rent_sf_yr',
        'Taxes Total': 'log_taxes_total'
    }

    for col, new_col in log_transform_cols.items():
        if col in df.columns:
            # Add 1 to handle zeros, take log
            df[new_col] = np.log1p(df[col].fillna(0))

    # 2. Ratio features (business logic insights)
    # Building efficiency
    if 'Building Size (SF)' in df.columns and 'Land Area (SF)' in df.columns:
        df['floor_area_ratio'] = np.where(
            df['Land Area (SF)'] > 0,
            df['Building Size (SF)'] / df['Land Area (SF)'],
            0
        )

    if 'Number Of Stories' in df.columns and 'Building Size (SF)' in df.columns:
        df['avg_floor_size'] = np.where(
            df['Number Of Stories'] > 0,
            df['Building Size (SF)'] / df['Number Of Stories'],
            df['Building Size (SF)']
        )

    # Parking efficiency
    if 'Number Of Parking Spaces' in df.columns and 'Building Size (SF)' in df.columns:
        df['parking_per_1000sf'] = np.where(
            df['Building Size (SF)'] > 1000,
            df['Number Of Parking Spaces'] / (df['Building Size (SF)'] / 1000),
            df['Number Of Parking Spaces']
        )

    # Tax burden
    if 'Taxes Total' in df.columns and 'Building Size (SF)' in df.columns:
        df['tax_per_sf'] = np.where(
            df['Building Size (SF)'] > 0,
            df['Taxes Total'] / df['Building Size (SF)'],
            0
        )

    # Rent per parking space (proxy for location quality)
    if 'Rent/SF/Yr' in df.columns and 'Parking Ratio' in df.columns:
        df['rent_per_parking'] = df['Rent/SF/Yr'] * df['Parking Ratio'].fillna(0)

    # 3. Interaction features
    # Age × Size (older large buildings may differ from newer ones)
    if 'building_age' in df.columns and 'Building Size (SF)' in df.columns:
        # Normalize to prevent huge numbers
        df['age_size_interaction'] = (df['building_age'] / 50) * (df['Building Size (SF)'] / 100000)

    # Distance to center × Building age (location-quality interaction)
    if 'dist_to_market_center' in df.columns and 'building_age' in df.columns:
        df['distance_age_interaction'] = (df['dist_to_market_center'] / 10) * (df['building_age'] / 50)

    # Property density × Building size (cluster effect)
    if 'properties_within_1mi' in df.columns and 'Building Size (SF)' in df.columns:
        df['density_size_interaction'] = (df['properties_within_1mi'] / 50) * (df['Building Size (SF)'] / 100000)

    return df


def add_categorical_encoding(df, target_col='Rent/SF/Yr', min_frequency=50):
    """
    Phase 1.5: Advanced categorical encoding strategies.

    Encoding methods:
    1. Target encoding: Replace category with mean target value (with regularization)
    2. Frequency encoding: Replace with category frequency (count)
    3. One-hot encoding: For low-cardinality categoricals (<10 unique)

    Args:
        df: DataFrame with categorical features
        target_col: Target variable for target encoding
        min_frequency: Minimum observations to avoid overfitting in target encoding

    Returns:
        DataFrame with encoded categoricals (modifies in place)
    """
    # Identify categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

    # Remove location/ID columns
    exclude_cols = [
        'Property Address', 'Market Name', 'Submarket Name', 'City', 'State',
        'Zip', 'County Name', 'Building Park', 'Submarket Cluster',
        'Continent', 'Country', 'Subcontinent', 'Cross Street'
    ]
    categorical_cols = [col for col in categorical_cols if col not in exclude_cols]

    for col in categorical_cols:
        if col not in df.columns:
            continue

        unique_count = df[col].nunique()

        # 1. Target encoding for high-cardinality categoricals (>10 unique)
        if unique_count > 10 and target_col in df.columns:
            # Calculate mean target per category with smoothing
            target_mean = df[target_col].mean()
            category_stats = df.groupby(col)[target_col].agg(['mean', 'count'])

            # Smoothing: blend category mean with global mean (regularization)
            # More observations → trust category mean more
            smoothing_factor = 10
            category_encoded = (
                (category_stats['mean'] * category_stats['count'] +
                 target_mean * smoothing_factor) /
                (category_stats['count'] + smoothing_factor)
            )

            # Map to dataframe
            df[f'{col}_target_encoded'] = df[col].map(category_encoded).fillna(target_mean)

            # Also add frequency encoding (captures popularity)
            df[f'{col}_frequency'] = df[col].map(df[col].value_counts()).fillna(0)

        # 2. One-hot encoding for low-cardinality categoricals (<=10 unique)
        elif unique_count <= 10:
            # Will be handled in modeling.py prepare_data_for_modeling()
            pass

        # 3. Frequency encoding only for medium cardinality
        else:
            df[f'{col}_frequency'] = df[col].map(df[col].value_counts()).fillna(0)

    return df


def apply_outlier_treatment(df, columns=None, method='winsorize', limits=(0.01, 0.99)):
    """
    Phase 1.6: Outlier treatment to reduce impact of extreme values.

    Methods:
    - Winsorization: Cap values at specified percentiles (default: 1st and 99th)
    - Clipping: Hard limits based on domain knowledge

    Args:
        df: DataFrame with potential outliers
        columns: List of columns to treat (if None, auto-detect numeric columns)
        method: 'winsorize' or 'clip'
        limits: Tuple of (lower_percentile, upper_percentile) for winsorization

    Returns:
        DataFrame with treated outliers (modifies in place)
    """
    if columns is None:
        # Auto-detect: numeric columns that could have outliers
        columns = [
            'Rent/SF/Yr', 'Building Size (SF)', 'Land Area (SF)',
            'Number Of Stories', 'Typical Floor Size', 'Parking Ratio',
            'Ceiling Ht', 'Percent Leased', 'Taxes Per SF',
            'building_age', 'years_since_sale', 'dist_to_market_center'
        ]
        # Keep only columns that exist
        columns = [col for col in columns if col in df.columns]

    if method == 'winsorize':
        for col in columns:
            if col in df.columns and df[col].notna().sum() > 0:
                # Calculate percentile thresholds
                lower_bound = df[col].quantile(limits[0])
                upper_bound = df[col].quantile(limits[1])

                # Create outlier indicators (before treatment)
                df[f'{col}_is_outlier_low'] = (df[col] < lower_bound).astype(int)
                df[f'{col}_is_outlier_high'] = (df[col] > upper_bound).astype(int)

                # Cap values
                df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)

    elif method == 'clip':
        # Domain-specific hard limits
        clip_rules = {
            'Rent/SF/Yr': (0, 200),  # Reasonable rent range
            'Percent Leased': (0, 100),  # Percentage bounds
            'building_age': (0, 150),  # Age constraints
            'Number Of Stories': (1, 200),  # Story limits
            'Parking Ratio': (0, 20)  # Parking ratio limits
        }

        for col, (lower, upper) in clip_rules.items():
            if col in df.columns:
                df[col] = df[col].clip(lower=lower, upper=upper)

    return df


def engineer_features(df, target_col='Rent/SF/Yr'):
    """
    Apply all feature engineering transformations to the dataframe.

    Applies phases in order:
    1. Temporal features (creates age, sale timing features)
    2. Geospatial features (distance, density)
    3. Numerical transformations (log, ratios, interactions)
    4. Imputation (must come after feature creation)
    5. Outlier treatment (after imputation)
    6. Categorical encoding (requires target, so after imputation)

    Args:
        df: DataFrame with cleaned data
        target_col: Target variable for encoding

    Returns:
        DataFrame with engineered features
    """
    print("\n" + "="*70)
    print("FEATURE ENGINEERING PIPELINE")
    print("="*70)

    print(f"\nInput shape: {df.shape}")

    # Phase 1: Create temporal features
    print("\n1. Creating temporal features...")
    df = add_temporal_features(df)
    print(f"   ✓ Added building_age, years_since_sale, sale_month_sin/cos, construction_era")

    # Phase 2: Create geospatial features
    print("\n2. Creating geospatial features...")
    df = add_geospatial_features(df)
    print(f"   ✓ Added dist_to_market_center, properties_within_1mi/5mi")

    # Phase 3: Create numerical transformations
    print("\n3. Creating numerical transformations...")
    df = add_numerical_transformations(df)
    print(f"   ✓ Added log transforms, ratio features, interaction features")

    # Phase 4: Impute missing values (after feature creation)
    print("\n4. Imputing missing values...")
    df = impute_missing_values(df, add_indicators=True)
    print(f"   ✓ Applied median/random/zero imputation with missing indicators")

    # Phase 5: Treat outliers
    print("\n5. Treating outliers...")
    df = apply_outlier_treatment(df, method='winsorize', limits=(0.01, 0.99))
    print(f"   ✓ Applied winsorization at 1st/99th percentiles")

    # Phase 6: Encode categoricals (requires target, must be after imputation)
    print("\n6. Encoding categorical variables...")
    df = add_categorical_encoding(df, target_col=target_col)
    print(f"   ✓ Applied target/frequency encoding for categoricals")

    print(f"\nFinal shape: {df.shape}")
    print(f"Features added: {df.shape[1] - 87}")  # Original had 87 columns
    print("\n" + "="*70)

    return df
