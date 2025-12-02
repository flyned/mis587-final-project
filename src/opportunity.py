"""
Opportunity Identification Module

This module implements Proposal Objective #4:
- Identify promising off-market or underpriced properties
- Create opportunity scores based on predicted vs actual values
- Flag value-add and distress indicators
- Generate investment recommendations

Framework:
1. Use rent prediction model to estimate fair market rent
2. Compare predicted rent vs actual rent
3. Score properties by upside potential
4. Flag opportunities by category
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import warnings
warnings.filterwarnings('ignore')


def calculate_rent_residuals(y_actual, y_predicted):
    """
    Calculate residuals (prediction errors) to identify pricing anomalies.

    Positive residual = Model predicts higher rent than actual (UNDERPRICED)
    Negative residual = Model predicts lower rent than actual (OVERPRICED)

    Args:
        y_actual: Actual rent values
        y_predicted: Model predicted rent values

    Returns:
        Dictionary with residual analysis
    """
    residuals = y_predicted - y_actual
    pct_diff = (y_predicted - y_actual) / y_actual * 100

    return {
        'residual': residuals,
        'pct_diff': pct_diff,
        'mean_residual': np.mean(residuals),
        'std_residual': np.std(residuals)
    }


def calculate_opportunity_score(y_actual, y_predicted, confidence=None):
    """
    Calculate opportunity score for each property.

    Score ranges from -100 to +100:
    - Positive scores = Underpriced (buying opportunity)
    - Negative scores = Overpriced (avoid or negotiate)

    The score considers:
    1. Percentage difference from predicted fair value
    2. Model confidence (if available)
    3. Magnitude of the opportunity

    Args:
        y_actual: Actual rent values
        y_predicted: Model predicted rent values
        confidence: Optional model confidence scores

    Returns:
        Array of opportunity scores
    """
    # Calculate percentage difference
    pct_diff = (y_predicted - y_actual) / y_actual * 100

    # Cap at ±50% to avoid extreme outliers dominating
    pct_diff_capped = np.clip(pct_diff, -50, 50)

    # Scale to -100 to +100
    opportunity_score = pct_diff_capped * 2

    # Adjust by confidence if available
    if confidence is not None:
        opportunity_score = opportunity_score * confidence

    return opportunity_score


def classify_opportunity(opportunity_score, y_actual, y_predicted):
    """
    Classify properties into opportunity categories.

    Categories:
    - Strong Buy: Score > 20 (>10% underpriced)
    - Buy: Score 10-20 (5-10% underpriced)
    - Hold/Fair Value: Score -10 to 10 (within 5% of fair value)
    - Avoid: Score -20 to -10 (5-10% overpriced)
    - Strong Avoid: Score < -20 (>10% overpriced)

    Args:
        opportunity_score: Opportunity scores
        y_actual: Actual rent values
        y_predicted: Predicted rent values

    Returns:
        Array of category labels
    """
    categories = np.empty(len(opportunity_score), dtype=object)

    categories[opportunity_score > 20] = 'Strong Buy'
    categories[(opportunity_score > 10) & (opportunity_score <= 20)] = 'Buy'
    categories[(opportunity_score >= -10) & (opportunity_score <= 10)] = 'Fair Value'
    categories[(opportunity_score >= -20) & (opportunity_score < -10)] = 'Avoid'
    categories[opportunity_score < -20] = 'Strong Avoid'

    return categories


def detect_distress_indicators(df):
    """
    Identify potential distress indicators from property characteristics.

    Distress signals:
    - High vacancy rate
    - Extended days on market
    - Below-market rent (actual < predicted)
    - Deferred maintenance indicators

    Args:
        df: DataFrame with property features

    Returns:
        DataFrame with distress flags
    """
    distress_flags = pd.DataFrame(index=df.index)

    # High vacancy
    if 'Vacancy %' in df.columns:
        distress_flags['high_vacancy'] = df['Vacancy %'] > 20

    # Extended time on market
    if 'Days On Market' in df.columns:
        distress_flags['extended_dom'] = df['Days On Market'] > 90

    # Low percent leased
    if 'Percent Leased' in df.columns:
        distress_flags['low_occupancy'] = df['Percent Leased'] < 70

    # Building age without renovation (potential deferred maintenance)
    if 'Year Built' in df.columns and 'Year Renovated' in df.columns:
        current_year = 2025
        building_age = current_year - df['Year Built']
        years_since_reno = current_year - df['Year Renovated'].fillna(df['Year Built'])
        distress_flags['potential_deferred_maintenance'] = (
            (building_age > 30) & (years_since_reno > 20)
        )

    # Count total distress flags
    distress_flags['distress_flag_count'] = distress_flags.sum(axis=1)
    distress_flags['is_distressed'] = distress_flags['distress_flag_count'] >= 2

    return distress_flags


def detect_value_add_opportunities(df, y_actual, y_predicted):
    """
    Identify value-add investment opportunities.

    Value-add signals:
    - Underpriced relative to prediction
    - Low current rent but good location
    - Older building with renovation potential
    - Below-market occupancy in strong market

    Args:
        df: DataFrame with property features
        y_actual: Actual rent values
        y_predicted: Predicted rent values

    Returns:
        DataFrame with value-add flags
    """
    value_add = pd.DataFrame(index=df.index)

    # Rent upside (underpriced)
    rent_upside_pct = (y_predicted - y_actual) / y_actual * 100
    value_add['rent_upside_pct'] = rent_upside_pct
    value_add['has_rent_upside'] = rent_upside_pct > 10

    # Renovation opportunity (old building, never renovated)
    if 'Year Built' in df.columns:
        current_year = 2025
        building_age = current_year - df['Year Built']
        value_add['building_age'] = building_age
        value_add['renovation_candidate'] = building_age > 25

        if 'Year Renovated' in df.columns:
            never_renovated = df['Year Renovated'].isna()
            value_add['never_renovated'] = never_renovated
            value_add['renovation_opportunity'] = (
                (building_age > 25) & never_renovated & (rent_upside_pct > 5)
            )

    # Lease-up opportunity (low occupancy in underpriced property)
    if 'Percent Leased' in df.columns:
        value_add['lease_up_opportunity'] = (
            (df['Percent Leased'] < 80) & (rent_upside_pct > 5)
        )

    # Overall value-add score
    value_add_cols = [col for col in value_add.columns if col.startswith('has_') or
                     col.endswith('_opportunity') or col == 'renovation_candidate']
    if value_add_cols:
        value_add['value_add_score'] = value_add[value_add_cols].sum(axis=1)
        value_add['is_value_add'] = value_add['value_add_score'] >= 2

    return value_add


def identify_anomalies(X, contamination=0.05):
    """
    Use Isolation Forest to identify anomalous properties.

    Anomalies may represent:
    - Data errors (verify before acting)
    - Unique properties (special circumstances)
    - Market inefficiencies (opportunities)

    Args:
        X: Feature matrix
        contamination: Expected proportion of anomalies

    Returns:
        Array of anomaly labels (-1 for anomaly, 1 for normal)
    """
    iso_forest = IsolationForest(
        contamination=contamination,
        random_state=42,
        n_jobs=1
    )

    anomaly_labels = iso_forest.fit_predict(X)
    anomaly_scores = iso_forest.decision_function(X)

    return anomaly_labels, anomaly_scores


def generate_opportunity_report(df, y_actual, y_predicted, X=None, top_n=20):
    """
    Generate comprehensive opportunity identification report.

    Args:
        df: Original DataFrame with property features
        y_actual: Actual rent values
        y_predicted: Model predicted rent values
        X: Optional feature matrix for anomaly detection
        top_n: Number of top opportunities to highlight

    Returns:
        Dictionary with opportunity analysis results
    """
    print("\n" + "="*70)
    print("OPPORTUNITY IDENTIFICATION REPORT")
    print("="*70)

    # Calculate opportunity scores
    scores = calculate_opportunity_score(y_actual, y_predicted)
    categories = classify_opportunity(scores, y_actual, y_predicted)

    # Create results DataFrame
    results = pd.DataFrame({
        'actual_rent': y_actual,
        'predicted_rent': y_predicted,
        'rent_diff': y_predicted - y_actual,
        'rent_diff_pct': (y_predicted - y_actual) / y_actual * 100,
        'opportunity_score': scores,
        'category': categories
    }, index=df.index)

    # Add property identifiers if available
    if 'Property Address' in df.columns:
        results['property_address'] = df['Property Address'].values
    if 'Market Name' in df.columns:
        results['market'] = df['Market Name'].values
    if 'City' in df.columns:
        results['city'] = df['City'].values

    # Summary statistics
    print(f"\n{'Category':<15} {'Count':>8} {'Avg Score':>12} {'Avg Upside':>12}")
    print("-"*50)

    for cat in ['Strong Buy', 'Buy', 'Fair Value', 'Avoid', 'Strong Avoid']:
        mask = results['category'] == cat
        count = mask.sum()
        if count > 0:
            avg_score = results.loc[mask, 'opportunity_score'].mean()
            avg_upside = results.loc[mask, 'rent_diff_pct'].mean()
            print(f"{cat:<15} {count:>8} {avg_score:>12.1f} {avg_upside:>11.1f}%")

    # Detect distress and value-add
    print("\n" + "-"*50)
    print("DISTRESS ANALYSIS")

    distress_flags = detect_distress_indicators(df)
    distress_count = distress_flags['is_distressed'].sum()
    print(f"Distressed properties identified: {distress_count} ({distress_count/len(df)*100:.1f}%)")

    print("\n" + "-"*50)
    print("VALUE-ADD ANALYSIS")

    value_add = detect_value_add_opportunities(df, y_actual, y_predicted)
    if 'is_value_add' in value_add.columns:
        value_add_count = value_add['is_value_add'].sum()
        print(f"Value-add opportunities identified: {value_add_count} ({value_add_count/len(df)*100:.1f}%)")

    # Combine all analysis
    results = results.join(distress_flags, rsuffix='_distress')
    results = results.join(value_add, rsuffix='_valueadd')

    # Anomaly detection
    if X is not None:
        print("\n" + "-"*50)
        print("ANOMALY DETECTION")

        anomaly_labels, anomaly_scores = identify_anomalies(X)
        results['is_anomaly'] = anomaly_labels == -1
        results['anomaly_score'] = anomaly_scores

        anomaly_count = (anomaly_labels == -1).sum()
        print(f"Anomalous properties detected: {anomaly_count} ({anomaly_count/len(df)*100:.1f}%)")

    # Top opportunities
    print("\n" + "="*70)
    print(f"TOP {top_n} BUYING OPPORTUNITIES")
    print("="*70)

    top_opportunities = results.nlargest(top_n, 'opportunity_score')

    print(f"\n{'#':>3} {'Score':>7} {'Upside':>8} {'Actual':>10} {'Predicted':>10} {'Category':<12}")
    print("-"*60)

    for i, (idx, row) in enumerate(top_opportunities.iterrows(), 1):
        print(f"{i:>3} {row['opportunity_score']:>7.1f} {row['rent_diff_pct']:>7.1f}% "
              f"${row['actual_rent']:>9.2f} ${row['predicted_rent']:>9.2f} {row['category']:<12}")

    return {
        'results': results,
        'summary': {
            'total_properties': len(df),
            'strong_buy_count': (categories == 'Strong Buy').sum(),
            'buy_count': (categories == 'Buy').sum(),
            'fair_value_count': (categories == 'Fair Value').sum(),
            'avoid_count': (categories == 'Avoid').sum(),
            'strong_avoid_count': (categories == 'Strong Avoid').sum(),
            'distressed_count': distress_count,
            'value_add_count': value_add_count if 'is_value_add' in value_add.columns else 0
        },
        'top_opportunities': top_opportunities,
        'distress_flags': distress_flags,
        'value_add': value_add
    }


def get_investment_recommendation(property_row):
    """
    Generate specific investment recommendation for a property.

    Args:
        property_row: Series with property analysis results

    Returns:
        Dictionary with detailed recommendation
    """
    score = property_row.get('opportunity_score', 0)
    category = property_row.get('category', 'Fair Value')
    is_distressed = property_row.get('is_distressed', False)
    is_value_add = property_row.get('is_value_add', False)
    rent_upside = property_row.get('rent_diff_pct', 0)

    # Base recommendation from category
    recommendations = {
        'Strong Buy': {
            'action': 'BUY',
            'confidence': 'High',
            'strategy': 'Acquire at or near asking price. Property appears significantly underpriced.',
            'timeline': 'Immediate'
        },
        'Buy': {
            'action': 'BUY',
            'confidence': 'Medium-High',
            'strategy': 'Pursue acquisition with standard due diligence. Moderate upside potential.',
            'timeline': '30-60 days'
        },
        'Fair Value': {
            'action': 'HOLD/MONITOR',
            'confidence': 'Medium',
            'strategy': 'Property is fairly priced. Consider if strategic fit or unique value.',
            'timeline': 'As needed'
        },
        'Avoid': {
            'action': 'PASS',
            'confidence': 'Medium',
            'strategy': 'Property appears overpriced. Only pursue if significant negotiation possible.',
            'timeline': 'N/A'
        },
        'Strong Avoid': {
            'action': 'PASS',
            'confidence': 'High',
            'strategy': 'Property is significantly overpriced. Do not pursue unless price drops substantially.',
            'timeline': 'N/A'
        }
    }

    rec = recommendations.get(category, recommendations['Fair Value']).copy()

    # Adjust for distress
    if is_distressed:
        rec['distress_note'] = 'DISTRESS DETECTED: May offer additional negotiation leverage, but verify underlying issues.'
        if category in ['Buy', 'Strong Buy']:
            rec['strategy'] += ' Distressed asset may provide additional upside through turnaround.'

    # Adjust for value-add
    if is_value_add:
        rec['value_add_note'] = f'VALUE-ADD OPPORTUNITY: Potential rent upside of {rent_upside:.1f}% through improvements/repositioning.'

    # Add specific metrics
    rec['opportunity_score'] = score
    rec['rent_upside_pct'] = rent_upside

    return rec
