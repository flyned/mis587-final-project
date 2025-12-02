"""
Rent Predictor Page
===================

Single property rent prediction with SHAP explanations.
Includes address-based property lookup from training data.
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add parent and project directories to path
parent_dir = Path(__file__).parent.parent
project_dir = parent_dir.parent
sys.path.insert(0, str(parent_dir))
sys.path.insert(0, str(project_dir))

from utils.predictor_wrapper import (
    load_model_and_metadata,
    get_market_options,
    get_property_type_options,
    predict_single_property
)
from utils.shap_explainer import (
    generate_shap_force_plot,
    create_shap_waterfall_text
)
from utils.data_validator import validate_property_input, show_validation_results
from utils.market_comparison import get_market_coordinates
from utils.address_search import search_properties_by_address, display_search_results

# Page config
st.set_page_config(page_title="Rent Predictor", page_icon="🏢", layout="wide")

st.title("Property Rent Prediction")
st.markdown("Get instant rent predictions with detailed explanations for any commercial property.")

# Load model (cached) with model type parameter
@st.cache_resource
def load_model(model_type="random_forest"):
    """Load model and metadata (cached)."""
    model_dir = project_dir / "models"

    if model_type == "neural_network":
        # Find neural network models
        model_files = list(model_dir.glob("nn_model_*.keras"))
        if not model_files:
            return None, None  # Return None if no NN model found

        latest_model = sorted(model_files)[-1]
        metadata_file = latest_model.with_name(latest_model.stem.replace('nn_model', 'nn_metadata') + '.json')
    else:
        # Find Random Forest models
        model_files = list(model_dir.glob("rf_model_local_*.pkl"))
        if not model_files:
            st.error("No trained model found. Please train a model first.")
            st.stop()

        latest_model = sorted(model_files)[-1]
        metadata_file = latest_model.with_name(latest_model.stem.replace('rf_model', 'rf_metadata') + '.json')

    if not metadata_file.exists():
        st.error(f"Metadata file not found: {metadata_file}")
        st.stop()

    return load_model_and_metadata(str(latest_model), str(metadata_file))

# Check available models
model_dir = project_dir / "models"
has_rf_model = len(list(model_dir.glob("rf_model_local_*.pkl"))) > 0
has_nn_model = len(list(model_dir.glob("nn_model_*.keras"))) > 0

# Model selection in sidebar
with st.sidebar:
    st.markdown("### Model Selection")

    # Build model options based on available models
    model_options = []
    model_option_labels = []

    if has_rf_model:
        model_options.append("random_forest")
        model_option_labels.append("Random Forest (Default)")

    if has_nn_model:
        model_options.append("neural_network")
        model_option_labels.append("Neural Network")

    if not model_options:
        st.error("No trained models found. Please train a model first.")
        st.stop()

    # Model selector
    selected_model_label = st.selectbox(
        "Select Model",
        options=model_option_labels,
        index=0,
        help="Choose which model to use for predictions"
    )

    # Map label back to model type
    selected_model_type = model_options[model_option_labels.index(selected_model_label)]

# Load the selected model
model, metadata = load_model(selected_model_type)

if model is None:
    st.error(f"Failed to load {selected_model_type} model.")
    st.stop()

# Sidebar info
with st.sidebar:
    val_metrics = metadata.get('validation_metrics', {})
    r2_val = val_metrics.get('r2', None)
    mae_val = val_metrics.get('mae', None)

    r2_str = f"{r2_val:.3f}" if r2_val is not None else "N/A"
    mae_str = f"${mae_val:.2f}/SF/Yr" if mae_val is not None else "N/A"

    st.info(f"""
    **Model Information**
    - Type: {metadata.get('model_type', 'Random Forest')}
    - Trained on: {metadata.get('training_samples', 'N/A')} properties
    - Validation R²: {r2_str}
    - MAE: {mae_str}
    """)

# Get market coordinates for auto-population
market_coords = get_market_coordinates()

# Data path for address search
data_dir = project_dir.parent / "data"
train_data_path = str(data_dir / "train.csv")

# Initialize session state for selected property
if 'selected_property' not in st.session_state:
    st.session_state.selected_property = None

# Address Search Section
st.markdown("---")
st.subheader("Search Existing Property")
st.markdown("Search for a property from our database to auto-fill details, or enter manually below.")

search_col1, search_col2 = st.columns([3, 1])
with search_col1:
    address_query = st.text_input(
        "Search by address",
        placeholder="e.g., 123 Main St, Boston",
        help="Type an address to search our property database"
    )
with search_col2:
    search_button = st.button("Search", use_container_width=True)

# Handle address search
if search_button and address_query:
    with st.spinner("Searching..."):
        search_results = search_properties_by_address(
            address_query,
            train_data_path,
            max_results=10,
            min_score=30
        )
        st.session_state.search_results = search_results

# Display search results if available
if 'search_results' in st.session_state and st.session_state.search_results:
    selected = display_search_results(st.session_state.search_results)
    if selected:
        st.session_state.selected_property = selected
        st.success("Property selected! Form auto-filled below. Click 'Predict Rent' to see prediction.")

st.markdown("---")

# Main content
col1, col2 = st.columns([1, 1])

# Get default values from selected property or use defaults
selected_prop = st.session_state.selected_property

with col1:
    st.subheader("Property Details")

    # Market selection OUTSIDE form so it can update coordinates dynamically
    st.markdown("**Location**")

    # Get market options and find default index
    market_options = get_market_options()
    default_market_idx = 0
    if selected_prop and selected_prop.get('Market Name') in market_options:
        default_market_idx = market_options.index(selected_prop['Market Name'])

    market_name = st.selectbox(
        "Market",
        options=market_options,
        index=default_market_idx,
        help="Major commercial real estate market",
        key="market_selector"
    )

    # Get coordinates for selected market (or from selected property)
    if selected_prop and selected_prop.get('Latitude'):
        default_lat = selected_prop['Latitude']
        default_lon = selected_prop['Longitude']
    elif market_name in market_coords:
        default_lat, default_lon = market_coords[market_name]
    else:
        default_lat, default_lon = 42.3601, -71.0589  # Boston fallback

    st.caption(f"Using {market_name} center coordinates: ({default_lat:.4f}, {default_lon:.4f})")

    # Form with coordinates now auto-populated based on market
    with st.form("property_form"):
        col_lat, col_lon = st.columns(2)
        with col_lat:
            latitude = st.number_input(
                "Latitude",
                min_value=24.0,
                max_value=50.0,
                value=default_lat,
                step=0.0001,
                format="%.4f",
                help="Auto-filled from market center (you can adjust for specific location)"
            )

        with col_lon:
            longitude = st.number_input(
                "Longitude",
                min_value=-125.0,
                max_value=-65.0,
                value=default_lon,
                step=0.0001,
                format="%.4f",
                help="Auto-filled from market center (you can adjust for specific location)"
            )

        # Building characteristics
        st.markdown("**Building Characteristics**")

        # Get property type options and find default index
        prop_type_options = get_property_type_options()
        default_type_idx = 0
        if selected_prop and selected_prop.get('Secondary Type') in prop_type_options:
            default_type_idx = prop_type_options.index(selected_prop['Secondary Type'])

        secondary_type = st.selectbox(
            "Secondary Type",
            options=prop_type_options,
            index=default_type_idx,
            help="Secondary property type/use (e.g., Warehouse, Manufacturing, Service)"
        )

        col_rba, col_year = st.columns(2)
        with col_rba:
            # Get default RBA from selected property
            default_rba = int(selected_prop['RBA']) if selected_prop and selected_prop.get('RBA') else 50_000
            default_rba = max(100, min(10_000_000, default_rba))  # Clamp to valid range

            rba = st.number_input(
                "RBA (Rentable Building Area)",
                min_value=100,
                max_value=10_000_000,
                value=default_rba,
                step=1000,
                help="Total rentable square footage"
            )

        with col_year:
            # Get default year from selected property
            default_year = int(selected_prop['Year Built']) if selected_prop and selected_prop.get('Year Built') else 2000
            default_year = max(1800, min(2025, default_year))  # Clamp to valid range

            year_built = st.number_input(
                "Year Built",
                min_value=1800,
                max_value=2025,
                value=default_year,
                step=1,
                help="Year property was constructed"
            )

        # Get default stories from selected property
        default_stories = int(selected_prop.get('Stories') or selected_prop.get('Number Of Stories') or 3) if selected_prop else 3
        default_stories = max(1, min(200, default_stories))  # Clamp to valid range

        stories = st.number_input(
            "Number of Stories",
            min_value=1,
            max_value=200,
            value=default_stories,
            step=1,
            help="Number of floors/stories"
        )

        # Submit button
        submit = st.form_submit_button("Predict Rent", use_container_width=True)

# Prediction results
if submit:
    # Build property data dictionary
    property_data = {
        'Market Name': market_name,
        'Secondary Type': secondary_type,
        'RBA': rba,
        'Year Built': year_built,
        'Stories': stories,
        'Latitude': latitude,
        'Longitude': longitude
    }

    # Validate input
    is_valid, errors = validate_property_input(property_data)

    if not show_validation_results(is_valid, errors):
        st.stop()

    # Make prediction
    with st.spinner("Analyzing property..."):
        try:
            result = predict_single_property(model, metadata, property_data)

            # Display results in right column
            with col2:
                st.subheader("Prediction Results")

                # Main prediction with confidence
                pred = result['prediction']
                ci_lower = result['ci_lower']
                ci_upper = result['ci_upper']
                confidence = result['confidence_level']

                # Color-code confidence
                conf_colors = {
                    'HIGH': '#4CAF50',
                    'MEDIUM': '#FF9800',
                    'LOW': '#F44336'
                }
                conf_color = conf_colors.get(confidence, '#666')

                st.markdown(f"""
                <div style="background-color: #F0F2F6; padding: 2rem; border-radius: 0.5rem; margin: 1rem 0;">
                    <h2 style="color: #0066CC; margin: 0;">${pred:.2f}/SF/Yr</h2>
                    <p style="color: #666; margin: 0.5rem 0 0 0;">Predicted Annual Rent</p>
                    <hr style="margin: 1rem 0;">
                    <p style="margin: 0.25rem 0;"><strong>95% Confidence Interval:</strong></p>
                    <p style="margin: 0.25rem 0; color: #666;">${ci_lower:.2f} - ${ci_upper:.2f} per SF/Yr</p>
                    <p style="margin: 0.25rem 0;"><strong>Confidence Level:</strong>
                        <span style="color: {conf_color}; font-weight: bold;">{confidence}</span>
                    </p>
                </div>
                """, unsafe_allow_html=True)

                # Confidence explanation
                if confidence == 'HIGH':
                    st.success("**High Confidence** - This prediction is reliable. The property matches our training data well.")
                elif confidence == 'MEDIUM':
                    st.warning("**Medium Confidence** - Use this prediction with caution. Consider additional validation.")
                else:
                    st.error("**Low Confidence** - Manual appraisal strongly recommended. Property characteristics are outside typical range.")

            # SHAP explanation section
            st.markdown("---")
            st.subheader("Why This Prediction?")
            st.markdown("See how different property features contribute to the predicted rent:")

            with st.spinner("Generating explanation..."):
                feature_impacts, base_value = generate_shap_force_plot(
                    model,
                    result['X'],
                    result['feature_names']
                )

                # Create waterfall explanation
                explanation_text = create_shap_waterfall_text(
                    feature_impacts,
                    base_value,
                    pred
                )

                # Display in two columns
                col_exp1, col_exp2 = st.columns(2)

                with col_exp1:
                    st.markdown("**Feature Contributions**")
                    st.markdown(explanation_text)

                with col_exp2:
                    st.markdown("**Top Impact Features**")

                    # Create bar chart of top features
                    import plotly.graph_objects as go

                    features = [f['feature'].replace('_', ' ').title()[:30] for f in feature_impacts]
                    values = [f['shap_value'] for f in feature_impacts]
                    colors = ['#4CAF50' if v > 0 else '#F44336' for v in values]

                    fig = go.Figure(go.Bar(
                        x=values,
                        y=features,
                        orientation='h',
                        marker=dict(color=colors),
                        text=[f"+${abs(v):.2f}" if v > 0 else f"-${abs(v):.2f}" for v in values],
                        textposition='auto',
                    ))

                    fig.update_layout(
                        title="Feature Impact on Rent Prediction",
                        xaxis_title="Impact ($/SF/Yr)",
                        yaxis_title="",
                        height=400,
                        margin=dict(l=0, r=0, t=40, b=0)
                    )

                    st.plotly_chart(fig, use_container_width=True)

            # Additional insights
            st.markdown("---")
            st.subheader("Additional Insights")

            col_insights1, col_insights2, col_insights3 = st.columns(3)

            with col_insights1:
                building_age = 2025 - year_built
                st.metric(
                    label="Building Age",
                    value=f"{building_age} years",
                    help="Age of the building"
                )

            with col_insights2:
                price_per_sf = pred
                annual_revenue = price_per_sf * rba
                st.metric(
                    label="Est. Annual Revenue",
                    value=f"${annual_revenue:,.0f}",
                    help="Predicted annual rental revenue (rent × RBA)"
                )

            with col_insights3:
                # Classify price range
                if pred < 5:
                    price_cat = "Budget"
                elif pred < 10:
                    price_cat = "Standard"
                elif pred < 20:
                    price_cat = "Premium"
                else:
                    price_cat = "Luxury"

                st.metric(
                    label="Price Category",
                    value=price_cat,
                    help="Classification based on $/SF/Yr"
                )

            # Reference Properties Section
            st.markdown("---")
            st.markdown("## Reference Properties")
            st.markdown("""
            See which actual properties from our training data are most similar to your input.
            This provides transparency and traditional "comp-based" validation.
            """)

            with st.spinner("Finding comparable properties..."):
                from utils.comparable_properties import (
                    find_market_comparables,
                    get_market_statistics,
                    display_market_comparables,
                    display_market_statistics
                )

                # Tab layout for different reference types
                tab1, tab2 = st.tabs(["Market Comps", "Market Stats"])

                with tab1:
                    # Find market comparables
                    comps = find_market_comparables(
                        property_data,
                        str(train_data_path),
                        n_comps=10
                    )
                    display_market_comparables(comps)

                with tab2:
                    # Get market statistics
                    stats = get_market_statistics(market_name, str(train_data_path))
                    display_market_statistics(stats)

        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            st.exception(e)

else:
    with col2:
        st.info("Enter property details and click **Predict Rent** to see results")
