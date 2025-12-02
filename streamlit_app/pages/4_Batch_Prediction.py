"""
Batch Prediction Page
=====================

Upload CSV files for batch rent predictions.
"""

import streamlit as st
import pandas as pd
from pathlib import Path
import sys
from io import BytesIO

# Add paths
parent_dir = Path(__file__).parent.parent
project_dir = parent_dir.parent
sys.path.insert(0, str(parent_dir))
sys.path.insert(0, str(project_dir))

from utils.predictor_wrapper import load_model_and_metadata, predict_batch
from utils.data_validator import validate_batch_csv, show_validation_results

# Page config
st.set_page_config(page_title="Batch Prediction", page_icon="📁", layout="wide")

st.title("Batch Rent Prediction")
st.markdown("Upload a CSV file to predict rents for multiple properties at once.")

# Load model
@st.cache_resource
def load_model():
    """Load model and metadata (cached)."""
    model_dir = project_dir / "models"
    model_files = list(model_dir.glob("rf_model_local_*.pkl"))
    if not model_files:
        st.error("No trained model found.")
        st.stop()

    latest_model = sorted(model_files)[-1]
    metadata_file = latest_model.with_name(latest_model.stem.replace('rf_model', 'rf_metadata') + '.json')

    return load_model_and_metadata(str(latest_model), str(metadata_file))

model, metadata = load_model()

# Instructions
with st.expander("CSV Format Requirements", expanded=False):
    st.markdown("""
    ### Required Columns:
    - **Market Name**: Market where property is located
    - **Property Type**: Type of commercial property (Office, Industrial, Retail, etc.)
    - **RBA**: Rentable Building Area in square feet
    - **Year Built**: Year property was constructed

    ### Optional Columns (Recommended):
    - **Latitude**: Property latitude (improves accuracy)
    - **Longitude**: Property longitude (improves accuracy)
    - **Stories**: Number of floors
    - **Property Address**: For reference (not used in prediction)

    ### Example CSV Structure:
    ```
    Market Name,Property Type,RBA,Year Built,Latitude,Longitude,Stories,Property Address
    Boston, MA,Office,50000,2000,42.3601,-71.0589,5,123 Main St
    Worcester, MA,Industrial,25000,1995,42.2626,-71.8023,3,456 Industrial Way
    ```
    """)

    # Download template button
    template_df = pd.DataFrame({
        'Market Name': ['Boston, MA', 'Worcester, MA'],
        'Property Type': ['Office', 'Industrial'],
        'RBA': [50000, 25000],
        'Year Built': [2000, 1995],
        'Latitude': [42.3601, 42.2626],
        'Longitude': [-71.0589, -71.8023],
        'Stories': [5, 3],
        'Property Address': ['123 Main St, Boston', '456 Industrial Way, Worcester']
    })

    csv_template = template_df.to_csv(index=False)

    st.download_button(
        label="📥 Download CSV Template",
        data=csv_template,
        file_name="batch_prediction_template.csv",
        mime="text/csv",
        help="Download a template CSV file to get started"
    )

# File upload
st.markdown("---")
st.subheader("Upload Your Data")

uploaded_file = st.file_uploader(
    "Choose a CSV file",
    type=['csv'],
    help="Upload a CSV file with your property data"
)

if uploaded_file is not None:
    try:
        # Load CSV
        df = pd.read_csv(uploaded_file)

        st.success(f"File uploaded successfully: {len(df)} properties found")

        # Preview
        st.subheader("Data Preview")
        st.dataframe(df.head(10), use_container_width=True)

        # Validate
        is_valid, errors, warnings = validate_batch_csv(df)

        if not show_validation_results(is_valid, errors, warnings):
            st.stop()

        # Prediction button
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 1, 1])

        with col2:
            predict_button = st.button(
                "Predict All",
                use_container_width=True,
                type="primary"
            )

        if predict_button:
            with st.spinner(f"Generating predictions for {len(df)} properties..."):
                try:
                    # Make predictions
                    results_df = predict_batch(model, metadata, df)

                    st.success(f"Predictions complete for {len(results_df)} properties!")

                    # Show results
                    st.markdown("---")
                    st.subheader("Prediction Results")

                    # Summary statistics
                    col_stats1, col_stats2, col_stats3, col_stats4 = st.columns(4)

                    with col_stats1:
                        avg_rent = results_df['Predicted_Rent_SF_Yr'].mean()
                        st.metric("Average Rent", f"${avg_rent:.2f}/SF/Yr")

                    with col_stats2:
                        min_rent = results_df['Predicted_Rent_SF_Yr'].min()
                        st.metric("Minimum Rent", f"${min_rent:.2f}/SF/Yr")

                    with col_stats3:
                        max_rent = results_df['Predicted_Rent_SF_Yr'].max()
                        st.metric("Maximum Rent", f"${max_rent:.2f}/SF/Yr")

                    with col_stats4:
                        high_conf_pct = (results_df['Confidence_Level'] == 'HIGH').sum() / len(results_df) * 100
                        st.metric("High Confidence", f"{high_conf_pct:.1f}%")

                    # Confidence breakdown
                    st.markdown("**Confidence Level Breakdown:**")
                    conf_counts = results_df['Confidence_Level'].value_counts()

                    col_conf1, col_conf2, col_conf3 = st.columns(3)

                    with col_conf1:
                        high_count = conf_counts.get('HIGH', 0)
                        st.success(f"**HIGH**: {high_count} properties ({high_count/len(results_df)*100:.1f}%)")

                    with col_conf2:
                        med_count = conf_counts.get('MEDIUM', 0)
                        st.warning(f"**MEDIUM**: {med_count} properties ({med_count/len(results_df)*100:.1f}%)")

                    with col_conf3:
                        low_count = conf_counts.get('LOW', 0)
                        st.error(f"**LOW**: {low_count} properties ({low_count/len(results_df)*100:.1f}%)")

                    # Full results table
                    st.markdown("---")
                    st.subheader("Detailed Results")

                    # Add color coding for confidence levels
                    def highlight_confidence(row):
                        if row['Confidence_Level'] == 'HIGH':
                            return ['background-color: #E8F5E9'] * len(row)
                        elif row['Confidence_Level'] == 'MEDIUM':
                            return ['background-color: #FFF3E0'] * len(row)
                        else:
                            return ['background-color: #FFEBEE'] * len(row)

                    st.dataframe(
                        results_df.style.apply(highlight_confidence, axis=1),
                        use_container_width=True
                    )

                    # Download results
                    st.markdown("---")
                    st.subheader("Download Results")

                    col_dl1, col_dl2 = st.columns(2)

                    with col_dl1:
                        # CSV download
                        csv_output = results_df.to_csv(index=False)

                        st.download_button(
                            label="📥 Download as CSV",
                            data=csv_output,
                            file_name="batch_predictions.csv",
                            mime="text/csv",
                            use_container_width=True
                        )

                    with col_dl2:
                        # Excel download
                        output = BytesIO()
                        with pd.ExcelWriter(output, engine='openpyxl') as writer:
                            results_df.to_excel(writer, index=False, sheet_name='Predictions')

                            # Add summary sheet
                            summary_df = pd.DataFrame({
                                'Metric': [
                                    'Total Properties',
                                    'Average Rent',
                                    'Minimum Rent',
                                    'Maximum Rent',
                                    'High Confidence Count',
                                    'Medium Confidence Count',
                                    'Low Confidence Count'
                                ],
                                'Value': [
                                    len(results_df),
                                    f"${results_df['Predicted_Rent_SF_Yr'].mean():.2f}/SF/Yr",
                                    f"${results_df['Predicted_Rent_SF_Yr'].min():.2f}/SF/Yr",
                                    f"${results_df['Predicted_Rent_SF_Yr'].max():.2f}/SF/Yr",
                                    conf_counts.get('HIGH', 0),
                                    conf_counts.get('MEDIUM', 0),
                                    conf_counts.get('LOW', 0)
                                ]
                            })
                            summary_df.to_excel(writer, index=False, sheet_name='Summary')

                        excel_output = output.getvalue()

                        st.download_button(
                            label="📥 Download as Excel",
                            data=excel_output,
                            file_name="batch_predictions.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            use_container_width=True
                        )

                except Exception as e:
                    st.error(f"Error during prediction: {str(e)}")
                    st.exception(e)

    except Exception as e:
        st.error(f"Error reading CSV file: {str(e)}")
        st.exception(e)

else:
    st.info("👆 Upload a CSV file to get started")
