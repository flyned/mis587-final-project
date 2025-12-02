# Lornell Real Estate - Commercial Rent Prediction Dashboard

Interactive Streamlit dashboard for predicting commercial real estate rents with AI.

## Features

- 🏢 **Property Lookup**: Single property predictions with SHAP explanations
- 📊 **Model Insights**: Interactive visualizations of model performance
- 📁 **Batch Prediction**: Upload CSV files for bulk predictions
- ℹ️ **About**: Comprehensive documentation and methodology

## Prerequisites

- Python 3.9 or higher
- Access to trained model files (in `../models/`)
- Generated interpretation figures (in `../figures/`)

## Local Installation

### 1. Install Dependencies

```bash
cd streamlit_app
pip install -r requirements.txt
```

### 2. Run Locally

```bash
streamlit run app.py
```

The app will open in your default browser at `http://localhost:8501`

## Deployment to Streamlit Community Cloud

### 1. Prepare Your Repository

1. Push code to GitHub:
   ```bash
   git init
   git add .
   git commit -m "Add Lornell Real Estate dashboard"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
   git push -u origin main
   ```

2. **Important:** Ensure these files are included:
   - `streamlit_app/app.py`
   - `streamlit_app/pages/*.py`
   - `streamlit_app/utils/*.py`
   - `streamlit_app/requirements.txt`
   - `streamlit_app/.streamlit/config.toml`
   - `models/rf_model_local_*.pkl`
   - `models/rf_metadata_local_*.json`
   - `figures/**/*.png` (all visualization files)
   - `figures/**/*.csv` (all data files)

### 2. Deploy on Streamlit Cloud

1. Go to [streamlit.io/cloud](https://streamlit.io/cloud)
2. Click "New app"
3. Connect your GitHub repository
4. Configure deployment:
   - **Repository**: `YOUR_USERNAME/YOUR_REPO`
   - **Branch**: `main`
   - **Main file path**: `streamlit_app/app.py`
5. Click "Deploy!"

### 3. Custom Subdomain (Optional)

After deployment, you can configure a custom subdomain:
- Go to App settings → General
- Set custom subdomain: `lornell-costar-predictor`
- Final URL: `https://lornell-costar-predictor.streamlit.app`

## File Structure

```
streamlit_app/
├── app.py                      # Main landing page
├── pages/
│   ├── 1_🏢_Property_Lookup.py  # Single property prediction
│   ├── 2_📊_Model_Insights.py   # Model visualizations
│   ├── 3_📁_Batch_Prediction.py # CSV batch processing
│   └── 4_ℹ️_About.py            # Documentation
├── utils/
│   ├── predictor_wrapper.py   # Model loading & prediction
│   ├── shap_explainer.py      # SHAP explanations
│   └── data_validator.py      # Input validation
├── .streamlit/
│   └── config.toml            # Theme & server config
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Configuration

### Theme Customization

Edit `.streamlit/config.toml` to customize colors:

```toml
[theme]
primaryColor = "#0066CC"        # Lornell blue
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
```

### Server Settings

Adjust upload limits and security in `config.toml`:

```toml
[server]
maxUploadSize = 50              # Max CSV size in MB
enableXsrfProtection = true
```

## Usage

### Property Lookup

1. Navigate to "🏢 Property Lookup"
2. Enter property details (market, type, RBA, year built, etc.)
3. Click "Predict Rent"
4. View prediction with confidence interval and SHAP explanation

### Batch Prediction

1. Navigate to "📁 Batch Prediction"
2. Download CSV template (optional)
3. Upload your property data CSV
4. Click "Predict All"
5. Download results as CSV or Excel

### Model Insights

1. Navigate to "📊 Model Insights"
2. Explore tabs:
   - **Feature Importance**: What drives rents?
   - **SHAP Analysis**: Feature impact distributions
   - **Performance by Segment**: Where model excels
   - **Geographic Performance**: Market-level accuracy

## Troubleshooting

### App Won't Start

**Error:** "No module named 'streamlit'"
- **Solution:** Run `pip install -r requirements.txt`

**Error:** "No trained model found"
- **Solution:** Ensure model files exist in `../models/`
- Check path: `models/rf_model_local_*.pkl`

### Slow Performance

**Issue:** Cold start delays (30+ seconds)
- **Solution:** Pre-warm app by visiting URL 5 minutes before demo
- **Alternative:** Run local backup (`streamlit run app.py`)

**Issue:** Large file uploads timeout
- **Solution:** Split batch CSVs into <500 properties each

### Deployment Issues

**Error:** "Module not found" on Streamlit Cloud
- **Solution:** Ensure all dependencies in `requirements.txt`
- Check Python version compatibility (3.9+)

**Error:** "File not found" for model/figures
- **Solution:** Verify all files committed to GitHub
- Check relative paths in code

## Performance Optimization

### Caching

The app uses Streamlit caching for performance:
- `@st.cache_resource`: Model loading (once per session)
- `@st.cache_data`: Static data (market lists, property types)

### Memory Management

- Model file: ~45MB (loaded once, cached)
- SHAP calculations: ~2MB per property
- Batch prediction: ~500 properties recommended max

## Security

### Data Privacy

- All predictions run client-side (no external API calls)
- Uploaded CSV files are not stored
- Session data cleared on browser close

### Authentication (Optional)

To add password protection:

```python
# Add to app.py
import streamlit_authenticator as stauth

authenticator = stauth.Authenticate(
    credentials,
    'lornell_dashboard',
    'secret_key',
    30
)

name, authentication_status, username = authenticator.login('Login', 'main')

if authentication_status:
    # Show app content
    main()
```

## Support

### Issues & Bugs

Report issues at: GitHub Issues or data-science@lornell.com

### Feature Requests

Submit requests: analytics@lornell.com

### Documentation

- **Technical Details**: `../PHASE_3_SUMMARY.md`
- **Model Training**: `../TRAINING_SUMMARY.md`
- **Project Overview**: `../README.md`

## License

Proprietary - Lornell Real Estate 2024

---

**Version:** 1.0
**Last Updated:** November 29, 2024
**Maintained by:** Lornell Real Estate Data Science Team
