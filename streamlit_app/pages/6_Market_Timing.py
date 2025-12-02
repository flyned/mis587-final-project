"""
Market Timing Page
==================

Days on Market prediction with bidding strategy recommendations.
Implements Proposal Objective #2: Market Timing Forecast.
"""

import streamlit as st
from pathlib import Path
import sys

# Add parent and project directories to path
parent_dir = Path(__file__).parent.parent
project_dir = parent_dir.parent
sys.path.insert(0, str(parent_dir))
sys.path.insert(0, str(project_dir))

from utils.predictor_wrapper import get_market_options, get_property_type_options
from utils.market_comparison import get_market_coordinates

# Page config
st.set_page_config(page_title="Market Timing", page_icon="⏱️", layout="wide")

st.title("Market Timing Forecast")
st.markdown("Predict how long a property will stay on market and get bidding strategy recommendations.")


def classify_market_timing(predicted_dom):
    """Classify into market timing categories."""
    if predicted_dom <= 7:
        return 'Quick Sale'
    elif predicted_dom <= 30:
        return 'Normal'
    elif predicted_dom <= 90:
        return 'Slow'
    else:
        return 'Extended'


def get_bidding_strategy(predicted_dom, timing_class):
    """Get bidding strategy based on DOM prediction."""
    strategies = {
        'Quick Sale': {
            'urgency': 'High',
            'urgency_color': '#F44336',
            'bid_premium': '+2-5%',
            'recommendation': 'Act quickly. Property likely to receive multiple offers. Consider offering above asking price or waiving contingencies.',
            'negotiation_power': 'Low',
            'icon': '🔥'
        },
        'Normal': {
            'urgency': 'Medium',
            'urgency_color': '#FF9800',
            'bid_premium': '0%',
            'recommendation': 'Standard market conditions. Offer at or slightly below asking. Normal contingencies acceptable.',
            'negotiation_power': 'Medium',
            'icon': '⚖️'
        },
        'Slow': {
            'urgency': 'Low',
            'urgency_color': '#4CAF50',
            'bid_premium': '-5-10%',
            'recommendation': 'Property has been on market longer than average. Seller may be motivated. Consider below-asking offer.',
            'negotiation_power': 'High',
            'icon': '💰'
        },
        'Extended': {
            'urgency': 'Very Low',
            'urgency_color': '#2196F3',
            'bid_premium': '-10-20%',
            'recommendation': 'Property has significant time on market. Strong negotiation position. Investigate reasons for slow sale.',
            'negotiation_power': 'Very High',
            'icon': '🎯'
        }
    }
    return strategies.get(timing_class, strategies['Normal'])


# Sidebar info
with st.sidebar:
    st.info("""
    **Model Information**
    - Type: Heuristic Model
    - Target: Days on Market
    - Based on: Domain expertise and data analysis

    *Note: Uses rule-based scoring based on property characteristics that influence market timing.*
    """)

    st.markdown("---")
    st.markdown("""
    **Market Timing Categories**
    - 🔥 **Quick Sale** (0-7 days)
    - ⚖️ **Normal** (8-30 days)
    - 💰 **Slow** (31-90 days)
    - 🎯 **Extended** (90+ days)
    """)

# Get market coordinates
market_coords = get_market_coordinates()

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Property Details")

    # Market selection
    st.markdown("**Location**")
    market_name = st.selectbox(
        "Market",
        options=get_market_options(),
        help="Major commercial real estate market",
        key="dom_market_selector"
    )

    # Get coordinates for selected market
    if market_name in market_coords:
        default_lat, default_lon = market_coords[market_name]
    else:
        default_lat, default_lon = 42.3601, -71.0589

    with st.form("dom_property_form"):
        col_lat, col_lon = st.columns(2)
        with col_lat:
            latitude = st.number_input(
                "Latitude",
                min_value=24.0, max_value=50.0,
                value=default_lat, step=0.0001, format="%.4f"
            )
        with col_lon:
            longitude = st.number_input(
                "Longitude",
                min_value=-125.0, max_value=-65.0,
                value=default_lon, step=0.0001, format="%.4f"
            )

        st.markdown("**Building Characteristics**")

        secondary_type = st.selectbox(
            "Secondary Type",
            options=get_property_type_options(),
            help="Property type/use"
        )

        col_rba, col_year = st.columns(2)
        with col_rba:
            rba = st.number_input(
                "RBA (SF)",
                min_value=100, max_value=10_000_000,
                value=50_000, step=1000
            )
        with col_year:
            year_built = st.number_input(
                "Year Built",
                min_value=1800, max_value=2025,
                value=2000, step=1
            )

        col_stories, col_rating = st.columns(2)
        with col_stories:
            stories = st.number_input(
                "Stories",
                min_value=1, max_value=200,
                value=3, step=1
            )
        with col_rating:
            star_rating = st.number_input(
                "Star Rating",
                min_value=1, max_value=5,
                value=3, step=1,
                help="Building quality rating (1-5 stars)"
            )

        st.markdown("**Current Listing Details**")

        col_rent, col_vacancy = st.columns(2)
        with col_rent:
            asking_rent = st.number_input(
                "Asking Rent ($/SF/Yr)",
                min_value=0.0, max_value=200.0,
                value=12.0, step=0.5,
                help="Current asking rent"
            )
        with col_vacancy:
            vacancy_pct = st.number_input(
                "Vacancy %",
                min_value=0.0, max_value=100.0,
                value=10.0, step=1.0
            )

        submit = st.form_submit_button("Predict Days on Market", use_container_width=True)

# Prediction
if submit:
    with st.spinner("Analyzing market timing..."):
        try:
            # Calculate building age
            building_age = 2025 - year_built

            # Heuristic-based prediction model
            # Note: The ML model predicts 0 for all cases due to highly skewed training data
            # (97.8% of properties have DOM=0). This heuristic provides more useful guidance.
            #
            # Based on domain knowledge and training data analysis:
            # - Median DOM is 0 days (most properties lease immediately)
            # - Mean DOM is ~6 days
            # - Properties that don't lease immediately tend to take much longer

            # Base prediction (most properties lease quickly in this market)
            base_dom = 3  # Slightly above median, reflecting typical quick turnaround

            # Vacancy impact (strongest predictor from model feature importance)
            # Higher vacancy signals oversupply or property issues
            if vacancy_pct >= 30:
                vacancy_impact = (vacancy_pct - 30) * 0.8 + 10  # Significant penalty
            elif vacancy_pct >= 15:
                vacancy_impact = (vacancy_pct - 15) * 0.5 + 3
            elif vacancy_pct >= 5:
                vacancy_impact = (vacancy_pct - 5) * 0.2
            else:
                vacancy_impact = -2  # Very low vacancy = quick absorption

            # Star rating impact (quality buildings lease faster)
            rating_impact = {
                5: -3,   # Class A - premium, quick lease
                4: -1,   # Above average
                3: 0,    # Average
                2: 4,    # Below average
                1: 8     # Poor quality - harder to lease
            }.get(star_rating, 0)

            # Building age impact
            if building_age <= 5:
                age_impact = -2  # New construction - desirable
            elif building_age <= 15:
                age_impact = 0   # Modern
            elif building_age <= 30:
                age_impact = 2   # Aging
            elif building_age <= 50:
                age_impact = 5   # Old
            else:
                age_impact = 10  # Very old - harder to lease

            # Size impact (larger properties take longer to absorb)
            if rba >= 500000:
                size_impact = 15  # Very large - long absorption
            elif rba >= 200000:
                size_impact = 8
            elif rba >= 100000:
                size_impact = 4
            elif rba >= 50000:
                size_impact = 2
            else:
                size_impact = 0  # Small - quick lease

            # Property type impact (some types harder to lease)
            slow_types = ['Food Processing', 'Refrigeration/Cold Storage', 'Manufacturing']
            fast_types = ['Warehouse', 'Distribution', 'Flex']
            if secondary_type in slow_types:
                type_impact = 7
            elif secondary_type in fast_types:
                type_impact = -2
            else:
                type_impact = 0

            # Rent impact (pricing vs market)
            # Assuming average rent is ~$12/SF/Yr based on training data
            if asking_rent > 20:
                rent_impact = 5  # Premium pricing
            elif asking_rent > 15:
                rent_impact = 2
            elif asking_rent < 8:
                rent_impact = -2  # Below market - quick interest
            else:
                rent_impact = 0

            # Calculate final prediction
            predicted_dom = base_dom + vacancy_impact + rating_impact + age_impact + size_impact + type_impact + rent_impact
            predicted_dom = max(0, round(predicted_dom, 1))

            # Classify and get strategy
            timing_class = classify_market_timing(predicted_dom)
            strategy = get_bidding_strategy(predicted_dom, timing_class)

            # Display results
            with col2:
                st.subheader("Market Timing Prediction")

                # Main prediction card
                st.markdown(f"""
                <div style="background-color: #F0F2F6; padding: 2rem; border-radius: 0.5rem; margin: 1rem 0;">
                    <h2 style="color: #0066CC; margin: 0;">{strategy['icon']} {predicted_dom:.0f} Days</h2>
                    <p style="color: #666; margin: 0.5rem 0 0 0;">Predicted Days on Market</p>
                    <hr style="margin: 1rem 0;">
                    <p style="margin: 0.25rem 0;">
                        <strong>Market Timing:</strong>
                        <span style="color: {strategy['urgency_color']}; font-weight: bold;">{timing_class}</span>
                    </p>
                    <p style="margin: 0.25rem 0;"><strong>Urgency Level:</strong> {strategy['urgency']}</p>
                </div>
                """, unsafe_allow_html=True)

                # Bidding Strategy
                st.markdown("### Bidding Strategy")

                col_s1, col_s2 = st.columns(2)
                with col_s1:
                    st.metric("Suggested Bid Adjustment", strategy['bid_premium'])
                with col_s2:
                    st.metric("Negotiation Power", strategy['negotiation_power'])

                st.info(f"**Recommendation:** {strategy['recommendation']}")

                # Timeline visualization
                st.markdown("### Market Timeline")

                # Create a simple timeline bar
                max_days = 120
                pct = min(predicted_dom / max_days * 100, 100)

                quick_end = 7/max_days*100
                normal_end = 30/max_days*100
                slow_end = 90/max_days*100

                st.markdown(f"""
                <div style="margin: 1rem 0;">
                    <div style="display: flex; height: 30px; border-radius: 5px; overflow: hidden;">
                        <div style="width: {quick_end}%; background: #F44336;" title="Quick Sale (0-7 days)"></div>
                        <div style="width: {normal_end - quick_end}%; background: #FF9800;" title="Normal (8-30 days)"></div>
                        <div style="width: {slow_end - normal_end}%; background: #4CAF50;" title="Slow (31-90 days)"></div>
                        <div style="width: {100 - slow_end}%; background: #2196F3;" title="Extended (90+ days)"></div>
                    </div>
                    <div style="position: relative; height: 20px;">
                        <div style="position: absolute; left: {pct}%; transform: translateX(-50%);">
                            <span style="font-size: 1.5rem;">📍</span>
                        </div>
                    </div>
                    <div style="display: flex; justify-content: space-between; font-size: 0.8rem; color: #666;">
                        <span>0</span>
                        <span>7 days</span>
                        <span>30 days</span>
                        <span>90 days</span>
                        <span>120+ days</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # Additional insights
            st.markdown("---")
            st.subheader("Market Insights")

            col_i1, col_i2, col_i3, col_i4 = st.columns(4)

            with col_i1:
                st.metric("Building Age", f"{building_age} years")
            with col_i2:
                st.metric("Vacancy Rate", f"{vacancy_pct}%")
            with col_i3:
                st.metric("Property Size", f"{rba:,} SF")
            with col_i4:
                st.metric("Star Rating", f"{star_rating} ⭐")

            # Factors affecting DOM
            st.markdown("### Key Factors Affecting Time on Market")

            factors = []
            if vacancy_pct > 15:
                factors.append(("⬆️", "High vacancy rate may extend listing time", "negative"))
            elif vacancy_pct < 5:
                factors.append(("⬇️", "Low vacancy indicates strong demand", "positive"))

            if building_age > 40:
                factors.append(("⬆️", "Older building may take longer to lease", "negative"))
            elif building_age < 10:
                factors.append(("⬇️", "Newer construction attracts faster interest", "positive"))

            if star_rating >= 4:
                factors.append(("⬇️", "High-quality building appeals to more tenants", "positive"))
            elif star_rating <= 2:
                factors.append(("⬆️", "Lower rating may limit tenant pool", "negative"))

            if rba > 200000:
                factors.append(("⬆️", "Large property may have longer absorption time", "negative"))

            if not factors:
                factors.append(("➡️", "Property characteristics are within normal range", "neutral"))

            for direction, text, sentiment in factors:
                color = "#4CAF50" if sentiment == "positive" else "#F44336" if sentiment == "negative" else "#666"
                st.markdown(f"<p style='color: {color};'>{direction} {text}</p>", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            st.exception(e)

else:
    with col2:
        st.info("Enter property details and click **Predict Days on Market** to see results")

        # Show what the tool provides
        st.markdown("""
        ### What You'll Get

        - **Days on Market Prediction** - Estimated time until property is leased/sold
        - **Market Timing Category** - Quick Sale, Normal, Slow, or Extended
        - **Bidding Strategy** - Recommended approach based on market timing
        - **Negotiation Power Assessment** - Your leverage in negotiations
        """)
