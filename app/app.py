import os
import sys

# Ensure project root is in Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk

from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable

from src.underwriting_engine import UnderwritingEngine

# Initialize Nominatim Geocoder (Free OpenStreetMap API)
geolocator = Nominatim(user_agent="wildfire_analytics_app")

# Define Geographic Bounding Box for Southern California Coverage Area
SOCAL_BOUNDS = {
    "min_lat": 32.50,
    "max_lat": 35.50,
    "min_lon": -120.50,
    "max_lon": -115.00
}


def is_within_bounds(lat: float, lon: float) -> bool:
    """Checks if coordinates fall within designated regional boundaries."""
    return (
            SOCAL_BOUNDS["min_lat"] <= lat <= SOCAL_BOUNDS["max_lat"] and
            SOCAL_BOUNDS["min_lon"] <= lon <= SOCAL_BOUNDS["max_lon"]
    )


def geocode_address(address_str: str):
    """Converts street address string to (latitude, longitude)."""
    try:
        location = geolocator.geocode(address_str, timeout=5)
        if location:
            return location.latitude, location.longitude, location.address
    except (GeocoderTimedOut, GeocoderUnavailable):
        st.sidebar.error("Geocoding service timed out. Please try again.")
    return None, None, None


# Page Configuration
st.set_page_config(
    page_title="Wildfire Risk Analytics & Underwriting Engine",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Initialize Engine
@st.cache_resource
def get_underwriting_engine():
    return UnderwritingEngine(base_premium=2200.0)


engine = get_underwriting_engine()

# Title Header
st.title("🔥 Enterprise Wildfire Risk Analytics & Underwriting Platform")
st.markdown(
    "Automated spatial risk assessment, TreeSHAP model interpretability, and interactive property hardening simulation engine."
)
st.divider()

# Sidebar - Property Selection & Input Controls
st.sidebar.header("📍 Property Location & Parameters")

input_mode = st.sidebar.radio(
    "Select Input Method:",
    ["Street Address Search", "Custom Lat/Lon Coordinates", "Preset Profiles"]
)

selected_lat, selected_lon = None, None

# Preset profiles with complete default baseline features
presets = {
    "Property A - Glendale Foothills": {
        "lat": 34.1425, "lon": -118.2551, "elevation_m": 285.0, "slope_pct": 18.5,
        "dist_nearest_fire_m": 916.0, "fires_within_5km": 2, "fires_within_10km": 5,
        "years_since_last_fire": 6, "max_vpd": 4.2, "max_wind_speed": 42.0, "total_precip_annual": 380.0
    },
    "Property B - Reseda / Valley": {
        "lat": 34.1975, "lon": -118.5413, "elevation_m": 240.0, "slope_pct": 2.1,
        "dist_nearest_fire_m": 8500.0, "fires_within_5km": 0, "fires_within_10km": 1,
        "years_since_last_fire": 18, "max_vpd": 3.8, "max_wind_speed": 28.0, "total_precip_annual": 410.0
    },
    "Property C - Santa Monica Coast": {
        "lat": 34.0195, "lon": -118.4912, "elevation_m": 40.0, "slope_pct": 4.5,
        "dist_nearest_fire_m": 12000.0, "fires_within_5km": 0, "fires_within_10km": 0,
        "years_since_last_fire": 25, "max_vpd": 2.1, "max_wind_speed": 35.0, "total_precip_annual": 350.0
    }
}

# Baseline default profile fallback for custom address or coordinate search
default_props = presets["Property A - Glendale Foothills"]

if input_mode == "Street Address Search":
    address_input = st.sidebar.text_input(
        "Enter Property Address:",
        value="100 N Brand Blvd, Glendale, CA"
    )

    if st.sidebar.button("Geocode & Assess Property"):
        lat, lon, full_addr = geocode_address(address_input)
        if lat and lon:
            if is_within_bounds(lat, lon):
                st.sidebar.success(f"Matched: {full_addr}")
                selected_lat, selected_lon = lat, lon
            else:
                st.sidebar.error(
                    f"⛔ Out of Bounds!\n\n"
                    f"Location ({lat:.4f}, {lon:.4f}) is outside our active Southern California underwriting coverage area."
                )
        else:
            st.sidebar.error("Address not found. Please enter a valid address including city and state.")
    else:
        # Default load if button not pressed yet
        selected_lat, selected_lon = default_props["lat"], default_props["lon"]

elif input_mode == "Custom Lat/Lon Coordinates":
    c1, c2 = st.sidebar.columns(2)
    lat_val = c1.number_input("Latitude", value=34.1425, format="%.4f")
    lon_val = c2.number_input("Longitude", value=-118.2551, format="%.4f")

    if is_within_bounds(lat_val, lon_val):
        selected_lat, selected_lon = lat_val, lon_val
    else:
        st.sidebar.error("⛔ Coordinates are outside active underwriting bounds!")

else:
    selected_preset = st.sidebar.selectbox("Select Profile:", list(presets.keys()))
    preset_data = presets[selected_preset]
    selected_lat, selected_lon = preset_data["lat"], preset_data["lon"]
    default_props = preset_data

# Stop execution if location is out of bounds or unselected
if selected_lat is None or selected_lon is None:
    st.warning(
        "⚠️ Please select or enter a valid property location within Southern California coverage bounds to proceed.")
    st.stop()

# Sliders and Overrides
st.sidebar.subheader("Terrain & Climate Overrides")
elevation_m = st.sidebar.slider("Elevation (m)", 0.0, 1000.0, float(default_props["elevation_m"]))
slope_pct = st.sidebar.slider("Slope (%)", 0.0, 40.0, float(default_props["slope_pct"]))
dist_nearest_fire_m = st.sidebar.slider("Distance to Historical Fire (m)", 100.0, 50000.0,
                                        float(default_props["dist_nearest_fire_m"]))
max_wind_speed = st.sidebar.slider("Max Annual Wind Speed (km/h)", 10.0, 80.0, float(default_props["max_wind_speed"]))

# Sidebar - What-If Mitigation Controls
st.sidebar.divider()
st.sidebar.header("🛡️ Property Hardening & Mitigation")

defensible_space = st.sidebar.select_slider(
    "Defensible Space Buffer (ft)",
    options=[0, 30, 100],
    value=0,
    help="Clearing flammable vegetation surrounding the structure."
)

roof_type = st.sidebar.selectbox(
    "Roofing Material",
    ["Standard", "Metal / Tile", "Class A Fire-Resistant"],
    index=0
)

ember_vents = st.sidebar.checkbox("Install Ember-Resistant Mesh Vents", value=False)

climate_scenario = st.sidebar.radio(
    "Climate Stress Test Scenario",
    ["Baseline", "Extreme Drought"],
    index=0
)

# Build Current Property Feature Vector
active_features = {
    "latitude": selected_lat,
    "longitude": selected_lon,
    "elevation_m": elevation_m,
    "slope_pct": slope_pct,
    "dist_nearest_fire_m": dist_nearest_fire_m,
    "fires_within_5km": default_props["fires_within_5km"],
    "fires_within_10km": default_props["fires_within_10km"],
    "years_since_last_fire": default_props["years_since_last_fire"],
    "max_vpd": default_props["max_vpd"],
    "max_wind_speed": max_wind_speed,
    "total_precip_annual": default_props["total_precip_annual"]
}

# Run Underwriting Calculation
result = engine.calculate_mitigation_adjustments(
    active_features,
    defensible_space_ft=defensible_space,
    roof_material=roof_type,
    ember_vents=ember_vents,
    climate_scenario=climate_scenario
)

# Layout Columns
col_left, col_right = st.columns([1.1, 1])

with col_left:
    st.subheader("🗺️ Property Location Map")

    # Render PyDeck Map
    map_df = pd.DataFrame([{
        "lat": selected_lat,
        "lon": selected_lon,
        "risk_score": result["final_risk_score"]
    }])

    view_state = pdk.ViewState(
        latitude=selected_lat,
        longitude=selected_lon,
        zoom=11,
        pitch=45
    )

    layer = pdk.Layer(
        "ScatterplotLayer",
        data=map_df,
        get_position=["lon", "lat"],
        get_color="[255, 65, 54, 200]" if result["final_risk_score"] >= 50 else "[46, 204, 113, 200]",
        get_radius=500,
        pickable=True
    )

    st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state))

    # SHAP Feature Attribution
    st.subheader("📊 SHAP Model Decision Breakdown")
    shap_dict = result["shap_contributions"]
    shap_df = pd.DataFrame({
        "Feature": list(shap_dict.keys()),
        "SHAP Contribution": list(shap_dict.values())
    }).sort_values(by="SHAP Contribution", ascending=True)

    fig_shap = px.bar(
        shap_df,
        x="SHAP Contribution",
        y="Feature",
        orientation="h",
        color="SHAP Contribution",
        color_continuous_scale="RdYlGn_r",
        title="Feature Attribution to Wildfire Risk Score"
    )
    st.plotly_chart(fig_shap, use_container_width=True)

with col_right:
    st.subheader("📋 Underwriting Assessment Dashboard")

    m1, m2, m3 = st.columns(3)
    m1.metric("Wildfire Risk Index", f"{result['final_risk_score']} / 100",
              delta=f"-{result['mitigation_credit_pts']} pts" if result['mitigation_credit_pts'] > 0 else None)
    m2.metric("Annual Premium", f"${result['recommended_annual_premium']:,.2f}")
    m3.metric("Policy Status", result["underwriting_status"])

    st.divider()

    # Mitigation Savings Impact
    st.subheader("💰 Mitigation Financial Savings")
    unmitigated_res = engine.calculate_mitigation_adjustments(
        active_features, defensible_space_ft=0, roof_material="Standard", ember_vents=False
    )

    annual_savings = unmitigated_res["recommended_annual_premium"] - result["recommended_annual_premium"]

    if annual_savings > 0:
        st.success(
            f"🎉 **Mitigation Active!** Policyholder saves **${annual_savings:,.2f} / year** by implementing selected risk reduction measures.")
    else:
        st.info("💡 Implement defensible space or fire-resistant roofing in the sidebar to simulate premium reduction.")

    st.subheader("🔍 Property Risk Indicators")
    indicator_df = pd.DataFrame([
        {"Indicator": "Slope Gradient", "Value": f"{slope_pct}%",
         "Impact": "High Risk" if slope_pct > 15 else "Normal"},
        {"Indicator": "Nearest Fire Proximity", "Value": f"{dist_nearest_fire_m:,.0f} meters",
         "Impact": "High Risk" if dist_nearest_fire_m < 2000 else "Normal"},
        {"Indicator": "Annual Max Wind", "Value": f"{max_wind_speed} km/h",
         "Impact": "High Risk" if max_wind_speed > 35 else "Normal"},
        {"Indicator": "Climate Scenario", "Value": climate_scenario,
         "Impact": "Elevated" if climate_scenario == "Extreme Drought" else "Standard"}
    ])
    st.table(indicator_df)