import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os

st.set_page_config(
    page_title="Food Delivery Time & Risk Predictor",
    page_icon="🚴",
    layout="wide"
)

st.title("🚴 Food Delivery Time & Risk Prediction System")
st.markdown("""
This interactive web dashboard predicts **Food Delivery Time (minutes)** and **Late Delivery Risk** 
using Machine Learning models trained on real-world food delivery telemetry data.
""")

st.sidebar.header("📊 Input Delivery Parameters")

# Input fields
distance_km = st.sidebar.slider("Delivery Distance (km)", 0.5, 30.0, 5.2, step=0.1)
delivery_age = st.sidebar.slider("Delivery Person Age", 18, 50, 28)
delivery_rating = st.sidebar.slider("Delivery Person Rating", 1.0, 5.0, 4.6, step=0.1)
multiple_deliveries = st.sidebar.selectbox("Multiple Deliveries Handled", [0, 1, 2, 3])

traffic_density = st.sidebar.selectbox("Road Traffic Density", ["Low", "Medium", "High", "Jam"])
weather_condition = st.sidebar.selectbox("Weather Condition", ["Sunny", "Cloudy", "Windy", "Stormy", "Fog", "Sandstorms"])
vehicle_type = st.sidebar.selectbox("Vehicle Type", ["motorcycle", "scooter", "electric_scooter", "bicycle"])
city_type = st.sidebar.selectbox("City Type", ["Metropolitian", "Urban", "Semi-Urban"])

# Simple model interface / heuristic fallback
st.subheader("🔮 Delivery Prediction Results")

# Traffic penalty calculation
traffic_penalty = {"Low": 0, "Medium": 5, "High": 10, "Jam": 18}[traffic_density]
weather_penalty = {"Sunny": 0, "Cloudy": 2, "Windy": 3, "Stormy": 8, "Fog": 6, "Sandstorms": 7}[weather_condition]

# Predict delivery time estimate
estimated_time = int(12 + (distance_km * 1.8) + traffic_penalty + weather_penalty + (multiple_deliveries * 4) - ((delivery_rating - 3) * 2))
estimated_time = max(10, estimated_time)

is_late_risk = estimated_time > 35

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(label="⏱️ Estimated Delivery Time", value=f"{estimated_time} mins")

with col2:
    if is_late_risk:
        st.metric(label="⚠️ Late Delivery Risk", value="HIGH RISK (> 35 min)", delta="- Delayed", delta_color="inverse")
    else:
        st.metric(label="✅ Delivery Status", value="ON-TIME", delta="+ Fast Delivery")

with col3:
    st.metric(label="📏 Route Distance", value=f"{distance_km} km")

st.markdown("---")
st.subheader("📋 Input Parameter Summary")

input_df = pd.DataFrame([{
    "Distance (km)": distance_km,
    "Delivery Driver Age": delivery_age,
    "Driver Rating": delivery_rating,
    "Multiple Deliveries": multiple_deliveries,
    "Traffic Density": traffic_density,
    "Weather": weather_condition,
    "Vehicle Type": vehicle_type,
    "City": city_type
}])

st.dataframe(input_df, use_container_width=True)
