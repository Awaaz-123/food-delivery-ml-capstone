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

# Load Trained Model and Preprocessing Objects
@st.cache_resource
def load_ml_assets():
    # Detect path whether running from app/ directory or root directory
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, 'models', 'best_regression_model.pkl')
    scaler_path = os.path.join(base_dir, 'models', 'scaler.pkl')
    features_path = os.path.join(base_dir, 'models', 'model_features.pkl')
    
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        features = joblib.load(features_path)
        return model, scaler, features
    return None, None, None

model, scaler, feature_names = load_ml_assets()

st.sidebar.header("📊 Input Delivery Parameters")

# Sidebar Input Widgets
distance_km = st.sidebar.slider("Delivery Distance (km)", 0.5, 30.0, 5.2, step=0.1)
delivery_age = st.sidebar.slider("Delivery Person Age", 18, 50, 28)
delivery_rating = st.sidebar.slider("Delivery Person Rating", 1.0, 5.0, 4.6, step=0.1)
multiple_deliveries = st.sidebar.selectbox("Multiple Deliveries Handled", [0, 1, 2, 3])

traffic_density = st.sidebar.selectbox("Road Traffic Density", ["Low", "Medium", "High", "Jam"])
weather_condition = st.sidebar.selectbox("Weather Condition", ["Sunny", "Cloudy", "Windy", "Stormy", "Fog", "Sandstorms"])
vehicle_type = st.sidebar.selectbox("Vehicle Type", ["motorcycle", "scooter", "electric_scooter", "bicycle"])
city_type = st.sidebar.selectbox("City Type", ["Metropolitian", "Urban", "Semi-Urban"])
order_type = st.sidebar.selectbox("Type of Order", ["Snack", "Meal", "Buffet", "Drinks"])
festival = st.sidebar.selectbox("Festival Period", ["No", "Yes"])

st.subheader("🔮 Machine Learning Prediction Results")

if model is not None and feature_names is not None:
    # Build Raw User Input DataFrame
    raw_input = pd.DataFrame([{
        'Delivery_person_Age': delivery_age,
        'Delivery_person_Ratings': delivery_rating,
        'distance_km': distance_km,
        'multiple_deliveries': multiple_deliveries,
        'Weatherconditions': weather_condition,
        'Road_traffic_density': traffic_density,
        'Type_of_order': order_type,
        'Type_of_vehicle': vehicle_type,
        'Festival': festival,
        'City': city_type
    }])
    
    # One-Hot Encoding and Column Alignment with Trained Model
    input_encoded = pd.get_dummies(raw_input, drop_first=True)
    input_aligned = input_encoded.reindex(columns=feature_names, fill_value=0)
    
    # Make Prediction using Trained Gradient Boosting Model
    predicted_time = model.predict(input_aligned)[0]
    estimated_time = int(round(predicted_time))
else:
    # Fallback heuristic calculation if model file not found
    traffic_penalty = {"Low": 0, "Medium": 5, "High": 10, "Jam": 18}[traffic_density]
    weather_penalty = {"Sunny": 0, "Cloudy": 2, "Windy": 3, "Stormy": 8, "Fog": 6, "Sandstorms": 7}[weather_condition]
    estimated_time = int(12 + (distance_km * 1.8) + traffic_penalty + weather_penalty + (multiple_deliveries * 4) - ((delivery_rating - 3) * 2))

estimated_time = max(10, estimated_time)
is_late_risk = estimated_time > 35

# Display Metrics
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(label="⏱️ ML Predicted Delivery Time", value=f"{estimated_time} mins")

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
