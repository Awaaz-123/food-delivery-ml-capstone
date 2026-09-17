import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os

# Set Page Config
st.set_page_config(
    page_title="Food Delivery ML Intelligence System",
    page_icon="🚴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Application Header
st.title("🚴 Food Delivery & Restaurant ML Intelligence System")
st.markdown("""
An interactive Machine Learning dashboard integrating **Regression** (Delivery ETA), 
**Classification** (Late Risk Probability), and **Unsupervised Clustering** (Order Logistics Segmentation).
""")

# Load Trained Models and Preprocessing Objects
@st.cache_resource
def load_ml_assets():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    reg_path = os.path.join(base_dir, 'models', 'best_regression_model.pkl')
    clf_path = os.path.join(base_dir, 'models', 'best_classification_model.pkl')
    cluster_path = os.path.join(base_dir, 'models', 'kmeans_model.pkl')
    scaler_cluster_path = os.path.join(base_dir, 'models', 'scaler_cluster.pkl')
    features_path = os.path.join(base_dir, 'models', 'model_features.pkl')
    
    reg_model = joblib.load(reg_path) if os.path.exists(reg_path) else None
    clf_model = joblib.load(clf_path) if os.path.exists(clf_path) else None
    kmeans_model = joblib.load(cluster_path) if os.path.exists(cluster_path) else None
    scaler_cluster = joblib.load(scaler_cluster_path) if os.path.exists(scaler_cluster_path) else None
    feature_names = joblib.load(features_path) if os.path.exists(features_path) else None
    
    return reg_model, clf_model, kmeans_model, scaler_cluster, feature_names

reg_model, clf_model, kmeans_model, scaler_cluster, feature_names = load_ml_assets()

# Sidebar Control Widgets
st.sidebar.header("⚙️ Delivery Input Parameters")

distance_km = st.sidebar.slider("📍 Delivery Distance (km)", 0.5, 30.0, 6.5, step=0.5)
delivery_rating = st.sidebar.slider("⭐ Delivery Driver Rating", 1.0, 5.0, 4.5, step=0.1)
delivery_age = st.sidebar.slider("👤 Delivery Driver Age", 18, 50, 29)
multiple_deliveries = st.sidebar.selectbox("📦 Multiple Deliveries Handled", [0, 1, 2, 3], index=0)

traffic_density = st.sidebar.selectbox("🚦 Road Traffic Density", ["Low", "Medium", "High", "Jam"], index=3)
weather_condition = st.sidebar.selectbox("🌤️ Weather Condition", ["Sunny", "Cloudy", "Windy", "Fog", "Sandstorms", "Stormy"], index=0)
vehicle_type = st.sidebar.selectbox("🛵 Vehicle Type", ["motorcycle", "scooter", "electric_scooter", "bicycle"])
city_type = st.sidebar.selectbox("🏙️ City Type", ["Metropolitian", "Urban", "Semi-Urban"])
order_type = st.sidebar.selectbox("🍔 Type of Order", ["Snack", "Meal", "Buffet", "Drinks"])
festival = st.sidebar.selectbox("🎉 Festival Period", ["No", "Yes"])

st.subheader("🔮 ML Predictive Intelligence Dashboard")

if reg_model is not None and feature_names is not None:
    # Construct 1-row feature DataFrame
    input_df = pd.DataFrame(0.0, index=[0], columns=feature_names)
    
    # Assign Numerical Features
    if 'Delivery_person_Age' in feature_names: input_df.loc[0, 'Delivery_person_Age'] = float(delivery_age)
    if 'Delivery_person_Ratings' in feature_names: input_df.loc[0, 'Delivery_person_Ratings'] = float(delivery_rating)
    if 'distance_km' in feature_names: input_df.loc[0, 'distance_km'] = float(distance_km)
    if 'multiple_deliveries' in feature_names: input_df.loc[0, 'multiple_deliveries'] = float(multiple_deliveries)
    if 'prep_time_min' in feature_names: input_df.loc[0, 'prep_time_min'] = 10.0
    if 'order_hour' in feature_names: input_df.loc[0, 'order_hour'] = 19.0
    
    # Assign One-Hot Categorical Features
    for col_prefix, val in [
        ('Road_traffic_density', traffic_density.strip()),
        ('Weatherconditions', weather_condition.strip()),
        ('Type_of_order', order_type.strip()),
        ('Type_of_vehicle', vehicle_type.strip()),
        ('Festival', festival.strip()),
        ('City', city_type.strip())
    ]:
        dummy_col = f"{col_prefix}_{val}"
        if dummy_col in feature_names:
            input_df.loc[0, dummy_col] = 1.0

    # Track 1: Regression ETA Prediction
    raw_pred_eta = reg_model.predict(input_df)[0]
    predicted_eta = int(round(raw_pred_eta))
    predicted_eta = max(10, predicted_eta)

    # Track 2: Classification Late Risk Prediction
    if clf_model is not None:
        late_prob = clf_model.predict_proba(input_df)[0][1]
        is_late_flag = clf_model.predict(input_df)[0]
    else:
        late_prob = 0.85 if predicted_eta > 30 else 0.15
        is_late_flag = 1 if predicted_eta > 30 else 0

    # Track 3: Clustering Segment Prediction
    if kmeans_model is not None and scaler_cluster is not None:
        cluster_cols = ['Delivery_person_Age', 'Delivery_person_Ratings', 'distance_km', 'multiple_deliveries', 'prep_time_min']
        cluster_df = pd.DataFrame([[delivery_age, delivery_rating, distance_km, multiple_deliveries, 10.0]], columns=cluster_cols)
        cluster_input_scaled = scaler_cluster.transform(cluster_df)
        cluster_id = kmeans_model.predict(cluster_input_scaled)[0]
        
        cluster_names = {
            0: "Cluster 0: Express Local Delivery (< 5 km)",
            1: "Cluster 1: Outer-City Long Transit (> 15 km)",
            2: "Cluster 2: Multi-Order Peak Hour Surge"
        }
        cluster_label = cluster_names.get(cluster_id, f"Cluster {cluster_id}")
    else:
        cluster_label = "Standard Delivery Segment"

else:
    # Heuristic fallback if models not loaded
    traffic_penalty = {"Low": 0, "Medium": 5, "High": 10, "Jam": 18}[traffic_density]
    weather_penalty = {"Sunny": 0, "Cloudy": 2, "Windy": 3, "Stormy": 8, "Fog": 6, "Sandstorms": 7}[weather_condition]
    predicted_eta = int(12 + (distance_km * 1.8) + traffic_penalty + weather_penalty + (multiple_deliveries * 4) - ((delivery_rating - 3) * 2))
    predicted_eta = max(10, predicted_eta)
    late_prob = 0.85 if predicted_eta > 30 else 0.15
    is_late_flag = 1 if predicted_eta > 30 else 0
    cluster_label = "Standard Delivery Segment"

# Display Metric Dashboard Cards
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="⏱️ Track 1: Predicted Delivery ETA",
        value=f"{predicted_eta} mins",
        delta=f"Distance: {distance_km} km"
    )

with col2:
    prob_pct = int(round(late_prob * 100))
    if is_late_flag == 1 or predicted_eta > 30:
        st.metric(
            label="⚠️ Track 2: Late Risk Classification",
            value=f"HIGH RISK ({prob_pct}% Prob)",
            delta="- SLA Breach Expected (>30m)",
            delta_color="inverse"
        )
    else:
        st.metric(
            label="✅ Track 2: Late Risk Classification",
            value=f"ON-TIME ({100 - prob_pct}% Confidence)",
            delta="+ Fast Delivery Expected",
            delta_color="normal"
        )

with col3:
    st.metric(
        label="🏷️ Track 3: Clustering Segment",
        value=f"Cluster {cluster_id if 'cluster_id' in locals() else 0}",
        delta=cluster_label.split(": ")[-1] if ":" in cluster_label else cluster_label
    )

# Visual Late Probability Bar
st.markdown("**Late Risk Probability Gauge:**")
st.progress(min(1.0, float(late_prob)))

st.markdown("---")
st.subheader("📋 Input Delivery Feature Summary")

summary_df = pd.DataFrame([{
    "Distance (km)": distance_km,
    "Traffic Density": traffic_density,
    "Weather Condition": weather_condition,
    "Driver Rating": delivery_rating,
    "Driver Age": delivery_age,
    "Multiple Deliveries": multiple_deliveries,
    "Vehicle Type": vehicle_type,
    "Order Type": order_type,
    "City Type": city_type,
    "Festival": festival
}])

st.dataframe(summary_df, width="stretch")
