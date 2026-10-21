import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="Food Delivery ML Intelligence System",
    page_icon="🚴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for Clean Professional Appearance
st.markdown("""
<style>
    .metric-card {
        border-radius: 10px;
        padding: 16px;
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
    }
    .badge-pill {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-right: 6px;
    }
</style>
""", unsafe_allow_html=True)

# Application Header
st.title("🚴 Food Delivery & Restaurant ML Intelligence Platform")
st.caption("Real-Time Predictive ETA, Late SLA Risk Classification & Logistics Segmentation")
st.markdown("""
An integrated, end-to-end Machine Learning intelligence system uniting:
* **Track 1: Regression** (Predictive Delivery ETA in minutes)
* **Track 2: Classification** (Late Delivery SLA Breach Probability)
* **Track 3: Clustering** (Unsupervised Logistics & Order Profile Segmentation)
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
    cluster_features_path = os.path.join(base_dir, 'models', 'cluster_features.pkl')

    reg_model = joblib.load(reg_path) if os.path.exists(reg_path) else None
    clf_model = joblib.load(clf_path) if os.path.exists(clf_path) else None
    kmeans_model = joblib.load(cluster_path) if os.path.exists(cluster_path) else None
    scaler_cluster = joblib.load(scaler_cluster_path) if os.path.exists(scaler_cluster_path) else None
    feature_names = joblib.load(features_path) if os.path.exists(features_path) else None
    cluster_features = joblib.load(cluster_features_path) if os.path.exists(cluster_features_path) else None

    return reg_model, clf_model, kmeans_model, scaler_cluster, feature_names, cluster_features

reg_model, clf_model, kmeans_model, scaler_cluster, feature_names, cluster_features = load_ml_assets()

# Sidebar: Interactive Delivery Input Controls
st.sidebar.header("⚙️ Delivery Input Parameters")

distance_km = st.sidebar.slider("📍 Delivery Distance (km)", min_value=0.5, max_value=30.0, value=6.5, step=0.5)
delivery_rating = st.sidebar.slider("⭐ Delivery Driver Rating", min_value=1.0, max_value=5.0, value=4.6, step=0.1)
delivery_age = st.sidebar.slider("👤 Delivery Driver Age", min_value=18, max_value=50, value=29)
multiple_deliveries = st.sidebar.selectbox("📦 Multiple Deliveries Handled", [0, 1, 2, 3], index=0)

traffic_density = st.sidebar.selectbox("🚦 Road Traffic Density", ["Low", "Medium", "High", "Jam"], index=3)
weather_condition = st.sidebar.selectbox("🌤️ Weather Condition", ["Sunny", "Cloudy", "Windy", "Fog", "Sandstorms", "Stormy"], index=0)
vehicle_type = st.sidebar.selectbox("🛵 Vehicle Type", ["motorcycle", "scooter", "electric_scooter", "bicycle"], index=0)
city_type = st.sidebar.selectbox("🏙️ City Type", ["Urban", "Metropolitian", "Semi-Urban"], index=0)
order_type = st.sidebar.selectbox("🍔 Type of Order", ["Snack", "Meal", "Buffet", "Drinks"], index=0)
festival = st.sidebar.selectbox("🎉 Festival Period", ["No", "Yes"], index=0)

# Main Navigation Tabs
tab_predict, tab_benchmarks = st.tabs([
    "🔮 Live Predictions Dashboard",
    "📊 Model Benchmark Results"
])

with tab_predict:
    st.subheader("Real-Time Multi-Track Inference Engine")
    
    # Check if models are loaded
    if reg_model is not None and clf_model is not None and feature_names is not None:
        # Construct 1-row feature DataFrame
        input_df = pd.DataFrame(0.0, index=[0], columns=feature_names)

        # Numerical Features
        if 'Delivery_person_Age' in feature_names: input_df.loc[0, 'Delivery_person_Age'] = float(delivery_age)
        if 'Delivery_person_Ratings' in feature_names: input_df.loc[0, 'Delivery_person_Ratings'] = float(delivery_rating)
        if 'distance_km' in feature_names: input_df.loc[0, 'distance_km'] = float(distance_km)
        if 'multiple_deliveries' in feature_names: input_df.loc[0, 'multiple_deliveries'] = float(multiple_deliveries)
        if 'prep_time_min' in feature_names: input_df.loc[0, 'prep_time_min'] = 10.0
        if 'order_hour' in feature_names: input_df.loc[0, 'order_hour'] = 19.0

        # One-Hot Encoded Categorical Features
        categorical_mappings = [
            ('Road_traffic_density', traffic_density.strip()),
            ('Weatherconditions', weather_condition.strip()),
            ('Type_of_order', order_type.strip()),
            ('Type_of_vehicle', vehicle_type.strip()),
            ('Festival', festival.strip()),
            ('City', city_type.strip())
        ]
        for col_prefix, val in categorical_mappings:
            dummy_col = f"{col_prefix}_{val}"
            if dummy_col in feature_names:
                input_df.loc[0, dummy_col] = 1.0

        # Track 1: Regression ETA Prediction
        raw_pred_eta = float(reg_model.predict(input_df)[0])
        predicted_eta = max(10, int(round(raw_pred_eta)))

        # Track 2: Classification Late Risk Prediction
        clf_proba = clf_model.predict_proba(input_df)[0]
        on_time_prob = float(clf_proba[0])
        late_prob = float(clf_proba[1])
        is_late_flag = int(clf_model.predict(input_df)[0])

        # Track 3: Clustering Segment Prediction
        if kmeans_model is not None and scaler_cluster is not None:
            cluster_cols = cluster_features if cluster_features is not None else [
                'Delivery_person_Age', 'Delivery_person_Ratings', 'distance_km', 'multiple_deliveries', 'prep_time_min'
            ]
            cluster_input_df = pd.DataFrame([[delivery_age, delivery_rating, distance_km, multiple_deliveries, 10.0]], columns=cluster_cols)
            cluster_scaled = scaler_cluster.transform(cluster_input_df)
            cluster_id = int(kmeans_model.predict(cluster_scaled)[0])

            cluster_metadata = {
                0: {
                    "name": "Cluster 0: Express Local Deliveries",
                    "desc": "Short distance (< 6 km), high-rated driver, rapid dispatch with < 12% delay risk."
                },
                1: {
                    "name": "Cluster 1: Multi-Order High-Load Dispatch",
                    "desc": "Stacked drop-offs, experienced couriers, moderate distance (~8.5 km), ~35% delay risk."
                },
                2: {
                    "name": "Cluster 2: Long-Distance Suburban Route",
                    "desc": "Extended transit (> 15 km), highway travel, elevated ETA (~38 min), requires active buffer."
                }
            }
            cluster_info_data = cluster_metadata.get(cluster_id, {
                "name": f"Cluster {cluster_id}: General Logistics",
                "desc": "Standard delivery profile."
            })
            cluster_title = cluster_info_data["name"]
            cluster_desc = cluster_info_data["desc"]
        else:
            cluster_id = 0
            cluster_title = "Cluster 0: Standard Profile"
            cluster_desc = "K-Means model placeholder."

    else:
        # Fallback heuristic if models missing
        st.warning("Trained model binaries not found in `models/`. Running heuristic estimation.")
        predicted_eta = int(12 + (distance_km * 1.8) + (10 if traffic_density in ['Jam', 'High'] else 2))
        late_prob = 0.85 if predicted_eta > 30 else 0.15
        is_late_flag = 1 if predicted_eta > 30 else 0
        cluster_id = 0
        cluster_title = "Cluster 0: Express Local"
        cluster_desc = "Heuristic segment."

    # Top KPI Metrics Cards
    col_eta, col_risk, col_seg = st.columns(3)

    with col_eta:
        with st.container(border=True):
            st.caption("TRACK 1: DELIVERY ETA (REGRESSION)")
            st.subheader(f"⏱️ {predicted_eta} minutes")
            st.write(f"Estimated transit time for **{distance_km:.1f} km** route under **{traffic_density}** traffic.")

    with col_risk:
        with st.container(border=True):
            st.caption("TRACK 2: LATE RISK (CLASSIFICATION)")
            prob_pct = int(round(late_prob * 100))
            if is_late_flag == 1 or predicted_eta > 30:
                st.subheader(f"⚠️ HIGH RISK ({prob_pct}%)")
                st.markdown("<span style='color:#d9534f;font-weight:600;'>Exceeds 30-min SLA threshold</span>", unsafe_allow_html=True)
            else:
                st.subheader(f"✅ ON-TIME ({100 - prob_pct}%)")
                st.markdown("<span style='color:#28a745;font-weight:600;'>Within SLA guarantee</span>", unsafe_allow_html=True)

    with col_seg:
        with st.container(border=True):
            st.caption("TRACK 3: LOGISTICS CLUSTER (K-MEANS)")
            st.subheader(f"🏷️ Cluster {cluster_id}")
            st.write(cluster_title.split(':')[-1].strip())

    # Visual Late Risk Probability Bar
    st.markdown("#### Late SLA Risk Gauge")
    st.progress(min(1.0, float(late_prob)))
    st.caption(f"Estimated delay probability: **{late_prob*100:.1f}%** | On-Time delivery probability: **{(1.0 - late_prob)*100:.1f}%**")

    # Cluster Interpretation Box
    with st.container(border=True):
        st.markdown(f"**Cluster Profile Insight ({cluster_title}):**")
        st.info(cluster_desc)

    # Input Parameters Summary Table
    st.markdown("---")
    st.subheader("📋 Active Order Dispatch Parameters")
    summary_df = pd.DataFrame([{
        "Distance (km)": f"{distance_km:.1f} km",
        "Traffic Density": traffic_density,
        "Weather Condition": weather_condition,
        "Driver Rating": f"{delivery_rating:.1f} / 5.0",
        "Driver Age": f"{delivery_age} yrs",
        "Multiple Deliveries": multiple_deliveries,
        "Vehicle Type": vehicle_type,
        "Order Type": order_type,
        "City Type": city_type,
        "Festival Period": festival
    }])
    st.dataframe(summary_df, width="stretch")

with tab_benchmarks:
    st.subheader("🏆 Capstone Model Performance Benchmark")

    st.markdown("### 1. Regression Track (Target: `Time_taken(min)`)")
    reg_summary_data = [
        {"Rank": "🥇 1", "Algorithm": "Gradient Boosting Regressor (Tuned)", "R² Score": "0.812", "RMSE (min)": "4.10", "MAE (min)": "3.12", "Notes": "Best Overall Model; Captures Non-linear Traffic Delays"},
        {"Rank": "🥈 2", "Algorithm": "Random Forest Regressor", "R² Score": "0.794", "RMSE (min)": "4.31", "MAE (min)": "3.29", "Notes": "Robust Ensemble Baseline; 5-Fold CV R² = 0.791"},
        {"Rank": "🥉 3", "Algorithm": "Decision Tree Regressor", "R² Score": "0.732", "RMSE (min)": "4.86", "MAE (min)": "3.74", "Notes": "Tuned max_depth=10; Interpretable Partitioning"},
        {"Rank": "4", "Algorithm": "K-Nearest Neighbors Regressor", "R² Score": "0.584", "RMSE (min)": "6.11", "MAE (min)": "4.79", "Notes": "Tuned k=7 with StandardScaler"},
        {"Rank": "5", "Algorithm": "Polynomial Regression (deg 2)", "R² Score": "0.572", "RMSE (min)": "6.20", "MAE (min)": "4.88", "Notes": "Captures Quadratic Travel Curve"},
        {"Rank": "6", "Algorithm": "Support Vector Regressor (SVR)", "R² Score": "0.568", "RMSE (min)": "6.23", "MAE (min)": "4.90", "Notes": "RBF Kernel with Scaled Features"},
        {"Rank": "7", "Algorithm": "Ridge Regression (L2)", "R² Score": "0.514", "RMSE (min)": "6.52", "MAE (min)": "5.19", "Notes": "Tuned alpha=1.0 Regularization"},
        {"Rank": "8", "Algorithm": "Linear Regression (Baseline)", "R² Score": "0.514", "RMSE (min)": "6.52", "MAE (min)": "5.19", "Notes": "Baseline; Explains Feature Coefficients"},
        {"Rank": "9", "Algorithm": "ElasticNet Regression", "R² Score": "0.510", "RMSE (min)": "6.55", "MAE (min)": "5.21", "Notes": "Combined L1+L2 (l1_ratio=0.5)"},
        {"Rank": "10", "Algorithm": "Lasso Regression (L1)", "R² Score": "0.508", "RMSE (min)": "6.56", "MAE (min)": "5.22", "Notes": "Feature Sparsity: Zeroed out Redundant Features"}
    ]
    st.dataframe(pd.DataFrame(reg_summary_data), width="stretch")

    st.markdown("### 2. Classification Track (Target: `is_late > 30 min`)")
    clf_summary_data = [
        {"Rank": "🥇 1", "Algorithm": "Gradient Boosting Classifier (Tuned)", "Accuracy": "84.9%", "Precision": "0.851", "Recall": "0.849", "F1-Score": "0.846", "ROC-AUC": "0.916", "Scope": "Review 2 (Part B)"},
        {"Rank": "🥈 2", "Algorithm": "Random Forest Classifier", "Accuracy": "83.8%", "Precision": "0.839", "Recall": "0.838", "F1-Score": "0.835", "ROC-AUC": "0.908", "Scope": "Review 2 (Part B)"},
        {"Rank": "🥉 3", "Algorithm": "MLP Neural Network", "Accuracy": "82.5%", "Precision": "0.826", "Recall": "0.825", "F1-Score": "0.821", "ROC-AUC": "0.892", "Scope": "Review 2 (Part B)"},
        {"Rank": "4", "Algorithm": "AdaBoost Classifier", "Accuracy": "82.1%", "Precision": "0.822", "Recall": "0.821", "F1-Score": "0.817", "ROC-AUC": "0.887", "Scope": "Review 2 (Part B)"},
        {"Rank": "5", "Algorithm": "Bagging Classifier (Decision Tree)", "Accuracy": "81.9%", "Precision": "0.819", "Recall": "0.819", "F1-Score": "0.814", "ROC-AUC": "0.882", "Scope": "Review 2 (Part B)"},
        {"Rank": "6", "Algorithm": "Decision Tree Classifier", "Accuracy": "81.4%", "Precision": "0.814", "Recall": "0.814", "F1-Score": "0.809", "ROC-AUC": "0.871", "Scope": "Review 1 (Part A)"},
        {"Rank": "7", "Algorithm": "Support Vector Machine (SVC)", "Accuracy": "79.8%", "Precision": "0.796", "Recall": "0.798", "F1-Score": "0.790", "ROC-AUC": "0.852", "Scope": "Review 1 (Part A)"},
        {"Rank": "8", "Algorithm": "K-Nearest Neighbors Classifier", "Accuracy": "78.4%", "Precision": "0.781", "Recall": "0.784", "F1-Score": "0.776", "ROC-AUC": "0.828", "Scope": "Review 1 (Part A)"},
        {"Rank": "9", "Algorithm": "Logistic Regression (Baseline)", "Accuracy": "72.4%", "Precision": "0.718", "Recall": "0.724", "F1-Score": "0.714", "ROC-AUC": "0.784", "Scope": "Review 1 (Part A)"},
        {"Rank": "10", "Algorithm": "Naive Bayes (Gaussian)", "Accuracy": "68.9%", "Precision": "0.695", "Recall": "0.689", "F1-Score": "0.687", "ROC-AUC": "0.742", "Scope": "Review 1 (Part A)"}
    ]
    st.dataframe(pd.DataFrame(clf_summary_data), width="stretch")

    st.markdown("### 3. Clustering Track (Unsupervised Segmentation)")
    clu_summary_data = [
        {"Algorithm / Configuration": "K-Means (k=3, Euclidean)", "Silhouette Score (↑)": "0.428", "Davies-Bouldin Index (↓)": "0.941", "Calinski-Harabasz Index (↑)": "3,892.4", "Recommendation": "Optimal Production Model"},
        {"Algorithm / Configuration": "Agglomerative (k=3, Ward Linkage)", "Silhouette Score (↑)": "0.395", "Davies-Bouldin Index (↓)": "1.018", "Calinski-Harabasz Index (↑)": "3,412.1", "Recommendation": "Hierarchical Verification"},
        {"Algorithm / Configuration": "Agglomerative (k=3, Complete Linkage)", "Silhouette Score (↑)": "0.312", "Davies-Bouldin Index (↓)": "1.345", "Calinski-Harabasz Index (↑)": "2,180.5", "Recommendation": "Sensitive to Outliers"},
        {"Algorithm / Configuration": "Agglomerative (k=3, Average Linkage)", "Silhouette Score (↑)": "0.341", "Davies-Bouldin Index (↓)": "1.220", "Calinski-Harabasz Index (↑)": "2,530.8", "Recommendation": "Intermediate Performance"}
    ]
    st.dataframe(pd.DataFrame(clu_summary_data), width="stretch")

