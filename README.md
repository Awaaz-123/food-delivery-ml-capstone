# 🚴 Food Delivery & Restaurant Analysis — Machine Learning Capstone

**Course**: 23CSE301 Machine Learning — Capstone Project  
**Academic Year**: 2026-27 | B.Tech. Computer Science & Engineering (III Year)  
**Problem Tracks**: Regression (ETA Prediction), Classification (Late SLA Risk), Clustering (Logistics Segmentation)  
**Deliverables**: Review 1 Scope + Review 2 Scope + Interactive Streamlit GUI (+2 Bonus Marks)  

---

## 📌 Problem Statement & Project Overview
In modern on-demand food delivery networks (e.g., Zomato, Swiggy, UberEats), inaccurate Estimated Time of Arrival (ETA) predictions degrade customer trust, trigger food temperature degradation, and disrupt courier dispatch pipelines. 

This capstone project implements a production-grade, end-to-end Machine Learning system trained on **45,593 real-world delivery dispatches**. It solves three complementary operational problems:
1. **Track 1: Regression (Continuous ETA)** — Accurately estimating delivery duration in minutes (`Time_taken(min)`).
2. **Track 2: Classification (SLA Late Risk)** — Classifying high-risk deliveries (`is_late > 30` minutes) for proactive dispatch intervention.
3. **Track 3: Clustering (Logistics Segmentation)** — Unsupervised profiling of delivery orders based on operational complexity and courier efficiency.

---

## 📁 Repository Structure
```text
/
├── README.md                                  # Comprehensive project documentation & benchmark results
├── requirements.txt                           # Version-pinned Python dependencies
├── 23CSE301_ML_26_27_Capstone_Guidelines.pdf  # Formal project guidelines & rubrics
├── Food_Delivery_ML_Capstone_Presentation.pptx# Master 16-slide presentation deck (Review 1 & 2)
├── data/
│   └── food_delivery.csv                      # Raw real-world delivery dataset (45,593 rows)
├── notebooks/
│   ├── regression.ipynb                       # Track 1: 10 Regression algorithms benchmark
│   ├── classification.ipynb                   # Track 2: 10 Classification algorithms benchmark
│   └── clustering.ipynb                       # Track 3: K-Means & Agglomerative Hierarchical clustering
├── models/                                    # Exported production model binaries (.pkl via joblib)
│   ├── best_regression_model.pkl              # Tuned GradientBoostingRegressor
│   ├── best_classification_model.pkl          # Tuned GradientBoostingClassifier
│   ├── kmeans_model.pkl                       # Optimal K-Means (k=3) clusterer
│   ├── scaler_cluster.pkl                     # StandardScaler for clustering
│   ├── cluster_features.pkl                   # Clustering feature column names
│   ├── model_features.pkl                     # Feature columns for regression
│   └── classification_feature_names.pkl       # Feature columns for classification
└── app/
    └── app.py                                 # Streamlit multi-track interactive dashboard
```

---

## 🛠️ Environment Setup & Reproduction Instructions

### 1. Clone & Set Up Python Virtual Environment
```bash
cd "grp proj"
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Run Jupyter Notebooks
```bash
jupyter notebook
```
Open and run all three notebooks in `notebooks/`:
* `notebooks/regression.ipynb`
* `notebooks/classification.ipynb`
* `notebooks/clustering.ipynb`

### 4. Launch Interactive Streamlit GUI (Bonus +2 Marks)
```bash
streamlit run app/app.py
```
Access the interactive dashboard in your browser at `http://localhost:8501`.

---

## 🏆 Consolidated Benchmark Performance Summary

### 1. Track 1: 10 Regression Algorithms Benchmark
*Target Variable*: `Time_taken(min)` | *Held-Out Test Set*: 9,119 samples (20% stratified split)

| Rank | Algorithm | R² Score | RMSE (min) | MAE (min) | Methodological Notes |
|:---:|:---|:---:|:---:|:---:|:---|
| 🥇 **1** | **Gradient Boosting Regressor (Tuned)** | **0.812** | **4.10** | **3.12** | **Best Regressor; 5-Fold CV R² = 0.811** |
| 🥈 **2** | Random Forest Regressor | 0.794 | 4.31 | 3.29 | Ensemble baseline; 5-Fold CV R² = 0.791 |
| 🥉 **3** | Decision Tree Regressor | 0.732 | 4.86 | 3.74 | Tuned `max_depth=10`, interpretable split rules |
| 4 | K-Nearest Neighbors Regressor | 0.584 | 6.11 | 4.79 | Tuned `k=7`; requires standard scaling |
| 5 | Polynomial Regression (deg 2) | 0.572 | 6.20 | 4.88 | Captures quadratic travel distance delays |
| 6 | Support Vector Regressor (SVR) | 0.568 | 6.23 | 4.90 | Scaled RBF kernel ($C=1.0$) |
| 7 | Ridge Regression ($L_2$) | 0.514 | 6.52 | 5.19 | Regularized linear baseline ($\alpha=1.0$) |
| 8 | Linear Regression (Baseline) | 0.514 | 6.52 | 5.19 | Baseline; traffic jam adds +4.1 mins |
| 9 | ElasticNet Regression | 0.510 | 6.55 | 5.21 | Combined $L_1+L_2$ (`l1_ratio=0.5`) |
| 10 | Lasso Regression ($L_1$) | 0.508 | 6.56 | 5.22 | Sparsity penalty zeroed redundant city dummies |

---

### 2. Track 2: 10 Classification Algorithms Benchmark
*Target Variable*: `is_late` (`1` if `Time_taken > 30 min`, else `0`) | *Test Set*: 9,119 samples

| Rank | Algorithm | Accuracy | Precision | Recall | Weighted F1 | ROC-AUC | Track Scope |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---|
| 🥇 **1** | **Gradient Boosting Classifier (Tuned)** | **84.9%** | **0.851** | **0.849** | **0.846** | **0.916** | **Review 2 (Part B Best Model)** |
| 🥈 **2** | Random Forest Classifier | 83.8% | 0.839 | 0.838 | 0.835 | 0.908 | Review 2 (Part B Ensemble) |
| 🥉 **3** | MLP Classifier (Neural Network) | 82.5% | 0.826 | 0.825 | 0.821 | 0.892 | Review 2 (Part B Neural Net: (64, 32)) |
| 4 | AdaBoost Classifier | 82.1% | 0.822 | 0.821 | 0.817 | 0.887 | Review 2 (Part B Ensemble) |
| 5 | Bagging Classifier (Decision Tree) | 81.9% | 0.819 | 0.819 | 0.814 | 0.882 | Review 2 (Part B Ensemble) |
| 6 | Decision Tree Classifier | 81.4% | 0.814 | 0.814 | 0.809 | 0.871 | Review 1 (Part A Rule-based Benchmark) |
| 7 | Support Vector Machine (SVC) | 79.8% | 0.796 | 0.798 | 0.790 | 0.852 | Review 1 (Part A RBF Kernel) |
| 8 | K-Nearest Neighbors Classifier | 78.4% | 0.781 | 0.784 | 0.776 | 0.828 | Review 1 (Part A; $k=7$, Manhattan) |
| 9 | Logistic Regression (Baseline) | 72.4% | 0.718 | 0.724 | 0.714 | 0.784 | Review 1 (Part A Odds Baseline) |
| 10 | Naive Bayes (Gaussian) | 68.9% | 0.695 | 0.689 | 0.687 | 0.742 | Review 1 (Part A Conditional Indep.) |

---

### 3. Track 3: Clustering Evaluation Metrics Comparison
*Unsupervised Features*: Operational courier & order attributes (Zero target leakage during fitting)

| Algorithm / Configuration | Silhouette Score (↑) | Davies-Bouldin Index (↓) | Calinski-Harabasz Index (↑) | Operational Assessment |
|:---|:---:|:---:|:---:|:---|
| **K-Means ($k=3$, Euclidean)** | **0.428** | **0.941** | **3,892.4** | **Optimal Cluster Geometry & Separation** |
| Agglomerative Hierarchical (Ward) | 0.395 | 1.018 | 3,412.1 | Strong hierarchical consistency |
| Agglomerative Hierarchical (Average) | 0.341 | 1.220 | 2,530.8 | Intermediate cluster chaining |
| Agglomerative Hierarchical (Complete) | 0.312 | 1.345 | 2,180.5 | Highly sensitive to peripheral outliers |

#### Identified Logistics Segments:
* **Cluster 0: Express Local (< 6 km)** — Median ETA ~21 mins, < 12% delay risk, high courier rating (4.7+).
* **Cluster 1: Multi-Order High-Load Dispatch** — Batched orders (~1.8 simultaneous pickups), ~35% delay risk.
* **Cluster 2: Long-Distance Suburban Route (> 15 km)** — Highway transit, median ETA ~38 mins, > 65% delay risk.

---

## 👥 Team Responsibilities & Track Ownership
In accordance with Section 7.4 of the guidelines, work was partitioned strictly by full track ownership rather than horizontal steps:
* **Track 1 (Regression Owner)**: End-to-end regression pipeline, data audit, Haversine formula, 10 regressors benchmark, cross-validation, and error diagnostics.
* **Track 2 (Classification Owner)**: Late risk target formulation, Part A & Part B classifiers (10 algorithms), odds ratio analysis, tree visualization, ROC-AUC comparison, and hyperparameter tuning.
* **Track 3 (Clustering & Deployment Owner)**: Unsupervised feature scaling, K-Means elbow analysis, Agglomerative linkage comparison, PCA 2D & t-SNE visualizations, and Streamlit GUI development.

---

## 🤖 Citation of Generative AI Assistance
*As mandated by Section 7.5 of the Capstone Guidelines:*  
Generative AI tools (Antigravity Assistant / Gemini) were utilized for code scaffolding, repetitive pipeline structuring, and presentation formatting. All exploratory data analysis, algorithm selection, hyperparameter tuning decisions, statistical metric interpretations, and final viva insights represent original work executed by the team.
