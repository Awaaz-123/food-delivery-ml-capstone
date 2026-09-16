# 🚴 Food Delivery & Restaurant Analysis — Machine Learning Capstone

**Course**: 23CSE301 Machine Learning — Capstone Project  
**Academic Year**: 2026-27  
**Problem Tracks**: Regression, Classification, Clustering  

---

## 📌 Project Overview
This project builds a complete, end-to-end Machine Learning pipeline analyzing real-world food delivery data. It covers exploratory data analysis (EDA), feature engineering, model training and comparison across 10 regression algorithms, 10 classification algorithms, and 2 clustering algorithms, capped with an interactive Streamlit web dashboard.

---

## 📁 Repository Structure
```text
/
├── README.md                  # Project overview, metric tables & instructions
├── requirements.txt           # Python dependencies
├── data/
│   └── food_delivery.csv      # Raw dataset
├── notebooks/
│   ├── regression.ipynb       # Track 1: 10 Regression algorithms (Delivery Time)
│   ├── classification.ipynb   # Track 2: 10 Classification algorithms (Late Risk)
│   └── clustering.ipynb       # Track 3: Customer & Restaurant segmentation
├── models/                    # Saved model binaries (.pkl / .joblib)
└── app/
    └── app.py                 # Streamlit Web GUI Application
```

---

## 🛠️ Environment Setup & Installation

### 1. Clone & Activate Virtual Environment
```bash
cd "grp proj"
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch Jupyter Notebooks
```bash
jupyter notebook
```

---

## 🏆 Model Performance Summary

### 1. Regression Track (Target: `Time_taken(min)`)
| Model | R² Score | RMSE (min) | MAE (min) |
|---|---|---|---|
| **Gradient Boosting Regressor** | **0.81** | **4.12** | **3.15** |
| Random Forest Regressor | 0.79 | 4.35 | 3.32 |
| Decision Tree Regressor | 0.73 | 4.88 | 3.76 |
| K-Nearest Neighbors | 0.58 | 6.12 | 4.80 |
| Linear Regression | 0.51 | 6.54 | 5.21 |

### 2. Classification Track (Target: `is_late` > 30 min)
| Model | Accuracy | Weighted F1 | ROC-AUC |
|---|---|---|---|
| **Gradient Boosting Classifier** | **84.5%** | **0.84** | **0.91** |
| Random Forest Classifier | 83.2% | 0.83 | 0.90 |
| Logistic Regression | 72.1% | 0.72 | 0.78 |

### 3. Clustering Track (Segmentation)
| Algorithm | Silhouette Score | Davies-Bouldin Index |
|---|---|---|
| **K-Means (k=3)** | **0.42** | **0.95** |
| Agglomerative Hierarchical | 0.39 | 1.02 |

---

## 🚀 Launching the Interactive GUI App 

Run the Streamlit application locally:
```bash
streamlit run app/app.py
```
