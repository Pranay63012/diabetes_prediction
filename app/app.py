import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path

# ------------------------------------------------------------
#                 PAGE CONFIG & CUSTOM THEME
# ------------------------------------------------------------
st.set_page_config(
    page_title="Diabetes Prediction System",
    layout="wide",
    page_icon="🩺"
)

# Blue & Black custom CSS
st.markdown("""
<style>

body {
    background-color: #0B0F19;
    color: #E8E8E8;
}

.sidebar .sidebar-content {
    background-color: #111827;
}

.css-1d391kg, .stButton>button {
    background-color: #2563EB !important;
    color: white !important;
    border-radius: 8px !important;
    padding: 0.6rem 1.2rem !important;
    font-size: 1.1rem !important;
    font-weight: 600 !important;
}

.stNumberInput>div>div>input {
    background-color: #1F2937 !important;
    color: #E5E7EB !important;
}

.stTable thead tr {
    background-color: #1E3A8A;
    color: white;
}

h1 {
    color: #3B82F6;
    font-weight: 900;
    letter-spacing: 2px;
}

h2, h3 {
    color: #60A5FA;
}

</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------
#                 PATHS & MODEL LOADING
# ------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"

MODEL_PATH = MODEL_DIR / "diabetes_rf_model.pkl"
SCALER_PATH = MODEL_DIR / "scaler.pkl"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_resource
def load_scaler():
    return joblib.load(SCALER_PATH)

model = load_model()
scaler = load_scaler()

# ------------------------------------------------------------
#         FUNCTION TO MAKE PREDICTION
# ------------------------------------------------------------
def predict_diabetes(values):
    arr = np.array(values).reshape(1, -1)
    arr_scaled = scaler.transform(arr)
    pred = model.predict(arr_scaled)[0]
    prob = model.predict_proba(arr_scaled)[0][1]
    return pred, prob


# ------------------------------------------------------------
#             HEADER SECTION
# ------------------------------------------------------------
st.markdown("<h1 style='text-align:center;'>🩺 DIABETES PREDICTION SYSTEM</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align:center;'>Enter health details & view estimated diabetes risk instantly</h3><br>", unsafe_allow_html=True)


# ------------------------------------------------------------
#             LAYOUT: SIDEBAR INPUTS
# ------------------------------------------------------------
st.sidebar.header("🔍 Input Features")

preg = st.sidebar.number_input("Pregnancies", 0, 20, 2)
glucose = st.sidebar.number_input("Glucose Level", 0, 250, 120)
bp = st.sidebar.number_input("Blood Pressure", 0, 200, 70)
skin = st.sidebar.number_input("Skin Thickness", 0, 100, 25)
insulin = st.sidebar.number_input("Insulin Level", 0, 900, 80)
bmi = st.sidebar.number_input("BMI (kg/m²)", 0.0, 70.0, 25.0)
dpf = st.sidebar.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
age = st.sidebar.number_input("Age", 1, 120, 30)

predict_btn = st.sidebar.button("Predict")


# ------------------------------------------------------------
#             INPUT FEATURE EXPLANATIONS
# ------------------------------------------------------------
st.markdown("## 📘 Input Feature Guide (Read Before Entering Values)")
st.write("""
- **Pregnancies** – Number of times the person has been pregnant (for males it's 0).
- **Glucose Level** – Plasma glucose concentration after a 2-hour oral glucose tolerance test.
- **Blood Pressure** – Diastolic blood pressure (mm Hg).
- **Skin Thickness** – Triceps skin fold thickness (mm).
- **Insulin** – Serum insulin level (μU/ml).
- **BMI** – Body Mass Index = weight (kg) / height (m²).
- **Diabetes Pedigree Function** – Indicates hereditary diabetes risk.
- **Age** – Patient age in years.
""")

# ------------------------------------------------------------
#           SMALL SAMPLE INPUT TABLE (Excel-like)
# ------------------------------------------------------------
st.markdown("### 📊 Sample Input Reference Table (Use These as Examples)")
sample_data = pd.DataFrame({
    "Preg": [2, 5, 1, 3, 0],
    "Glucose": [120, 155, 99, 140, 80],
    "BP": [70, 82, 64, 90, 75],
    "Skin": [22, 30, 18, 28, 20],
    "Insulin": [85, 130, 60, 100, 50],
    "BMI": [25.3, 32.5, 21.1, 29.7, 23.4],
    "DPF": [0.45, 1.20, 0.35, 0.90, 0.25],
    "Age": [29, 45, 22, 35, 28]
})

st.table(sample_data.style.set_table_attributes("style='display:inline-block; width:40%;'"))


# ------------------------------------------------------------
#                      PREDICTION OUTPUT
# ------------------------------------------------------------
if predict_btn:
    features = [preg, glucose, bp, skin, insulin, bmi, dpf, age]
    pred, prob = predict_diabetes(features)

    st.markdown("---")
    st.markdown("## 🔮 Prediction Result")

    if pred == 1:
        st.error(f"### ❗ High Risk of Diabetes\nProbability: **{prob*100:.2f}%**")
    else:
        st.success(f"### ✅ Low Risk of Diabetes\nProbability: **{prob*100:.2f}%**")


