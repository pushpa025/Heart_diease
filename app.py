import streamlit as st
import pandas as pd
import joblib
import numpy as np
from lime import lime_tabular
import matplotlib.pyplot as plt

# Load saved components
model = joblib.load('heart_model.pkl')
scaler = joblib.load('scaler.pkl')
X_train_scaled = joblib.load('X_train_scaled.pkl')

st.set_page_config(page_title="Heart AI Diagnostic", layout="wide")
st.title("🏥 Cardiac Risk AI-Diagnostic System")

# Form for user inputs
with st.sidebar:
    st.header("Patient Clinical Data")
    age = st.slider("Age", 20, 80, 50)
    sex = st.selectbox("Sex (1=M, 0=F)", [1, 0])
    cp = st.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3])
    trestbps = st.number_input("Blood Pressure", 90, 200, 120)
    chol = st.number_input("Cholesterol", 100, 500, 200)
    fbs = st.selectbox("Fasting Sugar > 120", [0, 1])
    restecg = st.selectbox("Resting ECG", [0, 1, 2])
    thalach = st.slider("Max Heart Rate", 70, 210, 150)
    exang = st.selectbox("Exercise Angina", [0, 1])
    oldpeak = st.slider("ST Depression", 0.0, 6.0, 1.0)
    slope = st.selectbox("ST Slope", [0, 1, 2])
    ca = st.selectbox("Major Vessels (0-3)", [0, 1, 2, 3])
    thal = st.selectbox("Thalassemia", [1, 2, 3])

# Run Diagnostic
if st.button("Predict Risk"):
    user_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
    scaled_data = scaler.transform(user_data)
    prediction = model.predict(scaled_data)
    prob = model.predict_proba(scaled_data)[0][1]
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Diagnostic Result")
        if prediction[0] == 1:
            st.error(f"⚠️ HIGH RISK DETECTED ({prob*100:.1f}%)")
        else:
            st.success(f"✅ LOW RISK ({prob*100:.1f}%)")
            
    with col2:
        st.subheader("Explanation (LIME)")
        explainer = lime_tabular.LimeTabularExplainer(
            training_data=X_train_scaled,
            feature_names=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'],
            class_names=['Healthy', 'Disease'], mode='classification'
        )
        exp = explainer.explain_instance(scaled_data[0], model.predict_proba)
        st.pyplot(exp.as_pyplot_figure())