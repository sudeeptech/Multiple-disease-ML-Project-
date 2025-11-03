import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Load ML models
diabetes_model = joblib.load("diabetes_model.sav")
heart_model = joblib.load("heart_disease_model.sav")
parkinsons_model = joblib.load("parkinsons_model.sav")

st.title("Health Prediction App")

# Sidebar for model selection
model_choice = st.sidebar.selectbox("Choose a Prediction Model",
                                    ("Diabetes", "Heart Disease", "Parkinsons"))

def predict_diabetes(input_data):
    input_array = np.array(input_data).reshape(1, -1)
    prediction = diabetes_model.predict(input_array)
    return "Diabetic" if prediction[0]==1 else "Not Diabetic"

def predict_heart(input_data):
    input_array = np.array(input_data).reshape(1, -1)
    prediction = heart_model.predict(input_array)
    return "Heart Disease" if prediction[0]==1 else "No Heart Disease"

def predict_parkinsons(input_data):
    input_array = np.array(input_data).reshape(1, -1)
    prediction = parkinsons_model.predict(input_array)
    return "Parkinsons" if prediction[0]==1 else "No Parkinsons"

if model_choice == "Diabetes":
    st.header("Diabetes Prediction")
    pregnancies = st.number_input("Number of Pregnancies", 0, 20)
    glucose = st.number_input("Glucose Level", 0, 200)
    bp = st.number_input("Blood Pressure", 0, 140)
    skin_thickness = st.number_input("Skin Thickness", 0, 100)
    insulin = st.number_input("Insulin Level", 0, 900)
    bmi = st.number_input("BMI", 0.0, 70.0)
    diabetes_pedigree = st.number_input("Diabetes Pedigree Function", 0.0, 2.5)
    age = st.number_input("Age", 0, 120)

    if st.button("Predict"):
        result = predict_diabetes([pregnancies, glucose, bp, skin_thickness,
                                   insulin, bmi, diabetes_pedigree, age])
        st.success(f"Prediction: {result}")

elif model_choice == "Heart Disease":
    st.header("Heart Disease Prediction")
    # Example input fields (customize as per your model)
    age = st.number_input("Age", 0, 120)
    sex = st.number_input("Sex (1=Male, 0=Female)", 0, 1)
    cp = st.number_input("Chest Pain Type (0-3)", 0, 3)
    trestbps = st.number_input("Resting Blood Pressure", 0, 200)
    chol = st.number_input("Serum Cholesterol", 0, 600)
    fbs = st.number_input("Fasting Blood Sugar > 120 mg/dl (1=True, 0=False)", 0, 1)
    restecg = st.number_input("Resting ECG (0-2)", 0, 2)
    thalach = st.number_input("Max Heart Rate Achieved", 0, 250)
    exang = st.number_input("Exercise Induced Angina (1=Yes, 0=No)", 0, 1)
    oldpeak = st.number_input("Oldpeak", 0.0, 10.0)
    slope = st.number_input("Slope of Peak Exercise ST segment (0-2)", 0, 2)
    ca = st.number_input("Number of Major Vessels (0-3)", 0, 3)
    thal = st.number_input("Thalassemia (1-3)", 0, 3)

    if st.button("Predict Heart Disease"):
        result = predict_heart([age, sex, cp, trestbps, chol, fbs, restecg,
                                thalach, exang, oldpeak, slope, ca, thal])
        st.success(f"Prediction: {result}")

elif model_choice == "Parkinsons":
    st.header("Parkinsons Prediction")
    # Example input fields (customize as per your model)
    fo = st.number_input("MDVP:Fo(Hz)", 0.0, 500.0)
    fhi = st.number_input("MDVP:Fhi(Hz)", 0.0, 500.0)
    flo = st.number_input("MDVP:Flo(Hz)", 0.0, 500.0)
    Jitter_percent = st.number_input("Jitter (%)", 0.0, 1.0)
    Jitter_Abs = st.number_input("Jitter(Abs)", 0.0, 1.0)
    RAP = st.number_input("RAP", 0.0, 1.0)
    PPQ = st.number_input("PPQ", 0.0, 1.0)
    DDP = st.number_input("DDP", 0.0, 1.0)
    Shimmer = st.number_input("Shimmer", 0.0, 1.0)
    Shimmer_dB = st.number_input("Shimmer(dB)", 0.0, 2.0)
    APQ3 = st.number_input("APQ3", 0.0, 1.0)
    APQ5 = st.number_input("APQ5", 0.0, 1.0)
    APQ = st.number_input("APQ", 0.0, 1.0)
    DDA = st.number_input("DDA", 0.0, 1.0)
    NHR = st.number_input("NHR", 0.0, 1.0)
    HNR = st.number_input("HNR", 0.0, 50.0)
    RPDE = st.number_input("RPDE", 0.0, 1.0)
    DFA = st.number_input("DFA", 0.0, 2.0)
    spread1 = st.number_input("spread1", -10.0, 10.0)
    spread2 = st.number_input("spread2", 0.0, 10.0)
    D2 = st.number_input("D2", 0.0, 5.0)
    PPE = st.number_input("PPE", 0.0, 1.0)

    if st.button("Predict Parkinsons"):
        result = predict_parkinsons([fo, fhi, flo, Jitter_percent, Jitter_Abs, RAP, PPQ, DDP,
                                     Shimmer, Shimmer_dB, APQ3, APQ5, APQ, DDA, NHR, HNR,
                                     RPDE, DFA, spread1, spread2, D2, PPE])
        st.success(f"Prediction: {result}")
