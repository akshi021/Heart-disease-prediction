import streamlit as st
import pickle
import numpy as np

# Load model and scaler
with open('model.pkl', 'rb') as file:
    data = pickle.load(file)
    model = data['model']
    scaler = data['scaler']

# App title
st.title("Heart Disease Prediction app ")
st.write("Enter patient details below:")

# Input fields
age = st.number_input("Age", min_value=1, max_value=120, value=50)
sex = st.selectbox("Sex (0 = Female, 1 = Male)", [0, 1])
cp = st.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure", value=120)
chol = st.number_input("Serum Cholesterol (mg/dl)", value=200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (1 = True; 0 = False)", [0, 1])
restecg = st.selectbox("Resting ECG (0-2)", [0, 1, 2])
thalach = st.number_input("Maximum Heart Rate Achieved", value=150)
exang = st.selectbox("Exercise Induced Angina (1 = Yes; 0 = No)", [0, 1])
oldpeak = st.number_input("Oldpeak (ST depression)", value=1.0, step=0.1)
slope = st.selectbox("Slope of ST segment (0-2)", [0, 1, 2])
ca = st.selectbox("Number of major vessels (0-4)", [0, 1, 2, 3, 4])
thal = st.selectbox("Thal (0 = Normal, 1 = Fixed Defect, 2 = Reversible Defect)", [0, 1, 2, 3])

# Predict button
if st.button("Predict"):
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                            thalach, exang, oldpeak, slope, ca, thal]])
    
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)

    if prediction[0] == 1:
        st.error("⚠️ The model predicts a **high risk** of heart disease.")
    else:
        st.success("✅ The model predicts a **low risk** of heart disease.")



