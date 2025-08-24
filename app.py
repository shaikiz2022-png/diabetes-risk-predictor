import streamlit as st
from diabetes_predictor import predict_diabetes  # Adjust import based on your script

# Title and instructions
st.title("Diabetes Risk Predictor")
st.write("Enter the following details to predict the likelihood of diabetes. All fields are required. Values should be based on medical measurements. If unsure, consult a healthcare professional.")

# Input fields based on common diabetes dataset features
pregnancies = st.number_input("Pregnancies", min_value=0.0, value=0.0, step=1.0)
glucose = st.number_input("Glucose", min_value=0.0, value=100.0, step=1.0)
blood_pressure = st.number_input("Blood Pressure", min_value=0.0, value=70.0, step=1.0)
skin_thickness = st.number_input("Skin Thickness", min_value=0.0, value=20.0, step=1.0)
insulin = st.number_input("Insulin", min_value=0.0, value=100.0, step=1.0)
bmi = st.number_input("BMI", min_value=0.0, value=25.0, step=0.1)
diabetes_pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.0, value=0.50, step=0.01)
age = st.number_input("Age", min_value=0.0, value=30.0, step=1.0)

# Predict button
if st.button("Predict"):
    if all([pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]):
        # Prepare input data as a list or dict (adjust based on your predict_diabetes function)
        input_data = [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]
        risk = predict_diabetes(input_data)  # Call your prediction function
        st.write(f"Risk: {min(max(risk, 0), 100):.1f}%")  # Ensure risk is between 0-100%
    else:
        st.error("Please fill all fields.")

# Optional: Load and display dataset info
import pandas as pd
st.write("Sample data from diabetes.csv:")
st.write(pd.read_csv("diabetes.csv").head())