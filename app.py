import streamlit as st

st.title("Diabetes Risk Predictor")
st.write("Enter a number for Age and click Predict.")
age = st.number_input("Age", min_value=0.0, value=30.0, step=1.0)
if st.button("Predict"):
    risk = age / 100 * 100  # Placeholder logic
    st.write(f"Risk: {risk:.1f}%")
