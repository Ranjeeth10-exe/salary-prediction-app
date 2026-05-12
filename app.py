import streamlit as st
import joblib
import numpy as np

# Load saved model
model = joblib.load('model.pkl')

# Website title
st.title("Salary Prediction App")

st.write("Predict salary using experience")

# User input
experience = st.number_input("Enter Years of Experience")

# Button
if st.button("Predict Salary"):

    prediction = model.predict(np.array([[experience]]))

    st.success(f"Predicted Salary: ₹ {prediction[0][0]:,.2f}")