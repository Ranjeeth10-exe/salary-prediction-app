import streamlit as st
import joblib
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Salary Prediction App",
    page_icon="💼",
    layout="centered"
)

# Custom CSS
st.markdown(
    """
    <style>
    .main {
        background-color: #f5f7fa;
    }

    .title {
        text-align: center;
        font-size: 40px;
        font-weight: bold;
        color: #1f2937;
        margin-bottom: 10px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #4b5563;
        margin-bottom: 30px;
    }

    .prediction-box {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0px 4px 15px rgba(0,0,0,0.1);
        text-align: center;
        margin-top: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Load model
model = joblib.load('model.pkl')

# Title
st.markdown('<div class="title">💼 Salary Prediction App</div>', unsafe_allow_html=True)

# Subtitle
st.markdown(
    '<div class="subtitle">Predict salary based on years of experience</div>',
     unsafe_allow_html=True
)

# Input section
experience = st.slider(
    "Years of Experience",
    min_value=0.0,
    max_value=20.0,
    value=1.0,
    step=0.5
)

# Predict button
if st.button("Predict Salary"):
    st.caption("Built using Machine Learning and Streamlit")