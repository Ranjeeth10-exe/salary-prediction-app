import streamlit as st
import pandas as pd
import numpy as np
import joblib

# =========================================================
# PAGE SETTINGS
# =========================================================

st.set_page_config(
    page_title="Salary Prediction",
    page_icon="💼",
    layout="centered"
)

# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

model = load_model()

# =========================================================
# PREDICTION FUNCTION
# =========================================================

def predict_salary(experience):

    exp = np.array([[experience]])

    salary = model.predict(exp)[0]

    # Safety Limits
    salary = max(salary, 30000)
    salary = min(salary, 5000000)

    return salary

# =========================================================
# TITLE
# =========================================================

st.title("💼 Salary Prediction System")

st.write(
    "Enter your years of experience to predict your salary using Machine Learning."
)

st.divider()

# =========================================================
# USER INPUT
# =========================================================

experience = st.slider(
    "Years of Experience",
    min_value=0.0,
    max_value=25.0,
    value=2.0,
    step=0.5
)

# =========================================================
# PREDICTION
# =========================================================

# Predicted annual salary
annual_salary = predict_salary(experience)

# Monthly salary
monthly_salary = annual_salary / 12

# =========================================================
# OUTPUT
# =========================================================

st.subheader("Predicted Salary")

col1, col2 = st.columns(2)

with col1:
    st.success(f"Annual Salary\n\n₹ {annual_salary:,.0f}")

with col2:
    st.info(f"Monthly Salary\n\n₹ {monthly_salary:,.0f}")

# =========================================================
# GRAPH
# =========================================================

st.divider()

st.subheader("Salary Growth Graph")

x = np.linspace(0, 25, 100)

y = [predict_salary(i) for i in x]

chart_data = pd.DataFrame({
    "Experience": x,
    "Annual Salary": y
})

st.line_chart(
    chart_data,
    x="Experience",
    y="Annual Salary",
    use_container_width=True
)

# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption("Polynomial Regression Model | Streamlit Project")