import streamlit as st
import pickle
import pandas as pd

# Page Settings
st.set_page_config(
    page_title="Loan Approval Prediction",
    page_icon="🏦",
    layout="centered"
)

# Load Model
model = pickle.load(open("model.pkl", "rb"))

# Title
st.title("🏦 Loan Approval Prediction System")

st.write(
    "CREATED BY SUJAL VERMA ROLLNO-2300270310184"
    "This application predicts whether a loan is likely to be approved based on applicant information using a trained Decision Tree model."
)

st.markdown("---")

st.header("Applicant Information")

# Input Fields
age = st.number_input("Age", 18, 100, 30)

income = st.number_input(
    "Income",
    min_value=1000,
    value=50000,
    step=1000
)

loan_amount = st.number_input(
    "Loan Amount",
    min_value=1000,
    value=20000,
    step=1000
)

credit_score = st.number_input(
    "Credit Score",
    min_value=300,
    max_value=900,
    value=700
)

# Predict Button
if st.button("Predict Loan Approval"):

    input_data = pd.DataFrame({
        "Age":[age],
        "Income":[income],
        "LoanAmount":[loan_amount],
        "CreditScore":[credit_score]
    })

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(input_data)

    confidence = max(probability[0])*100

    st.markdown("---")

    st.header("Prediction Result")

    if prediction == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")

    st.write(f"### Confidence : {confidence:.2f}%")

    st.progress(int(confidence))
