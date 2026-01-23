import streamlit as st
import numpy as np
import pickle

# Load model & scaler only
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

st.set_page_config(page_title="Loan Approval Prediction")

st.title("🏦 Loan Approval Prediction (SVM)")

# Inputs
applicant_income = st.number_input("Applicant Income", min_value=0.0, step=1000.0)
loan_amount = st.number_input("Loan Amount", min_value=0.0, step=100.0)
credit_history = st.selectbox("Credit History", [0.0, 1.0])
self_employed = st.selectbox("Self Employed", ["No", "Yes"])

# ✅ Manual encoding (SAFE)
self_employed_encoded = 1 if self_employed == "Yes" else 0

if st.button("Predict Loan Status"):
    input_data = np.array([[
        applicant_income,
        loan_amount,
        credit_history,
        self_employed_encoded
    ]])

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]

    if prediction == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")
