import streamlit as st
import pandas as pd
import joblib

# Load saved model and scaler
model = joblib.load('logistic_model.pkl')
scaler = joblib.load('scaler.pkl')

st.set_page_config(page_title="Credit Card Fraud Detection", layout="centered")
st.title("💳 Credit Card Fraud Detection App")
st.write("Enter transaction details below to check if it's fraud or not 👇")

v1 = st.number_input("V1", value=0.0)
v2 = st.number_input("V2", value=0.0)
v3 = st.number_input("V3", value=0.0)
v4 = st.number_input("V4", value=0.0)
amount = st.number_input("Amount", value=0.0)

if st.button("Predict"):
    input_data = pd.DataFrame([[v1, v2, v3, v4, amount]], columns=['V1','V2','V3','V4','Amount'])
    scaled_data = scaler.transform(input_data)
    pred = model.predict(scaled_data)[0]
    proba = model.predict_proba(scaled_data)[0][1]

    if pred == 1:
        st.error(f"🚨 Fraud Detected! (Probability: {proba:.2f})")
    else:
        st.success(f"✅ Transaction is Safe. (Probability: {proba:.2f})")

