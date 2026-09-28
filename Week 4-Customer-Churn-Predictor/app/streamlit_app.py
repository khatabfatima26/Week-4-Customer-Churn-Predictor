# app/streamlit_app.py
# Live demo app for the Customer Churn Predictor

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉")

# ---------- Load trained model + encoders ----------
model = joblib.load("models/churn_model.pkl")
encoders = joblib.load("models/label_encoders.pkl")

st.title("📉 Customer Churn Predictor")
st.write("Enter customer details to predict the probability of churn.")

# ---------- User inputs (same features the model was trained on) ----------
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", encoders["gender"].classes_)
    senior = st.selectbox("Senior Citizen", [0, 1])
    partner = st.selectbox("Partner", encoders["Partner"].classes_)
    dependents = st.selectbox("Dependents", encoders["Dependents"].classes_)
    tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
    phoneservice = st.selectbox("Phone Service", encoders["PhoneService"].classes_)
    multiplelines = st.selectbox("Multiple Lines", encoders["MultipleLines"].classes_)
    internetservice = st.selectbox("Internet Service", encoders["InternetService"].classes_)
    onlinesecurity = st.selectbox("Online Security", encoders["OnlineSecurity"].classes_)
    onlinebackup = st.selectbox("Online Backup", encoders["OnlineBackup"].classes_)

with col2:
    deviceprotection = st.selectbox("Device Protection", encoders["DeviceProtection"].classes_)
    techsupport = st.selectbox("Tech Support", encoders["TechSupport"].classes_)
    streamingtv = st.selectbox("Streaming TV", encoders["StreamingTV"].classes_)
    streamingmovies = st.selectbox("Streaming Movies", encoders["StreamingMovies"].classes_)
    contract = st.selectbox("Contract", encoders["Contract"].classes_)
    paperless = st.selectbox("Paperless Billing", encoders["PaperlessBilling"].classes_)
    paymentmethod = st.selectbox("Payment Method", encoders["PaymentMethod"].classes_)
    monthlycharges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=70.0)
    totalcharges = st.number_input("Total Charges ($)", min_value=0.0, value=800.0)

# ---------- Prediction ----------
if st.button("Predict Churn", type="primary"):
    raw = {
        "gender": gender, "SeniorCitizen": senior, "Partner": partner,
        "Dependents": dependents, "tenure": tenure, "PhoneService": phoneservice,
        "MultipleLines": multiplelines, "InternetService": internetservice,
        "OnlineSecurity": onlinesecurity, "OnlineBackup": onlinebackup,
        "DeviceProtection": deviceprotection, "TechSupport": techsupport,
        "StreamingTV": streamingtv, "StreamingMovies": streamingmovies,
        "Contract": contract, "PaperlessBilling": paperless,
        "PaymentMethod": paymentmethod, "MonthlyCharges": monthlycharges,
        "TotalCharges": totalcharges,
    }
    row = pd.DataFrame([raw])
    # Encode categorical inputs exactly like during training
    for col, le in encoders.items():
        if col in row.columns:
            row[col] = le.transform(row[col].astype(str))

    prob = model.predict_proba(row)[0][1]
    st.metric("Churn Probability", f"{prob * 100:.1f}%")

    if prob >= 0.5:
        st.error("⚠️ This customer is LIKELY to churn. Consider retention offers!")
    else:
        st.success("✅ This customer is likely to STAY.")

# ---------- Model evaluation dashboard (confusion matrix PR curve) ----------
st.divider()
st.header("📊 Model Evaluation")
st.subheader("Confusion Matrix - Logistic Regression")
st.image("plots/confusion_matrix_logistic.png")
st.subheader("Confusion Matrix - Decision Tree")
st.image("plots/confusion_matrix_decision.png")
st.subheader("Precision-Recall Curve")
st.image("plots/precision_recall_curve.png")