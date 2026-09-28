# src/predict.py
# Helper functions for loading the model and making predictions

import joblib
import pandas as pd

def load_model(model_path="models/churn_model.pkl"):
    """Load the trained churn model."""
    return joblib.load(model_path)

def load_encoders(encoders_path="models/label_encoders.pkl"):
    """Load the label encoders used during training."""
    return joblib.load(encoders_path)

def predict_churn(customer_data: dict) -> float:
    """
    Predict churn probability for a single customer.
    customer_data: dict of raw feature values (same keys as training columns).
    Returns: churn probability between 0 and 1.
    """
    model = load_model()
    encoders = load_encoders()

    row = pd.DataFrame([customer_data])
    for col, le in encoders.items():
        if col in row.columns:
            row[col] = le.transform(row[col].astype(str))

    probability = model.predict_proba(row)[0][1]
    return probability

# Quick test (only runs when you execute this file directly)
if __name__ == "__main__":
    sample = {
        "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes",
        "Dependents": "No", "tenure": 12, "PhoneService": "Yes",
        "MultipleLines": "No", "InternetService": "Fiber optic",
        "OnlineSecurity": "No", "OnlineBackup": "No",
        "DeviceProtection": "No", "TechSupport": "No",
        "StreamingTV": "Yes", "StreamingMovies": "No",
        "Contract": "Month-to-month", "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 85.0, "TotalCharges": 1000.0,
    }
    prob = predict_churn(sample)
    print(f"Churn probability: {prob * 100:.1f}%")
    print("Prediction:", "CHURN" if prob >= 0.5 else "STAY")