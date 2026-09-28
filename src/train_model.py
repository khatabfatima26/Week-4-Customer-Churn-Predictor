# src/script/train_model.py
# Week 4 - Customer Churn Predictor
# Trains Logistic Regression and Decision Tree, compares them,
# saves plots + trained models.

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, precision_recall_curve)

# ---------- 1. Load data ----------
df = pd.read_csv("data/churn.csv")
print("First 5 rows:")
print(df.head())
print("\nMissing values:\n", df.isnull().sum())
print("\nClass balance:\n", df["Churn"].value_counts())

# ---------- 2. Preprocessing ----------
df = df.drop(columns=["customerID"])                      # useless for prediction
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(0)

# Label-encode every non-numeric column and SAVE the encoders for the app
# (works on both old "object" and new "str" pandas column types)
encoders = {}
for col in df.columns:
    if not pd.api.types.is_numeric_dtype(df[col]):
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

X = df.drop(columns=["Churn"])
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# ---------- 3. Model 1: Logistic Regression (with scaling) ----------
lr_model = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=1000))
])
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)
lr_prob = lr_model.predict_proba(X_test)[:, 1]

# ---------- 4. Model 2: Decision Tree (limited depth to avoid overfitting) ----------
dt_model = DecisionTreeClassifier(max_depth=5, random_state=42)
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)
dt_prob = dt_model.predict_proba(X_test)[:, 1]

# ---------- 5. Compare results ----------
print("\n===== LOGISTIC REGRESSION =====")
print("Accuracy:", accuracy_score(y_test, lr_pred))
print(classification_report(y_test, lr_pred, target_names=["No Churn", "Churn"]))

print("\n===== DECISION TREE =====")
print("Accuracy:", accuracy_score(y_test, dt_pred))
print(classification_report(y_test, dt_pred, target_names=["No Churn", "Churn"]))

# ---------- 6. Save confusion matrices as images ----------
os.makedirs("plots", exist_ok=True)
os.makedirs("models", exist_ok=True)

for name, pred in [("Logistic Regression", lr_pred), ("Decision Tree", dt_pred)]:
    cm = confusion_matrix(y_test, pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Churn", "Churn"], yticklabels=["No Churn", "Churn"])
    plt.title(f"Confusion Matrix - {name}")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(f"plots/confusion_matrix_{name.split()[0].lower()}.png", dpi=150)
    plt.close()

# ---------- 7. Precision-Recall curves on one graph ----------
plt.figure(figsize=(6, 5))
for name, prob in [("Logistic Regression", lr_prob), ("Decision Tree", dt_prob)]:
    precision, recall, _ = precision_recall_curve(y_test, prob)
    plt.plot(recall, precision, label=name)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve Comparison")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("plots/precision_recall_curve.png", dpi=150)
plt.close()

# ---------- 8. Save models + encoders for the Streamlit app ----------
joblib.dump(lr_model, "models/churn_model.pkl")
joblib.dump(encoders, "models/label_encoders.pkl")
print("\n✅ Saved: models/churn_model.pkl, models/label_encoders.pkl, and plots/")