# 📉 Customer Churn Predictor

**Week 4 Project — Supervised Classification (Machine Learning Internship)**

## Objective
Predict whether a customer will churn using demographic and account data,
and compare **Logistic Regression** vs **Decision Tree** classifiers using
confusion matrices and precision-recall curves.

## Dataset
Telco Customer Churn dataset (`data/churn.csv`) — customer demographics,
account info (tenure, contract type, charges), and a Churn label.

## Project Structure
customer-churn-predictor/
├── data/churn.csv               # dataset
├── src/train_model.py           # training + evaluation script
├── models/                      # saved model + encoders (.pkl)
├── plots/                       # confusion matrices + PR curve images
├── app/streamlit_app.py         # live demo app
├── requirements.txt
└── README.md