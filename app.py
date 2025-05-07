import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load trained model
model = joblib.load("random_forest_model.pkl")

# Load dataset
df = pd.read_csv("laptop_data_cleaned.csv")

# Define features (drop target column)
target_col = "Price"
features = df.drop(columns=[target_col])

st.title("💻 Laptop Price Predictor")

user_input = {}

# Include touchscreen and IPS toggle switches if they exist
binary_cols = ['TouchScreen', 'Ips']
for col in binary_cols:
    if col in features.columns:
        user_input[col] = int(st.checkbox(f"{col}?"))

# Build input form dynamically (skip already-handled binary toggles)
for col in features.columns:
    if col in binary_cols:
        continue
    if features[col].dtype == 'object':
        options = sorted(df[col].unique().tolist())
        user_input[col] = st.selectbox(f"{col}", options)
    else:
        min_val = float(df[col].min())
        max_val = float(df[col].max())
        mean_val = float(df[col].mean())
        user_input[col] = st.number_input(f"{col}", min_value=min_val, max_value=max_val, value=mean_val)

# Prediction
if st.button("Predict"):
    input_df = pd.DataFrame([user_input])

    # One-hot encode and align with model features
    input_df = pd.get_dummies(input_df)
    if hasattr(model, 'feature_names_in_'):
        model_features = model.feature_names_in_
        for col in model_features:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[model_features]

    prediction = model.predict(input_df)
    prediction *= 300  # Adjusted for currency or scaling

    acc=open("model_accuracy.txt","r")
    model_accuracy = float(acc.read())
    acc.close()
    st.success(f"Predicted Laptop Price: {prediction[0]:,.2f}$ with {model_accuracy * 100:.2f}% Accuracy")
