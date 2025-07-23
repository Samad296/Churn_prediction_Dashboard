import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Load trained model and scaler
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Streamlit App UI
st.set_page_config(page_title="Churn Predictor", layout="wide")
st.title("📉 Customer Churn Prediction")
st.markdown("Predict whether a customer will churn based on their attributes.")

# Sidebar Inputs
st.sidebar.header("🧾 Input Customer Details")

SeniorCitizen = st.sidebar.selectbox("Senior Citizen", ["No", "Yes"])
partner = st.sidebar.selectbox("Has Partner?", ["No", "Yes"])
dependents = st.sidebar.selectbox("Has Dependents?", ["No", "Yes"])
tenure = st.sidebar.slider("Tenure (in months)", 0, 72, 24)
monthly_charges = st.sidebar.slider("Monthly Charges ($)", 0.0, 200.0, 70.0)
total_charges = st.sidebar.slider("Total Charges ($)", 0.0, 10000.0, 2000.0)

online_security = st.sidebar.selectbox("Online Security", ["No", "Yes", "No internet service"])
online_backup = st.sidebar.selectbox("Online Backup", ["No", "Yes", "No internet service"])
device_protection = st.sidebar.selectbox("Device Protection", ["No", "Yes", "No internet service"])
tech_support = st.sidebar.selectbox("Tech Support", ["No", "Yes", "No internet service"])

payment_method = st.sidebar.selectbox("Payment Method", [
    "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
])

paperless = st.sidebar.selectbox("Paperless Billing", ["No", "Yes"])

# Encode Inputs
binary_map = {"No": 0, "Yes": 1}
internet_map = {"No": 0, "Yes": 1, "No internet service": 2}
payment_map = {
    "Electronic check": 0,
    "Mailed check": 1,
    "Bank transfer (automatic)": 2,
    "Credit card (automatic)": 3
}

input_data = {
    'SeniorCitizen': binary_map[SeniorCitizen],
    'Partner': binary_map[partner],
    'Dependents': binary_map[dependents],
    'tenure': tenure,
    'OnlineSecurity': internet_map[online_security],
    'OnlineBackup': internet_map[online_backup],
    'DeviceProtection': internet_map[device_protection],
    'TechSupport': internet_map[tech_support],
    'PaymentMethod': payment_map[payment_method],
    'PaperlessBilling': binary_map[paperless],
    'MonthlyCharges': monthly_charges,
    'TotalCharges': total_charges
}

input_df = pd.DataFrame([input_data])

# Scale numerical features
num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
input_df[num_cols] = scaler.transform(input_df[num_cols])

# Predict button
if st.sidebar.button("🔮 Predict Churn"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.markdown("---")
    st.subheader("🎯 Prediction Result")
    if prediction == 1:
        st.error(f"⚠️ The customer is likely to **churn** (Probability: {probability:.2%})")
    else:
        st.success(f"✅ The customer is **not likely to churn** (Probability: {probability:.2%})")

    # Example Model Metrics & Confusion Matrix
    st.markdown("### 📊 Model Evaluation Summary")

    accuracy = 0.7679
    precision = 0.55
    recall = 0.73
    f1 = 0.62
    cm = [[811, 225], [102, 271]]  # Example confusion matrix

    st.markdown(f"""
    - **Accuracy**: `{accuracy:.2%}`
    - **Precision (Churn)**: `{precision:.2f}`
    - **Recall (Churn)**: `{recall:.2f}`
    - **F1 Score (Churn)**: `{f1:.2f}`
    """)

    st.markdown("### 📉 Confusion Matrix")
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=["No Churn", "Churn"],
                yticklabels=["No Churn", "Churn"], ax=ax)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    st.pyplot(fig)
