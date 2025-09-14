import streamlit as st
import pandas as pd
import joblib

# Load trained model and columns
model = joblib.load("churn_model.pkl")
model_columns = joblib.load("model_columns.pkl")

st.title("📊 Customer Churn Prediction App")
st.write("Enter customer details below to predict churn:")

# Input widgets for all 50 features (example: 0/1 for boolean columns)
inputs = {
    "SeniorCitizen": [st.selectbox("Senior Citizen", [0,1])],
    "Charges_Month": [st.number_input("Monthly Charges", 0.0, 5000.0)],
    "TotalCharges": [st.number_input("Total Charges", 0.0, 20000.0)],
    
    # Gender
    "gender_Female": [st.selectbox("Gender Female", [0,1])],
    "gender_Male": [st.selectbox("Gender Male", [0,1])],
    
    # Partner
    "Partner_No": [st.selectbox("Partner No", [0,1])],
    "Partner_Yes": [st.selectbox("Partner Yes", [0,1])],
    
    # Dependents
    "Dependents_No": [st.selectbox("Dependents No", [0,1])],
    "Dependents_Yes": [st.selectbox("Dependents Yes", [0,1])],
    
    # PhoneService
    "PhoneService_No": [st.selectbox("PhoneService No", [0,1])],
    "PhoneService_Yes": [st.selectbox("PhoneService Yes", [0,1])],
    
    # MultipleLines
    "MultipleLines_No": [st.selectbox("MultipleLines No", [0,1])],
    "MultipleLines_Yes": [st.selectbox("MultipleLines Yes", [0,1])],
    
    # InternetService
    "InternetService_DSL": [st.selectbox("InternetService DSL", [0,1])],
    "InternetService_Fiber optic": [st.selectbox("InternetService Fiber optic", [0,1])],
    "InternetService_No": [st.selectbox("InternetService No", [0,1])],
    
    # OnlineSecurity
    "OnlineSecurity_No": [st.selectbox("OnlineSecurity No", [0,1])],
    "OnlineSecurity_Yes": [st.selectbox("OnlineSecurity Yes", [0,1])],
    
    # OnlineBackup
    "OnlineBackup_No": [st.selectbox("OnlineBackup No", [0,1])],
    "OnlineBackup_Yes": [st.selectbox("OnlineBackup Yes", [0,1])],
    
    # DeviceProtection
    "DeviceProtection_No": [st.selectbox("DeviceProtection No", [0,1])],
    "DeviceProtection_Yes": [st.selectbox("DeviceProtection Yes", [0,1])],
    
    # TechSupport
    "TechSupport_No": [st.selectbox("TechSupport No", [0,1])],
    "TechSupport_Yes": [st.selectbox("TechSupport Yes", [0,1])],
    
    # StreamingTV
    "StreamingTV_No": [st.selectbox("StreamingTV No", [0,1])],
    "StreamingTV_Yes": [st.selectbox("StreamingTV Yes", [0,1])],
    
    # StreamingMovies
    "StreamingMovies_No": [st.selectbox("StreamingMovies No", [0,1])],
    "StreamingMovies_Yes": [st.selectbox("StreamingMovies Yes", [0,1])],
    
    # Contract
    "Contract_Month-to-month": [st.selectbox("Contract Month-to-month", [0,1])],
    "Contract_One year": [st.selectbox("Contract One year", [0,1])],
    "Contract_Two year": [st.selectbox("Contract Two year", [0,1])],
    
    # PaperlessBilling
    "PaperlessBilling_No": [st.selectbox("PaperlessBilling No", [0,1])],
    "PaperlessBilling_Yes": [st.selectbox("PaperlessBilling Yes", [0,1])],
    
    # PaymentMethod
    "Method_Payment_Bank transfer (automatic)": [st.selectbox("Payment Method Bank transfer", [0,1])],
    "Method_Payment_Credit card (automatic)": [st.selectbox("Payment Method Credit card", [0,1])],
    "Method_Payment_Electronic check": [st.selectbox("Payment Method Electronic check", [0,1])],
    "Method_Payment_Mailed check": [st.selectbox("Payment Method Mailed check", [0,1])],
    
    # Tenure groups
    "tenure_group_1 - 12": [st.selectbox("Tenure 1-12", [0,1])],
    "tenure_group_13 - 24": [st.selectbox("Tenure 13-24", [0,1])],
    "tenure_group_25 - 36": [st.selectbox("Tenure 25-36", [0,1])],
    "tenure_group_37 - 48": [st.selectbox("Tenure 37-48", [0,1])],
    "tenure_group_49 - 60": [st.selectbox("Tenure 49-60", [0,1])],
    "tenure_group_61 - 72": [st.selectbox("Tenure 61-72", [0,1])]
}

# Convert input to DataFrame
input_df = pd.DataFrame(inputs)

# Fill missing columns and reorder
for col in model_columns:
    if col not in input_df.columns:
        input_df[col] = 0
input_df = input_df[model_columns]

# Predict
if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[:,1][0]
    
    st.success(f"Prediction: {'Churn' if prediction == 1 else 'Not Churn'}")
    st.info(f"Confidence: {probability*100:.2f}%")
