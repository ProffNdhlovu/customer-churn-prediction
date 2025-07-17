import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

st.set_page_config(page_title="Customer Churn Prediction", page_icon="📊", layout="wide")

@st.cache_data
def load_models():
    try:
        model = joblib.load('models/best_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        label_encoders = joblib.load('models/label_encoders.pkl')
        return model, scaler, label_encoders
    except FileNotFoundError:
        st.error("Models not found. Please run main.py first.")
        return None, None, None

def main():
    st.title("🔮 Customer Churn Prediction Dashboard")
    st.markdown("---")
    
    model, scaler, label_encoders = load_models()
    
    if model is None:
        return
    
    st.sidebar.title("Customer Information")
    
    # Input fields
    gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    senior_citizen = st.sidebar.selectbox("Senior Citizen", [0, 1])
    partner = st.sidebar.selectbox("Partner", ["Yes", "No"])
    dependents = st.sidebar.selectbox("Dependents", ["Yes", "No"])
    tenure = st.sidebar.slider("Tenure (months)", 1, 72, 12)
    phone_service = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.sidebar.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
    internet_service = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security = st.sidebar.selectbox("Online Security", ["Yes", "No", "No internet service"])
    online_backup = st.sidebar.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    device_protection = st.sidebar.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    tech_support = st.sidebar.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    streaming_tv = st.sidebar.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    streaming_movies = st.sidebar.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
    contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])
    payment_method = st.sidebar.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer", "Credit card"])
    monthly_charges = st.sidebar.number_input("Monthly Charges", min_value=0.0, max_value=200.0, value=50.0)
    total_charges = st.sidebar.number_input("Total Charges", min_value=0.0, max_value=10000.0, value=1000.0)
    
    if st.sidebar.button("Predict Churn", type="primary"):
        # Create input data
        input_data = pd.DataFrame({
            'gender': [gender],
            'SeniorCitizen': [senior_citizen],
            'Partner': [partner],
            'Dependents': [dependents],
            'tenure': [tenure],
            'PhoneService': [phone_service],
            'MultipleLines': [multiple_lines],
            'InternetService': [internet_service],
            'OnlineSecurity': [online_security],
            'OnlineBackup': [online_backup],
            'DeviceProtection': [device_protection],
            'TechSupport': [tech_support],
            'StreamingTV': [streaming_tv],
            'StreamingMovies': [streaming_movies],
            'Contract': [contract],
            'PaperlessBilling': [paperless_billing],
            'PaymentMethod': [payment_method],
            'MonthlyCharges': [monthly_charges],
            'TotalCharges': [total_charges]
        })
        
        # Preprocess and predict
        categorical_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 
                           'MultipleLines', 'InternetService', 'OnlineSecurity',
                           'OnlineBackup', 'DeviceProtection', 'TechSupport',
                           'StreamingTV', 'StreamingMovies', 'Contract',
                           'PaperlessBilling', 'PaymentMethod']
        
        for col in categorical_cols:
            if col in label_encoders:
                try:
                    input_data[col] = label_encoders[col].transform(input_data[col])
                except ValueError:
                    input_data[col] = 0
        
        scaled_data = scaler.transform(input_data)
        prediction = model.predict(scaled_data)[0]
        probability = model.predict_proba(scaled_data)[0, 1]
        
        # Display results
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Churn Prediction", "Yes" if prediction == 1 else "No")
        
        with col2:
            st.metric("Churn Probability", f"{probability:.2%}")
        
        with col3:
            risk_level = "High" if probability > 0.7 else "Medium" if probability > 0.3 else "Low"
            st.metric("Risk Level", risk_level)
        
        # Risk assessment
        if probability > 0.7:
            st.error(f"🚨 HIGH RISK: This customer has a high probability of churning ({probability:.1%})")
        elif probability > 0.3:
            st.warning(f"⚠️ MEDIUM RISK: This customer has a moderate probability of churning ({probability:.1%})")
        else:
            st.success(f"✅ LOW RISK: This customer has a low probability of churning ({probability:.1%})")

if __name__ == "__main__":
    main()
