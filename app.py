from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from typing import List

app = FastAPI(title="Customer Churn Prediction API", version="1.0.0")

# Load models at startup
try:
    model = joblib.load('models/best_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    label_encoders = joblib.load('models/label_encoders.pkl')
    
    with open('models/feature_columns.txt', 'r') as f:
        feature_columns = [line.strip() for line in f.readlines()]
except FileNotFoundError:
    print("Models not found. Please run main.py first.")
    model = None

class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

@app.get("/")
async def root():
    return {"message": "Customer Churn Prediction API is running!"}

@app.get("/health")
async def health_check():
    if model is None:
        return {"status": "unhealthy", "message": "Models not loaded"}
    return {"status": "healthy", "message": "Models loaded successfully"}

@app.post("/predict")
async def predict_churn(customer: CustomerData):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Convert to DataFrame
        customer_dict = customer.dict()
        df = pd.DataFrame([customer_dict])
        
        # Encode categorical variables
        categorical_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 
                          'MultipleLines', 'InternetService', 'OnlineSecurity',
                          'OnlineBackup', 'DeviceProtection', 'TechSupport',
                          'StreamingTV', 'StreamingMovies', 'Contract',
                          'PaperlessBilling', 'PaymentMethod']
        
        for col in categorical_cols:
            if col in label_encoders:
                try:
                    df[col] = label_encoders[col].transform(df[col])
                except ValueError:
                    df[col] = 0
        
        # Scale features
        X_scaled = scaler.transform(df)
        
        # Make prediction
        prediction = model.predict(X_scaled)[0]
        probability = model.predict_proba(X_scaled)[0, 1]
        
        # Determine risk level
        if probability < 0.3:
            risk_level = "Low"
        elif probability < 0.7:
            risk_level = "Medium"
        else:
            risk_level = "High"
        
        return {
            "churn_prediction": int(prediction),
            "churn_probability": float(probability),
            "risk_level": risk_level
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
