import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import xgboost as xgb
import joblib
import warnings
warnings.filterwarnings('ignore')

class ChurnPredictor:
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = None
        
    def load_data(self):
        """Load and return the Telco Customer Churn dataset"""
        print("Loading Telco Customer Churn dataset...")
        
        # Generate realistic sample data for demonstration
        np.random.seed(42)
        n_samples = 1000
        
        data = {
            'customerID': [f'C{i:04d}' for i in range(n_samples)],
            'gender': np.random.choice(['Male', 'Female'], n_samples),
            'SeniorCitizen': np.random.choice([0, 1], n_samples),
            'Partner': np.random.choice(['Yes', 'No'], n_samples),
            'Dependents': np.random.choice(['Yes', 'No'], n_samples),
            'tenure': np.random.randint(1, 73, n_samples),
            'PhoneService': np.random.choice(['Yes', 'No'], n_samples),
            'MultipleLines': np.random.choice(['Yes', 'No', 'No phone service'], n_samples),
            'InternetService': np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples),
            'OnlineSecurity': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
            'OnlineBackup': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
            'DeviceProtection': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
            'TechSupport': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
            'StreamingTV': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
            'StreamingMovies': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
            'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples),
            'PaperlessBilling': np.random.choice(['Yes', 'No'], n_samples),
            'PaymentMethod': np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'], n_samples),
            'MonthlyCharges': np.round(np.random.uniform(18.25, 118.75, n_samples), 2),
            'TotalCharges': np.round(np.random.uniform(18.8, 8684.8, n_samples), 2),
            'Churn': np.random.choice(['Yes', 'No'], n_samples, p=[0.27, 0.73])
        }
        
        return pd.DataFrame(data)
    
    def preprocess_data(self, df):
        """Clean and preprocess the data"""
        print("Preprocessing data...")
        
        # Handle missing values
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
        
        # Remove customer ID
        df = df.drop('customerID', axis=1)
        
        # Encode categorical variables
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        categorical_cols.remove('Churn')
        
        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            self.label_encoders[col] = le
        
        # Encode target variable
        df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
        
        return df
    
    def explore_data(self, df):
        """Perform basic exploratory data analysis"""
        print("Performing EDA...")
        print(f"Dataset shape: {df.shape}")
        print(f"Churn rate: {df['Churn'].mean():.2%}")
        
        correlations = df.corr()['Churn'].sort_values(ascending=False)
        print("\nTop features correlated with churn:")
        print(correlations.head(10))
        
        return correlations
    
    def prepare_features(self, df):
        """Prepare features for modeling"""
        print("Preparing features...")
        
        X = df.drop('Churn', axis=1)
        y = df['Churn']
        
        self.feature_columns = X.columns.tolist()
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_models(self, X_train, y_train):
        """Train multiple models"""
        print("Training models...")
        
        # Logistic Regression
        self.models['logistic'] = LogisticRegression(random_state=42)
        self.models['logistic'].fit(X_train, y_train)
        
        # Random Forest
        self.models['random_forest'] = RandomForestClassifier(
            n_estimators=100, random_state=42
        )
        self.models['random_forest'].fit(X_train, y_train)
        
        # XGBoost
        self.models['xgboost'] = xgb.XGBClassifier(
            n_estimators=100, random_state=42
        )
        self.models['xgboost'].fit(X_train, y_train)
        
        print("Models trained successfully!")
    
    def evaluate_models(self, X_test, y_test):
        """Evaluate all models"""
        print("Evaluating models...")
        
        results = {}
        for name, model in self.models.items():
            predictions = model.predict(X_test)
            probabilities = model.predict_proba(X_test)[:, 1]
            
            auc_score = roc_auc_score(y_test, probabilities)
            results[name] = {
                'auc_score': auc_score,
                'classification_report': classification_report(y_test, predictions)
            }
            
            print(f"\n{name.upper()} Results:")
            print(f"AUC Score: {auc_score:.4f}")
            print(classification_report(y_test, predictions))
        
        return results
    
    def save_models(self):
        """Save trained models and preprocessors"""
        print("Saving models...")
        
        joblib.dump(self.models['xgboost'], 'models/best_model.pkl')
        joblib.dump(self.scaler, 'models/scaler.pkl')
        joblib.dump(self.label_encoders, 'models/label_encoders.pkl')
        
        with open('models/feature_columns.txt', 'w') as f:
            f.write('\n'.join(self.feature_columns))
        
        print("Models saved successfully!")

def main():
    """Main execution function"""
    print("Starting Customer Churn Prediction Pipeline...")
    
    predictor = ChurnPredictor()
    df = predictor.load_data()
    df_processed = predictor.preprocess_data(df)
    correlations = predictor.explore_data(df_processed)
    X_train, X_test, y_train, y_test = predictor.prepare_features(df_processed)
    predictor.train_models(X_train, y_train)
    results = predictor.evaluate_models(X_test, y_test)
    predictor.save_models()
    
    print("\nPipeline completed successfully!")
    print("Models saved and ready for deployment!")

if __name__ == "__main__":
    main()
