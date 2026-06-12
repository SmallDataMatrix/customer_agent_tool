"""
Machine Learning Model Training for SupportIQ

This script trains machine learning models for:
1. Ticket Priority Prediction
2. Ticket Routing (Issue Type Classification)
3. CSAT Score Prediction
4. SLA Breach Prediction

Models are saved to the models/ directory.
"""

import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    mean_absolute_error, r2_score, f1_score
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

def load_training_data():
    """Load data from CSV file for training."""
    df = pd.read_csv('data/processed/tickets_cleaned.csv')
    df = df.dropna(subset=['priority', 'csat_score'])
    return df

def train_priority_model(df):
    """Train ticket priority prediction model."""
    print("Training Priority Prediction Model...")
    
    X = df[['channel', 'issue_type', 'sub_issue', 'language', 'is_weekend', 'is_business_hours']]
    y = df['priority']
    
    # Encode target
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    
    # Preprocessing pipeline
    categorical_features = ['channel', 'issue_type', 'sub_issue', 'language']
    numeric_features = ['is_weekend', 'is_business_hours']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ]), categorical_features),
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), numeric_features)
        ])
    
    # One-hot encode categorical features after column transformer
    from sklearn.preprocessing import OneHotEncoder
    full_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    # Train
    full_pipeline.fit(X_train, y_train)
    
    # Evaluate
    y_pred = full_pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"Priority Model - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    
    # Save model
    model_data = {
        'model': full_pipeline,
        'label_encoder': le,
        'accuracy': accuracy,
        'f1_score': f1
    }
    
    with open('models/priority_model.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    
    return model_data

def train_routing_model(df):
    """Train ticket routing (issue type classification) model."""
    print("\nTraining Routing Model...")
    
    X = df[['channel', 'priority', 'language', 'is_weekend', 'is_business_hours', 
            'first_response_time_min', 'handling_time_min']]
    y = df['issue_type']
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    
    categorical_features = ['channel', 'priority', 'language']
    numeric_features = ['is_weekend', 'is_business_hours', 'first_response_time_min', 'handling_time_min']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ]), categorical_features),
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), numeric_features)
        ])
    
    full_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    full_pipeline.fit(X_train, y_train)
    y_pred = full_pipeline.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"Routing Model - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    
    model_data = {
        'model': full_pipeline,
        'label_encoder': le,
        'accuracy': accuracy,
        'f1_score': f1
    }
    
    with open('models/routing_model.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    
    return model_data

def train_csat_model(df):
    """Train CSAT score prediction model."""
    print("\nTraining CSAT Prediction Model...")
    
    X = df[['channel', 'priority', 'issue_type', 'is_weekend', 'is_business_hours',
            'first_response_time_min', 'resolution_time_min', 'handling_time_min']]
    y = df['csat_score']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    categorical_features = ['channel', 'priority', 'issue_type']
    numeric_features = ['is_weekend', 'is_business_hours', 'first_response_time_min', 
                        'resolution_time_min', 'handling_time_min']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ]), categorical_features),
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), numeric_features)
        ])
    
    full_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    
    full_pipeline.fit(X_train, y_train)
    y_pred = full_pipeline.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"CSAT Model - MAE: {mae:.4f}, R2: {r2:.4f}")
    
    model_data = {
        'model': full_pipeline,
        'mae': mae,
        'r2_score': r2
    }
    
    with open('models/csat_model.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    
    return model_data

def train_sla_model(df):
    """Train SLA breach prediction model."""
    print("\nTraining SLA Breach Prediction Model...")
    
    X = df[['channel', 'priority', 'issue_type', 'is_weekend', 'is_business_hours',
            'first_response_time_min', 'handling_time_min', 'wait_time_min']]
    y = df['sla_breach'].astype(int)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    categorical_features = ['channel', 'priority', 'issue_type']
    numeric_features = ['is_weekend', 'is_business_hours', 'first_response_time_min', 
                        'handling_time_min', 'wait_time_min']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ]), categorical_features),
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), numeric_features)
        ])
    
    full_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
        ('classifier', RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42))
    ])
    
    full_pipeline.fit(X_train, y_train)
    y_pred = full_pipeline.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"SLA Model - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
    print(classification_report(y_test, y_pred))
    
    model_data = {
        'model': full_pipeline,
        'accuracy': accuracy,
        'f1_score': f1
    }
    
    with open('models/sla_model.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    
    return model_data

def main():
    print("=" * 60)
    print("ML Model Training Pipeline")
    print("=" * 60)
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Load data
    df = load_training_data()
    print(f"Loaded {len(df)} training records")
    
    # Train all models
    try:
        train_priority_model(df)
        train_routing_model(df)
        train_csat_model(df)
        train_sla_model(df)
        
        print("\n" + "=" * 60)
        print("Model training completed successfully!")
        print("Models saved to: models/")
        print("=" * 60)
        
    except Exception as e:
        print(f"Model training failed: {e}")
        raise

if __name__ == "__main__":
    main()
