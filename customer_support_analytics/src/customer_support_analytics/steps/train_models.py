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
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    mean_absolute_error, r2_score, f1_score
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from pathlib import Path

# Import from SupportIQ package
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from supportiq_web import path_manager, logger, log_pipeline_start, log_pipeline_complete


def load_training_data() -> pd.DataFrame:
    """Load data from CSV file for training."""
    data_path = path_manager.get_processed_data_path('tickets_cleaned.csv')
    logger.info(f"Loading training data from: {data_path}")
    
    if not data_path.exists():
        logger.error(f"Training data not found: {data_path}")
        raise FileNotFoundError(f"Training data not found: {data_path}")
    
    df = pd.read_csv(data_path)
    initial_count = len(df)
    
    # Drop rows with missing target values
    df = df.dropna(subset=['priority', 'csat_score'])
    dropped_count = initial_count - len(df)
    
    logger.log_data_loaded('tickets_cleaned.csv', len(df), len(df.columns))
    if dropped_count > 0:
        logger.info(f"Removed {dropped_count} rows with missing target values")
    
    return df


def train_priority_model(df: pd.DataFrame):
    """Train ticket priority prediction model."""
    logger.info("Training Priority Prediction Model...")
    
    X = df[['channel', 'issue_type', 'sub_issue', 'language', 'is_weekend', 'is_business_hours']]
    y = df['priority']
    
    # Encode target
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    logger.debug(f"Train: {len(X_train)}, Test: {len(X_test)}")
    
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
    
    full_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    # Train
    logger.debug("Fitting priority model...")
    full_pipeline.fit(X_train, y_train)
    
    # Evaluate
    y_pred = full_pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    logger.info(f"Priority Model - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
    logger.debug(f"Classification Report:\n{classification_report(y_test, y_pred, target_names=le.classes_)}")
    
    # Save model
    model_data = {
        'model': full_pipeline,
        'label_encoder': le,
        'accuracy': accuracy,
        'f1_score': f1
    }
    
    model_path = path_manager.get_model_path('priority_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    logger.info(f"Priority model saved to: {model_path}")
    
    return model_data


def train_routing_model(df: pd.DataFrame):
    """Train ticket routing (issue type classification) model."""
    logger.info("\nTraining Routing Model...")
    
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
    
    logger.info(f"Routing Model - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
    logger.debug(f"Classification Report:\n{classification_report(y_test, y_pred, target_names=le.classes_)}")
    
    model_data = {
        'model': full_pipeline,
        'label_encoder': le,
        'accuracy': accuracy,
        'f1_score': f1
    }
    
    model_path = path_manager.get_model_path('routing_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    logger.info(f"Routing model saved to: {model_path}")
    
    return model_data


def train_csat_model(df: pd.DataFrame):
    """Train CSAT score prediction model."""
    logger.info("\nTraining CSAT Prediction Model...")
    
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
    
    logger.info(f"CSAT Model - MAE: {mae:.4f}, R2: {r2:.4f}")
    
    model_data = {
        'model': full_pipeline,
        'mae': mae,
        'r2_score': r2
    }
    
    model_path = path_manager.get_model_path('csat_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    logger.info(f"CSAT model saved to: {model_path}")
    
    return model_data


def train_sla_model(df: pd.DataFrame):
    """Train SLA breach prediction model."""
    logger.info("\nTraining SLA Breach Prediction Model...")
    
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
    
    logger.info(f"SLA Model - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
    logger.debug(f"Classification Report:\n{classification_report(y_test, y_pred)}")
    
    model_data = {
        'model': full_pipeline,
        'accuracy': accuracy,
        'f1_score': f1
    }
    
    model_path = path_manager.get_model_path('sla_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    logger.info(f"SLA model saved to: {model_path}")
    
    return model_data


def main():
    import time
    start_time = time.time()
    
    log_pipeline_start("ML Model Training Pipeline")
    
    # Ensure models directory exists
    path_manager._create_directories()
    logger.info(f"Models directory: {path_manager.MODELS_DIR}")
    
    # Load data
    df = load_training_data()
    
    # Train all models
    try:
        logger.log_step(1, 4, "Training Priority Prediction Model")
        priority_model = train_priority_model(df)
        
        logger.log_step(2, 4, "Training Routing Model")
        routing_model = train_routing_model(df)
        
        logger.log_step(3, 4, "Training CSAT Prediction Model")
        csat_model = train_csat_model(df)
        
        logger.log_step(4, 4, "Training SLA Breach Prediction Model")
        sla_model = train_sla_model(df)
        
        # Log summary
        logger.info("\n=== Model Training Summary ===")
        logger.info(f"Priority Model: Accuracy={priority_model['accuracy']:.4f}, F1={priority_model['f1_score']:.4f}")
        logger.info(f"Routing Model: Accuracy={routing_model['accuracy']:.4f}, F1={routing_model['f1_score']:.4f}")
        logger.info(f"CSAT Model: MAE={csat_model['mae']:.4f}, R2={csat_model['r2_score']:.4f}")
        logger.info(f"SLA Model: Accuracy={sla_model['accuracy']:.4f}, F1={sla_model['f1_score']:.4f}")
        
        duration = time.time() - start_time
        log_pipeline_complete("ML Model Training Pipeline", duration)
        
    except Exception as e:
        logger.error("Model training failed", exception=e)
        raise


if __name__ == "__main__":
    main()