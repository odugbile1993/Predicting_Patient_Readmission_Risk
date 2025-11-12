"""
Inference API Module for Patient Readmission Prediction
Provides a Flask API for model inference in production.
"""

from flask import Flask, request, jsonify
import pandas as pd
import joblib
import numpy as np
import os

app = Flask(__name__)

# Global variables for model and preprocessor
model = None
preprocessor = None

def load_artifacts():
    """
    Load the trained model and preprocessor artifacts.
    In production, these would be loaded from cloud storage or a model registry.
    """
    global model, preprocessor
    
    try:
        # Load model (in real scenario, this would be from a secure location)
        model = joblib.load('trained_readmission_model.pkl')
        
        # Initialize preprocessor (in real scenario, this would be saved and loaded)
        from data_preprocessing import DataPreprocessor
        preprocessor = DataPreprocessor()
        
        print("Model and preprocessor loaded successfully")
        return True
    except Exception as e:
        print(f"Error loading artifacts: {e}")
        return False

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for monitoring."""
    return jsonify({'status': 'healthy', 'model_loaded': model is not None})

@app.route('/predict', methods=['POST'])
def predict_readmission_risk():
    """
    Predict readmission risk for a patient.
    
    Expected JSON input:
    {
        "patient_data": {
            "age": 65,
            "number_of_medications": 7,
            "length_of_stay": 5,
            "previous_admissions": 2,
            ...
        }
    }
    
    Returns:
        JSON response with prediction and risk score
    """
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        # Get JSON data from request
        data = request.get_json()
        
        if 'patient_data' not in data:
            return jsonify({'error': 'patient_data field required'}), 400
        
        # Convert to DataFrame for preprocessing
        patient_df = pd.DataFrame([data['patient_data']])
        
        # Preprocess the data (in real scenario, use saved preprocessor)
        # For demonstration, we'll use simple preprocessing
        processed_data = preprocess_patient_data(patient_df)
        
        # Make prediction
        prediction = model.predict(processed_data)[0]
        prediction_proba = model.predict_proba(processed_data)[0]
        
        # Prepare response
        response = {
            'prediction': int(prediction),
            'risk_score': float(prediction_proba[1]),  # Probability of readmission
            'risk_category': 'High' if prediction == 1 else 'Low',
            'confidence': max(prediction_proba)
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

def preprocess_patient_data(patient_df):
    """
    Preprocess patient data for prediction.
    In a real scenario, this would mirror the training preprocessing.
    
    Args:
        patient_df (pandas.DataFrame): Raw patient data
        
    Returns:
        numpy.ndarray: Preprocessed data ready for model
    """
    # This is a simplified version - in production, use the exact same preprocessing
    # as during training, potentially using a saved preprocessor
    
    # Example preprocessing steps
    processed_df = patient_df.copy()
    
    # Handle numerical features - scale them
    numerical_cols = processed_df.select_dtypes(include=[np.number]).columns
    if len(numerical_cols) > 0:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        processed_df[numerical_cols] = scaler.fit_transform(processed_df[numerical_cols])
    
    # Ensure feature alignment with training data
    # In production, you would have the expected feature order from training
    return processed_df.values

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """
    Batch prediction endpoint for multiple patients.
    
    Expected JSON input:
    {
        "patients": [
            {"age": 65, "medications": 7, ...},
            {"age": 45, "medications": 3, ...}
        ]
    }
    """
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        data = request.get_json()
        
        if 'patients' not in data:
            return jsonify({'error': 'patients field required'}), 400
        
        # Convert to DataFrame
        patients_df = pd.DataFrame(data['patients'])
        
        # Preprocess (simplified)
        processed_data = preprocess_patient_data(patients_df)
        
        # Batch prediction
        predictions = model.predict(processed_data)
        predictions_proba = model.predict_proba(processed_data)
        
        # Prepare batch response
        results = []
        for i, (pred, proba) in enumerate(zip(predictions, predictions_proba)):
            results.append({
                'patient_id': i,
                'prediction': int(pred),
                'risk_score': float(proba[1]),
                'risk_category': 'High' if pred == 1 else 'Low'
            })
        
        return jsonify({'predictions': results})
    
    except Exception as e:
        return jsonify({'error': f'Batch prediction failed: {str(e)}'}), 500

if __name__ == '__main__':
    # Load model artifacts when starting the app
    print("Starting Readmission Prediction API...")
    
    if load_artifacts():
        # Run Flask app
        # In production, use a proper WSGI server like Gunicorn
        app.run(host='0.0.0.0', port=5000, debug=True)
    else:
        print("Failed to load model artifacts. API cannot start.")
