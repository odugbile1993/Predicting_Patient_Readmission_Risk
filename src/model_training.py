"""
Model Training Module for Patient Readmission Prediction
Handles model training, evaluation, and hyperparameter tuning.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score
import joblib

class ReadmissionPredictor:
    """
    A class to train and evaluate models for readmission prediction.
    
    Supports multiple algorithms and includes comprehensive evaluation.
    """
    
    def __init__(self, model_type='logistic_regression'):
        """
        Initialize the predictor with specified model type.
        
        Args:
            model_type (str): Type of model to use ('logistic_regression' or 'random_forest')
        """
        self.model_type = model_type
        self.model = None
        self.best_params_ = None
        
        if model_type == 'logistic_regression':
            self.model = LogisticRegression(random_state=42, max_iter=1000)
        elif model_type == 'random_forest':
            self.model = RandomForestClassifier(random_state=42)
        else:
            raise ValueError("Model type must be 'logistic_regression' or 'random_forest'")
    
    def split_data(self, X, y, test_size=0.2, val_size=0.2):
        """
        Split data into training, validation, and test sets.
        
        Args:
            X (pandas.DataFrame): Feature matrix
            y (pandas.Series): Target variable
            test_size (float): Proportion for test set
            val_size (float): Proportion for validation set from training data
            
        Returns:
            tuple: X_train, X_val, X_test, y_train, y_val, y_test
        """
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Second split: separate validation set from temporary set
        val_size_adj = val_size / (1 - test_size)  # Adjust validation size
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adj, random_state=42, stratify=y_temp
        )
        
        print(f"Data split completed:")
        print(f"  Training set: {X_train.shape[0]} samples")
        print(f"  Validation set: {X_val.shape[0]} samples")
        print(f"  Test set: {X_test.shape[0]} samples")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def tune_hyperparameters(self, X_train, y_train, X_val, y_val):
        """
        Perform hyperparameter tuning based on model type.
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data for evaluation
            
        Returns:
            dict: Best hyperparameters found
        """
        print(f"Tuning hyperparameters for {self.model_type}...")
        
        if self.model_type == 'logistic_regression':
            param_grid = {
                'C': [0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear']
            }
        elif self.model_type == 'random_forest':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 10, None],
                'min_samples_split': [2, 5, 10]
            }
        
        # Perform grid search with cross-validation
        grid_search = GridSearchCV(
            self.model, param_grid, cv=5, scoring='f1', n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        
        self.best_params_ = grid_search.best_params_
        self.model = grid_search.best_estimator_
        
        print(f"Best parameters: {self.best_params_}")
        
        # Evaluate on validation set
        val_score = f1_score(y_val, self.model.predict(X_val))
        print(f"Validation F1-score: {val_score:.4f}")
        
        return self.best_params_
    
    def train_model(self, X_train, y_train, tune_hyperparams=True, X_val=None, y_val=None):
        """
        Train the model with optional hyperparameter tuning.
        
        Args:
            X_train, y_train: Training data
            tune_hyperparams (bool): Whether to perform hyperparameter tuning
            X_val, y_val: Validation data (required if tune_hyperparams=True)
        """
        if tune_hyperparams:
            if X_val is None or y_val is None:
                raise ValueError("Validation data required for hyperparameter tuning")
            self.tune_hyperparameters(X_train, y_train, X_val, y_val)
        else:
            print(f"Training {self.model_type} without hyperparameter tuning...")
            self.model.fit(X_train, y_train)
    
    def evaluate_model(self, X_test, y_test):
        """
        Comprehensive model evaluation on test set.
        
        Args:
            X_test, y_test: Test data for evaluation
            
        Returns:
            dict: Dictionary of evaluation metrics
        """
        print("\n" + "="*50)
        print("MODEL EVALUATION RESULTS")
        print("="*50)
        
        # Generate predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': np.mean(y_pred == y_test),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        # Print detailed report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        print("\nKey Metrics:")
        for metric, value in metrics.items():
            print(f"  {metric.replace('_', ' ').title()}: {value:.4f}")
        
        return metrics
    
    def save_model(self, filepath):
        """
        Save the trained model to disk.
        
        Args:
            filepath (str): Path where to save the model
        """
        if self.model is not None:
            joblib.dump(self.model, filepath)
            print(f"Model saved to {filepath}")
        else:
            print("No trained model to save.")
    
    def load_model(self, filepath):
        """
        Load a trained model from disk.
        
        Args:
            filepath (str): Path from where to load the model
        """
        self.model = joblib.load(filepath)
        print(f"Model loaded from {filepath}")

# Example usage and demonstration
def demonstrate_training():
    """Demonstrate the model training workflow with sample data."""
    
    # Generate sample data for demonstration
    from sklearn.datasets import make_classification
    
    print("Generating sample patient data...")
    X, y = make_classification(
        n_samples=1000, n_features=10, n_informative=8, 
        n_redundant=2, n_clusters_per_class=1, random_state=42
    )
    
    # Convert to DataFrame for better demonstration
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    X_df = pd.DataFrame(X, columns=feature_names)
    
    # Initialize predictor
    print("\nInitializing Logistic Regression predictor...")
    predictor = ReadmissionPredictor(model_type='logistic_regression')
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = predictor.split_data(X_df, y)
    
    # Train model with hyperparameter tuning
    predictor.train_model(X_train, y_train, tune_hyperparams=True, X_val=X_val, y_val=y_val)
    
    # Evaluate model
    metrics = predictor.evaluate_model(X_test, y_test)
    
    # Save model
    predictor.save_model('trained_readmission_model.pkl')
    
    return predictor, metrics

if __name__ == "__main__":
    # Run demonstration
    predictor, metrics = demonstrate_training()
    
    print("\n" + "="*50)
    print("DEMONSTRATION COMPLETE")
    print("="*50)
