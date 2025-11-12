"""
Data Preprocessing Module for Patient Readmission Prediction
Handles data cleaning, feature engineering, and preparation for model training.
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

class DataPreprocessor:
    """
    A class to preprocess patient data for readmission prediction.
    
    Steps include:
    - Handling missing values
    - Feature engineering
    - Encoding categorical variables
    - Scaling numerical features
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.imputer_num = SimpleImputer(strategy='median')
        self.imputer_cat = SimpleImputer(strategy='most_frequent')
        self.encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        
    def load_data(self, filepath):
        """
        Load patient data from CSV file.
        
        Args:
            filepath (str): Path to the CSV file
            
        Returns:
            pandas.DataFrame: Loaded dataset
        """
        try:
            df = pd.read_csv(filepath)
            print(f"Data loaded successfully with {df.shape[0]} rows and {df.shape[1]} columns")
            return df
        except FileNotFoundError:
            print(f"Error: File {filepath} not found")
            return None
    
    def handle_missing_values(self, df):
        """
        Handle missing values in the dataset.
        
        Args:
            df (pandas.DataFrame): Input dataframe
            
        Returns:
            pandas.DataFrame: Dataframe with handled missing values
        """
        print("Handling missing values...")
        
        # Separate numerical and categorical columns
        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        # Impute numerical columns with median
        if len(numerical_cols) > 0:
            df[numerical_cols] = self.imputer_num.fit_transform(df[numerical_cols])
        
        # Impute categorical columns with mode
        if len(categorical_cols) > 0:
            df[categorical_cols] = self.imputer_cat.fit_transform(df[categorical_cols])
            
        return df
    
    def engineer_features(self, df):
        """
        Create new features from existing data.
        
        Args:
            df (pandas.DataFrame): Input dataframe
            
        Returns:
            pandas.DataFrame: Dataframe with new features
        """
        print("Engineering features...")
        
        # Example feature: Create polypharmacy flag (5+ medications)
        if 'number_of_medications' in df.columns:
            df['polypharmacy'] = (df['number_of_medications'] >= 5).astype(int)
        
        # Example feature: Age groups
        if 'age' in df.columns:
            df['age_group'] = pd.cut(df['age'], 
                                   bins=[0, 40, 60, 80, 100],
                                   labels=['Young', 'Middle', 'Senior', 'Elderly'])
        
        return df
    
    def preprocess_pipeline(self, df, target_column='readmitted_30_days'):
        """
        Complete preprocessing pipeline.
        
        Args:
            df (pandas.DataFrame): Raw input data
            target_column (str): Name of the target variable
            
        Returns:
            tuple: (X_processed, y_processed, feature_names)
        """
        print("Starting preprocessing pipeline...")
        
        # Create a copy to avoid modifying original data
        data = df.copy()
        
        # Step 1: Handle missing values
        data = self.handle_missing_values(data)
        
        # Step 2: Feature engineering
        data = self.engineer_features(data)
        
        # Step 3: Separate features and target
        X = data.drop(columns=[target_column])
        y = data[target_column]
        
        # Step 4: Preprocess numerical features
        numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
        if len(numerical_cols) > 0:
            X[numerical_cols] = self.scaler.fit_transform(X[numerical_cols])
        
        # Step 5: Preprocess categorical features
        categorical_cols = X.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            encoded_cats = self.encoder.fit_transform(X[categorical_cols])
            encoded_df = pd.DataFrame(encoded_cats, 
                                    columns=self.encoder.get_feature_names_out(categorical_cols))
            
            # Combine numerical and encoded categorical features
            X = pd.concat([X[numerical_cols], encoded_df], axis=1)
        
        print(f"Preprocessing complete. Final feature shape: {X.shape}")
        return X, y, X.columns.tolist()

# Example usage
if __name__ == "__main__":
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Load sample data (replace with actual data path)
    # df = preprocessor.load_data('data/patient_data.csv')
    
    # Run preprocessing pipeline
    # X_processed, y, feature_names = preprocessor.preprocess_pipeline(df)
    
    print("Data preprocessing module ready for use.")
