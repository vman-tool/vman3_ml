# vman3_ml/train.py
from ccva_ml.processing import DataPreprocessor
from ccva_ml.training import ModelTrainer
import argparse
import pandas as pd
import numpy as np
import sys
import os

# Add the parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    parser = argparse.ArgumentParser(description='CCVA Model Training')
    parser.add_argument("--input", required=True, help="Path to training CSV")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--min_vc", type=float, default=130, help="Minimum value counts for the target value")
    parser.add_argument("--na_threshold", type=float, default=0.7, 
                       help="Confidence threshold for dropping NA in individual datasets")
    
    args = parser.parse_args()
    
    # Load and preprocess data
    preprocessor = DataPreprocessor(verbose=args.verbose, min_vc=args.min_vc, na_threshold = args.na_threshold)
    df = preprocessor.load_data(args.input)
    df = preprocessor._preprocess_data(df)
    
    X, y = preprocessor._prepare_training_data(df)
    
    # Ensure y is 1D array
    if isinstance(y, pd.DataFrame):
        y = y.squeeze()  # Convert single-column DataFrame to Series
    y = np.array(y).ravel()  # Ensure 1D numpy array

    # Train models
    trainer = ModelTrainer(verbose=args.verbose)
    trainer.train(X, y)
    trainer.save_model(preprocessor=preprocessor)


    # cross validation
    from sklearn.model_selection import train_test_split
    X_encoded, _ = preprocessor._encode_features(X)
    X_scaled, _ = preprocessor._scale_features(X_encoded)
    y_encoded, _ = preprocessor._encode_target(y)
    X_train, X_test, y_train, y_test = train_test_split(
        #np.asarray(X_scaled), np.asarray(y_encoded), test_size=0.2, random_state=42
        np.asarray(X_scaled), np.asarray(y_encoded),test_size=0.2, random_state=42, stratify=y_encoded
    )
    trainer.evaluate_with_cross_validation(X_test, y_test, cv=5, save_path='cv_results.json')


    
    print(f"Training completed.")

if __name__ == "__main__":
    main()