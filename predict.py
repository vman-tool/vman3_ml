# vman3_ml/predict.py
from ccva_ml.processing import DataPreprocessor
import joblib
import pandas as pd
import numpy as np
import sys
import os
import argparse

# Add the parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class CCVAPredictor:
    def __init__(self, model_path='models/ccva_model.pkl', verbose=False):
        """Load trained model and preprocessing objects"""
        artifacts = joblib.load(model_path)
        self.model = artifacts['model']
        self.scaler = artifacts['scaler']
        self.label_encoder = artifacts['label_encoder']
        self.feature_encoders = artifacts['feature_encoders']
        self.preprocessor = artifacts['preprocessor']
        self.original_classes = set(artifacts['original_classes'])
        self.ood_threshold = artifacts.get('ood_threshold')
        self.dk_threshold = artifacts.get('dk_threshold', 0.5)  # Default if not present
        self.verbose = verbose
        self.expected_columns = self.preprocessor.final_training_columns
        self.encoders = None
    
    def _validate_columns(self, df):
        """Check if all expected columns are present"""
        missing_columns = [col.lower() for col in self.expected_columns if col not in df.columns]
        extra_columns = [col.lower() for col in df.columns if col not in self.expected_columns]
        
        if missing_columns:
            print(f"WARNING: Missing {len(missing_columns)} columns that were used in training:")
            print(f"The missing columns are {missing_columns}")
        
        if extra_columns and self.verbose:
            print(f"Note: Found {len(extra_columns)} additional columns not used in training")
        
        return len(missing_columns) == 0
    
    def predict(self, cleaned_df):
        """Make predictions with comprehensive data cleaning"""
        try:
            cleaned_df = cleaned_df.copy()
            
            # 1. First handle missing values (same as original)
            for col in self.feature_encoders.keys():
                if col in cleaned_df.columns:
                    if pd.api.types.is_string_dtype(cleaned_df[col]):
                        cleaned_df[col] = cleaned_df[col].fillna('dk').replace({'':'dk'})
                    else:
                        cleaned_df[col] = cleaned_df[col].fillna(-999)

            # 2. Calculate DK ratios (same as original)
            dk_ratios = (cleaned_df=='dk').mean(axis=1)
            dk_ood_mask = dk_ratios > self.dk_threshold

            if self.verbose and dk_ood_mask.any():
                print(f"found {dk_ood_mask.sum()} records with > {self.dk_threshold:.0%} dk values")
                        
            # 6. Rest of your original processing
            # for col in cleaned_df.select_dtypes(include=['object']):
            #     cleaned_df[col] = cleaned_df[col].astype(str)
            
            # Encode and scale features
            preprocessor = DataPreprocessor(verbose=self.verbose)
            encoded, self.encoders  = preprocessor._encode_features(cleaned_df)
            encoded = encoded[self.expected_columns]

            # Check for duplicate columns
            #print("Duplicates in encoded.columns:", encoded.columns.duplicated().any())
            #print("Duplicates in scaler.feature_names_in_:", pd.Series(self.scaler.feature_names_in_).duplicated().any())

            if encoded.columns.duplicated().any():
                dupes_encoded = encoded.columns[encoded.columns.duplicated(keep=False)].unique()
                print(f"\nFound {len(dupes_encoded)} duplicate column names in encoded data:")
                for i, col in enumerate(dupes_encoded, 1):
                    print(f"{i}. {col} (appears {sum(encoded.columns == col)} times)")
                    
                # Show positions of duplicates
                print("\nDuplicate positions in encoded.columns:")
                dup_positions = {}
                for idx, col in enumerate(encoded.columns):
                    if col in dupes_encoded:
                        dup_positions.setdefault(col, []).append(idx)
                for col, positions in dup_positions.items():
                    print(f"'{col}': positions {positions}")

                # remove duplicates
                duplicates = encoded.columns.duplicated()
                encoded = encoded.loc[:, ~duplicates]

            array_to_scale = encoded.values.astype('float32')
            scaled = self.scaler.transform(array_to_scale)
        
            # 7. Prediction logic
            if hasattr(self.model, 'predict_proba'):
                probs = self.model.predict_proba(scaled)
                predictions = self.model.classes_[np.argmax(probs, axis=1)]
                confidence = np.max(probs, axis=1)
                ood_mask = confidence < self.ood_threshold if self.ood_threshold else np.zeros_like(confidence, dtype=bool)
                
                if self.ood_threshold is not None:
                    if not 0 <= self.ood_threshold <= 1:
                        print(f"WARNING: Unusual OOD threshold {self.ood_threshold}")
                    ood_mask = confidence < self.ood_threshold
                else:
                    ood_mask = np.zeros_like(confidence, dtype=bool)
                    
            elif hasattr(self.model, 'decision_function'):
                scores = self.model.decision_function(scaled)
                predictions = self.model.predict(scaled)
                
                if len(scores.shape) == 1:  # Binary
                    confidence = self._normalize_decision_scores(scores)
                else:  # Multiclass
                    confidence = self._normalize_decision_scores(scores.max(axis=1))
                
                if self.ood_threshold is not None:
                    ood_mask = confidence < self.ood_threshold
                else:
                    ood_mask = np.zeros_like(confidence, dtype=bool)
            else:
                predictions = self.model.predict(scaled)
                ood_mask = np.zeros(len(predictions), dtype=bool)
            
            final_ood_mask = ood_mask | dk_ood_mask
            decoded = self.label_encoder.inverse_transform(predictions)
            decoded[final_ood_mask] = 'out_of_distribution'
            decoded[~np.isin(decoded, list(self.original_classes))] = 'out_of_distribution'
            
            return decoded
            
        except Exception as e:
            print(f"Prediction error: {str(e)}")
            return np.array(['PREDICTION_ERROR'] * len(cleaned_df))
    
    def _generate_data_quality_report(self, df):
        """Generate a comprehensive report on data quality issues"""
        report = {
            "missing_columns": [],
            "empty_columns": [],
            "type_mismatches": [],
            "problematic_records": {},
            "dk_stats": {
                "total_dk": 0,
                "columns_with_dk": [],
                "records_with_dk": 0
            }
        }
        
        # Check columns
        report["missing_columns"] = list(set(self.expected_columns) - set(df.columns))
        
        # Check each column
        for col in self.expected_columns:
            if col in df.columns:
                # Check for empty columns - NEW FIX: Explicit boolean conversion
                try:
                    is_na = bool(df[col].isna().all())
                    is_empty_string = bool(df[col].eq('').all())
                    is_empty = is_na or is_empty_string
                except Exception as e:
                    if self.verbose:
                        print(f"Error checking empty status for {col}: {str(e)}")
                    is_empty = False
                
                if is_empty:
                    report["empty_columns"].append(col)
                
                # Check type mismatches
                if col in self.feature_encoders:
                    if not pd.api.types.is_string_dtype(df[col]):
                        report["type_mismatches"].append(col)
                
                # Find problematic records
                if col in self.feature_encoders:
                    try:
                        # Convert to string safely
                        str_col = df[col].astype(str)
                        # Check for problematic values
                        mask = ~str_col.str.match(r'^[\w\s-]+$')  # Simple pattern
                        if mask.any():
                            report["problematic_records"][col] = df[mask][col].unique().tolist()
                    except Exception as e:
                        if self.verbose:
                            print(f"Error checking problematic records for {col}: {str(e)}")
                        report["problematic_records"][col] = ["ERROR_CHECKING_VALUES"]
        
        return report

def main():
    parser = argparse.ArgumentParser(description='CCVA Model Prediction')
    parser.add_argument("--model", required=True, help="Path to trained model")
    parser.add_argument("--input", required=True, help="Path to prediction CSV")
    parser.add_argument("--output", help="Output file for predictions")
    parser.add_argument("--report", help="Run data quality report")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    
    # Initialize and load data
    preprocessor = DataPreprocessor(verbose=args.verbose)
    df = preprocessor.load_data(args.input)

    # process data
    df = preprocessor._preprocess_data(df)
    
    # # Handle missing values with dk
    for col in df.columns:
        if pd.api.types.is_string_dtype(df[col]):
            df[col] = df[col].fillna('dk').replace({'': 'dk'})
        else:
            df[col] = df[col].fillna(-999)  # Numeric placeholder

    
    # Make predictions
    predictor = CCVAPredictor(args.model, verbose=args.verbose)
    if args.report == "dqr":
        dqr = predictor._generate_data_quality_report(df)
        print(dqr)
        return
    
    # keep columns used in the training, and also ensure no dulicates
    model_cols = []
    seen = set()
    for col in predictor.expected_columns:
        if col in df.columns and col not in seen:
            seen.add(col)
            model_cols.append(col)

    topredict =  df.loc[:,model_cols].copy()
    predictions = predictor.predict(topredict)
    print(f"prediction shape: {np.array(predictions).shape}")

    # Save results
    output_file = args.output or "predictions.csv" 
    results = pd.DataFrame({
        'prediction': predictions,
        **{col: df[col] for col in df.columns}
    })
    results.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")

if __name__ == "__main__":
    main()