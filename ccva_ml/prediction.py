import joblib
import pandas as pd
import numpy as np

class CCVAPredictor:
    def __init__(self, model_path='models/ccva_model.pkl'):
        """Load trained model and preprocessing objects"""
        artifacts = joblib.load(model_path)
        self.model = artifacts['model']
        self.scaler = artifacts['scaler']
        self.label_encoder = artifacts['label_encoder']
        self.feature_encoders = artifacts['feature_encoders']
        self.original_classes = artifacts['original_classes']
        self.ood_threshold = artifacts.get('ood_threshold')

        # Validate threshold
        if self.ood_threshold is not None:
            if not (0 < self.ood_threshold < 1):
                print(f"WARNING: Invalid OOD threshold {self.ood_threshold}. Resetting to None")
                self.ood_threshold = None

        # Validation
        self.original_classes = set(artifacts['original_classes'])  # Ensure it's a set
        self.label_classes = set(self.label_encoder.classes_)

        if self.original_classes != self.label_classes:
            print("WARNING: Original classes don't match label encoder classes")
            print(f"Original: {self.original_classes}")
            print(f"Encoder: {self.label_classes}")
    
    def predict(self, new_data):
        """Make predictions with OOD detection"""
        try:
            processed = self._preprocess_data(new_data.copy())
            encoded = self._encode_features(processed)

            if encoded.shape[1] != self.scaler.n_features_in_:
                raise ValueError(
                    f"Input has {encoded.shape[1]} features, "
                    f"but scaler expects {self.scaler.n_features_in_}"
                )
            
            scaled = self.scaler.transform(encoded)
            
            # Get predictions and confidence
            if hasattr(self.model, 'predict_proba'):
                probs = self.model.predict_proba(scaled)
                predictions = self.model.classes_[np.argmax(probs, axis=1)]
                confidence = np.max(probs, axis=1)
                ood_mask = confidence < self.ood_threshold if self.ood_threshold else np.zeros_like(confidence, dtype=bool)
            else:
                predictions = self.model.predict(scaled)
                ood_mask = np.zeros(len(predictions), dtype=bool)
            
            # Label OOD samples
            decoded = self.label_encoder.inverse_transform(predictions)
            decoded[ood_mask] = 'out_of_distribution'

            # Additional validation for unexpected classes
            valid_classes = set(self.original_classes) | {'out_of_distribution'}
            decoded[~np.isin(decoded, list(valid_classes))] = 'UNKNOWN'
            
            return decoded
        except Exception as e:
            print(f"Prediction error: {str(e)}")
            return np.array(['UNKNOWN'] * len(new_data))

    def _preprocess_data(self, df):
        """Validate features before prediction"""
        # Check for required columns
        required_cols = set(self.feature_encoders.keys())
        missing_cols = required_cols - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Standardize column names
        df.columns = df.columns.str.lower().str.replace(r'[\s/]', '_')

        available_cols = [col for col in df.columns if col in required_cols]
        return df[available_cols]
    
    
    def _validate_input_data(self, df):
        """Comprehensive data quality checks before prediction"""
        errors = []
        warnings = []
        
        # 1. Check column presence
        missing_cols = set(self.expected_columns) - set(df.columns)
        if missing_cols:
            errors.append(f"Missing columns: {missing_cols}")
        
        # 2. Check data types
        for col in self.feature_encoders.keys():
            if col in df.columns:
                if not pd.api.types.is_string_dtype(df[col]) and not pd.api.types.is_numeric_dtype(df[col]):
                    warnings.append(f"Column {col} has unexpected dtype: {df[col].dtype}")
        
        # 3. Check for empty/NA columns
        empty_cols = []
        for col in self.feature_encoders.keys():
            if col in df.columns:
                if df[col].isna().all() or df[col].eq('').all():
                    empty_cols.append(col)
        if empty_cols:
            warnings.append(f"Completely empty columns: {empty_cols}")
        
        # 4. Check for problematic values
        problematic_records = {}
        for col in self.feature_encoders.keys():
            if col in df.columns:
                # Check for non-string values in categorical columns
                if col in self.feature_encoders and not pd.api.types.is_string_dtype(df[col]):
                    problematic_records[col] = df[~df[col].astype(str).str.isalnum()].index.tolist()
        
        if problematic_records:
            warnings.append(f"Potential problematic values in columns: {problematic_records}")
        
        # Return or raise errors
        if errors:
            raise ValueError("\n".join(errors))
        
        if warnings and self.verbose:
            print("\nData Quality Warnings:")
            print("\n".join(warnings))
        
        return True