# vman3_ml/vman3_ml/training.py
from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from .processing import DataPreprocessor
import pandas as pd
import numpy as np
import joblib
import os

class ModelTrainer:
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.models = {
            'RandomForest': RandomForestClassifier(),
            # Add other models...
        }
        self.original_classes = None  # Initialize here
        self.ood_threshold = None
        self.dk_threshold = None
        self.best_model = None
        self.scaler = None
        self.label_encoder = None
        self.encoders = None
        self.classes_ = None

        
        
    def train(self, X, y, test_size=0.3, n_iter_search=10):
        """Train and evaluate models"""
        
        # Validate input shapes
        if len(X) != len(y):
            raise ValueError(
                f"Mismatched input shapes: X has {len(X)} samples, y has {len(y)}. "
                "They must have the same number of samples."
            )
    
        # Store original classes for OOD detection
        try:
            self.original_classes = set(pd.Series(y).unique())
        except Exception as e:
            raise ValueError(f"Could not determine unique classes from target variable: {str(e)}")
    

        # Encode features and target
        preprocessor = DataPreprocessor(verbose=self.verbose)

        X_encoded, self.encoders = preprocessor._encode_features(X)
        #print("Duplicates in  X_encoded:", X_encoded.columns.duplicated().any())
        X_scaled, self.scaler = preprocessor._scale_features(X_encoded)
        
        # check and remove duplicates
        if isinstance(X_scaled, pd.DataFrame):
            duplicates = X_scaled.columns.duplicated()
            if duplicates.any():
                print(f"Warning: Found {duplicates.sum()} duplicate columns after scaling:")
                print(X_scaled.columns[duplicates].tolist())
                
                # Remove duplicates while preserving order
                X_scaled = X_scaled.loc[:, ~duplicates]

        y_encoded, self.label_encoder = preprocessor._encode_target(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_encoded, test_size=test_size, random_state=42
        )

        # Hyperparameter search space
        param_dist = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        # Train and tune models
        best_score = 0
        for name, model in self.models.items():
            if self.verbose:
                print(f"\nTraining {name} with {n_iter_search} iterations...")
            
            # Randomized parameter search
            search = RandomizedSearchCV(
                model,
                param_distributions=param_dist,
                n_iter=n_iter_search,
                cv=5,
                verbose=self.verbose
            )
            search.fit(X_train, y_train)
            
            # Get best model from search
            model = search.best_estimator_
            score = model.score(X_test, y_test)
            
            if score > best_score:
                best_score = score
                self.best_model = model
                if self.verbose:
                    print(f"New best model: {name} with accuracy {score:.2%}")
                    print("Best parameters:", search.best_params_)
            
            # Set OOD and dk threshold based on available methods
            if hasattr(model, 'predict_proba'):
                probs = model.predict_proba(X_test)
                self.ood_threshold = np.percentile(probs.max(axis=1), 5)  # 5th percentile of confidence
                if self.verbose:
                    print(f"Set OOD threshold (probability): {self.ood_threshold:.2f}")

                # calculate 'dk' - dont know ratio
                dk_ratios = (X == 'dk').mean(axis=1)
                self.dk_threshold = np.percentile(dk_ratios, 95)  # 95th percentile as threshold
                if self.verbose:
                    print(f"Set DK ratio threshold: {self.dk_threshold:.2f}")

            elif hasattr(model, 'decision_function'):
                scores = model.decision_function(X_test)
                if len(scores.shape) == 1:  # Binary classification
                    self.ood_threshold = np.percentile(scores, 5)
                else:  # Multiclass
                    self.ood_threshold = np.percentile(scores.max(axis=1), 5)
                if self.verbose:
                    print(f"Set OOD threshold (decision score): {self.ood_threshold:.2f}")
            else:
                self.ood_threshold = None
                if self.verbose:
                    print("No OOD threshold available for this model type")
        
        return self.best_model

    def save_model(self, path='models', preprocessor=None):
        """Save trained model and preprocessing objects
        
        Args:
            path (str): Directory to save model
            preprocessor (DataPreprocessor): The preprocessor instance used during training
        """
        os.makedirs(path, exist_ok=True)
        
        artifacts = {
            'model': self.best_model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_encoders': self.encoders,
            'preprocessor': preprocessor,
            'original_classes': self.original_classes,
            'ood_threshold': self.ood_threshold,
            'dk_threshold': getattr(self, 'dk_threshold', 0.5),  # Default if not set
            'best_params': getattr(self.best_model, 'best_params_', None),
            'final_feature_order': preprocessor.final_training_columns
        }
        
        joblib.dump(artifacts, f"{path}/ccva_model.pkl")
        
        if self.verbose:
            print(f"Model saved with {len(preprocessor.final_training_columns)} features")
            if hasattr(self.best_model, 'best_params_'):
                print("Best parameters:", self.best_model.best_params_)
    