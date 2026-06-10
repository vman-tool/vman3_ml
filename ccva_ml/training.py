# vman3_ml/vman3_ml/training.py
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from .processing import DataPreprocessor

import pandas as pd
import numpy as np
import joblib
import os
import json


class ModelTrainer:
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.models = {
            'RandomForest': RandomForestClassifier(
                class_weight='balanced_subsample',
                random_state=42,
                n_jobs=-1,
            ),
        }
        self.original_classes = None
        self.ood_threshold = None
        self.dk_threshold = None
        self.best_model = None
        self.best_params_ = None
        self.scaler = None
        self.label_encoder = None
        self.encoders = None
        self.classes_ = None
        self._X_test = None
        self._y_test = None

    def train(self, X, y, test_size=0.2, n_iter_search=10):
        """Train and evaluate models"""
        if len(X) != len(y):
            raise ValueError(
                f"Mismatched input shapes: X has {len(X)} samples, y has {len(y)}. "
                "They must have the same number of samples."
            )

        try:
            self.original_classes = set(pd.Series(y).unique())
        except Exception as e:
            raise ValueError(f"Could not determine unique classes from target variable: {str(e)}")

        preprocessor = DataPreprocessor(verbose=self.verbose)
        X_encoded, self.encoders = preprocessor._encode_features(X)
        X_scaled, self.scaler = preprocessor._scale_features(X_encoded)

        if isinstance(X_scaled, pd.DataFrame):
            duplicates = X_scaled.columns.duplicated()
            if duplicates.any():
                print(f"Warning: Found {duplicates.sum()} duplicate columns after scaling:")
                print(X_scaled.columns[duplicates].tolist())
                X_scaled = X_scaled.loc[:, ~duplicates]

        y_encoded, self.label_encoder = preprocessor._encode_target(y)

        # Drop classes with fewer samples than needed for stratified split (≥2) and
        # cross-validation (cv=5 needs ≥5 per class in the training fold, so ≥7 total).
        _cv_folds = 5
        _min_class_size = _cv_folds + 2  # enough for at least 1 sample per CV fold after split
        class_counts = pd.Series(y_encoded).value_counts()
        tiny_classes = class_counts[class_counts < _min_class_size].index
        if len(tiny_classes):
            keep_mask = ~pd.Series(y_encoded).isin(tiny_classes).values
            n_dropped = (~keep_mask).sum()
            dropped_labels = self.label_encoder.inverse_transform(sorted(tiny_classes))
            print(
                f"Warning: dropping {n_dropped} sample(s) from {len(tiny_classes)} class(es) "
                f"with < {_min_class_size} members (too few for stratified CV): {list(dropped_labels)}"
            )
            X_scaled = np.asarray(X_scaled)[keep_mask]
            y_encoded = np.asarray(y_encoded)[keep_mask]

        X_train, X_test, y_train, y_test = train_test_split(
            np.asarray(X_scaled), np.asarray(y_encoded),
            test_size=test_size, random_state=42, stratify=y_encoded,
        )
        self._X_test = X_test
        self._y_test = y_test

        print('Shape of X_train', X_train.shape)
        print('Shape of X_test', X_test.shape)
        print('Shape of y_train', y_train.shape)
        print('Shape of y_test', y_test.shape)
        print('The number of training examples:\n', Counter(y_train))

        param_dist = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
        }

        best_score = 0
        for name, model in self.models.items():
            if self.verbose:
                print(f"\nTraining {name} with {n_iter_search} iterations...")

            search = RandomizedSearchCV(
                model,
                param_distributions=param_dist,
                n_iter=n_iter_search,
                cv=5,
                verbose=self.verbose,
                scoring='f1_macro',
                random_state=42,
                n_jobs=-1,
            )
            search.fit(X_train, y_train)

            self.best_params_ = search.best_params_
            print("Best: %.2f using %s" % (search.best_score_, self.best_params_))

            best_classifier = model.set_params(**self.best_params_)
            self.best_model = best_classifier.fit(X_train, y_train)

            y_pred = self.best_model.predict(X_test)
            score = accuracy_score(y_test, y_pred)

            if score > best_score:
                best_score = score
                if self.verbose:
                    print(f"New best model: {name} with accuracy {score:.2%}")

            if hasattr(self.best_model, 'predict_proba'):
                probs = self.best_model.predict_proba(X_test)
                # 5th-percentile: only the bottom 5% of training confidences
                # are treated as OOD, so ~95% of in-distribution data passes.
                self.ood_threshold = np.percentile(probs.max(axis=1), 5)
                if self.verbose:
                    print(f"Set OOD threshold (5th-pct probability): {self.ood_threshold:.3f}")

                dk_ratios = (X == 'dk').mean(axis=1)
                self.dk_threshold = np.percentile(dk_ratios, 95)
                if self.verbose:
                    print(f"Set DK ratio threshold: {self.dk_threshold:.2f}")

            elif hasattr(self.best_model, 'decision_function'):
                scores = self.best_model.decision_function(X_test)
                if len(scores.shape) == 1:
                    self.ood_threshold = np.percentile(scores, 5)
                else:
                    self.ood_threshold = np.percentile(scores.max(axis=1), 5)
                if self.verbose:
                    print(f"Set OOD threshold (decision score): {self.ood_threshold:.2f}")
            else:
                self.ood_threshold = None
                if self.verbose:
                    print("No OOD threshold available for this model type")

        return self.best_model

    def save_model(self, path='models', preprocessor=None, version=None):
        """Save trained model and preprocessing objects."""
        os.makedirs(path, exist_ok=True)

        version = version or getattr(preprocessor, 'instrument_version', None)
        model_filename = f"ccva_model_{version}.pkl" if version else "ccva_model.pkl"

        artifacts = {
            'model': self.best_model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_encoders': self.encoders,
            'preprocessor': preprocessor,
            'original_classes': self.original_classes,
            'ood_threshold': self.ood_threshold,
            'dk_threshold': self.dk_threshold if self.dk_threshold is not None else 0.5,
            'best_params': self.best_params_,
            'final_feature_order': preprocessor.final_training_columns,
            'training_quality_report': getattr(preprocessor, 'training_quality_report_', {}),
            'rare_label_mapping': getattr(preprocessor, 'rare_label_mapping_', {}),
            'instrument_version': version,
            'instrument_detection': getattr(preprocessor, 'instrument_detection_', {}),
            'instrument_dictionary': getattr(preprocessor, 'instrument_dictionary', None),
        }

        joblib.dump(artifacts, f"{path}/{model_filename}")
        if version:
            joblib.dump(artifacts, f"{path}/ccva_model.pkl")

        if self.verbose:
            print(f"Model saved with {len(preprocessor.final_training_columns)} features")
            if self.best_params_:
                print("Best parameters:", self.best_params_)

    def evaluate(self, X_test, y_test, save_path=None):
        """Evaluate the trained model on the held-out test set.

        Args:
            X_test (np.ndarray): Test features (already encoded and scaled).
            y_test (np.ndarray): Test labels (already encoded).
            save_path (str, optional): Path to save the JSON report.
        """
        if self.best_model is None:
            raise ValueError("No trained model found. Please train a model before evaluation.")

        y_pred = self.best_model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        y_pred_labels = self.label_encoder.inverse_transform(y_pred)
        y_test_labels = self.label_encoder.inverse_transform(y_test)

        print(f"Test Accuracy: {acc:.4f}")
        print(classification_report(y_test_labels, y_pred_labels, zero_division=0))

        if save_path:
            output_data = {
                'test_accuracy': acc,
                'classification_report': classification_report(
                    y_test_labels, y_pred_labels, zero_division=0, output_dict=True
                ),
            }
            with open(save_path, 'w') as f:
                json.dump(output_data, f, indent=4)
            print(f"Evaluation results saved to {save_path}")
