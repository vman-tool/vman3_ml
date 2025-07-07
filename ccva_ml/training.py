# vman3_ml/vman3_ml/training.py
from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV

from collections import Counter 
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from .processing import DataPreprocessor

import pandas as pd
import numpy as np
import joblib
import os
import json
import matplotlib.pyplot as plt


from imblearn.under_sampling import RandomUnderSampler

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
   
    def train(self, X, y, test_size=0.2, n_iter_search=10):
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
            #np.asarray(X_scaled), np.asarray(y_encoded), test_size=test_size, random_state=42
            np.asarray(X_scaled), np.asarray(y_encoded), test_size=test_size, random_state=42, stratify=y_encoded
        )
        print('Shape of X_train', X_train.shape)
        print('Shape of X_test', X_test.shape)
        print('Shape of y_train', y_train.shape)
        print('Shape of y_test', y_test.shape)

        # balance the training classes
        print('The number of training examples before resampling:\n', Counter(y_train))

        # balance the labels using RandomUnderSampler
        rus = RandomUnderSampler(random_state=42)
        X_train_res, y_train_res = rus.fit_resample(X_train, y_train)
        print('The number of samples after resampling:\n', Counter(y_train_res))

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
            search.fit(X_train_res, y_train_res)
            
            # Get best model from search
            # model = search.best_estimator_
            print("Best: %.2f using %s" % (search.best_score_,search.best_params_))

            # Insert the best parameters identified by randomized grid search into the base classifier
            best_classifier = model.set_params(**search.best_params_)

            # fit the final model with best parameters
            self.best_model = best_classifier.fit(X_train_res, y_train_res)

            # predict model test set
            y_pred = self.best_model.predict(X_test)
            score = accuracy_score(y_test, y_pred)
            
            if score > best_score:
                best_score = score
                # self.best_model = best_classifier
                if self.verbose:
                    print(f"New best model: {name} with accuracy {score:.2%}")
                    # print("Best parameters:", search.best_params_)
            
            # Set OOD and dk threshold based on available methods
            if hasattr(model, 'predict_proba'):
                probs = model.predict_proba(X_test)
                # print(f"The value for probs max is\n {probs.max(axis=1)}")
                # print(f"The value for nth percentile is\n {np.percentile(probs.max(axis=1), 5)}")
                self.ood_threshold = np.percentile(probs.max(axis=1), 55)  # 5th percentile of confidence
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
    
    def evaluate_with_cross_validation(self, X_test, y_test, cv=5, save_path=None):
        """Evaluate the trained model using cross-validation on the test data.

        Args:
            X_test (np.ndarray): Test features.
            y_test (np.ndarray): Test labels.
            cv (int): Number of cross-validation folds.
            save_path (str, optional): Path to save the results. If None, results are printed.
        """
        if self.best_model is None:
            raise ValueError("No trained model found. Please train a model before evaluation.")

        #kf = KFold(n_splits=cv, shuffle=True, random_state=42)

        # use stratified sampling
        from sklearn.model_selection import StratifiedKFold
        kf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)


        results = []
        accuracies = []
        all_reports = []

        #for fold, (train_idx, val_idx) in enumerate(kf.split(X_test)):
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_test, y_test)):
            X_val_train, X_val_test = X_test[train_idx], X_test[val_idx]
            y_val_train, y_val_test = y_test[train_idx], y_test[val_idx]

            model = self.best_model
            #model.fit(X_val_train, y_val_train) # no need to re-fit the model in a cross validation
            y_pred = model.predict(X_val_test)

            acc = accuracy_score(y_val_test, y_pred)
            accuracies.append(acc)
            #report = classification_report(y_val_test, y_pred, output_dict=True)
            report = classification_report(y_val_test, y_pred, output_dict=True, zero_division=0)
            all_reports.append(report)
            results.append({
                'fold': fold + 1,
                'accuracy': acc,
                'classification_report': report
            })

            print(f"\nFold {fold + 1} Accuracy: {acc:.4f}")
            #print(classification_report(y_val_test, y_pred))
            print(classification_report(y_val_test, y_pred, zero_division=0))

        # print label mapping
        if self.label_encoder is not None:
            print("\nLabel Mapping:")
            for i, label in enumerate(self.label_encoder.classes_):
                print(f"  {i}: {label}")

        # Compute and print average and std deviation
        avg_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)
        print(f"\nAverage Accuracy across {cv} folds: {avg_acc:.4f}")
        print(f"Standard Deviation of Accuracy: {std_acc:.4f}")


        # Compute average classification report
        avg_report = defaultdict(lambda: defaultdict(float))
        count_report = defaultdict(int)

        for report in all_reports:
            for label, metrics in report.items():
                if isinstance(metrics, dict):
                    for metric, value in metrics.items():
                        avg_report[label][metric] += value
                    count_report[label] += 1
                    

        for label in avg_report:
            if isinstance(avg_report[label], dict):
                for metric in avg_report[label]:
                    avg_report[label][metric] /= count_report[label]

        # print("\nAverage Classification Report:")
        # for label, metrics in avg_report.items():
        #     print(f"{label}:")
        #     for metric, value in metrics.items():
        #         print(f"  {metric}: {value:.4f}")



        # # Format and print the averaged classification report as a table
        # header = f"{'Label':<15}{'Precision':>10}{'Recall':>10}{'F1-Score':>10}{'Support':>10}"
        # print("\nAveraged Classification Report:")
        # print(header)
        # print("-" * len(header))

        # for label, metrics in avg_report.items():
        #     if label in ['accuracy']:
        #         continue
        #     label_name = label
        #     if label.isdigit():
        #         idx = int(label)
        #         if idx < len(self.label_encoder.classes_):
        #             label_name = self.label_encoder.classes_[idx]
        #     precision = metrics.get('precision', 0.0)
        #     recall = metrics.get('recall', 0.0)
        #     f1 = metrics.get('f1-score', 0.0)
        #     support = metrics.get('support', 0.0)
        #     print(f"{label_name:<15}{precision:10.2f}{recall:10.2f}{f1:10.2f}{support:10.0f}")



        # Replace numeric keys with actual label names
        mapped_report = {}
        for key, metrics in avg_report.items():
            if key.isdigit():
                label = self.label_encoder.classes_[int(key)]
            else:
                label = key
            mapped_report[label] = metrics

        # Determine column widths
        label_width = max(len(label) for label in mapped_report.keys())
        metric_names = ['precision', 'recall', 'f1-score', 'support']
        col_widths = {metric: max(len(metric), 9) for metric in metric_names}

        # Print header
        header = f"{'Label':<{label_width}}"
        for metric in metric_names:
            header += f"  {metric.capitalize():>{col_widths[metric]}}"
        print(header)
        print('-' * len(header))

        # Print each row
        for label, metrics in mapped_report.items():
            row = f"{label:<{label_width}}"
            for metric in metric_names:
                value = metrics.get(metric, 0)
                if metric == 'support':
                    row += f"  {int(value):>{col_widths[metric]}}"
                else:
                    row += f"  {value:>{col_widths[metric]}.2f}"
            print(row)


        if save_path:
            output_data = {
                            'fold_results': results,
                            'average_accuracy': avg_acc,
                            'std_accuracy': std_acc,
                            'average_classification_report': {label: dict(metrics) for label, metrics in avg_report.items()}
                        }
            with open(save_path, 'w') as f:
                json.dump(output_data, f, indent=4)
            #pd.DataFrame(results).to_json(save_path, orient='records', lines=True)
            print(f"\nCross-validation results saved to {save_path}")

        # visualize
        plt.figure(figsize=(8, 5))
        plt.bar(range(1, cv + 1), accuracies, color='skyblue')
        plt.xlabel('Fold')
        plt.ylabel('Accuracy')
        plt.title('Cross-Validation Accuracy per Fold')
        plt.xticks(range(1, cv + 1))
        plt.ylim(0, 1)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig('cv_accuracy_plot.png')
        print("\nAccuracy plot saved to cv_accuracy_plot.png")

