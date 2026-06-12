# vman3_ml/vman3_ml/training.py
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from collections import Counter
from .processing import DataPreprocessor

import pandas as pd
import numpy as np
import joblib
import os
import json

try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False


_PARAM_GRIDS = {
    'RandomForest': {
        'n_estimators':       [100, 200, 300],
        'max_depth':          [None, 10, 20, 30],
        'min_samples_split':  [2, 5, 10],
        'min_samples_leaf':   [1, 2, 4],
    },
    'XGBoost': {
        'n_estimators':      [100, 200, 300],
        'max_depth':         [3, 6, 9],
        'learning_rate':     [0.05, 0.1, 0.2],
        'subsample':         [0.8, 1.0],
        'colsample_bytree':  [0.8, 1.0],
        'min_child_weight':  [1, 3],
    },
}

# All searches run sequentially (n_jobs=1) to avoid macOS loky/OpenMP
# interaction that segfaults XGBoost when RF's parallel workers ran first.
# Each model still uses its own internal n_jobs for tree building.
_SEARCH_N_JOBS = {
    'RandomForest': 1,
    'XGBoost': 1,
}


class ModelTrainer:
    def __init__(self, verbose=False):
        self.verbose = verbose
        # XGBoost must be inserted first — RF's loky workers contaminate XGBoost's
        # OpenMP runtime on macOS, causing a SIGSEGV if RF runs before XGBoost.
        self.models = {}
        if _HAS_XGB:
            self.models['XGBoost'] = XGBClassifier(
                objective='multi:softprob',
                eval_metric='mlogloss',
                tree_method='hist',
                device='cpu',
                nthread=-1,
                random_state=42,
                verbosity=0,
            )
        self.models['RandomForest'] = RandomForestClassifier(
            class_weight='balanced_subsample',
            random_state=42,
            n_jobs=-1,
        )
        if not _HAS_XGB:
            print("WARNING: xgboost not installed — training with RandomForest only.")

        self.original_classes = None
        self.ood_threshold     = None
        self.dk_threshold      = None
        self.best_model        = None
        self.best_model_name_  = None
        self.best_params_      = None
        self.scaler            = None
        self.label_encoder     = None
        self.encoders          = None
        self.classes_          = None
        self._X_test           = None
        self._y_test           = None
        self.comparison_report_: dict = {}

    # ---------------------------------------------------------------------- #
    #  Training                                                                #
    # ---------------------------------------------------------------------- #

    def train(self, X, y, test_size=0.2, n_iter_search=10):
        """Train all configured models and select the best by test macro-F1."""
        if len(X) != len(y):
            raise ValueError(
                f"Mismatched shapes: X has {len(X)} rows, y has {len(y)}."
            )

        try:
            self.original_classes = set(pd.Series(y).unique())
        except Exception as exc:
            raise ValueError(f"Could not determine unique classes: {exc}")

        preprocessor = DataPreprocessor(verbose=self.verbose)
        X_encoded, self.encoders = preprocessor._encode_features(X)
        X_scaled, self.scaler   = preprocessor._scale_features(X_encoded)

        if isinstance(X_scaled, pd.DataFrame):
            dup = X_scaled.columns.duplicated()
            if dup.any():
                print(f"Warning: {dup.sum()} duplicate columns after scaling — removing.")
                X_scaled = X_scaled.loc[:, ~dup]

        y_encoded, self.label_encoder = preprocessor._encode_target(y)

        # Drop classes too small for stratified CV (need ≥ cv_folds + 2 samples)
        _cv_folds        = 5
        _min_class_size  = _cv_folds + 2
        class_counts     = pd.Series(y_encoded).value_counts()
        tiny_classes     = class_counts[class_counts < _min_class_size].index
        if len(tiny_classes):
            keep_mask    = ~pd.Series(y_encoded).isin(tiny_classes).values
            n_dropped    = (~keep_mask).sum()
            dropped_labels = self.label_encoder.inverse_transform(sorted(tiny_classes))
            print(
                f"Warning: dropping {n_dropped} sample(s) from {len(tiny_classes)} "
                f"class(es) with < {_min_class_size} members: {list(dropped_labels)}"
            )
            X_scaled  = np.asarray(X_scaled)[keep_mask]
            y_encoded = np.asarray(y_encoded)[keep_mask]

        # Re-map class indices to be consecutive starting at 0.
        # Dropping tiny classes leaves gaps (e.g. [0,1,2,4,5]) which causes
        # XGBoost to raise "Invalid classes" because it expects [0..N-1].
        unique_remaining = sorted(set(y_encoded))
        if unique_remaining != list(range(len(unique_remaining))):
            remap = {old: new for new, old in enumerate(unique_remaining)}
            y_encoded = np.array([remap[c] for c in y_encoded])
            # Keep the label_encoder in sync so inverse_transform still works
            self.label_encoder.classes_ = self.label_encoder.classes_[unique_remaining]

        X_train, X_test, y_train, y_test = train_test_split(
            np.asarray(X_scaled), np.asarray(y_encoded),
            test_size=test_size, random_state=42, stratify=y_encoded,
        )
        self._X_test = X_test
        self._y_test = y_test
        # Per-class training counts — used by CCVAPredictor for Wilson CIs
        self.training_class_counts_ = {
            self.label_encoder.classes_[i]: int((y_train == i).sum())
            for i in range(len(self.label_encoder.classes_))
        }

        import sys
        print(f"Train: {X_train.shape}  Test: {X_test.shape}", flush=True)
        print(f"Class distribution (train): {Counter(y_train)}\n", flush=True)

        best_score = -1.0

        for name, model in self.models.items():
            print(f"─── {name} ───────────────────────────────────────", flush=True)
            sys.stdout.flush()
            param_grid = _PARAM_GRIDS.get(name, {})

            search_n_jobs = _SEARCH_N_JOBS.get(name, -1)
            try:
                search = RandomizedSearchCV(
                    model,
                    param_distributions=param_grid,
                    n_iter=n_iter_search,
                    cv=_cv_folds,
                    scoring='f1_macro',
                    random_state=42,
                    n_jobs=search_n_jobs,
                    verbose=int(self.verbose),
                )
                search.fit(X_train, y_train)
            except Exception as exc:
                import traceback; traceback.print_exc()
                print(f"  FAILED: {exc} — skipping {name}.", flush=True)
                continue

            best_classifier = model.set_params(**search.best_params_)
            best_classifier.fit(X_train, y_train)

            y_pred = best_classifier.predict(X_test)
            acc    = accuracy_score(y_test, y_pred)
            f1     = f1_score(y_test, y_pred, average='macro', zero_division=0)

            print(f"  CV macro-F1 : {search.best_score_:.4f}", flush=True)
            print(f"  Test accuracy: {acc:.4f}   Test macro-F1: {f1:.4f}", flush=True)
            print(f"  Best params : {search.best_params_}", flush=True)

            self.comparison_report_[name] = {
                'cv_f1_macro':    round(float(search.best_score_), 4),
                'test_accuracy':  round(float(acc), 4),
                'test_f1_macro':  round(float(f1), 4),
                'best_params':    search.best_params_,
            }

            if f1 > best_score:
                best_score            = f1
                self.best_model       = best_classifier
                self.best_model_name_ = name
                self.best_params_     = search.best_params_

                if hasattr(best_classifier, 'predict_proba'):
                    probs = best_classifier.predict_proba(X_test)
                    # 2nd-percentile: only the bottom 2% of test confidences are OOD.
                    self.ood_threshold = np.percentile(probs.max(axis=1), 2)
                    dk_ratios = (X == 'dk').mean(axis=1) if hasattr(X, 'mean') else np.zeros(len(X))
                    self.dk_threshold = float(np.percentile(dk_ratios, 95))

                elif hasattr(best_classifier, 'decision_function'):
                    scores = best_classifier.decision_function(X_test)
                    self.ood_threshold = float(np.percentile(
                        scores.max(axis=1) if scores.ndim > 1 else scores, 2
                    ))
                else:
                    self.ood_threshold = None

        # Print comparison table
        self._print_comparison()
        return self.best_model

    def _print_comparison(self):
        if not self.comparison_report_:
            return
        lines = [
            f"\n{'═'*58}",
            f"  MODEL COMPARISON",
            f"{'═'*58}",
            f"  {'Model':<16} {'CV F1':>8}  {'Test Acc':>9}  {'Test F1':>8}",
            f"  {'─'*14}  {'─'*8}  {'─'*9}  {'─'*8}",
        ]
        for name, m in self.comparison_report_.items():
            marker = ' ◀ best' if name == self.best_model_name_ else ''
            lines.append(
                f"  {name:<16} {m['cv_f1_macro']:>8.4f}  "
                f"{m['test_accuracy']:>9.4f}  {m['test_f1_macro']:>8.4f}{marker}"
            )
        lines.append(f"{'═'*58}\n")
        print('\n'.join(lines), flush=True)

    # ---------------------------------------------------------------------- #
    #  Save                                                                    #
    # ---------------------------------------------------------------------- #

    def save_model(self, path='models', preprocessor=None, version=None):
        """Save the best trained model and all preprocessing objects."""
        os.makedirs(path, exist_ok=True)

        versions = getattr(preprocessor, 'training_instrument_versions_', None)
        if versions and len(versions) > 1:
            version = 'combined'
        else:
            version = version or getattr(preprocessor, 'instrument_version', None)
        model_filename = f"ccva_model_{version}.pkl" if version else "ccva_model.pkl"

        artifacts = {
            'model':                   self.best_model,
            'model_name':              self.best_model_name_,
            'scaler':                  self.scaler,
            'label_encoder':           self.label_encoder,
            'feature_encoders':        self.encoders,
            'preprocessor':            preprocessor,
            'original_classes':        self.original_classes,
            'ood_threshold':           self.ood_threshold,
            'dk_threshold':            self.dk_threshold if self.dk_threshold is not None else 0.5,
            'best_params':             self.best_params_,
            'comparison_report':       self.comparison_report_,
            'final_feature_order':     preprocessor.final_training_columns,
            'training_quality_report': getattr(preprocessor, 'training_quality_report_', {}),
            'rare_label_mapping':      getattr(preprocessor, 'rare_label_mapping_', {}),
            'instrument_version':      version,
            'instrument_detection':    getattr(preprocessor, 'instrument_detection_', {}),
            'instrument_dictionary':   getattr(preprocessor, 'instrument_dictionary', None),
            'training_instrument_versions': getattr(
                preprocessor, 'training_instrument_versions_', [version]
            ),
            'union_feature_columns': getattr(
                preprocessor, 'union_feature_columns_', preprocessor.final_training_columns
            ),
            # Narrative embedding metadata
            'narrative_model_name': (
                preprocessor.narrative_embedder.model_name
                if getattr(preprocessor, 'narrative_embedder', None) is not None
                   and getattr(preprocessor, 'narrative_dims_', 0) > 0
                else None
            ),
            'narrative_dims': getattr(preprocessor, 'narrative_dims_', 0),
            'training_class_counts': getattr(self, 'training_class_counts_', {}),
        }

        joblib.dump(artifacts, f"{path}/{model_filename}")
        if version:
            joblib.dump(artifacts, f"{path}/ccva_model.pkl")

        if self.verbose:
            print(f"Model saved: {path}/{model_filename}")
            print(f"  Winner: {self.best_model_name_}  |  "
                  f"Features: {len(preprocessor.final_training_columns)}  |  "
                  f"Narrative dims: {getattr(preprocessor, 'narrative_dims_', 0)}")
            if self.best_params_:
                print(f"  Best params: {self.best_params_}")

    # ---------------------------------------------------------------------- #
    #  Evaluate                                                                #
    # ---------------------------------------------------------------------- #

    def evaluate(self, X_test, y_test, save_path=None):
        """Evaluate the best model on the held-out test set."""
        if self.best_model is None:
            raise ValueError("No trained model. Call train() first.")

        y_pred        = self.best_model.predict(X_test)
        acc           = accuracy_score(y_test, y_pred)
        y_pred_labels = self.label_encoder.inverse_transform(y_pred)
        y_test_labels = self.label_encoder.inverse_transform(y_test)

        print(f"Test Accuracy: {acc:.4f}  ({self.best_model_name_})")
        print(classification_report(y_test_labels, y_pred_labels, zero_division=0))

        if save_path:
            output_data = {
                'winner':        self.best_model_name_,
                'test_accuracy': acc,
                'comparison':    self.comparison_report_,
                'classification_report': classification_report(
                    y_test_labels, y_pred_labels, zero_division=0, output_dict=True
                ),
            }
            with open(save_path, 'w') as fh:
                json.dump(output_data, fh, indent=4)
            print(f"Evaluation saved to {save_path}")
