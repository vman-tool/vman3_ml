import joblib
import pandas as pd
import numpy as np


class CCVAPredictor:
    def __init__(self, model_path='models/ccva_model.pkl', verbose=False):
        """Load trained model and all preprocessing objects from a saved artifact."""
        artifacts = joblib.load(model_path)
        self.model = artifacts['model']
        self.scaler = artifacts['scaler']
        self.label_encoder = artifacts['label_encoder']
        self.feature_encoders = artifacts['feature_encoders']
        self.preprocessor = artifacts['preprocessor']
        self.original_classes = set(artifacts['original_classes'])
        self.ood_threshold = artifacts.get('ood_threshold')
        self.dk_threshold = artifacts.get('dk_threshold', 0.5)
        self.instrument_version = artifacts.get('instrument_version')
        self.verbose = verbose
        self.expected_columns = list(self.preprocessor.final_training_columns)

        if self.ood_threshold is not None and not (0 < self.ood_threshold < 1):
            print(f"WARNING: Unusual OOD threshold {self.ood_threshold:.4f} — disabling OOD filter.")
            self.ood_threshold = None

        label_classes = set(self.label_encoder.classes_)
        if self.original_classes != label_classes:
            print("WARNING: original_classes don't match label encoder classes")

        if self.verbose:
            print(f"Model loaded: {len(self.expected_columns)} features, "
                  f"{len(self.original_classes)} classes, "
                  f"OOD threshold={self.ood_threshold}, "
                  f"DK threshold={self.dk_threshold}")

    def _validate_columns(self, df):
        """Check if all expected columns are present."""
        missing = [col for col in self.expected_columns if col not in df.columns]
        extra   = [col for col in df.columns if col not in self.expected_columns]
        if missing:
            print(f"WARNING: Missing {len(missing)} columns used in training: {missing[:10]}")
        if extra and self.verbose:
            print(f"Note: {len(extra)} additional columns not used in training")
        return len(missing) == 0

    def _apply_encoders(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply the saved training-time encoders to a feature DataFrame.

        Uses the encoders fitted during training (stored in self.feature_encoders)
        so that categorical values are mapped to the same integer codes the model
        learned on.  Unseen category values are mapped to the 'dk' sentinel (or
        the first known class when 'dk' is absent).
        """
        X = X.copy()
        for col, encoder in self.feature_encoders.items():
            if col not in X.columns:
                continue
            col_data = X[col].astype(str).replace({'nan': 'dk', '': 'dk', ' ': 'dk'})
            if isinstance(encoder, dict):
                # Hardcoded yes/no/dk mapping — deterministic, no unseen-value risk
                X[col] = col_data.map(encoder).fillna(-1).astype(float)
            else:
                # LabelEncoder fitted on training data — transform only, never refit
                known    = set(encoder.classes_)
                fallback = 'dk' if 'dk' in known else encoder.classes_[0]
                X[col]   = encoder.transform(
                    col_data.apply(lambda v: v if v in known else fallback)
                )
        # Any remaining object columns not covered by saved encoders
        for col in X.select_dtypes(include=['object']).columns:
            X[col] = pd.factorize(X[col])[0]
        return X

    def predict(self, cleaned_df):
        """Make predictions with DK-ratio and OOD confidence thresholding."""
        try:
            cleaned_df = cleaned_df.copy()

            # Fill NA before computing dk_ratio so the ratio reflects actual missingness
            for col in self.feature_encoders:
                if col in cleaned_df.columns:
                    if pd.api.types.is_string_dtype(cleaned_df[col]):
                        cleaned_df[col] = cleaned_df[col].fillna('dk').replace({'': 'dk'})
                    else:
                        cleaned_df[col] = cleaned_df[col].fillna(-999)

            dk_ratios   = (cleaned_df == 'dk').mean(axis=1)
            dk_ood_mask = dk_ratios > self.dk_threshold
            if self.verbose and dk_ood_mask.any():
                print(f"DK-OOD: {dk_ood_mask.sum()} records exceed "
                      f"DK threshold ({self.dk_threshold:.0%})")

            # Feature selection using the saved preprocessor (prediction mode, no target)
            X_raw, _ = self.preprocessor._prepare_training_data(
                cleaned_df,
                target_col=None,
                instrument_version=self.instrument_version,
            )

            # Encode using the SAVED training encoders (not re-fitted ones)
            encoded = self._apply_encoders(X_raw)

            # Align to the exact column order the scaler was fitted on.
            # _encode_features reorders columns (numeric first, then object), so
            # scaler.feature_names_in_ differs from final_training_columns order.
            encoded = encoded.reindex(columns=self.scaler.feature_names_in_, fill_value=0)
            scaled  = self.scaler.transform(encoded)

            if hasattr(self.model, 'predict_proba'):
                probs      = self.model.predict_proba(scaled)
                predictions = self.model.classes_[np.argmax(probs, axis=1)]
                confidence  = np.max(probs, axis=1)
                ood_mask    = (
                    (confidence < self.ood_threshold)
                    if self.ood_threshold is not None
                    else np.zeros(len(predictions), dtype=bool)
                )
                if self.verbose:
                    print(f"Confidence — min={confidence.min():.3f}, "
                          f"median={np.median(confidence):.3f}, "
                          f"max={confidence.max():.3f}, "
                          f"OOD threshold={self.ood_threshold}")
            else:
                predictions = self.model.predict(scaled)
                ood_mask    = np.zeros(len(predictions), dtype=bool)

            final_ood_mask = ood_mask | dk_ood_mask.values
            decoded = self.label_encoder.inverse_transform(predictions)
            decoded[final_ood_mask] = 'out_of_distribution'
            # Catch any label that somehow fell outside the known class set
            valid = set(self.original_classes) | {'out_of_distribution'}
            decoded[~np.isin(decoded, list(valid))] = 'out_of_distribution'

            if self.verbose:
                n_ood = final_ood_mask.sum()
                print(f"Predictions: {len(decoded) - n_ood} classified, {n_ood} OOD")

            return decoded

        except Exception as e:
            import traceback
            print(f"Prediction error: {e}")
            traceback.print_exc()
            return np.array(['PREDICTION_ERROR'] * len(cleaned_df))

    def generate_data_quality_report(self, df):
        """Generate a report on data quality issues in the input DataFrame."""
        report = {
            'missing_columns': list(set(self.expected_columns) - set(df.columns)),
            'empty_columns': [],
            'type_mismatches': [],
            'problematic_records': {},
            'dk_stats': {
                'total_dk': 0,
                'columns_with_dk': [],
                'records_with_dk': 0,
            },
        }

        for col in self.expected_columns:
            if col not in df.columns:
                continue

            try:
                is_empty = bool(df[col].isna().all()) or bool(df[col].eq('').all())
            except Exception:
                is_empty = False

            if is_empty:
                report['empty_columns'].append(col)

            if col in self.feature_encoders and not pd.api.types.is_string_dtype(df[col]):
                report['type_mismatches'].append(col)

            if col in self.feature_encoders:
                try:
                    mask = ~df[col].astype(str).str.match(r'^[\w\s-]+$')
                    if mask.any():
                        report['problematic_records'][col] = df.loc[mask, col].unique().tolist()
                except Exception:
                    report['problematic_records'][col] = ['ERROR_CHECKING_VALUES']

        return report
