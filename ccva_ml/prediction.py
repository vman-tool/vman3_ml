import joblib
import pandas as pd
import numpy as np
from .instrument_dictionary import detect_instrument_version
from .narrative import NarrativeEmbedder, NARRATIVE_COLS

try:
    import shap as _shap
    _HAS_SHAP = True
except ImportError:
    _HAS_SHAP = False


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
        self.ood_threshold         = artifacts.get('ood_threshold')
        self.ood_entropy_threshold = artifacts.get('ood_entropy_threshold')
        # DK threshold: enforce a minimum of 0.60.
        # Old artifacts were calibrated on quality-filtered training data, giving
        # unrealistically tight thresholds (~0.15) that over-flag prediction data.
        # 0.60 means ">60 % of feature columns are missing" — genuinely unusable.
        raw_dk = artifacts.get('dk_threshold', 0.60)
        self.dk_threshold = max(raw_dk, 0.60)
        self.instrument_version          = artifacts.get('instrument_version')
        self.training_instrument_versions = artifacts.get('training_instrument_versions',
                                                          [self.instrument_version])
        self.verbose          = verbose
        self.expected_columns = list(self.preprocessor.final_training_columns)
        self.union_feature_columns = list(
            artifacts.get('union_feature_columns') or self.expected_columns
        )
        # Per-class training counts for Wilson confidence intervals
        self.training_class_counts = artifacts.get('training_class_counts', {})

        # Narrative embedding metadata
        self.narrative_dims       = int(artifacts.get('narrative_dims', 0) or 0)
        narrative_model_name      = artifacts.get('narrative_model_name')
        self._narrative_embedder  = (
            NarrativeEmbedder(model_name=narrative_model_name, verbose=verbose)
            if self.narrative_dims > 0 and narrative_model_name
            else None
        )

        if self.ood_threshold is not None and not (0 < self.ood_threshold < 1):
            print(f"WARNING: Unusual OOD threshold {self.ood_threshold:.4f} — disabling OOD filter.")
            self.ood_threshold = None

        label_classes = set(self.label_encoder.classes_)
        if self.original_classes != label_classes:
            print("WARNING: original_classes don't match label encoder classes")

        if self.verbose:
            versions_str = ', '.join(str(v) for v in self.training_instrument_versions if v)
            model_name   = artifacts.get('model_name', type(self.model).__name__)
            ood_str = (
                f"entropy>{self.ood_entropy_threshold:.3f}"
                if self.ood_entropy_threshold is not None
                else f"conf<{self.ood_threshold}"
            )
            print(f"Model loaded: {model_name} | instrument={versions_str or self.instrument_version}, "
                  f"{len(self.union_feature_columns)} union features, "
                  f"{len(self.original_classes)} classes, "
                  f"narrative_dims={self.narrative_dims}, "
                  f"OOD={ood_str}, DK threshold={self.dk_threshold:.0%}")

        # Column order the scaler was fit on — used for reindexing before transform.
        # New artifacts store this explicitly; old ones require reconstruction.
        _stored_cols = artifacts.get('scaler_feature_columns')
        if _stored_cols:
            self.scaler_feature_columns = list(_stored_cols)
        else:
            # Old artifact: _encode_features puts numeric cols first then categorical.
            # Reconstruct that order: anything NOT in feature_encoders is numeric.
            _enc_keys = set(self.feature_encoders.keys())
            _all = list(self.preprocessor.final_training_columns)
            _numeric = [c for c in _all if c not in _enc_keys]
            _categ   = [c for c in _all if c in _enc_keys]
            self.scaler_feature_columns = _numeric + _categ

        # Feature-label lookup and SHAP explainer (lazy-init on first predict_detailed call)
        self._feature_labels: dict = {}
        self._shap_explainer = None

    # ---------------------------------------------------------------------- #
    #  Internal helpers                                                        #
    # ---------------------------------------------------------------------- #

    def _build_feature_labels(self):
        """Build {column_name_lower → English question label} from instrument dictionaries."""
        labels: dict = {}
        instr_dict = getattr(self.preprocessor, 'instrument_dictionary', None)
        if not isinstance(instr_dict, dict):
            return labels
        # Flat dict with a top-level 'survey' key (most common case)
        if 'survey' in instr_dict:
            survey_lists = [instr_dict['survey']]
        else:
            # Nested {version: sub_dict} mapping — collect survey from each sub-dict
            survey_lists = [v['survey'] for v in instr_dict.values()
                            if isinstance(v, dict) and 'survey' in v]
        for survey in survey_lists:
            for q in (survey if isinstance(survey, list) else []):
                name = q.get('name') or q.get('normalized_name') or ''
                label = (q.get('label_en') or q.get('label') or '').strip()
                if name and label:
                    labels[name.lower()] = label
        return labels

    def _get_shap_explainer(self):
        if self._shap_explainer is None and _HAS_SHAP:
            self._shap_explainer = _shap.TreeExplainer(self.model)
        return self._shap_explainer

    # Minimum fraction of total |SHAP| that the narrative block must contribute
    # before it appears in the notes.  0.15 = 15 %.
    _NARRATIVE_NOTE_THRESHOLD = 0.15

    # Values that carry no interpretive meaning for a human reader.
    _UNINFORMATIVE_VALUES = frozenset({'nan', 'none', 'skipped', '?', '', 'null', 'na', 'n/a', '########'})

    # Feature columns that are administrative/metadata, not clinical signals.
    _SUPPRESSED_FEATURES = frozenset({'id10010c', 'id10010b', 'id10010a', 'id10010'})

    @classmethod
    def _is_informative(cls, val) -> bool:
        """True when val adds something a human reader can act on."""
        if val is None:
            return False
        s = str(val).strip().lower()
        return s not in cls._UNINFORMATIVE_VALUES

    @staticmethod
    def _ascii_notes(text: str) -> str:
        """Replace non-ASCII punctuation with plain ASCII equivalents."""
        return (text
                .replace('…', '...')   # … ellipsis
                .replace('↑', '(+)')   # ↑ up arrow
                .replace('↓', '(-)')   # ↓ down arrow
                )

    def _narrative_text(self, df_row) -> str:
        """Concatenate non-empty narrative fields for a single record row."""
        parts = []
        for col in NARRATIVE_COLS:
            val = df_row.get(col, '') if hasattr(df_row, 'get') else ''
            val = str(val).strip()
            if val and val.lower() not in ('nan', 'none', 'skipped', ''):
                parts.append(val)
        return ' '.join(parts)

    def _generate_notes(self, X_scaled, predictions, cleaned_df, top_n=5):
        """Return an array of human-readable notes, one per record.

        Each note lists the top contributing structured features (by SHAP
        magnitude for the predicted class) with their raw values and direction.
        When the narrative embedding block contributes ≥ _NARRATIVE_NOTE_THRESHOLD
        of the total |SHAP| for a record, a narrative note is prepended showing
        the raw narrative text and the narrative's aggregate direction/percentage.
        """
        feature_names = list(self.scaler_feature_columns)
        is_narrative   = np.array([n.startswith('narr_emb_') for n in feature_names])

        if not _HAS_SHAP:
            return np.array(['shap not installed'] * len(predictions), dtype=object)

        explainer = self._get_shap_explainer()
        if explainer is None:
            return np.array([''] * len(predictions), dtype=object)

        try:
            raw = explainer.shap_values(X_scaled)
            # Normalise to a 3-D array (N, F, C) regardless of shap/model version
            if isinstance(raw, list):
                # list of C arrays each (N, F)  →  (N, F, C)
                shap_3d = np.stack(raw, axis=-1)
            elif isinstance(raw, np.ndarray) and raw.ndim == 2:
                # Single-output model: (N, F) → treat as (N, F, 1)
                shap_3d = raw[:, :, np.newaxis]
            else:
                shap_3d = raw  # already (N, F, C)
        except Exception as exc:
            if self.verbose:
                print(f"[notes] SHAP failed: {exc}")
            return np.array([''] * len(predictions), dtype=object)

        if not self._feature_labels:
            self._feature_labels = self._build_feature_labels()

        has_df = hasattr(cleaned_df, 'iloc')

        notes = []
        for i, pred_class_idx in enumerate(predictions):
            c = int(pred_class_idx) if shap_3d.shape[2] > 1 else 0
            c = min(c, shap_3d.shape[2] - 1)
            sv = shap_3d[i, :, c].copy()

            # ── Narrative contribution ─────────────────────────────────────
            narr_abs_total = float(np.sum(np.abs(sv[is_narrative])))
            total_abs      = float(np.sum(np.abs(sv)))
            narr_frac      = narr_abs_total / total_abs if total_abs > 1e-6 else 0.0
            narr_signed    = float(np.sum(sv[is_narrative]))

            sv[is_narrative] = 0.0          # blank out uninterpretable dims

            # ── Top structured features ────────────────────────────────────
            # Rank all features by |SHAP| descending; search through the top
            # candidates to collect top_n whose raw values are interpretable.
            # Uninformative values (skipped, nan, none, ?) are skipped.
            abs_sv     = np.abs(sv)
            candidates = np.argsort(abs_sv)[::-1]
            candidates = [j for j in candidates if abs_sv[j] > 1e-6]

            parts = []

            # Narrative entry (prepended when it carries meaningful weight)
            if narr_frac >= self._NARRATIVE_NOTE_THRESHOLD:
                arrow    = '↑' if narr_signed > 0 else '↓'
                pct      = int(round(narr_frac * 100))
                row      = cleaned_df.iloc[i] if has_df else {}
                text     = self._narrative_text(row)
                if text:
                    snippet = text if len(text) <= 80 else text[:77] + '...'
                    parts.append(f'Narrative ({pct}%): {snippet} {arrow}')
                else:
                    parts.append(f'Narrative ({pct}%): (no text) {arrow}')

            for j in candidates:
                if len(parts) - int(narr_frac >= self._NARRATIVE_NOTE_THRESHOLD) >= top_n:
                    break
                fname  = feature_names[j]
                if fname.lower() in self._SUPPRESSED_FEATURES:
                    continue
                raw_val = (cleaned_df.iloc[i].get(fname, '?')
                           if has_df and fname in cleaned_df.columns
                           else '?')
                if isinstance(raw_val, float):
                    raw_val = '?' if pd.isna(raw_val) else f'{raw_val:.2g}'
                if not self._is_informative(raw_val):
                    continue
                label = self._feature_labels.get(fname.lower(), fname)
                label = label if len(label) <= 45 else label[:42] + '...'
                arrow = '(+)' if sv[j] > 0 else '(-)'
                parts.append(f'{label}: {raw_val} {arrow}')

            notes.append(self._ascii_notes('; '.join(parts)))

        return np.array(notes, dtype=object)

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
        """Apply the saved training-time encoders to a feature DataFrame."""
        X = X.copy()
        for col, encoder in self.feature_encoders.items():
            if col not in X.columns:
                continue
            col_data = X[col].astype(str).replace({'nan': 'dk', '': 'dk', ' ': 'dk'})
            if isinstance(encoder, dict):
                X[col] = col_data.map(encoder).fillna(-1).astype(float)
            else:
                known    = set(encoder.classes_)
                fallback = 'dk' if 'dk' in known else encoder.classes_[0]
                X[col]   = encoder.transform(
                    col_data.apply(lambda v: v if v in known else fallback)
                )
        for col in X.select_dtypes(include=['object']).columns:
            X[col] = pd.factorize(X[col])[0]
        return X

    def _prepare_features(self, cleaned_df):
        """Encode and scale input features; return (scaled_array, dk_ood_mask)."""
        cleaned_df = cleaned_df.copy()

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

        pred_detection = detect_instrument_version(cleaned_df)
        pred_version   = pred_detection['version']
        if self.verbose:
            print(f"Prediction: detected instrument version {pred_version} "
                  f"(model trained on: {', '.join(str(v) for v in self.training_instrument_versions if v)})")

        X_raw, _ = self.preprocessor._prepare_training_data(
            cleaned_df,
            target_col=None,
            instrument_version=pred_version,
        )

        missing_union = [c for c in self.union_feature_columns if c not in X_raw.columns]
        if missing_union:
            if self.verbose:
                print(f"Filling {len(missing_union)} missing union features with 'dk'")
            for col in missing_union:
                X_raw[col] = 'dk'

        if self.narrative_dims > 0:
            if self._narrative_embedder is not None:
                emb_df = self._narrative_embedder.embed(cleaned_df)
            else:
                emb_df = None

            if emb_df is not None:
                emb_aligned = emb_df.reindex(X_raw.index).fillna(0.0)
                X_raw = pd.concat([X_raw, emb_aligned], axis=1)
            else:
                if self.verbose:
                    print(f"WARNING: narrative embedder unavailable; "
                          f"filling {self.narrative_dims} narr_emb_* cols with zeros.")
                for i in range(self.narrative_dims):
                    X_raw[f'narr_emb_{i}'] = 0.0

        encoded = self._apply_encoders(X_raw)
        encoded = encoded.reindex(columns=self.scaler_feature_columns, fill_value=0)
        scaled  = self.scaler.transform(encoded)
        return scaled, dk_ood_mask

    @staticmethod
    def _wilson_ci(p, n, z=1.96):
        """Wilson score 95% confidence interval for a proportion p with sample size n."""
        n      = np.maximum(n, 1).astype(float)
        n_adj  = n + z ** 2
        center = (p * n + z ** 2 / 2) / n_adj
        margin = z * np.sqrt(n * p * (1 - p) + z ** 2 / 4) / n_adj
        return np.clip(center - margin, 0, 1), np.clip(center + margin, 0, 1)

    # ---------------------------------------------------------------------- #
    #  Public API                                                              #
    # ---------------------------------------------------------------------- #

    def predict(self, cleaned_df):
        """Return a 1-D array of predicted cause-of-death labels (or 'out_of_distribution')."""
        return self.predict_detailed(cleaned_df)['prediction'].values

    def predict_detailed(self, cleaned_df):
        """Predict cause of death and return a DataFrame with confidence statistics.

        Columns returned
        ----------------
        prediction              : Predicted cause of death (or 'out_of_distribution')
        pred_probability        : P(predicted class) from the model's softmax output
        pred_confidence_lower   : Lower bound of 95% Wilson CI on pred_probability
        pred_confidence_upper   : Upper bound of 95% Wilson CI on pred_probability
        pred_margin             : P(top-1) - P(top-2) — how decisively the class won
        pred_entropy            : Normalized Shannon entropy (0=certain, 1=maximally
                                  uncertain across all classes)
        pred_second_prediction  : Runner-up class (useful when margin is small)

        The Wilson CI uses per-class training counts as the effective sample size,
        so rarer classes produce wider intervals even at the same raw probability.
        """
        try:
            scaled, dk_ood_mask = self._prepare_features(cleaned_df)

            if hasattr(self.model, 'predict_proba'):
                probs       = self.model.predict_proba(scaled)          # (N, C)
                pred_idx    = np.argmax(probs, axis=1)                  # integer indices
                predictions = self.model.classes_[pred_idx]

                # Top-1 probability
                confidence = probs[np.arange(len(probs)), pred_idx]

                # Margin: P(top-1) - P(top-2)
                sorted_probs = np.sort(probs, axis=1)[:, ::-1]
                margin = sorted_probs[:, 0] - (sorted_probs[:, 1] if probs.shape[1] > 1
                                               else np.zeros(len(probs)))

                # Runner-up prediction
                second_idx        = np.argsort(probs, axis=1)[:, -2]
                second_pred_enc   = self.model.classes_[second_idx]
                second_prediction = self.label_encoder.inverse_transform(second_pred_enc)

                # Normalized Shannon entropy (0 = certain, 1 = uniform)
                eps          = 1e-12
                raw_entropy  = -np.sum(probs * np.log(probs + eps), axis=1)
                max_entropy  = np.log(probs.shape[1])
                norm_entropy = raw_entropy / max_entropy if max_entropy > 0 else raw_entropy

                # Wilson 95% CI — width reflects how many training examples the model
                # saw for this class (rarer class → wider interval)
                class_labels = self.label_encoder.classes_
                n_arr = np.array(
                    [self.training_class_counts.get(class_labels[i], 10) for i in predictions],
                    dtype=float,
                )
                lower, upper = self._wilson_ci(confidence, n_arr)

                # Entropy-based OOD (primary signal for new models):
                # Normalised entropy ∈ [0,1] — 0 = certain, 1 = uniform.
                # More robust than raw confidence because it measures spread
                # across ALL classes, not just the top-1 probability.
                if self.ood_entropy_threshold is not None:
                    ood_mask = norm_entropy > self.ood_entropy_threshold
                elif self.ood_threshold is not None:
                    # Fallback for old models without entropy threshold
                    ood_mask = confidence < self.ood_threshold
                else:
                    ood_mask = np.zeros(len(predictions), dtype=bool)

                if self.verbose:
                    print(
                        f"Confidence — min={confidence.min():.3f}, "
                        f"median={np.median(confidence):.3f}, "
                        f"max={confidence.max():.3f}"
                    )
                    if self.ood_entropy_threshold is not None:
                        print(
                            f"Entropy   — median={np.median(norm_entropy):.3f}, "
                            f"OOD threshold={self.ood_entropy_threshold:.3f} "
                            f"({ood_mask.sum()} conf-OOD)"
                        )
            else:
                predictions      = self.model.predict(scaled)
                nan_col          = np.full(len(predictions), np.nan)
                confidence       = nan_col.copy()
                margin           = nan_col.copy()
                norm_entropy     = nan_col.copy()
                lower            = nan_col.copy()
                upper            = nan_col.copy()
                second_prediction = np.full(len(predictions), '', dtype=object)
                ood_mask         = np.zeros(len(predictions), dtype=bool)

            final_ood_mask = ood_mask | dk_ood_mask.values
            decoded = self.label_encoder.inverse_transform(predictions)
            decoded[final_ood_mask] = 'out_of_distribution'
            valid   = set(self.original_classes) | {'out_of_distribution'}
            decoded[~np.isin(decoded, list(valid))] = 'out_of_distribution'

            if self.verbose:
                n_ood = final_ood_mask.sum()
                print(f"Predictions: {len(decoded) - n_ood} classified, {n_ood} OOD")

            notes = self._generate_notes(scaled, predictions, cleaned_df)

            return pd.DataFrame({
                'prediction':            decoded,
                'pred_probability':      np.round(confidence, 4),
                'pred_confidence_lower': np.round(lower, 4),
                'pred_confidence_upper': np.round(upper, 4),
                'pred_margin':           np.round(margin, 4),
                'pred_entropy':          np.round(norm_entropy, 4),
                'pred_second_prediction': second_prediction,
                'pred_notes':            notes,
            }, index=cleaned_df.index)

        except Exception as e:
            import traceback
            print(f"Prediction error: {e}")
            traceback.print_exc()
            nan_col = np.full(len(cleaned_df), np.nan)
            return pd.DataFrame({
                'prediction':            np.array(['PREDICTION_ERROR'] * len(cleaned_df)),
                'pred_probability':      nan_col,
                'pred_confidence_lower': nan_col,
                'pred_confidence_upper': nan_col,
                'pred_margin':           nan_col,
                'pred_entropy':          nan_col,
                'pred_second_prediction': np.full(len(cleaned_df), '', dtype=object),
                'pred_notes':            np.full(len(cleaned_df), '', dtype=object),
            }, index=cleaned_df.index)

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
