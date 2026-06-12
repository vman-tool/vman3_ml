# vman3_ml/vman3_ml/processing.py
import pandas as pd
import numpy as np
from vman3_dq import change_null_toskipped
from sklearn.preprocessing import StandardScaler, LabelEncoder
import chardet
import re
import json
from pathlib import Path

from .instrument_dictionary import (
    detect_instrument_version,
    load_instrument_dictionary,
    load_instrument_dictionaries,
    normalize_column_name,
    normalize_target_aliases,
    select_feature_columns,
)
from .mapcauselist import map_causelist, map_ucod_text_to_who
from .narrative import NARRATIVE_COLS, NarrativeEmbedder, HAS_SENTENCE_TRANSFORMERS

class DataPreprocessor:
    def __init__(self, verbose=False, min_vc=130, na_threshold=0.7, use_quality_filter=True, taxonomy_path=None, instrument_version=None):
        self.verbose = verbose
        self.min_vc = min_vc
        self.na_threshold = na_threshold
        self.use_quality_filter = use_quality_filter
        self.taxonomy_path = Path(taxonomy_path) if taxonomy_path else Path(__file__).resolve().parent / 'data' / 'cause_taxonomy.json'
        self.taxonomy = self._load_cause_taxonomy(self.taxonomy_path)
        self.instrument_version = instrument_version
        self.instrument_dictionary = None
        self.instrument_detection_ = {}
        self.final_training_columns = None  # To store the final columns used in training
        self.training_quality_report_ = {}
        self.training_audit_report_ = {}
        self.rare_label_mapping_ = {}
        self.label_family_mapping_ = {}
        self.source_path = None
        self.training_instrument_versions_ = []
        self.union_feature_columns_ = None
        self.narrative_dims_ = 0
        self.narrative_embedder = (
            NarrativeEmbedder(verbose=verbose) if HAS_SENTENCE_TRANSFORMERS else None
        )

    def load_data(self, file_path):
        """Load and preprocess raw data"""
        self.source_path = str(file_path)
        with open(file_path, 'rb') as file:
            encoding = chardet.detect(file.read())['encoding']
        df = pd.read_csv(file_path, encoding=encoding, low_memory=False)
        #return self._preprocess_data(df)
        return df
    
    def _preprocess_data(self, df):
        """Internal preprocessing steps"""
        if self.verbose:
            print("Starting data preprocessing")
        
        # Drop unnecessary columns
        df = df.drop(columns=[col for col in df.columns if "_check" in col])

        # Clean data using vman3_dq
        df = change_null_toskipped(df, verbose=self.verbose)

        # Standardize column names
        df.columns = (
            df.columns
            .str.strip()  # First remove any whitespace
            .str.lower()  # Convert to lowercase
            .str.replace(r'[\s/]+', '_', regex=True)  # Replace spaces and slashes with underscore
            .str.replace(r'_+', '_', regex=True)  # Replace multiple underscores with single
            .str.replace(r'[^\w_]', '', regex=True)  # Remove any other special chars
            .str.strip('_')  # Remove leading/trailing underscores
        )
       
        clean_df = self._merge_duplicate_columns(df)
        clean_df = normalize_target_aliases(clean_df)
        clean_df = self._merge_duplicate_columns(clean_df)
        return clean_df

    def _load_instrument_context(self, df, source_path=None, instrument_version=None):
        """Detect and load the version-specific WHO VA instrument dictionary."""
        if instrument_version:
            detection = {"version": str(instrument_version), "scores": {}, "reason": "explicit"}
        else:
            detection = detect_instrument_version(
                df,
                source_path=source_path or self.source_path,
                dictionaries=load_instrument_dictionaries(),
            )

        version = str(detection["version"])
        self.instrument_version = version
        self.instrument_detection_ = detection
        self.instrument_dictionary = load_instrument_dictionary(version)
        return detection, self.instrument_dictionary

    def _get_feature_columns(self, df, instrument_dictionary=None):
        """Return the feature columns that are present in the dataframe for the detected instrument."""
        if instrument_dictionary:
            selected, missing = select_feature_columns(df, instrument_dictionary)
            if selected:
                if self.verbose and missing:
                    print(f"Instrument dictionary missing {len(missing)} feature columns in the input data")
                return selected, missing

        legacy_sets = self._get_feature_sets()
        selected = []
        for _, columns in legacy_sets.items():
            for column in columns:
                if column in df.columns and column not in selected:
                    selected.append(column)
        return selected, []
    
    def _apply_who_causelist_mapping(self, df):
        """Add WHO standardised cause columns, choosing the right mapping path.

        Priority:
          1. pcva_who_cod already present (e.g. NG pre-mapped data) → skip.
          2. pcva_ucod_icd present → ICD-based mapping via map_causelist().
          3. pcva_ucod present → text-label matching via map_ucod_text_to_who().
        """
        if 'pcva_who_cod' in df.columns:
            return df
        if 'pcva_ucod_icd' in df.columns:
            return map_causelist(df, icd_col='pcva_ucod_icd', verbose=self.verbose)
        if 'pcva_ucod' in df.columns:
            return map_ucod_text_to_who(df, ucod_col='pcva_ucod', verbose=self.verbose)
        if self.verbose:
            print("Warning: no ICD or ucod column found — skipping WHO cause mapping.")
        return df

    def align_to_version(self, df, source_version, target_version):
        """Project a DataFrame from source_version feature space into target_version feature space.

        - Drops columns that exist only in source_version (not in target)
        - Fills columns that exist only in target_version with 'dk' (string) or -999 (numeric)

        This does NOT harmonise cause labels — it only aligns predictor features.
        Use this for prediction robustness or cross-version experiments, not for
        mixing training labels across versions (their label schemas are incompatible).

        Returns the aligned DataFrame.
        """
        src_dict = load_instrument_dictionary(str(source_version))
        tgt_dict = load_instrument_dictionary(str(target_version))

        src_features = set(src_dict.get('feature_columns', []))
        tgt_features = set(tgt_dict.get('feature_columns', []))

        src_only = src_features - tgt_features
        tgt_only = tgt_features - src_features

        aligned = df.copy()

        # Drop source-only columns
        cols_to_drop = [c for c in aligned.columns if c in src_only]
        aligned = aligned.drop(columns=cols_to_drop)

        # Fill target-only columns with appropriate sentinel
        for col in sorted(tgt_only):
            if col not in aligned.columns:
                tgt_col_info = next(
                    (s for s in tgt_dict.get('survey', []) if s.get('name') == col), {}
                )
                col_type = tgt_col_info.get('type', 'select_one')
                if col_type in ('integer', 'decimal'):
                    aligned[col] = -999
                else:
                    aligned[col] = 'dk'

        if self.verbose:
            print(
                f"align_to_version {source_version}→{target_version}: "
                f"dropped {len(cols_to_drop)} source-only cols, "
                f"filled {len(tgt_only)} target-only cols with sentinels"
            )
        return aligned

    def combine_datasets(
        self,
        dataset_specs,
    ) -> tuple[pd.DataFrame, list[str], list[str]]:
        """Load, preprocess, WHO-map and feature-align multiple datasets.

        Each dataset is processed with its own instrument version (auto-detected
        from filename / column overlap).  Feature columns from all versions are
        unioned; records missing a version-specific column receive the 'dk'
        sentinel so the model can learn the absence pattern.

        Args:
            dataset_specs: iterable of paths (str/Path) OR dicts with keys
                           'path' (required) and 'version' (optional override).

        Returns:
            combined_df:           All records stacked with union feature set.
            versions_used:         Deduplicated ordered list of detected versions.
            union_feature_columns: Ordered union of feature columns across all datasets.
        """
        records: list[tuple[pd.DataFrame, str, list[str]]] = []
        versions_seen: list[str] = []

        for spec in dataset_specs:
            if isinstance(spec, (str, Path)):
                path, version_hint = str(spec), None
            else:
                path = str(spec['path'])
                version_hint = spec.get('version')

            df = self.load_data(path)
            df = self._preprocess_data(df)

            detection, instrument_dictionary = self._load_instrument_context(
                df, source_path=path, instrument_version=version_hint,
            )
            version = detection['version']
            versions_seen.append(version)

            # WHO target mapping — handles ICD (TZ/NG), text labels (ES), pre-mapped (NG)
            df = self._apply_who_causelist_mapping(df)

            feature_columns, _ = self._get_feature_columns(df, instrument_dictionary)

            # Carry feature columns + WHO passthrough + narrative text side-by-side
            passthrough = [
                c for c in ['pcva_who_cod', 'pcva_ucod_icd',
                             'pcva_who_major', 'pcva_who_broad', 'pcva_who_id']
                            + NARRATIVE_COLS
                if c in df.columns
            ]
            keep = list(dict.fromkeys(feature_columns + passthrough))
            records.append((df[keep].copy(), version, feature_columns))

            if self.verbose:
                print(f"  Loaded {path}: version={version}, "
                      f"{len(df)} rows, {len(feature_columns)} features")

        # Ordered union of feature columns (first-seen version comes first)
        union_features: list[str] = []
        seen_cols: set[str] = set()
        for _, _, feat_cols in records:
            for col in feat_cols:
                if col not in seen_cols:
                    seen_cols.add(col)
                    union_features.append(col)

        # Align each dataset to the full union (fill absent version cols with 'dk')
        aligned: list[pd.DataFrame] = []
        for df_slice, _, _ in records:
            missing = [c for c in union_features if c not in df_slice.columns]
            if missing:
                df_slice = df_slice.copy()
                for col in missing:
                    df_slice[col] = 'dk'
            aligned.append(df_slice)

        combined_df   = pd.concat(aligned, axis=0, ignore_index=True)
        versions_used = list(dict.fromkeys(versions_seen))

        if self.verbose:
            print(f"combine_datasets: {len(records)} datasets "
                  f"({', '.join(versions_used)}) → "
                  f"{len(combined_df)} rows, {len(union_features)} union features")

        return combined_df, versions_used, union_features

    def _prepare_training_data(self, df, target_col='pcva_who_cod', source_path=None, instrument_version=None,
                               preselected_feature_columns=None):
        """Prepare features and optionally target for training/prediction.

        Args:
            df:                          Input DataFrame.
            target_col:                  Target column name. Default 'pcva_who_cod'.
                                         Pass None for prediction mode.
            source_path:                 Path hint used for instrument version detection.
            instrument_version:          Override instrument version ('2016', '2022', or
                                         'combined' when multi-dataset training).
            preselected_feature_columns: When provided (multi-dataset mode), skip
                                         instrument detection and feature selection and
                                         use this list directly.  combine_datasets()
                                         passes the union feature set here.

        Returns:
            (X, y) in training mode; (X, None) in prediction mode.
        """
        df = normalize_target_aliases(df.copy())
        target_col = normalize_column_name(target_col) if target_col else None

        if preselected_feature_columns is not None:
            # Multi-dataset combined mode — features already aligned by combine_datasets()
            feature_columns          = [c for c in preselected_feature_columns if c in df.columns]
            missing_dictionary_columns = []
            instrument_dictionary    = None
            if instrument_version:
                self.instrument_version = str(instrument_version)
            self.instrument_detection_ = {
                'version': self.instrument_version or 'combined',
                'reason':  'preselected_combined',
                'scores':  {},
            }
            detection = self.instrument_detection_
        else:
            detection, instrument_dictionary = self._load_instrument_context(
                df,
                source_path=source_path,
                instrument_version=instrument_version,
            )
            feature_columns, missing_dictionary_columns = self._get_feature_columns(df, instrument_dictionary)
            if not feature_columns:
                raise ValueError("No usable feature columns found for the detected instrument")

        # In training mode, map ICD codes to WHO standardised causes before anything else.
        # _apply_who_causelist_mapping is a no-op when pcva_who_cod already exists (combined mode).
        if target_col is not None:
            df = self._apply_who_causelist_mapping(df)

        dfs = [df[feature_columns]]

        # Non-feature columns carried through the pipeline for QC and audit
        # (dropped from X before returning)
        _NON_FEATURE = ['pcva_ucod_icd', 'pcva_who_major', 'pcva_who_broad', 'pcva_who_id']
        passthrough_cols = [
            col for col in _NON_FEATURE
            if col in df.columns and col != target_col
        ]
        # Keep pcva_ucod_icd as quality_col for the ICD-consistency check
        quality_columns = [col for col in ['pcva_ucod_icd'] if col in passthrough_cols]

        if target_col is not None:
            if target_col not in df.columns:
                raise ValueError(
                    f"Target column '{target_col}' not found in data. "
                    "If training with WHO standardised causes, ensure pcva_ucod_icd is present "
                    "so map_causelist() can create 'pcva_who_cod' automatically."
                )
            full_df = pd.concat(dfs + [df[[target_col]]], axis=1)
            if passthrough_cols:
                full_df = pd.concat([full_df, df[passthrough_cols]], axis=1)
        else:
            full_df = pd.concat(dfs, axis=1)

        if target_col is not None:
            clean_df, quality_report = self._apply_training_quality_filter(
                full_df,
                target_col=target_col,
                quality_col=quality_columns[0] if quality_columns else None,
            )
            self.training_quality_report_ = quality_report

            clean_df, rare_mapping, cluster_report = self._cluster_rare_causes(clean_df, target_col=target_col)
            self.rare_label_mapping_ = rare_mapping
            self.training_audit_report_ = self._build_training_audit_report(
                initial_rows=len(full_df),
                quality_report=quality_report,
                quality_details=self.training_quality_report_.get('details', {}),
                cluster_report=cluster_report,
                final_rows=len(clean_df),
                instrument_report={
                    'detected_version': detection.get('version'),
                    'detection_reason': detection.get('reason'),
                    'version_scores': detection.get('scores', {}),
                    'survey_columns': len(instrument_dictionary.get('survey', [])) if instrument_dictionary else 0,
                    'feature_columns_used': feature_columns,
                    'missing_dictionary_columns': missing_dictionary_columns,
                    'source_file': instrument_dictionary.get('source_file') if instrument_dictionary else None,
                },
            )

            clean_df = self._dropcols_by_threshold(clean_df, self.na_threshold, target_col=target_col)
            
            if self.verbose:
                print(f"Before dropping NA\n{clean_df[target_col].value_counts()}") 
        else:
            clean_df = full_df.copy()

        # Handle missing values
        for col in clean_df.columns:
            if pd.api.types.is_string_dtype(clean_df[col]):
                clean_df[col] = clean_df[col].fillna('dk').replace({'':'dk'})
            else:
                clean_df[col] = clean_df[col].fillna(-999)
        
        if target_col is not None and self.verbose:
            print(f"Training dataset contains the following causes\n{clean_df[target_col].value_counts()}") 

        # check for duplicates
        duplicates = clean_df.columns.duplicated()
        if duplicates.any():
                print(f"Warning: Found {duplicates.sum()} duplicates while preparing training data:")
                print(clean_df.columns[duplicates].tolist())
                # Remove duplicates while preserving order
                clean_df = clean_df.loc[:, ~duplicates]


        if target_col is not None:
            # Drop target, ICD quality col, and all WHO meta cols — none are features
            drop_columns = [
                col for col in [target_col] + passthrough_cols
                if col in clean_df.columns
            ]
            self.final_training_columns = list(dict.fromkeys(clean_df.drop(columns=drop_columns).columns))
            y = clean_df[target_col].copy()
            X = clean_df.drop(columns=drop_columns)
        else:
            self.final_training_columns = list(dict.fromkeys(clean_df.columns))
            X = clean_df.copy()
            y = None

        return X, y

    def _load_cause_taxonomy(self, taxonomy_path):
        """Load the explicit cause-family taxonomy from disk."""
        default_taxonomy = {
            'default_cluster': 'cluster_other',
            'unknown_patterns': ['unknown', 'unspecified', 'ill-defined', 'ill defined', 'symptom', 'signs and symptoms', 'cause of death unknown', 'not stated'],
            'families': []
        }

        if taxonomy_path and Path(taxonomy_path).exists():
            with open(taxonomy_path, 'r', encoding='utf-8') as handle:
                taxonomy = json.load(handle)
            taxonomy.setdefault('default_cluster', 'cluster_other')
            taxonomy.setdefault('unknown_patterns', [])
            taxonomy.setdefault('families', [])
            return taxonomy

        return default_taxonomy

    def _apply_training_quality_filter(self, df, target_col='pcva_ucod', quality_col=None):
        """Filter obvious low-quality labels and flag inconsistencies against ICD chapter knowledge."""
        working = df.copy()
        report = {
            'dropped_missing_or_unknown': 0,
            'qc_comparable_rows': 0,
            'qc_agree_rows': 0,
            'qc_mismatch_rows': 0,
            'qc_kept_rows': 0,
            'qc_dropped_rows': 0,
            'details': {},
        }

        label_series = working[target_col].astype(str).str.strip()
        unknown_patterns = self.taxonomy.get('unknown_patterns', [])
        invalid_mask = (
            label_series.eq('') |
            label_series.str.lower().isin({'nan', 'none'}) |
            label_series.str.contains('|'.join(re.escape(pattern) for pattern in unknown_patterns), case=False, na=False)
        )
        report['dropped_missing_or_unknown'] = int(invalid_mask.sum())
        if invalid_mask.any():
            report['details']['filtered_missing_or_unknown'] = (
                working.loc[invalid_mask, target_col]
                .value_counts()
                .rename_axis('cause')
                .reset_index(name='count')
                .to_dict('records')
            )
        working = working.loc[~invalid_mask].copy()

        if quality_col and quality_col in working.columns:
            icd_version = self._detect_icd_version(working[quality_col])
            if self.verbose:
                print(f"Detected ICD version for quality filter: ICD-{icd_version}")
            label_family = working[target_col].apply(self._infer_cause_family_from_label)
            icd_family = working[quality_col].apply(
                lambda c: self._infer_icd_family(c, icd_version=icd_version)
            )
            self.label_family_mapping_ = pd.DataFrame({
                target_col: working[target_col],
                'label_family': label_family,
                'icd_family': icd_family,
            }, index=working.index).to_dict('index')

            comparable = (
                label_family.notna() &
                icd_family.notna() &
                ~label_family.isin({'other', 'review'}) &
                ~icd_family.isin({'other', 'unknown'})
            )
            matched = comparable & (label_family == icd_family)
            mismatched = comparable & ~matched

            report['qc_comparable_rows'] = int(comparable.sum())
            report['qc_agree_rows'] = int(matched.sum())
            report['qc_mismatch_rows'] = int(mismatched.sum())

            if mismatched.any():
                mismatch_details = (
                    working.loc[mismatched, [target_col, quality_col]]
                    .assign(label_family=label_family[mismatched], icd_family=icd_family[mismatched])
                    .groupby([target_col, 'label_family', 'icd_family'], dropna=False)
                    .size()
                    .reset_index(name='count')
                    .sort_values('count', ascending=False)
                )
                report['details']['inconsistent_causes'] = mismatch_details.to_dict('records')

            if self.use_quality_filter and mismatched.any():
                working = working.loc[~mismatched].copy()

            report['qc_kept_rows'] = int(len(working))
            report['qc_dropped_rows'] = int(len(df) - len(working))

        return working, report

    def _build_training_audit_report(self, initial_rows, quality_report, quality_details, cluster_report, final_rows, instrument_report=None):
        """Build a pre-training audit report that explains how records were filtered or clustered."""
        report = {
            'input_rows': int(initial_rows),
            'rows_after_quality_filter': int(quality_report.get('qc_kept_rows', 0)),
            'rows_after_clustering': int(cluster_report.get('rows_after_clustering', 0)),
            'rows_final_for_training': int(final_rows),
            'missing_or_unknown_labels': int(quality_report.get('dropped_missing_or_unknown', 0)),
            'quality_mismatch_rows': int(quality_report.get('qc_mismatch_rows', 0)),
            'quality_agree_rows': int(quality_report.get('qc_agree_rows', 0)),
            'clustered_rare_labels': int(cluster_report.get('clustered_label_count', 0)),
            'filtered_causes': quality_details.get('filtered_missing_or_unknown', []),
            'inconsistent_causes': quality_details.get('inconsistent_causes', []),
            'clustered_causes': cluster_report.get('clustered_causes', []),
        }

        if instrument_report:
            report['instrument_context'] = instrument_report

        return report

    @staticmethod
    def _detect_icd_version(icd_series):
        """Return '10' or '11' based on the dominant coding pattern in the series."""
        sample = icd_series.dropna().astype(str).head(50)
        icd10_matches = sample.str.match(r'^[A-Z]\d{2}').sum()
        return '10' if icd10_matches / max(len(sample), 1) > 0.5 else '11'

    def _infer_icd_family(self, icd_code, icd_version=None):
        """Map an ICD code to a broad disease family.

        Accepts ICD-10 (e.g. 'A09', 'I50') and ICD-11 (e.g. '1F40.Y', 'KB21.0').
        icd_version is inferred from the code format when not supplied.
        """
        if pd.isna(icd_code):
            return None
        code = str(icd_code).strip().upper()
        if not code or code in {'NAN', 'NONE'}:
            return None

        # Auto-detect if not provided
        if icd_version is None:
            icd_version = '10' if re.match(r'^[A-Z]\d{2}', code) else '11'

        if icd_version == '10':
            return self._infer_icd10_family(code)
        return self._infer_icd11_family(code)

    def _infer_icd10_family(self, code):
        """Map an ICD-10 code to a broad disease family."""
        match = re.match(r'^([A-Z])(\d{2})', code)
        if not match:
            return None
        letter = match.group(1)
        chapter_number = int(match.group(2))

        if letter in {'A', 'B'}:
            return 'infectious'
        if letter == 'C' or (letter == 'D' and chapter_number <= 48):
            return 'neoplasms'
        if letter == 'D' and chapter_number >= 50:
            return 'blood'
        if letter == 'E':
            return 'endocrine'
        if letter == 'F':
            return 'mental'
        if letter == 'G':
            return 'neurological'
        if letter == 'H':
            return 'sensory'
        if letter == 'I':
            return 'cardiovascular'
        if letter == 'J':
            return 'respiratory'
        if letter == 'K':
            return 'digestive'
        if letter == 'L':
            return 'skin'
        if letter == 'M':
            return 'musculoskeletal'
        if letter == 'N':
            return 'genitourinary'
        if letter == 'O':
            return 'maternal'
        if letter == 'P':
            return 'perinatal'
        if letter == 'Q':
            return 'congenital'
        if letter == 'R':
            return 'symptoms'
        if letter in {'S', 'T', 'V', 'W', 'X', 'Y'}:
            return 'injury'
        if letter == 'U':
            return 'other'
        return None

    def _infer_icd11_family(self, code):
        """Map an ICD-11 code to a broad disease family.

        ICD-11 chapter prefixes (first character):
          1-9  → chapters 1-9 (infectious, neoplasms, blood, immune, endocrine,
                                mental, sleep, neurological, visual)
          A    → ear/mastoid
          B    → cardiovascular
          C    → respiratory
          D    → digestive
          E    → skin
          F    → musculoskeletal
          G    → genitourinary
          H    → sexual health
          J    → maternal/obstetric
          K    → perinatal
          L    → congenital/developmental
          M    → symptoms/signs
          N    → injury
          P    → external causes
        """
        _ICD11_MAP = {
            '1': 'infectious',   '2': 'neoplasms',      '3': 'blood',
            '4': 'immune',       '5': 'endocrine',       '6': 'mental',
            '7': 'other',        '8': 'neurological',    '9': 'sensory',
            'A': 'sensory',      'B': 'cardiovascular',  'C': 'respiratory',
            'D': 'digestive',    'E': 'skin',             'F': 'musculoskeletal',
            'G': 'genitourinary','H': 'other',            'J': 'maternal',
            'K': 'perinatal',    'L': 'congenital',       'M': 'symptoms',
            'N': 'injury',       'P': 'injury',           'Q': 'other',
            'X': 'other',
        }
        if not code:
            return None
        return _ICD11_MAP.get(code[0])

    def _infer_cause_family_from_label(self, label):
        """Map a verbal autopsy cause label to a broad training family."""
        text = str(label).strip().lower()
        if not text or text in {'nan', 'none'}:
            return None

        if any(pattern in text for pattern in self.taxonomy.get('unknown_patterns', [])):
            return 'review'

        for family_rule in self.taxonomy.get('families', []):
            family = family_rule.get('name')
            keywords = family_rule.get('keywords', [])
            if any(keyword in text for keyword in keywords):
                return family

        return 'other'

    def _who_cluster_label(self, cause_label: str, who_meta: dict) -> str:
        """Derive a cluster label for a rare cause using the WHO hierarchy.

        Priority:
          1. pcva_who_major  (e.g. "Infectious and parasitic diseases" → cluster_infectious_and_parasitic_diseases)
          2. pcva_who_broad  (e.g. "Communicable" → cluster_communicable)
          3. Text-keyword fallback via _infer_cause_family_from_label
        """
        meta = who_meta.get(cause_label, {})
        for key in ('pcva_who_major', 'pcva_who_broad'):
            val = str(meta.get(key, '') or '').strip()
            if val and val.lower() not in ('nan', 'none', ''):
                slug = re.sub(r'[^a-z0-9]+', '_', val.lower()).strip('_')
                return f'cluster_{slug}'

        family = self._infer_cause_family_from_label(cause_label)
        if family in {None, 'review'}:
            family = 'other'
        return f'cluster_{family}'

    def _cluster_rare_causes(self, df, target_col='pcva_who_cod'):
        """Collapse sparse causes into WHO-hierarchy cluster groups before training.

        Uses pcva_who_major / pcva_who_broad (if present in df) to derive cluster
        labels, so clusters align with the same WHO cause hierarchy used for mapping.
        Falls back to text-keyword inference when WHO metadata is not available.
        """
        if target_col not in df.columns:
            return df.copy(), {}, {'clustered_label_count': 0, 'clustered_causes': [], 'rows_after_clustering': len(df)}

        working = df.copy()
        counts = working[target_col].value_counts(dropna=False)
        rare_labels = [label for label in counts[counts < self.min_vc].index if pd.notna(label)]

        if not rare_labels:
            return working, {}, {'clustered_label_count': 0, 'clustered_causes': [], 'rows_after_clustering': len(working)}

        # Build a per-cause WHO metadata lookup (one row per unique cause label)
        who_meta_cols = [c for c in ('pcva_who_major', 'pcva_who_broad') if c in working.columns]
        if who_meta_cols:
            who_meta = (
                working[[target_col] + who_meta_cols]
                .dropna(subset=[target_col])
                .drop_duplicates(subset=[target_col])
                .set_index(target_col)
                .to_dict('index')
            )
        else:
            who_meta = {}

        cluster_map = {}
        cluster_summary = []
        for label in sorted(rare_labels):
            cluster_label = self._who_cluster_label(str(label), who_meta)
            cluster_map[str(label)] = cluster_label
            cluster_summary.append({
                'cause': str(label),
                'count': int(counts.get(label, 0)),
                'cluster': cluster_label,
            })
            working.loc[working[target_col] == label, target_col] = cluster_label

        # Second sweep: any label that still has < 2 records after clustering
        # (e.g. a cluster that received only one rare cause with one record)
        # cannot be used in a stratified split — merge into cluster_other.
        post_counts = working[target_col].value_counts(dropna=False)
        tiny_labels = [label for label in post_counts[post_counts < 2].index if pd.notna(label)]
        for label in tiny_labels:
            cluster_map[str(label)] = 'cluster_other'
            working.loc[working[target_col] == label, target_col] = 'cluster_other'
            cluster_summary.append({
                'cause': str(label),
                'count': int(post_counts.get(label, 0)),
                'cluster': 'cluster_other',
                'reason': 'post-cluster singleton',
            })

        if self.verbose:
            print("Rare cause clustering applied (WHO hierarchy):")
            for original, cluster_label in list(cluster_map.items())[:20]:
                print(f"  {original!r} → {cluster_label}")
            print(f"  Classes after clustering: {working[target_col].nunique()}")

        return working, cluster_map, {
            'clustered_label_count': len(cluster_map),
            'clustered_causes': cluster_summary,
            'rows_after_clustering': len(working),
        }
    
    
        # 'Malaria': ['id10077','id10126','id10127','id10128','id10130','id10131','id10133','id10134','id10135','id10136','id10137',
        #         'id10142','id10143','id10144','id10148','id10149','id10152','id10153','id10159','id10166','id10168','id10173',
        #         'id10174','id10181','id10186','id10188','id10189','id10193','id10194','id10200','id10204','id10207','id10208',
        #         'id10210','id10214','id10219','id10223','id10227','id10230','id10233','id10238','id10241','id10243','id10244',
        #         'id10245','id10246','id10247','id10249','id10252','id10253','id10258','id10261','id10264','id10265','id10267',
        #         'id10268','id10270']

        # 'HIV/AIDS' : ['id10126','id10127','id10128','id10129','id10130','id10131','id10132','id10133','id10134','id10135','id10136',
        #               'id10137','id10138','id10139','id10140','id10141','id10142','id10143','id10144','id10152','id10153','id10159',
        #               'id10166','id10168','id10173','id10174','id10181','id10186','id10188','id10189','id10193','id10194','id10200',
        #               'id10204','id10207','id10208','id10210','id10212','id10214','id10219','id10223','id10227','id10228','id10230',
        #               'id10233','id10237','id10238','id10241','id10243','id10244','id10245','id10246','id10249','id10252','id10253',
        #               'id10258','id10261','id10264','id10265','id10267','id10268','id10270']

    def _get_feature_sets(self):
        """Define feature groupings"""
        return {
            'demographic': ['id10019','id10059','id10063','id10064','id10065','ageinyears', 'age_group', 'isneonatal','ischild', 'isadult'],
            'accident_injuries': ['id10077', 'id10079', 'id10080', 'id10081','id10082', 'id10083', 'id10084', 'id10085', 'id10086', 
                                  'id10087', 'id10088', 'id10089', 'id10090','id10091', 'id10092', 'id10093', 'id10094', 'id10095', 
                                  'id10096', 'id10097', 'id10098', 'id10099','id10100'],
            'medical_history': ['id10123','id10125', 'id10126', 'id10127', 'id10128','id10129', 'id10130', 'id10131', 'id10132', 'id10133',
                                'id10134', 'id10135', 'id10136', 'id10137','id10138', 'id10139', 'id10140', 'id10141', 'id10142',
                                'id10143', 'id10144','id10148','id10149'],
            'general_symptoms': ['id10147', 'id10148_a','id10173_a', 'id10173', 'id10174','id10181', 'id10186', 'id10188', 'id10189', 'id10193',
                                 'id10194', 'id10195','id10200', 'id10204', 'id10207','id10208', 'id10210', 'id10212', 'id10214', 'id10219',
                                 'id10223', 'id10227', 'id10228', 'id10230','id10233', 'id10237', 'id10238', 'id10241', 'id10243',
                                 'id10244', 'id10245', 'id10246','id10247', 'id10249', 'id10252', 'id10253', 'id10258', 'id10261',
                                 'id10264', 'id10265','id10267', 'id10268', 'id10270', 'id10295', 'id10296', 'id10304', 'id10305',
                                 'id10306','id10307', 'id10310','id10152','id10153','id10158','id10159','id10162','id10163','id10161',
                                 'id10166','id10168','id10182','id10183','id10185','id10187','id10191','id10192','id10199','id10201','id10202'],
            'risk_factors': ['id10411', 'id10412', 'id10413','id10414','id10415','id10416'],
            'hs_utilization': ['id10418', 'id10419', 'id10420', 'id10421', 'id10422', 'id10423', 'id10424','id10425','id10426', 'id10427',
                               'id10428','id10429','id10430','id10431','id10432','id10433','id10435','id10437','id10438','id10445','id10446'],
            'background_context':['id10450','id10451','id10452','id10453','id10454', 'id10455', 'id10456', 'id10457', 'id10458', 'id10459']
        }
    
    def _droprows_by_value_counts(self, df, column, threshold):
        """Filter dataframe by value counts in specified column"""
        vc = df[column].value_counts()
        return df[df[column].isin(vc[vc > threshold].index)] 
    

    def _drop_na_columns(self, dataframe:pd.DataFrame, th:float=0.7):
        return dataframe.dropna(thresh = th * len(dataframe), axis=1)  # Drop columns with more than 70% NA values

    def _dropcols_by_threshold(self, df, th:float=0.7, target_col='pcva_ucod'):
        cod_dfs = {cod: df[df[target_col] == cod] for cod in df[target_col].dropna().unique()}

        temp_df = []
        for cod in cod_dfs:
            cod_na_dropped = self._drop_na_columns(cod_dfs[cod], th)
            if self.verbose:
                print(f"Dataframe: {cod}, Shape before dropping NA: {cod_dfs[cod].shape}, Shape after dropping NA: {cod_na_dropped.shape}")
            temp_df.append(cod_na_dropped)
        return pd.concat(temp_df, axis=0)
    
    def _encode_features(self, X):
        """Encode categorical features with robust 1D handling"""
        # Store original index to ensure consistent length
        original_index = X.index if hasattr(X, 'index') else pd.RangeIndex(len(X))
        
        # Ensure we're working with a DataFrame
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
        # Make a copy to avoid modifying original data
        X = X.copy()
        
        # Process each categorical column
        encoders = {}
        encoded_dfs = []
        
        for col in X.select_dtypes(include=['object']).columns:
            try:
                # Convert column to 1D Series
                col_data = X[col]
                if isinstance(col_data, pd.DataFrame):  # Handle case where column is 2D
                    col_data = col_data.iloc[:, 0]  # Take first column
                
                # Standardize missing values
                col_data = col_data.astype(str).replace(['nan', '', ' '], 'dk')
                
                # Get unique values safely
                unique_vals = pd.Series(col_data).unique()
                
                # Handle yes/no/dk columns specially
                if set(unique_vals).issubset({'yes', 'no', 'dk'}):
                    mapping = {'yes': 1, 'no': 0, 'dk': -1}
                    encoded = col_data.map(mapping)
                    encoders[col] = mapping
                else:
                    le = LabelEncoder()
                    encoded = le.fit_transform(col_data)
                    encoders[col] = le
                    
                encoded_dfs.append(pd.DataFrame({col: encoded}, index=original_index))
                
            except Exception as e:
                if self.verbose:
                    print(f"Error encoding column {col}: {str(e)}")
                # Fallback to simple numeric encoding
                col_data = X[col].iloc[:, 0] if isinstance(X[col], pd.DataFrame) else X[col]
                encoded = pd.factorize(col_data.astype(str))[0]
                encoders[col] = None
                encoded_dfs.append(pd.DataFrame({col: encoded}, index=original_index))
        
        # Combine features while preserving original index
        numeric_df = X.select_dtypes(exclude=['object'])
        full_df = pd.concat([numeric_df] + encoded_dfs, axis=1)
        
        # Verify we maintained the same number of samples
        if len(full_df) != len(original_index):
            raise ValueError(
                f"Encoding changed number of samples from {len(original_index)} to {len(full_df)}"
            )
        
        return full_df, encoders
    
    def _scale_features(self, X, fit=True):
        """Scale features while preserving column names.

        fit=True (default) for training; fit=False to transform with an already-fitted scaler.
        """
        if not hasattr(self, 'scaler'):
            self.scaler = StandardScaler()

        if isinstance(X, np.ndarray):
            if not hasattr(self, 'feature_names_in_'):
                raise ValueError("Can't scale numpy array without feature names")
            X = pd.DataFrame(X, columns=self.feature_names_in_)

        X_scaled = self.scaler.fit_transform(X) if fit else self.scaler.transform(X)

        scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
        duplicates = scaled_df.columns.duplicated()
        if duplicates.any():
            print(f"Warning: The following columns were found/created and removed while preparing the training dataset: {scaled_df.columns[duplicates].tolist()}")
            scaled_df = scaled_df.loc[:, ~duplicates]

        return scaled_df, self.scaler
    
    def _encode_target(self, y):
        """Encode target variable"""
        if len(y.shape) > 1:
            raise ValueError("Target variable y must be 1-dimensional. Got shape {}".format(y.shape))
    
        le = LabelEncoder()
        try:
            y_encoded = le.fit_transform(y)
            self.classes_ = le.classes_  # Store the original class labels
            self.class_mapping = dict(zip(y_encoded, y))
            return y_encoded, le
        except Exception as e:
            raise ValueError(f"Error encoding target variable: {str(e)}")
        
    def _validate_input_data(self, df):
        """Comprehensive data quality checks before prediction"""
        errors = []
        warnings = []
        
        # 1. Check column presence
        missing_cols = set(self.final_training_columns) - set(df.columns)
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

    def _merge_duplicate_columns(self, df):
        """
        Merge duplicate columns in a DataFrame by column name, keeping non-empty content,
        and return a DataFrame with unique column names.
        
        Returns:
            DataFrame with duplicates merged
            Dictionary of merge operations performed
        """
        # Make a copy to avoid changing the original
        df_clean = df.copy()
        duplicates_report = {}
        
        # Find duplicate columns (case-sensitive)
        col_counts = df_clean.columns.value_counts()
        duplicate_cols = col_counts[col_counts > 1].index.tolist()
        
        for col in duplicate_cols:
            # Find all columns with this duplicate name
            matching_cols = [c for c in df_clean.columns if c == col]
            
            if len(matching_cols) > 1:
                # Create merged column (first non-null value across duplicates)
                merged = (
                    df_clean[matching_cols]
                    .astype(str)
                    .replace({'nan': None, '': None})
                    .bfill(axis=1)
                    .iloc[:, 0]
                )
                
                # Record merge operation
                duplicates_report[col] = {
                    'kept': col,
                    'dropped': matching_cols[1:],
                    'action': 'merged with first non-empty value kept'
                }
                
                # Remove all duplicates and add merged column
                df_clean = df_clean.drop(columns=matching_cols)
                df_clean[col] = merged
        
        # Optional verbose reporting
        if self.verbose and duplicates_report:
            print(f"Merged {len(duplicates_report)} duplicate column groups")
            for col, info in duplicates_report.items():
                print(f"- Kept '{info['kept']}', dropped {info['dropped']}")
        
        return df_clean

        
    