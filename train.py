# vman3_ml/train.py
# sentence_transformers/PyTorch and XGBoost both ship libomp; on macOS loading
# two OpenMP runtimes in the same process causes a SIGSEGV.  Setting both vars
# before any imports prevents the crash while keeping sklearn's joblib parallelism.
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('HF_HUB_VERBOSITY', 'error')

from vman_ml.processing import DataPreprocessor
from vman_ml.training import ModelTrainer
from vman_ml.mapcauselist import export_mapping_excel
from vman_ml.label_audit import LabelAuditor, build_feature_labels
import argparse
import pandas as pd
import numpy as np
import sys
import os
import json
from pathlib import Path
from sklearn.model_selection import train_test_split as _holdout_split

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    parser = argparse.ArgumentParser(
        description='CCVA Model Training — supports single or multiple input datasets.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single dataset (version auto-detected from filename)
  python train.py --input data/va_2016_tz.csv

  # Multiple datasets — trains one unified model across versions
  python train.py --input data/va_2016_tz.csv data/va_2022_ng.csv data/va_2022_es.csv

  # Override version for a single dataset
  python train.py --input data/va_2016_tz.csv --version 2016
""",
    )
    parser.add_argument('--input', required=True, nargs='+',
                        help='Path(s) to training CSV file(s). When multiple paths are '
                             'given a single unified model is trained on the combined data.')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--min_vc', type=int, default=130,
                        help='Minimum value counts for the target value')
    parser.add_argument('--na_threshold', type=float, default=0.7,
                        help='Threshold for dropping NA columns per cause group')
    parser.add_argument('--taxonomy_file',
                        help='Path to an explicit cause taxonomy JSON file')
    parser.add_argument('--audit_report', default='reports/training_audit_report.json',
                        help='Path to save the pre-training audit report')
    parser.add_argument('--version', choices=['2016', '2022'],
                        help='Override instrument version (only used for single-input training)')
    parser.add_argument('--target', default='pcva_who_cod',
                        help='Target column. Default: pcva_who_cod (WHO standardised cause). '
                             'Pass pcva_ucod to use raw cause labels.')
    parser.add_argument('--export-mapping', metavar='XLSX',
                        help='Export a WHO cause-mapping audit workbook before training '
                             '(e.g. reports/who_mapping.xlsx)')
    parser.add_argument('--label-audit', action='store_true',
                        help='Run cleanlab label quality audit after training and write '
                             'results to --audit-report directory.')
    parser.add_argument('--llm-review', action='store_true',
                        help='Send top-N cleanlab-flagged records to Claude API for a '
                             'second-opinion review (requires ANTHROPIC_API_KEY env var). '
                             'Only active when --label-audit is also set.')
    parser.add_argument('--audit-top-n', type=int, default=50,
                        help='Number of flagged records to send for LLM review (default 50).')
    parser.add_argument('--audit-cv-folds', type=int, default=5,
                        help='Cross-validation folds for cleanlab (default 5).')

    args = parser.parse_args()

    preprocessor = DataPreprocessor(
        verbose=args.verbose,
        min_vc=args.min_vc,
        na_threshold=args.na_threshold,
        taxonomy_path=args.taxonomy_file,
        instrument_version=args.version if len(args.input) == 1 else None,
    )

    # ------------------------------------------------------------------ #
    # Load and prepare data                                                #
    # ------------------------------------------------------------------ #
    if len(args.input) > 1:
        # Multi-dataset: combine into one aligned DataFrame
        print(f"Multi-dataset training: {len(args.input)} input files")
        combined_df, versions_used, union_feature_columns = preprocessor.combine_datasets(args.input)

        if args.export_mapping:
            mapping_path = Path(args.export_mapping)
            mapping_path.parent.mkdir(parents=True, exist_ok=True)
            export_mapping_excel(combined_df, output_path=str(mapping_path))

        X, y = preprocessor._prepare_training_data(
            combined_df,
            target_col=args.target,
            instrument_version='combined',
            preselected_feature_columns=union_feature_columns,
        )
        preprocessor.training_instrument_versions_ = versions_used
        preprocessor.union_feature_columns_        = union_feature_columns
        _source_df = combined_df   # kept for narrative embedding below

    else:
        # Single dataset
        df = preprocessor.load_data(args.input[0])
        df = preprocessor._preprocess_data(df)

        if args.export_mapping:
            mapping_path = Path(args.export_mapping)
            mapping_path.parent.mkdir(parents=True, exist_ok=True)
            export_mapping_excel(df, output_path=str(mapping_path))

        X, y = preprocessor._prepare_training_data(
            df,
            target_col=args.target,
            source_path=args.input[0],
            instrument_version=args.version,
        )
        preprocessor.training_instrument_versions_ = [preprocessor.instrument_version]
        preprocessor.union_feature_columns_        = preprocessor.final_training_columns
        _source_df = df   # kept for narrative embedding below

    # ------------------------------------------------------------------ #
    # Narrative embeddings                                                 #
    # ------------------------------------------------------------------ #
    if preprocessor.narrative_embedder is not None:
        emb_df = preprocessor.narrative_embedder.embed(_source_df)
        if emb_df is not None:
            # Align to rows that survived quality-filtering / clustering
            emb_aligned = emb_df.reindex(X.index).fillna(0.0)
            X = pd.concat([X, emb_aligned], axis=1)
            preprocessor.narrative_dims_       = emb_df.shape[1]
            preprocessor.final_training_columns = list(X.columns)
            print(f"Narrative embeddings added: {emb_df.shape[1]} dims")
        else:
            print("Narrative embedding skipped (no columns found or encoder unavailable).")
    else:
        print("Narrative embedding disabled (sentence-transformers not installed).")

    # ------------------------------------------------------------------ #
    # Audit report                                                         #
    # ------------------------------------------------------------------ #
    audit_report_path = Path(args.audit_report)
    audit_report_path.parent.mkdir(parents=True, exist_ok=True)
    audit_report = {
        'input_files':                 args.input,
        'target_column':               args.target,
        'instrument_versions':         preprocessor.training_instrument_versions_,
        'union_feature_count':         len(preprocessor.union_feature_columns_ or []),
        'taxonomy_file':               str(Path(preprocessor.taxonomy_path).resolve()) if preprocessor.taxonomy_path else None,
        'quality_report':              preprocessor.training_quality_report_,
        'training_audit_report':       preprocessor.training_audit_report_,
        'rare_label_mapping':          preprocessor.rare_label_mapping_,
    }
    with open(audit_report_path, 'w', encoding='utf-8') as handle:
        json.dump(audit_report, handle, indent=2, ensure_ascii=False)
    if args.verbose:
        print(f"Audit report saved to {audit_report_path}")

    # ------------------------------------------------------------------ #
    # Holdout split — 20% stratified, never seen during training          #
    # ------------------------------------------------------------------ #
    if isinstance(y, pd.DataFrame):
        y = y.squeeze()
    y = np.array(y).ravel()

    X_train_raw, X_holdout_raw, y_train_raw, y_holdout_raw = _holdout_split(
        X, y, test_size=0.2, stratify=y, random_state=0,
    )
    print(
        f"Hold-out split: {len(X_train_raw):,} train / "
        f"{len(X_holdout_raw):,} test ({100 * len(X_holdout_raw) / len(y):.0f}%)",
        flush=True,
    )

    # ------------------------------------------------------------------ #
    # Train                                                                #
    # ------------------------------------------------------------------ #
    trainer = ModelTrainer(verbose=args.verbose)
    trainer.train(X_train_raw, y_train_raw)
    trainer.save_model(preprocessor=preprocessor, version=preprocessor.instrument_version)

    # Internal val results (used for model selection during training)
    trainer.evaluate(trainer._X_test, trainer._y_test,
                     save_path='cv_results.json', label='Val')
    print()

    # ------------------------------------------------------------------ #
    # Final hold-out evaluation — 20% never touched during training       #
    # ------------------------------------------------------------------ #
    print('═' * 62, flush=True)
    print('  HOLD-OUT TEST RESULTS  (20% — never seen during training)', flush=True)
    print('═' * 62, flush=True)

    # Filter out any classes the trainer dropped (tiny classes in training set).
    # Use list (not set) for np.isin — numpy silently fails when test_elements is a set.
    known_classes = list(trainer.label_encoder.classes_)
    ho_mask = np.isin(y_holdout_raw, known_classes)
    if not ho_mask.all():
        n_excl = int((~ho_mask).sum())
        print(f"  Note: {n_excl} holdout record(s) excluded — class not in training set.")
        X_holdout_raw = X_holdout_raw.iloc[ho_mask] if hasattr(X_holdout_raw, 'iloc') else X_holdout_raw[ho_mask]
        y_holdout_raw = y_holdout_raw[ho_mask]

    X_holdout_scaled  = trainer.transform_features(X_holdout_raw)
    y_holdout_encoded = trainer.label_encoder.transform(y_holdout_raw)

    holdout_path = str(audit_report_path.parent / 'holdout_test_results.json')
    trainer.evaluate(X_holdout_scaled, y_holdout_encoded,
                     save_path=holdout_path, label='Hold-out')

    print('Training completed.')

    # ------------------------------------------------------------------ #
    # Label quality audit                                                  #
    # ------------------------------------------------------------------ #
    if args.label_audit:
        feature_labels = build_feature_labels(preprocessor)
        auditor = LabelAuditor(
            cv_folds  = args.audit_cv_folds,
            top_n_llm = args.audit_top_n,
            verbose   = args.verbose,
        )
        auditor.run(
            X_scaled           = trainer._X_all_scaled,
            y_encoded          = trainer._y_all,
            model              = trainer.best_model,
            label_encoder      = trainer.label_encoder,
            source_df          = _source_df,
            audit_source_index = trainer._audit_source_index,
            feature_labels     = feature_labels,
            output_dir         = str(audit_report_path.parent),
            run_llm            = args.llm_review,
            top_n_llm          = args.audit_top_n,
        )


if __name__ == '__main__':
    main()
