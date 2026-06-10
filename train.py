# vman3_ml/train.py
from ccva_ml.processing import DataPreprocessor
from ccva_ml.training import ModelTrainer
from ccva_ml.mapcauselist import export_mapping_excel
import argparse
import pandas as pd
import numpy as np
import sys
import os
import json
from pathlib import Path

# Add the parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    parser = argparse.ArgumentParser(description='CCVA Model Training')
    parser.add_argument('--input', required=True, help='Path to training CSV')
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
                        help='Override instrument version detection')
    parser.add_argument('--target', default='pcva_who_cod',
                        help='Target column for training. Default: pcva_who_cod '
                             '(WHO standardised cause mapped from pcva_ucod_icd). '
                             'Pass pcva_ucod to use the raw cause labels instead.')
    parser.add_argument('--export-mapping', metavar='XLSX',
                        help='Export a WHO cause-mapping audit workbook to this path '
                             '(e.g. reports/who_mapping.xlsx)')

    args = parser.parse_args()

    preprocessor = DataPreprocessor(
        verbose=args.verbose,
        min_vc=args.min_vc,
        na_threshold=args.na_threshold,
        taxonomy_path=args.taxonomy_file,
        instrument_version=args.version,
    )
    df = preprocessor.load_data(args.input)
    df = preprocessor._preprocess_data(df)

    # Optionally export WHO cause-mapping audit workbook before training
    if args.export_mapping:
        mapping_path = Path(args.export_mapping)
        mapping_path.parent.mkdir(parents=True, exist_ok=True)
        export_mapping_excel(df, output_path=str(mapping_path))

    X, y = preprocessor._prepare_training_data(
        df,
        target_col=args.target,
        source_path=args.input,
        instrument_version=args.version,
    )

    audit_report_path = Path(args.audit_report)
    audit_report_path.parent.mkdir(parents=True, exist_ok=True)
    audit_report = {
        'input_file': str(Path(args.input).resolve()),
        'target_column': args.target,
        'taxonomy_file': str(Path(preprocessor.taxonomy_path).resolve()) if preprocessor.taxonomy_path else None,
        'quality_report': preprocessor.training_quality_report_,
        'training_audit_report': preprocessor.training_audit_report_,
        'rare_label_mapping': preprocessor.rare_label_mapping_,
    }
    with open(audit_report_path, 'w', encoding='utf-8') as handle:
        json.dump(audit_report, handle, indent=2, ensure_ascii=False)

    if args.verbose:
        print(f"Saved pre-training audit report to {audit_report_path}")

    # Ensure y is a 1D array
    if isinstance(y, pd.DataFrame):
        y = y.squeeze()
    y = np.array(y).ravel()

    # Train model
    trainer = ModelTrainer(verbose=args.verbose)
    trainer.train(X, y)
    trainer.save_model(preprocessor=preprocessor, version=preprocessor.instrument_version)

    # Evaluate on the held-out test set captured during training
    trainer.evaluate(trainer._X_test, trainer._y_test, save_path='cv_results.json')

    print('Training completed.')

if __name__ == '__main__':
    main()
