# vman3_ml/predict.py
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('HF_HUB_VERBOSITY', 'error')

from ccva_ml.processing import DataPreprocessor
from ccva_ml.prediction import CCVAPredictor
from ccva_ml.instrument_dictionary import detect_instrument_version
import pandas as pd
import numpy as np
import sys
import os
import argparse
from pathlib import Path

# Add the parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    parser = argparse.ArgumentParser(description='CCVA Model Prediction')
    parser.add_argument("--model", required=True, help="Path to trained model file or model directory")
    parser.add_argument("--input", required=True, help="Path to prediction CSV")
    parser.add_argument("--output", help="Output file for predictions")
    parser.add_argument("--report", help="Run data quality report (use 'dqr')")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--ood_threshold", type=float, default=None,
                        help="Override the model's OOD confidence threshold (0–1). "
                             "Only applies when the model has no entropy threshold "
                             "(old models). Lower = fewer OOD flags.")
    parser.add_argument("--dk-threshold", type=float, default=None,
                        help="Override the DK-missingness OOD threshold (0–1). "
                             "Records where more than this fraction of feature columns "
                             "are 'dk'/missing are flagged OOD. Default: 0.60. "
                             "Set to 1.0 to disable DK-based OOD entirely.")

    args = parser.parse_args()

    preprocessor = DataPreprocessor(verbose=args.verbose)
    df = preprocessor.load_data(args.input)
    df = preprocessor._preprocess_data(df)
    detection = detect_instrument_version(df, source_path=args.input)

    model_path = Path(args.model)
    if model_path.is_dir():
        # Directory: prefer combined model, fall back to version-specific, then generic
        combined_path  = model_path / "ccva_model_combined.pkl"
        versioned_path = model_path / f"ccva_model_{detection['version']}.pkl"
        generic_path   = model_path / "ccva_model.pkl"
        if combined_path.exists():
            model_path = combined_path
        elif versioned_path.exists():
            model_path = versioned_path
        else:
            model_path = generic_path
    # If the user passed an explicit file path, use it exactly as specified.

    predictor = CCVAPredictor(str(model_path), verbose=args.verbose)

    if args.ood_threshold is not None:
        if not (0 < args.ood_threshold < 1):
            print(f"WARNING: --ood_threshold {args.ood_threshold} is outside (0,1); ignoring.")
        else:
            predictor.ood_threshold         = args.ood_threshold
            predictor.ood_entropy_threshold = None   # force confidence-based path
            if args.verbose:
                print(f"OOD threshold overridden to {args.ood_threshold}")

    if args.dk_threshold is not None:
        if not (0 < args.dk_threshold <= 1):
            print(f"WARNING: --dk-threshold {args.dk_threshold} is outside (0,1]; ignoring.")
        else:
            predictor.dk_threshold = args.dk_threshold
            if args.verbose:
                print(f"DK threshold overridden to {args.dk_threshold:.0%}")

    if args.report == "dqr":
        dqr = predictor.generate_data_quality_report(df)
        print(dqr)
        return

    pred_df = predictor.predict_detailed(df)
    print(f"prediction shape: {pred_df.shape}")

    output_file = args.output or "predictions.csv"
    # Confidence columns first, then original input columns
    results = pd.concat([pred_df, df.reindex(pred_df.index)], axis=1)
    results.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"Predictions saved to {output_file}")


if __name__ == "__main__":
    main()
