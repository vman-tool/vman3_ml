# vman3_ml/predict.py
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

    args = parser.parse_args()

    preprocessor = DataPreprocessor(verbose=args.verbose)
    df = preprocessor.load_data(args.input)
    df = preprocessor._preprocess_data(df)
    detection = detect_instrument_version(df, source_path=args.input)

    model_path = Path(args.model)
    if model_path.is_dir():
        versioned_path = model_path / f"ccva_model_{detection['version']}.pkl"
        generic_path = model_path / "ccva_model.pkl"
        model_path = versioned_path if versioned_path.exists() else generic_path
    else:
        versioned_path = model_path.with_name(f"ccva_model_{detection['version']}.pkl")
        if versioned_path.exists():
            model_path = versioned_path

    predictor = CCVAPredictor(str(model_path), verbose=args.verbose)

    if args.report == "dqr":
        dqr = predictor.generate_data_quality_report(df)
        print(dqr)
        return

    predictions = predictor.predict(df)
    print(f"prediction shape: {np.array(predictions).shape}")

    output_file = args.output or "predictions.csv"
    results = pd.DataFrame({
        'prediction': predictions,
        **{col: df[col] for col in df.columns},
    })
    results.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")


if __name__ == "__main__":
    main()
