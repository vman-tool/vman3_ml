# VMan ML 1.0 — Verbal Autopsy Cause-of-Death Prediction

VMan ML is the machine-learning module for [VMan3](https://github.com/vman-tool/vman3). It uses an XGBoost classifier to assign a cause of death from WHO Verbal Autopsy (VA) interview data, supporting both the **2016** and **2022** WHO VA instrument versions.

---

## Model Performance (v1.0.0)

| Metric | Value |
|---|---|
| Algorithm | XGBoost |
| Hold-out accuracy | **93.8%** |
| Hold-out F1 macro | **92.3%** |
| Hold-out F1 weighted | **93.6%** |
| Cross-validation F1 macro | **90.1%** (5-fold) |
| Training samples | 6,656 |
| Hold-out test samples | 1,332 |
| Cause classes | 23 |

Training data covers three countries:  Tanzania (2016 instrument), Nigeria (2022), Spain (2022).

See `reports/v1.0.0/` for full classification report, confusion matrix, and training audit.

---

## Project Structure

```
vman_ml/              ← repo root
  setup.py            ← pip packaging (install with: pip install .)
  train.py            ← model training script
  predict.py          ← standalone prediction script
  vman_ml/            ← Python package (imported by VMan3 backend)
    __init__.py
    processing.py     ← VA data preprocessor
    prediction.py     ← CCVAPredictor class
    instrument_dictionary.py  ← detects WHO instrument version (2016/2022)
    mapcauselist.py   ← ICD → cause label mapping
    label_audit.py    ← cause label QC
    narrative.py      ← narrative cause descriptions
    training.py       ← training utilities
    data/             ← lookup files (packaged with pip install)
      cause_taxonomy.json
      va_instr_2016.xlsx
      va_instr_2022.xlsx
      who_target_list.csv
    models/           ← trained model artifacts (NOT distributed via pip)
      ccva_model_combined.pkl   ← combined 2016+2022 model
      ccva_model_2016.pkl       ← 2016-instrument-only model
      ccva_model_2022.pkl       ← 2022-instrument-only model
  reports/
    v1.0.0/           ← training outputs for model version 1.0.0
      holdout_test_results.json
      training_audit_report.json
      cv_results.json
      cv_accuracy_plot.png
      results_combined.csv
```

---

## Training

```bash
python3 train.py \
    --input data/va_2016_tz.csv data/va_2022_ng.csv data/va_2022_es.csv \
    --report-version v1.1.0 \
    --verbose
```

This saves all report outputs (`cv_results.json`, `training_audit_report.json`, `holdout_test_results.json`) to `reports/v1.1.0/`. Omit `--report-version` and they go to `reports/` as before.

Additional options:
```bash
python3 train.py \
    --input data/va_2016_tz.csv data/va_2022_ng.csv data/va_2022_es.csv \
    --report-version v1.1.0 \
    --verbose \
    --min_vc 50
```

The target column must be labelled `pcva_ucod`. After training, the model is saved to `vman_ml/models/`.

---

## Standalone Prediction

```bash
python3 predict.py --model models/ccva_model_combined.pkl --input va_test.csv --output results.csv --verbose
```

Relax the Don't-Know (DK) threshold:
```bash
python3 predict.py --model models/ --input va_test.csv --dk-threshold 1.0
```

Relax both DK and out-of-distribution (OOD) thresholds:
```bash
python3 predict.py --model models/ --input va_test.csv --dk-threshold 1.0 --ood-threshold 0.05
```

---

## Deployment in VMan3

The Python package (code + data files) is installed in the VMan3 backend via pip:

```
git+https://github.com/vman-tool/vman3_ml.git@v1.0.0
```

The **model file (`.pkl`) is NOT distributed through pip** — it is maintained separately in the VMan3 backend under `backend/app/ccva/ml_models/ccva_model_combined.pkl`. This allows the model to be updated independently of the code without rebuilding the Docker image.

### Model update workflow

1. Retrain: `python3 train.py ...` → new `vman_ml/models/ccva_model_v1.1_2026-09.pkl`
2. Upload the new `.pkl` via the VMan3 admin panel (Settings → Configurations → VMan ML)
3. The admin panel archives the old model and updates `model_registry.json` automatically

### Code update workflow

1. Update code in this repo and push a new tag (e.g. `v1.1.0`)
2. Update the tag in `backend/requirements.txt`
3. Rebuild the Docker image

---

## Installation (development)

```bash
pip install -e /path/to/vman3_ml
```
