# VMan ML 1.0 — Verbal Autopsy Cause-of-Death Prediction

VMan ML is the machine-learning module for [VMan3](https://github.com/vman-tool/vman3). It uses an XGBoost classifier to assign a cause of death from WHO Verbal Autopsy (VA) interview data, supporting both the **2016** and **2022** WHO VA instrument versions.

---

## Model Performance (v1.2.0)

| Metric | Value |
|---|---|
| Algorithm | XGBoost |
| Hold-out accuracy | **90.7%** |
| Hold-out F1 macro | **84.7%** |
| Hold-out F1 weighted | **89.9%** |
| Cross-validation F1 macro | **85.4%** (5-fold) |
| Training samples | 9,095 |
| Hold-out test samples | 2,274 |
| Cause classes | 32 |

Training data covers three countries: Tanzania (2016 instrument), Eswatini (2022), Nigeria (2022, two independently-coded source files). See [Training Data & Quality Filters](#training-data--quality-filters) below for how records are vetted before training.

See `reports/v1.2.0/` for the full classification report, holdout results, and training audit trail.

Earlier versions:

| Version | Hold-out accuracy | Training samples | Notes |
|---|---|---|---|
| v1.0.0 | 93.8% | 6,656 | Tanzania + Nigeria + Eswatini, no data-quality filters |
| v1.2.0 | 90.7% | 9,095 | Adds demographic-plausibility + vman_dq filters; fixes a causelist-mapping bug that had been silently dropping ~83% of one Nigeria source file. Accuracy reads lower mainly because the class count nearly doubled (23 → 32) once that file's causes were correctly recovered — several previously-clustered maternal/neonatal causes (e.g. "Obstetric haemorrhage", "Pregnancy-induced hypertension", "Neonatal sepsis") now have enough volume to stand as their own classes, which is a harder, more granular, more clinically useful problem. |

---

## Training Data & Quality Filters

Training data currently comes from four files (`vman_ml/data/`, gitignored — not distributed with the package):

| File | Rows | Instrument | Cause coding |
|---|---|---|---|
| `train_2016_tz.csv` | 3,601 | 2016 (ICD-10) | ICD-10 code (`pcva_ucod_icd`) |
| `train_2022_es.csv` | 380 | 2022 (ICD-11) | Free text (`pcva_ucod`) |
| `train_2022_ng-1.csv` | 5,474 | 2022 (ICD-11) | ICD-11 code (`pcva_icd10`, aliased to `pcva_ucod_icd`) |
| `train_2022_ng-2.csv` | 2,589 | 2022 (ICD-11) | Free text (`pcva_ucod`) |

Before a record is used for training, `DataPreprocessor._prepare_training_data()` runs it through several checks (all counts below are from the actual v1.2.0 run, see `reports/v1.2.0/training_audit_report.json`):

1. **Missing/unknown cause label** — blank, `"unknown"`, `"unspecified"`, etc. → **287** dropped.
2. **WHO causelist mapping failure** — the raw ICD code or free-text cause couldn't be matched to a WHO standardised cause (`pcva_who_cod` stays null) → **364** dropped. This step used to discard **4,903** rows (41% of the dataset) because `train_2022_ng-1.csv`'s ICD column (`pcva_icd10`) wasn't aliased to the name the pipeline looks for, so it silently fell back to free-text matching on granular clinical labels ("Other severe and complicated plasmodium falciparum malaria") that the WHO causelist matcher can't resolve. Fixed by adding the alias in `instrument_dictionary.py`'s `TARGET_ALIASES`.
3. **Demographic-cause plausibility filter** (new) — drops records where the assigned cause is biologically implausible for the decedent's recorded sex, e.g. a maternal/pregnancy cause on a male decedent, or "Male reproductive neoplasms" on a female decedent → **23** dropped.
4. **vman_dq data-quality filter** (new) — calls [`vman_dq.compute_ici()`](https://github.com/vman-tool/vman3_dq) directly (not reimplemented) to drop records where the raw interview data itself is contradictory on a pregnancy/maternal-vs-sex question (rule `C5`), plus any record in vman_dq's own "Critical" ICI tier (< 70%) → **1** dropped.
   - vman_dq's ICI rule set was redesigned after this integration was first built: `id10109`/`id10110` (whether a delivered baby moved/breathed after birth - stillbirth/live-birth questions) were originally mischecked against sex under the old rule "C7", producing ~400 false violations per ~12k rows (virtually all genuine male neonatal deaths). ICI's rules are now C1-C9 with different definitions - `C5` is the current pregnancy/maternal-vs-sex check (5 hand-verified fields), and the old id10109/id10110 mischeck is fixed by checking them against neonatal status instead, as part of the new `C4` (neonate-only questions). See `vman_dq`'s own README/docstring for the current rule definitions.
5. **Rare-cause clustering** — causes with fewer than `--min_vc` (default 130) records are grouped into WHO-hierarchy clusters (e.g. `cluster_neonatal_causes_of_death`) rather than dropped, so rare causes still contribute training signal → 42 causes clustered into 12 cluster groups.

Net: **12,044** input rows → **11,369** used for training (94.4%).

---

## Project Structure

```
vman_ml/              ← repo root
  setup.py            ← pip packaging (install with: pip install .)
  train.py            ← model training script
  predict.py          ← standalone prediction script
  vman_ml/            ← Python package (imported by VMan3 backend)
    __init__.py
    processing.py     ← VA data preprocessor + training data-quality filters
    prediction.py     ← CCVAPredictor class
    instrument_dictionary.py  ← detects WHO instrument version (2016/2022)
    mapcauselist.py   ← ICD/text → WHO cause label mapping
    label_audit.py    ← cause label QC
    narrative.py      ← narrative-text embeddings (id10476/id10477/id10479/id10436)
    training.py       ← training utilities
    resources/        ← lookup files (packaged with pip install; never gitignored)
      va_instr_2016.xlsx
      va_instr_2022.xlsx
      who_target_list.py   ← WHO PCVA causelist (Python module, not JSON/CSV —
                              see comment in the file for why)
      dictionaries/         ← cached per-version instrument dictionaries
    data/             ← training CSVs (gitignored, not packaged — see
                         "Training Data & Quality Filters" above)
  models/             ← trained model artifacts (repo root, not packaged)
    ccva_model_combined.pkl   ← combined 2016+2022 model
    ccva_model_2016.pkl       ← 2016-instrument-only model
    ccva_model_2022.pkl       ← 2022-instrument-only model
  reports/
    v1.2.0/           ← training outputs for model version 1.2.0 (gitignored,
                         regenerate locally via the Training command below)
      holdout_test_results.json
      training_audit_report.json
      cv_results.json
      who_mapping.xlsx
```

---

## Training

```bash
python3 train.py \
    --input vman_ml/data/train_2016_tz.csv vman_ml/data/train_2022_es.csv \
            vman_ml/data/train_2022_ng-1.csv vman_ml/data/train_2022_ng-2.csv \
    --report-version v1.2.0 \
    --export-mapping reports/v1.2.0/who_mapping.xlsx \
    --verbose
```

This saves all report outputs (`cv_results.json`, `training_audit_report.json`, `holdout_test_results.json`, and — with `--export-mapping` — a `who_mapping.xlsx` audit workbook) to `reports/v1.2.0/`. Omit `--report-version` and they go to `reports/` as before.

Additional options:
```bash
python3 train.py \
    --input vman_ml/data/train_2016_tz.csv vman_ml/data/train_2022_es.csv \
            vman_ml/data/train_2022_ng-1.csv vman_ml/data/train_2022_ng-2.csv \
    --report-version v1.2.0 \
    --verbose \
    --min_vc 50
```

The target column can be `pcva_ucod` (raw text) or `pcva_who_cod` (default — WHO standardised cause; requires an ICD or free-text cause column so `map_causelist`/`map_ucod_text_to_who` can populate it). After training, the model is saved to `models/`.

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

1. Retrain: `python3 train.py ...` → new `models/ccva_model_combined.pkl`
2. Upload the new `.pkl` via the VMan3 admin panel (Settings → Configurations → VMan ML)
3. The admin panel archives the old model and updates `model_registry.json` automatically

### Code update workflow

1. Update code in this repo and push a new tag (e.g. `v1.2.0`)
2. Update the tag in `backend/requirements.txt`
3. Rebuild the Docker image

---

## Installation (development)

```bash
pip install -e /path/to/vman3_ml
```
