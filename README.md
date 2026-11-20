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

Training data currently comes from three files in `vman_ml/resources/` (gitignored — not distributed with the package, despite otherwise living alongside the packaged lookup files in that folder). Each filename is `train<version><country>.csv`: the numeric prefix is the WHO VA instrument version (`16` = 2016, `22` = 2022) and the suffix is the ISO-ish country code (`tz` = Tanzania, `es` = Eswatini, `ng` = Nigeria).

| File | Rows | Instrument | Cause coding |
|---|---|---|---|
| `train16_tz.csv` | 3,601 | 2016 (ICD-10) | ICD-10 code (`pcva_ucod_icd`) |
| `train22_es.csv` | 380 | 2022 (ICD-11) | Free text (`pcva_ucod`) |
| `train22_ng.csv` | 5,474 | 2022 (ICD-11) | ICD-11 code (`pcva_icd10`, aliased to `pcva_ucod_icd`) |

(Row counts above are a current snapshot — these files are working data, not version-pinned, so they can drift. `train22_ng.csv` is the renamed former `train_2022_ng-1.csv`; the separate free-text-coded `train_2022_ng-2.csv` from the v1.2.0 run below is not part of the current fileset. The quality-filter counts below are frozen to that specific v1.2.0 run.)

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
    narrative.py      ← narrative-text embeddings (id10476/id10477/id10479/id10436) -
                         narrative-less records get an all-zero vector + a 0/1
                         presence flag, not the model's own encoding of "" (see
                         module docstring: keeps narrative a pure supplement -
                         harmless when absent, informative when present)
    shap_report.py    ← global SHAP figures saved after training (see Training below)
    training.py       ← training utilities
    resources/        ← lookup files (packaged with pip install) PLUS the
                         gitignored training CSVs (not packaged) — see
                         "Training Data & Quality Filters" above
      va_instr_2016.xlsx
      va_instr_2022.xlsx
      who_target_list.py   ← WHO PCVA causelist (Python module, not JSON/CSV —
                              see comment in the file for why)
      dictionaries/         ← cached per-version instrument dictionaries
      train16_tz.csv        ← training data, gitignored (2016 instrument, Tanzania)
      train22_es.csv        ← training data, gitignored (2022 instrument, Eswatini)
      train22_ng.csv        ← training data, gitignored (2022 instrument, Nigeria)
  models/             ← trained model artifacts (repo root, not packaged)
    vman_ccva.pkl             ← combined 2016+2022 model (standard name - this
                                 is the one deployed into VMan3, see below)
    ccva_model_2016.pkl       ← 2016-instrument-only model
    ccva_model_2022.pkl       ← 2022-instrument-only model
  reports/
    v1.2.0/           ← training outputs for model version 1.2.0 (gitignored,
                         regenerate locally via the Training command below)
      holdout_test_results.json
      training_audit_report.json
      cv_results.json
      who_mapping.xlsx
      training_summary.md      ← human-readable Markdown digest of the three
                                  JSON files above (tables, not raw JSON)
```

---

## Training

```bash
python3 train.py \
    --input vman_ml/resources/train16_tz.csv vman_ml/resources/train22_es.csv \
            vman_ml/resources/train22_ng.csv \
    --report-version v1.2.0 \
    --export-mapping reports/v1.2.0/who_mapping.xlsx \
    --verbose
```

This saves all report outputs (`cv_results.json`, `training_audit_report.json`, `holdout_test_results.json`, and — with `--export-mapping` — a `who_mapping.xlsx` audit workbook) to `reports/v1.2.0/`. Omit `--report-version` and they go to `reports/` as before.

A `training_summary.md` is also written alongside them — a single Markdown file with the same information as those three JSON files, as tables: the data-pipeline/drop-reason counts, the model comparison (CV/validation scores per candidate model), validation and hold-out metrics, and a per-cause precision/recall/F1/AUROC table on the hold-out set (worst 15 shown, rest collapsed). Easier to read than the raw JSON when you just want to check how a run went.

Validation and hold-out metrics also include a macro-AUROC headline number (`auroc_macro` in `cv_results.json`/`holdout_test_results.json`) and a per-cause AUROC column (`auroc_per_class`), both one-vs-rest. Per-class AUROC is computed per class independently (not via a single `multi_class='ovr'` call) so one cause with too few hold-out records to define an ROC curve (no positive, or no negative, examples in that split - genuinely possible given some clusters have single-digit support) is just excluded from that cause's row (shown as `n/a` in the table) rather than failing the whole metric.

### Generating a report without retraining

`training_summary.md` can be (re)generated on its own from an existing report directory, without re-running `train.py`:
```bash
python3 -m vman_ml.report reports/v1.2.0
```
This is useful after manually editing a JSON report, or if `training_summary.md` predates a change to how it's rendered. `reports/v1.2.0` can be any directory containing `training_audit_report.json`, `cv_results.json`, and/or `holdout_test_results.json` — missing files are skipped rather than erroring.

Additional options:
```bash
python3 train.py \
    --input vman_ml/resources/train16_tz.csv vman_ml/resources/train22_es.csv \
            vman_ml/resources/train22_ng.csv \
    --report-version v1.2.0 \
    --verbose \
    --min_vc 50
```

The target column can be `pcva_ucod` (raw text) or `pcva_who_cod` (default — WHO standardised cause; requires an ICD or free-text cause column so `map_causelist`/`map_ucod_text_to_who` can populate it). After training, the model is saved to `models/`.

### Short flags

Every frequently-used option also has a short form (both work identically):

| Short | Long |
|---|---|
| `-i` | `--input` |
| `-v` | `--verbose` |
| `-e` | `--export-mapping` |
| `-r` | `--report-version` |
| `-a` | `--audit_report` |
| `-t` | `--target` |
| `-V` | `--version` |
| `-l` | `--label-audit` |

The example above, written with short flags:
```bash
python3 train.py -i vman_ml/resources/train16_tz.csv vman_ml/resources/train22_es.csv \
                     vman_ml/resources/train22_ng.csv \
    -r v1.2.0 -e reports/v1.2.0/who_mapping.xlsx -v
```

The remaining options (`--min_vc`, `--na_threshold`, `--taxonomy_file`, `--llm-review`, `--audit-top-n`, `--audit-cv-folds`, `--no-confirmation-weights`) are long-form only — see `train.py --help` for the full list.

Each model searches hyperparameters with `HalvingRandomSearchCV` (5-fold CV, 10 initial candidates by default, successive-halving — see [scikit-learn's docs](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.HalvingRandomSearchCV.html)): weaker candidates are eliminated early on a small sample of the training data, and only survivors are refit on progressively larger samples, which is substantially cheaper than giving every candidate a full 5-fold fit on the whole set (the previous `RandomizedSearchCV` behavior). Training prints a live progress line per model — fit count and elapsed time, so a multi-minute search doesn't look hung — without a percentage, since successive halving's real fit count isn't known upfront (it depends on how many candidates survive each round).

Timing is also recorded: per-model fit time in `cv_results.json`/`holdout_test_results.json` (`comparison.<model>.fit_seconds`), and in `training_audit_report.json`, `training_seconds` (hyperparameter search only) and `total_run_seconds` (the whole run — data loading, preprocessing, narrative embeddings, search, evaluation). `training_summary.md` renders both under a "Timing" section.

After training, two global SHAP figures are saved to the report directory and embedded in `training_summary.md`: `shap_feature_importance.png` (top 20 features by mean |SHAP value|, computed with `shap.TreeExplainer` on a sample of up to 500 hold-out records) and `shap_narrative_contribution.png` (the narrative-embedding block's combined contribution vs. everything else — see `vman_ml/shap_report.py`). This needs `matplotlib`, now a regular dependency; it's a reporting extra and never fails the training run itself if `shap`/`matplotlib` are unavailable.

---

## Standalone Prediction

```bash
python3 predict.py --model models/vman_ccva.pkl --input va_test.csv --output results.csv --verbose
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

The **model file (`.pkl`) is NOT distributed through pip** — it is maintained separately in the VMan3 backend under `backend/app/ccva/ml_models/ccva_model_combined.pkl`. This allows the model to be updated independently of the code without rebuilding the Docker image. (That backend-side filename is the backend's own internal convention, unrelated to this repo's output name below — the admin panel upload endpoint saves whatever `.pkl` you upload to that fixed internal path regardless of what it was named when you uploaded it, so renaming this repo's output requires no backend change.)

### Model update workflow

1. Retrain: `python3 train.py ...` → new `models/vman_ccva.pkl`
2. Upload the new `.pkl` via the VMan3 admin panel (Settings → Configurations → VMan ML)
3. The admin panel archives the old model and updates `model_registry.json` automatically

### Code update workflow

1. Update code in this repo and push a new tag (e.g. `v1.2.0`)
2. Update the tag in `backend/requirements.txt`
3. Rebuild the Docker image

---

## Fine-Tuning Considerations (LoRA / QLoRA)

LoRA/QLoRA adapt a pretrained *neural network's* weight matrices cheaply (low-rank
adapters, optionally over a 4-bit-quantized base) instead of updating all its
weights. They don't apply uniformly across this pipeline — the two trained
components are architecturally different:

- **XGBoost / RandomForest (the actual cause-of-death classifier)** — trees,
  no weight matrices, nothing for LoRA to attach to. The nearest equivalent
  goal (cheap adaptation to new data without refitting from scratch) is
  XGBoost's own warm-start (`xgb_model=`, continuing to boost an existing
  model on new rows) — same goal, unrelated mechanism. Relevant once a 4th
  country's data arrives and a full `RandomizedSearchCV` refit feels wasteful.
- **The narrative embedder** (`paraphrase-multilingual-MiniLM-L12-v2` in
  `vman_ml/narrative.py`) — this *is* a transformer, currently used entirely
  frozen as a generic multilingual sentence encoder. This is the one place
  LoRA is actually relevant: the model has no notion of "cause of death", it
  just measures generic paraphrase similarity. Fine-tuning it on this
  project's own (narrative text, WHO cause label) pairs — already available
  in the training data — could produce embeddings that cluster by clinical
  similarity instead of surface wording.

  LoRA is arguably a *better* fit here than full fine-tuning precisely
  because the labeled-narrative set is small relative to the model's original
  pretraining corpus (measured: only ~7,100 unique non-empty narratives
  across the full TZ+ES+NG set) — full fine-tuning on that little data risks
  catastrophic forgetting of the base model's general multilingual knowledge,
  while LoRA's constrained low-rank updates are specifically suited to narrow
  domain adaptation without damaging it. QLoRA's quantization benefit doesn't
  matter here — MiniLM-L12 is only ~118M params, already cheap to run at full
  precision on CPU.

  Not yet implemented — would need a training objective (classification or
  contrastive loss on cause labels), a split for the embedder's own
  training/validation that doesn't leak into the downstream XGBoost
  evaluation, and a controlled comparison against the frozen-embedding
  baseline (e.g. via `shap_narrative_contribution.png`, see Training below)
  to confirm it actually improves downstream macro-F1 before adopting it.

---

## Installation (development)

This repo has its own virtual environment, separate from the VMan3 backend's.
Always activate `vman_ml/.venv` (not the backend's) before running `train.py`
or `predict.py` — check with `which python3` if unsure.

```bash
cd /path/to/vman3_ml
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt   # pinned third-party deps
pip install -e .                  # installs this package itself, editable
```

`requirements.txt` pins the exact versions known to work (including
`vman_dq`, pulled from its git tag). Reinstalling from it should reproduce
the environment exactly, without re-resolving newer releases.

If you add or upgrade a dependency, update `pyproject.toml`'s `dependencies`
list first, then regenerate the pin file from the active venv:

```bash
pip freeze | grep -v '^-e ' > requirements.txt
```

(the `-e git+...#egg=vman_ml` self-reference that `pip freeze` adds for the
editable install should be dropped — the package is installed separately via
`pip install -e .` above, not from `requirements.txt`).
