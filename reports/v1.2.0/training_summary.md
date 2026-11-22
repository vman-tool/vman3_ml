# Training Summary — v1.2.0

## Inputs
- Input files: vman_ml/resources/train16_tz.csv, vman_ml/resources/train22_es.csv, vman_ml/resources/train22_ng.csv
- Target column: `pcva_who_cod`
- Instrument version(s): 2016_tz, 2022_es, 2022_tz
- Feature count: 499

## Data Pipeline
| Stage | Rows |
|---|---|
| Input rows | 9455 |
| After quality filter | 9433 |
| After rare-cause clustering | 9433 |
| Final rows used for training | 5451 |

### Rows dropped, by reason
| Reason | Rows dropped |
|---|---|
| Missing/unknown cause label | 1 |
| WHO causelist mapping failure | 3982 |
| Demographic-cause implausibility | 20 |
| vman_dq pregnancy/sex contradiction | 1 |
| vman_dq low ICI (< 70%) | 0 |

vman_dq ICI: 9434 rows scored, mean ICI 99.87%, rules applied: C2, C3, C4, C5

**Rare-cause clustering:** 33 causes grouped into 10 cluster group(s) (causes below `--min_vc` records).

## Model Comparison (internal validation split)
| Model | CV Macro-F1 | Val Accuracy | Val Macro-F1 |
|---|---|---|---|
| XGBoost | 0.7213 | 0.8163 | 0.743 |
| RandomForest | 0.6793 | 0.7577 | 0.6815 |

**Selected model:** XGBoost

## Validation Results
| Metric | Value |
|---|---|
| Records | 871 |
| Accuracy | 0.8163 |
| Macro-F1 | 0.743 |
| Weighted-F1 | 0.8116 |

## Hold-out Test Results (20%, never seen during training)
| Metric | Value |
|---|---|
| Records | 1090 |
| Accuracy | 0.7945 |
| Macro-F1 | 0.7492 |
| Weighted-F1 | 0.7902 |

### Per-cause performance on hold-out (worst 15 by F1)
| Cause | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| cluster_non_communicable_diseases | 0.000 | 0.000 | 0.000 | 5 |
| Sickle cell with crisis | 0.727 | 0.381 | 0.500 | 21 |
| cluster_nutritional_and_endocrine_disorders | 0.640 | 0.667 | 0.653 | 24 |
| cluster_gastrointestinal_disorders | 1.000 | 0.500 | 0.667 | 2 |
| Acute respiratory infection, including pneumonia | 0.769 | 0.633 | 0.694 | 79 |
| Diarrheal diseases | 0.738 | 0.667 | 0.700 | 156 |
| Other and unspecified maternal cause | 0.795 | 0.738 | 0.765 | 42 |
| cluster_pregnancy_childbirth_and_puerperium_related_disorders | 0.833 | 0.714 | 0.769 | 28 |
| Measles | 0.803 | 0.754 | 0.778 | 65 |
| Pregnancy-induced hypertension | 0.829 | 0.744 | 0.784 | 39 |
| Malaria | 0.740 | 0.877 | 0.803 | 285 |
| cluster_neonatal_causes_of_death | 0.842 | 0.800 | 0.821 | 40 |
| Neonatal sepsis | 0.828 | 0.869 | 0.848 | 61 |
| cluster_respiratory_disorders | 1.000 | 0.750 | 0.857 | 4 |
| Birth asphyxia | 0.906 | 0.906 | 0.906 | 53 |

<details><summary>All causes</summary>

| Cause | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| cluster_non_communicable_diseases | 0.000 | 0.000 | 0.000 | 5 |
| Sickle cell with crisis | 0.727 | 0.381 | 0.500 | 21 |
| cluster_nutritional_and_endocrine_disorders | 0.640 | 0.667 | 0.653 | 24 |
| cluster_gastrointestinal_disorders | 1.000 | 0.500 | 0.667 | 2 |
| Acute respiratory infection, including pneumonia | 0.769 | 0.633 | 0.694 | 79 |
| Diarrheal diseases | 0.738 | 0.667 | 0.700 | 156 |
| Other and unspecified maternal cause | 0.795 | 0.738 | 0.765 | 42 |
| cluster_pregnancy_childbirth_and_puerperium_related_disorders | 0.833 | 0.714 | 0.769 | 28 |
| Measles | 0.803 | 0.754 | 0.778 | 65 |
| Pregnancy-induced hypertension | 0.829 | 0.744 | 0.784 | 39 |
| Malaria | 0.740 | 0.877 | 0.803 | 285 |
| cluster_neonatal_causes_of_death | 0.842 | 0.800 | 0.821 | 40 |
| Neonatal sepsis | 0.828 | 0.869 | 0.848 | 61 |
| cluster_respiratory_disorders | 1.000 | 0.750 | 0.857 | 4 |
| Birth asphyxia | 0.906 | 0.906 | 0.906 | 53 |
| Obstetric haemorrhage | 0.854 | 0.967 | 0.907 | 91 |
| cluster_infectious_and_parasitic_diseases | 0.977 | 0.857 | 0.913 | 49 |
| Pregnancy-related sepsis | 1.000 | 0.857 | 0.923 | 28 |
| cluster_external_causes_of_death | 0.900 | 1.000 | 0.947 | 18 |

</details>
