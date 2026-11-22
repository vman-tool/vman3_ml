# Training Summary — v1.3.0

## Inputs
- Input files: vman_ml/resources/train16_tz.csv, vman_ml/resources/train22_es.csv
- Target column: `pcva_who_cod`
- Instrument version(s): 2016_tz, 2022_es
- Feature count: 499

## Data Pipeline
| Stage | Rows |
|---|---|
| Input rows | 3981 |
| After quality filter | 3695 |
| After rare-cause clustering | 3695 |
| Final rows used for training | 3521 |

### Rows dropped, by reason
| Reason | Rows dropped |
|---|---|
| Missing/unknown cause label | 286 |
| WHO causelist mapping failure | 174 |
| Demographic-cause implausibility | 0 |
| vman_dq pregnancy/sex contradiction | 0 |
| vman_dq low ICI (< 70%) | 0 |

vman_dq ICI: 3695 rows scored, mean ICI 99.68%, rules applied: C2, C3, C4, C5

**Rare-cause clustering:** 50 causes grouped into 13 cluster group(s) (causes below `--min_vc` records).

## Model Comparison (internal validation split)
| Model | CV Macro-F1 | Val Accuracy | Val Macro-F1 | Fit Time (s) |
|---|---|---|---|---|
| XGBoost | 0.8897 | 0.9326 | 0.9184 | n/a |
| RandomForest | 0.8245 | 0.867 | 0.8132 | n/a |

**Selected model:** XGBoost

## Validation Results
| Metric | Value |
|---|---|
| Records | 564 |
| Accuracy | 0.9326 |
| Macro-F1 | 0.9184 |
| Weighted-F1 | 0.9311 |

## Hold-out Test Results (20%, never seen during training)
| Metric | Value |
|---|---|
| Records | 705 |
| Accuracy | 0.9078 |
| Macro-F1 | 0.8559 |
| Weighted-F1 | 0.9062 |

### Per-cause performance on hold-out (worst 15 by F1)
| Cause | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| cluster_nutritional_and_endocrine_disorders | 0.250 | 0.222 | 0.235 | 9 |
| cluster_gastrointestinal_disorders | 0.636 | 0.700 | 0.667 | 20 |
| cluster_neoplasms | 0.810 | 0.708 | 0.756 | 48 |
| cluster_diseases_of_the_circulatory_system | 0.800 | 0.769 | 0.784 | 26 |
| cluster_respiratory_disorders | 0.846 | 0.733 | 0.786 | 15 |
| cluster_stillbirths | 1.000 | 0.667 | 0.800 | 3 |
| cluster_pregnancy_childbirth_and_puerperium_related_disorders | 1.000 | 0.714 | 0.833 | 7 |
| cluster_renal_disorders | 1.000 | 0.769 | 0.870 | 13 |
| cluster_infectious_and_parasitic_diseases | 0.871 | 0.931 | 0.900 | 29 |
| Other and unspecified cardiac disease | 0.886 | 0.944 | 0.914 | 124 |
| Diabetes mellitus | 1.000 | 0.867 | 0.929 | 30 |
| cluster_mental_and_nervous_system_disorders | 0.941 | 0.941 | 0.941 | 17 |
| HIV/AIDS related death | 0.933 | 0.962 | 0.947 | 130 |
| cluster_external_causes_of_death | 0.977 | 0.956 | 0.966 | 45 |
| Pulmonary tuberculosis | 0.952 | 0.984 | 0.968 | 61 |

<details><summary>All causes</summary>

| Cause | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| cluster_nutritional_and_endocrine_disorders | 0.250 | 0.222 | 0.235 | 9 |
| cluster_gastrointestinal_disorders | 0.636 | 0.700 | 0.667 | 20 |
| cluster_neoplasms | 0.810 | 0.708 | 0.756 | 48 |
| cluster_diseases_of_the_circulatory_system | 0.800 | 0.769 | 0.784 | 26 |
| cluster_respiratory_disorders | 0.846 | 0.733 | 0.786 | 15 |
| cluster_stillbirths | 1.000 | 0.667 | 0.800 | 3 |
| cluster_pregnancy_childbirth_and_puerperium_related_disorders | 1.000 | 0.714 | 0.833 | 7 |
| cluster_renal_disorders | 1.000 | 0.769 | 0.870 | 13 |
| cluster_infectious_and_parasitic_diseases | 0.871 | 0.931 | 0.900 | 29 |
| Other and unspecified cardiac disease | 0.886 | 0.944 | 0.914 | 124 |
| Diabetes mellitus | 1.000 | 0.867 | 0.929 | 30 |
| cluster_mental_and_nervous_system_disorders | 0.941 | 0.941 | 0.941 | 17 |
| HIV/AIDS related death | 0.933 | 0.962 | 0.947 | 130 |
| cluster_external_causes_of_death | 0.977 | 0.956 | 0.966 | 45 |
| Pulmonary tuberculosis | 0.952 | 0.984 | 0.968 | 61 |
| cluster_neonatal_causes_of_death | 0.966 | 1.000 | 0.982 | 28 |
| Malaria | 0.969 | 1.000 | 0.984 | 63 |
| Road traffic accident | 1.000 | 1.000 | 1.000 | 33 |
| cluster_non_communicable_diseases | 1.000 | 1.000 | 1.000 | 4 |

</details>
