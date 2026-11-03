# Sequence models vs simple baselines for post-48h sepsis mortality prediction

Analysis code for the manuscript:

> **Sequence models do not outperform simple baselines for post-48h sepsis mortality prediction: a head-to-head comparison**
> Tian T, Liu Y, Li H, Lu M, Sun Q, Li C, Du G, Li Y. (under review)

Using routinely recorded data from hours 0–47 after ICU admission, the study predicts in-hospital death after hour 48 in adult patients with sepsis. It compares six models built on summary features (logistic regression, random forest, SVM, KNN, Gaussian naive Bayes, decision tree) with an attention-based bidirectional LSTM built on hourly sequences. SOFA and qSOFA are reference comparators. Models were developed and evaluated in **MIMIC-IV v3.1** and examined in an exploratory evaluation in **eICU-CRD v2.0**.

## Data access

This repository contains **code only**. It contains no patient-level data, no extracted or intermediate datasets, and no trained model files.

The source databases are credentialed-access resources on PhysioNet:

- MIMIC-IV v3.1 — https://doi.org/10.13026/kpb9-mt58
- eICU-CRD v2.0 — https://doi.org/10.13026/C2WM1R

To reproduce the analyses you must obtain PhysioNet credentialing, complete the required human-subjects training, and sign the relevant data use agreements. Then build local PostgreSQL copies of both databases, including the standard MIMIC-IV derived concepts (`mimiciv_derived`, from the MIT-LCP `mimic-code` repository) and the eICU derived concepts. Under the PhysioNet data use agreements, outputs that contain patient-level rows must not be redistributed.

## Repository structure

```
pipeline/     Cohort extraction (SQL), feature assembly, model training and evaluation (Python)
reanalysis/   Final analyses reported in the manuscript (see reanalysis/README.md)
```

### `pipeline/` — run order

Run the scripts from inside `pipeline/`. They read and write intermediate files (`*.csv`, `*.npz`) in the working directory and write tables to `tables/` and figures to `figures/`.

**MIMIC-IV (development cohort)**

| Step | Scripts | Purpose |
|---|---|---|
| 1 | `01_sepsis_cohort.sql` | Sepsis-3 adult ICU cohort |
| 2 | `02_extract_hourly.sql` | Hourly vital signs and laboratory values, hours 0–47 |
| 3 | `04_labels.sql`, `04b_extract_sofa.sql`, `04c_extract_demographics.sql` | Outcome labels, SOFA/qSOFA inputs, demographics |
| 4 | `03_impute.py`, `05_assemble_training_data.py`, `split_utils.py` | Imputation, sequence/summary feature assembly, subject-level split (70/15/15, seed 42) |
| 5 | `06_train_baseline.py`, `06b_train_sequence.py`, `06_cross_validation.py`, `06b_cv_sequence.py`, `06c_ablate_timepoint.py` | Model training, cross-validation, time-point ablation |
| 6 | `07_*.py`, `10_*.py`, `11_*.py` | Discrimination, bootstrap CIs (record- and subject-clustered), DeLong tests, calibration, decision curves, SOFA/qSOFA comparators, subgroups |
| 7 | `08_*.py`, `09_*.py`, `12*.py`, `shap_cache.py` | SHAP and attention analyses, descriptive figures |
| 8 | `make_table*.py`, `make_merged_predictions.py` | Summary tables |

**eICU-CRD (exploratory external evaluation)**

`eicu_01_cohort.sql` → `eicu_02_extract_hourly.sql` → `eicu_03_sofa_qsofa.sql` → `eicu_04b_extract_demographics.sql` → `eicu_04_assemble.py` → `eicu_05_impute.py` → `eicu_06_external_validate.py` → `eicu_07`–`eicu_09` (figures, subgroups, decision curves), and `07_calibration_dca_external.py`, `07e_bootstrap_eicu_patient.py`.

> **Note:** `pipeline/` is the original end-to-end workflow. The numbers reported in the manuscript come from the final re-analysis in `reanalysis/`. That re-analysis fits imputation means and standardisation parameters on the training set only, retrains all seven models, and uses a time-bounded 48-hour SOFA comparator. Comments in some scripts are in Chinese.

## Environment

The primary re-analysis used Python 3.13 with PyTorch 2.13.0, scikit-learn 1.9.0, NumPy 2.5.2, pandas 3.0.5, SciPy 1.18, SHAP 0.52 and Matplotlib 3.11 (see `requirements.txt`). The SQL scripts target PostgreSQL. Unless an analysis-specific seed is stated, seed 42 is used for data splitting, model training and bootstrap resampling.

## Reporting

The study is reported following TRIPOD+AI.

## Licence

Code is released under the MIT Licence (see `LICENSE`). This licence does not cover the MIMIC-IV or eICU-CRD data, which remain subject to their PhysioNet licences and data use agreements.

## Contact

Correspondence: Ge Du (duge492904559@163.com); Yanbing Li (cd_lyb@126.com).
