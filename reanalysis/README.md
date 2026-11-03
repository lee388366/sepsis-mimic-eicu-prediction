# Final re-analysis (manuscript results)

The scripts in this folder produced the results reported in the manuscript. The folder layout matches the layout in which the scripts were run, because the scripts locate their inputs relative to their own position:

- `reanalysis/` corresponds to the analysis workspace.
- The repository root corresponds to the project root, which holds the outputs of `../pipeline/` (see the top-level README).

Code comments and some console messages are in Chinese. The scripts never write to the source databases: every SQL query is read-only.

## 1. Primary analysis: train-only preprocessing, 48-hour landmark cohort

`revision_2026-09-13/landmark_trainonly_reanalysis/scripts/`

Imputation means and standardisation parameters are fitted on the training partition only, and all seven models are retrained with the hyperparameters fixed in the original pipeline. Variant **A** is the primary cohort. Variant **B** is the restricted sensitivity cohort (suspected-infection time ≤ 48 h).

| Order | Script | Purpose |
|---|---|---|
| 0 | `s00_snapshot.py` | Snapshot of earlier outputs and environment (provenance) |
| 1 | `s01_landmark_audit.py`, `eicu_cohort_index.sql` | 48-hour eligibility audit; cohort variants A/B; infection-timing distribution |
| 2 | `s02_build_trainonly_data.py` | Train-only static imputation and scaling |
| 3 | `s03_train_models.py A` / `B` | Retrain six summary-feature models and the attention BiLSTM |
| 4 | `s04_internal_performance.py A` / `B` | Internal AUC/AP with subject-clustered bootstrap CIs, DeLong tests, calibration, decision curves, thresholds |
| 5 | `s05_subgroup.py A` / `B` | Subgroups, paired differences, interaction tests |
| 6 | `s06_lstm_seed_stability.py` | LSTM retraining across seeds 42–47 |
| 7 | `s07_eicu_external.py A` / `B` | eICU-CRD evaluation; uniquepid-clustered CIs; patient-split recalibration |
| 8 | `s08_explain.py A` / `B`, `s08b_occlusion_lowarm.py` | SHAP, attention, occlusion faithfulness, time-window ablation |
| 9 | `s09_cv.py A` / `B` | Subject-level cross-validation (RF 10-fold, LSTM 5-fold) |
| 10 | `s10_figures.py A` / `B`, `s25_regenerate_figures.py` | Figures |
| 11 | `s11_compare.py`, `s15_old_vs_new_prediction_agreement.py` | Comparison with the earlier preprocessing pipeline |
| 12 | `s12_finalize.py`, `s19_rebuild_table1.py` | Record-level calibration sensitivity analysis; Table 1 |

`common_eval.py` contains the shared metric, bootstrap and calibration functions.

## 2. Time-bounded 48-hour SOFA comparators

`revision_2026-09-21/resubmission_repair/`

- `score_lineage/`: the MIMIC-IV SOFA, restricted to hours 0–47. `build_strict_sql.py` assembles `mimic_sofa_strict48.sql` from the MIT-LCP `mimic-code` concept queries (`official_*.sql`), and `run_strict_extract.py` executes it. The remaining scripts test the query against a synthetic dataset (`synthetic_strict48.sql`, `test_strict_sql_synthetic.py`) and run boundary and quality checks (`check_*.py`, `*_qa.sql`, `validate_new_scores.py`).
- `eicu_sofa_rebuild/`: the eICU-CRD SOFA, restricted to hours 0–47. `build_score_sql.py` assembles `eicu_sofa_strict48.sql` from `cohort_cte.sql`, `score_body.sql` and the eICU-code pivoted concepts in `sources/`, and `run_extract.py` executes it. `validate_scores.py` and `independent_synthetic_review.py` run the checks.
- `score_evaluation/evaluate_strict_sofa.py` and `eicu_score_evaluation/evaluate_eicu_sofa.py` evaluate the score comparators in MIMIC-IV and eICU-CRD.
- `figures/` holds the scripts that redraw figures from frozen prediction arrays. They do no model fitting.

## 3. Post hoc GCS recording and response-trajectory analyses

`revision_2026-09-13/dag_gcs_*`

These analyses did not retrain the prediction models.

1. `dag_gcs_next_stage/`: read-only extraction of the cohort index, raw and derived GCS, RASS/CAM-ICU, sedative and opioid exposure, airway and ventilation records, and neurological diagnoses (`sql/`, `run_extract.sh`, QA scripts).
2. `dag_gcs_mimic_interpretation/` (v1) and `dag_gcs_mimic_interpretation_v2/` (v2): GCS time-base tables with component provenance (`build_base_table.py`, `qa_base_table.py`). v2 corrects the alignment of the background variables.
3. `dag_gcs_interpretation_analysis/` and `dag_gcs_treatment_analysis/`: clinical context of GCS recordings (RASS and treatment context). They read the v1 table, and the treatment analysis compares v1 with v2.
4. `dag_gcs_fixed_window_trajectories/`: eye- and motor-score trajectories at 6, 12, 24, 36 and 48 h, with the 6- vs 48-hour paired comparison (subject-clustered bootstrap, seed 20260913). This analysis reads the v2 table.

Several scripts verify SHA-256 hashes of their inputs against values recorded in the original environment. When re-running on a new extraction, remove or update those assertions.

## Not included

This folder contains no patient-level data, extracted datasets, trained models or figures. It also leaves out exploratory database probes used while designing the eICU SOFA rules, and helper scripts that synchronised results into manuscript files. Both are available from the corresponding author on request.
