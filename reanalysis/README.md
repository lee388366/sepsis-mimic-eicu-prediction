# Final re-analysis (manuscript results)

This folder will contain the scripts that produced the results reported in the manuscript:

1. **Train-only preprocessing re-analysis.** Imputation means and standardisation parameters are fitted on the training set only. All seven models are retrained. The folder covers internal performance, subject-clustered bootstrap, subgroups, LSTM seed stability, eICU evaluation with patient-split recalibration, explanation and occlusion analyses, cross-validation, and figures.
2. **Time-bounded 48-hour SOFA / qSOFA comparators.** These are reconstructed in MIMIC-IV (from the MIT-LCP `mimic-code` concepts) and in eICU-CRD, together with their evaluation scripts.
3. **Post hoc GCS recording and response-trajectory analyses.** These analyses did not retrain the prediction models.

These scripts are being added to this folder.
