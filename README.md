# COPD-XAI-Revision

Code, predictor definitions, cross-validation splits and outputs for the manuscript
*Leakage Audited Explainable Machine Learning for COPD Severity and Quality of Life Assessment with Propensity Score Adjusted Analysis* (Discover Computing, manuscript 2026-3956).

## Data
`dataset.csv` is the public COPD student dataset (101 patients) from Kaggle:
https://www.kaggle.com/datasets/prakharrathi25/copd-student-dataset (Rathi, 2023). It is redistributed here unchanged for reproducibility; please cite the original source.
Data-quality step applied in the code: CAT > 40 and HAD > 42 (outside the instruments' ranges) are set to missing and imputed inside the training folds.

## Targets and predictors
See `results/targets_and_predictors.csv`.
- **Severity classification:** target `COPDSEVERITY` (4 GOLD grades). 13 predictors: AGE, gender, PackHistory, smoking, MWT1Best, CAT, HAD, SGRQ, Diabetes, muscular, hypertension, AtrialFib, IHD.
- **QoL regression:** target `SGRQ` (0–100, higher = worse QoL). 16 predictors: AGE, gender, PackHistory, smoking, MWT1Best, FEV1, FEV1PRED, FVC, FVCPRED, CAT, HAD, Diabetes, muscular, hypertension, AtrialFib, IHD. Sensitivity analysis without CAT and HAD.
- **Excluded everywhere:** ID, row index, `copd` (exact duplicate of the label), AGEquartiles, MWT1, MWT2; for classification also FEV1, FEV1PRED, FVC, FVCPRED (they define the GOLD grade).
- **Adjusted association:** exposure = severe or very severe grade; outcome = SGRQ; adjustment set = AGE, gender, PackHistory, smoking, Diabetes, muscular, hypertension, AtrialFib, IHD.

## Validation protocol
Nested, repeated cross-validation: outer stratified 5-fold repeated 10 times (50 test folds), inner stratified 3-fold grid search. Median imputation, standard scaling, SMOTE (k = 3) and tuning are fitted only on training folds. Random seed 42. Outer test-fold patient IDs: `results/cv_splits_classification.csv`.

## How to reproduce
```
pip install -r requirements.txt
python copd_revision_analysis.py --data dataset.csv --out results   # main analysis, about 1.5 h on one CPU core
python supplementary_checks.py                                       # weighting balance, matched regression, QoL without CAT/HAD
python make_love_plot.py                                              # Figure 10 and unadjusted difference
```

## Where each manuscript item comes from
| Manuscript item | Output |
|---|---|
| Table 3 (leakage audit) | `results.json` → `leakage_audit` |
| Tables 4, 5, 7 | `results.json` → `cls_smote`, `classwise`, `cls_nosmote`, `pvals_*` |
| Table 6 (SMOTE class counts) | `results.json` → `smote_log` |
| Table 8 | `results.json` → `reg`; `supplementary_results.json` → `reg_without_cat_had` |
| Tables 9, 10 | `results.json` → `psm`; `supplementary_results.json` → `ipw_smd`, `matched_regression`, `unadjusted` |
| Table 11 | `results.json` → `xai_cls`, `xai_reg` |
| Figure 5 | `figures/fig_cm.png` |
| Figure 6 | `figures/fig_roc.png` |
| Figure 7 | `figures/fig_calibration.png` |
| Figure 8 | `figures/fig_reg_calibration.png` |
| Figure 9 | `figures/fig_ps_overlap.png` |
| Figure 10 | `figures/fig_love_plot.png` |
| Figure 11 | `figures/fig_shap_global_cls.png` |
| Figure 12 | `figures/fig_shap_rank_stability_cls.png` |
| Figures 13, 14 | `figures/fig_patient_shap_cls.png`, `figures/fig_patient_lime_cls.png` |
| Figure 15 | `figures/fig_shap_global_reg.png` |
| Figure 16 | `figures/fig_patient_shap_reg.png` |

Additional outputs (not in the manuscript): `fig_shap_rank_stability_reg.png`, `fig_patient_lime_reg.png`.

## Software
Python 3.12.3, numpy 2.4.4, pandas 3.0.2, scikit-learn 1.8.0, xgboost 3.4.1, imbalanced-learn 0.14.2, shap 0.53.0, lime 0.2.0.1, statsmodels 0.15.0. Full log: `results/run_log.txt`.

## Note
No survival analysis is included: the dataset has no follow-up time, event indicator or censoring variable.

## License
Code: MIT (see `LICENSE`). Data: subject to the terms of the original Kaggle dataset.
