# EEG Data Science Playbook

This document is a practical workflow for doing EEG data science in a rigorous, reproducible way.

## 1) Define the task first

- Problem type: binary classification, multi-class classification, regression, or anomaly detection.
- Primary target: clearly define what label is predicted and at what granularity (epoch, trial, session, or subject).
- Success metric: choose one primary metric before modeling.
- Baselines: define at least one simple baseline (majority class or logistic regression).

Checklist:
- [ ] Target variable is fixed and documented.
- [ ] Primary metric is fixed.
- [ ] Baseline models are defined.

## 2) Build a data inventory

Use a fixed table for every source file and transform.

Suggested inventory columns:
- source_path
- modality (EEG, ACC, BVP, EDA, etc.)
- sampling_rate_hz
- start_time
- end_time
- number_of_samples
- missing_ratio
- notes

Checklist:
- [ ] Every raw file has metadata.
- [ ] Time alignment assumptions are documented.

## 3) Split strategy before feature engineering

Always choose split policy before feature extraction.

Recommended policies:
- Subject-wise split for generalization to unseen people.
- Session-wise split if multiple sessions per subject exist.
- Time-aware split if there is temporal drift.

Hard rule:
- No information from test subjects can influence scaling, imputation, feature selection, or hyperparameter tuning.

Checklist:
- [ ] Group labels for split are defined (subject_id or session_id).
- [ ] Train/validation/test policy is written down.
- [ ] Leakage checks are included.

## 4) Preprocessing pipeline

Typical EEG preprocessing stages:
- Re-referencing
- Band-pass filter
- Notch filter
- Artifact rejection/correction
- Epoching
- Baseline correction

Store all chosen parameters in one place.

Checklist:
- [ ] Filter cutoffs and order are documented.
- [ ] Artifact handling rules are documented.
- [ ] Number of dropped epochs/channels is tracked.

## 5) Feature engineering

Use features aligned with the hypothesis.

Common feature families:
- Time-domain summary features
- Frequency band power
- Time-frequency features
- Connectivity features
- Learned features from deep models (only if data is large enough)

Checklist:
- [ ] Feature windows are documented.
- [ ] Feature computation is train-only then applied to val/test.
- [ ] Feature set size and missingness are reported.

## 6) Modeling and tuning

Start simple, then increase complexity.

Recommended order:
1. Logistic regression / linear SVM
2. Tree-based models
3. Sequence or deep models if justified by sample size

Checklist:
- [ ] Baseline beat by meaningful margin.
- [ ] Hyperparameter search uses grouped CV.
- [ ] Class imbalance handling is justified.

## 7) Evaluation and uncertainty

Report beyond one score.

Minimum reporting set:
- Primary metric (with confidence interval)
- Secondary metrics (precision/recall/F1, ROC-AUC or PR-AUC where relevant)
- Confusion matrix
- Per-subject performance

Checklist:
- [ ] Confidence intervals included.
- [ ] Per-subject breakdown included.
- [ ] Error analysis includes common failure patterns.

## 8) Statistical testing and robustness

For biomarker claims or model comparisons:
- Use permutation testing when assumptions are weak.
- Correct for multiple comparisons when testing many channels/features.
- Run sensitivity analysis on preprocessing choices.

Checklist:
- [ ] Statistical test and assumptions documented.
- [ ] Multiple-testing correction method documented.
- [ ] Sensitivity analysis completed.

## 9) Reproducibility package

Each experiment run should save:
- dataset version
- code commit hash
- random seed policy
- preprocessing parameters
- model hyperparameters
- feature list
- metrics and plots

Checklist:
- [ ] Run metadata is saved.
- [ ] Results are reproducible from one command.

## 10) Clinical or portfolio reporting

For health-focused projects, include:
- Intended use statement
- Population scope and exclusion criteria
- Bias and fairness checks
- Limitations and non-claims

Checklist:
- [ ] Intended use is written in plain language.
- [ ] Limitations are explicit.
- [ ] Clinical overclaims are avoided.

---

## Suggested folder conventions

- EEG/: cleaned tabular exports and metadata
- notebooks/: exploratory and modeling notebooks
- docs/: methods, assumptions, and experiment logs
- reports/: generated figures and result summaries
- templates/: reusable report and checklist templates

Current repo can adopt these incrementally without breaking existing files.
