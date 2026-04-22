# EEG: ElectroEncephaloGram

A repositoruy for EEG data processing and analysis.

## Workflow Starter

- [EEG data science playbook](docs/eeg-data-science-playbook.md)
- [Model report template](templates/model-report-template.md)

## Step-by-step quick start

1. Read the playbook and lock your prediction target plus split strategy.
2. Use the model report template to define experiment metadata before training.
3. Implement preprocessing in notebooks with leakage-safe train/validation/test boundaries.
4. Record per-subject performance, confidence intervals, and failure modes.

## Implementation

Run preprocessing and a leakage-safe subject-wise split:

```bash
python scripts/preprocess_and_split.py --drop-unlabeled
```

Generated files:

- outputs/processed/features_with_split.csv
- outputs/processed/train.csv
- outputs/processed/val.csv
- outputs/processed/test.csv
- outputs/processed/summary.json

## Known issues (shortcomings) of the dataset:

- Extremely imbalanced multiclass target.
  - Train imbalance is about 31.6x (largest vs smallest class), and many classes have very few examples. That crushes macro F1 even with class weights.
- Too many classes for the available per-class signal.
  - Solving a large multiclass problem with sparse support in many labels, so each one-vs-rest boundary is weak.
- Group split is correct but makes generalization harder.
  - Training on 21 subjects and tested on 5 unseen subjects. That is the right leakage-safe evaluation, but subject shift is large in EEG.
- Features are coarse summaries.
  - Current features are aggregate stats + bandpower snapshots. They may not capture temporal dynamics that separate many fine-grained labels