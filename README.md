# EEG

## ElectroEncephaloGram

A repositoruy for EEG data processing and analysis.

## Workflow Starter

- EEG data science playbook: [docs/eeg-data-science-playbook.md](docs/eeg-data-science-playbook.md)
- Model report template: [templates/model-report-template.md](templates/model-report-template.md)

## Step-by-step quick start

1. Read the playbook and lock your prediction target plus split strategy.
2. Use the model report template to define experiment metadata before training.
3. Implement preprocessing in notebooks with leakage-safe train/validation/test boundaries.
4. Record per-subject performance, confidence intervals, and failure modes.

## Step 2 implementation (in this repo)

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
