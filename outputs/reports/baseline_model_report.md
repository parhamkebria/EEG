# Baseline Model Report

- Generated at (UTC): 
- Input file: outputs/processed/features_with_split.csv
- Feature count: 21
- Train rows: 7058
- Test rows: 1628
- Train imbalance ratio (max/min): 31.62x
- Automatic class-weight handling enabled: True

## Baseline Model
- Algorithm: DummyClassifier(strategy='most_frequent')
- Purpose: lower-bound sanity baseline

## GroupKFold CV On Train
- Accuracy: 0.0935 +/- 0.0003
- Macro F1: 0.0026 +/- 0.0000
- Balanced Accuracy: 0.0149 +/- 0.0000

## Final Test Evaluation
- Accuracy: 0.0915
- Macro F1: 0.0025
- Balanced Accuracy: 0.0149

## Interpretation
- If your candidate model does not beat this baseline on macro F1, it is not learning useful class-discriminative structure.
- Macro F1 is prioritized because label frequencies are imbalanced.
