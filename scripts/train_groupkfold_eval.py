#!/usr/bin/env python3
"""Train EEG classifiers with GroupKFold on train split, then evaluate on test split.

Features are loaded from outputs/processed/features_with_split.csv by default.
The script:
1) Uses only rows where split == 'train' for GroupKFold model selection.
2) Keeps subject groups isolated during CV with GroupKFold.
3) Retrains the selected model on full train split.
4) Evaluates once on held-out test split.
5) Writes a baseline model report and JSON metrics artifact.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class MetricSummary:
    accuracy_mean: float
    accuracy_std: float
    f1_macro_mean: float
    f1_macro_std: float
    balanced_accuracy_mean: float
    balanced_accuracy_std: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train with GroupKFold on train split, evaluate once on test split."
    )
    parser.add_argument(
        "--input",
        default="outputs/processed/features_with_split.csv",
        help="Input feature table with split column (default: outputs/processed/features_with_split.csv)",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/reports",
        help="Output directory for reports (default: outputs/reports)",
    )
    parser.add_argument("--group-col", default="id", help="Group column (default: id)")
    parser.add_argument("--label-col", default="label", help="Label column (default: label)")
    parser.add_argument("--split-col", default="split", help="Split column (default: split)")
    parser.add_argument(
        "--n-splits",
        type=int,
        default=3,
        help="GroupKFold n_splits on train data (default: 3)",
    )
    parser.add_argument(
        "--imbalance-threshold",
        type=float,
        default=2.0,
        help="If max_class_count/min_class_count >= threshold, enable class-weight balancing (default: 2.0)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for stochastic models (default: 42)",
    )
    return parser.parse_args()


def _ensure_columns(df: pd.DataFrame, required: List[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required column(s): {missing}")


def _select_feature_columns(df: pd.DataFrame, group_col: str, label_col: str, split_col: str) -> List[str]:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    excluded = {group_col, label_col, split_col}
    feature_cols = [c for c in numeric_cols if c not in excluded]
    if not feature_cols:
        raise RuntimeError("No numeric feature columns found for training.")
    return feature_cols


def _imbalance_ratio(y: pd.Series) -> Tuple[float, Dict[str, int]]:
    counts = y.value_counts()
    if counts.empty:
        raise RuntimeError("Label column is empty after filtering splits.")
    min_count = int(counts.min())
    max_count = int(counts.max())
    ratio = float(max_count / max(min_count, 1))
    return ratio, {str(k): int(v) for k, v in counts.to_dict().items()}


def _cv_groupkfold_metrics(
    estimator: Pipeline,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    groups: pd.Series,
    n_splits: int,
) -> Tuple[MetricSummary, List[Dict[str, float]]]:
    unique_groups = groups.nunique()
    if unique_groups < 2:
        raise RuntimeError("Need at least 2 unique groups for GroupKFold.")

    effective_splits = min(n_splits, unique_groups)
    if effective_splits < 2:
        raise RuntimeError("Effective GroupKFold n_splits is < 2.")

    gkf = GroupKFold(n_splits=effective_splits)
    fold_metrics: List[Dict[str, float]] = []

    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(x_train, y_train, groups=groups), start=1):
        model = clone(estimator)
        model.fit(x_train.iloc[train_idx], y_train.iloc[train_idx])
        y_pred = model.predict(x_train.iloc[val_idx])
        y_true = y_train.iloc[val_idx]

        metrics = {
            "fold": float(fold_idx),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        }
        fold_metrics.append(metrics)

    acc = np.asarray([m["accuracy"] for m in fold_metrics], dtype=float)
    f1m = np.asarray([m["f1_macro"] for m in fold_metrics], dtype=float)
    bacc = np.asarray([m["balanced_accuracy"] for m in fold_metrics], dtype=float)

    summary = MetricSummary(
        accuracy_mean=float(acc.mean()),
        accuracy_std=float(acc.std(ddof=0)),
        f1_macro_mean=float(f1m.mean()),
        f1_macro_std=float(f1m.std(ddof=0)),
        balanced_accuracy_mean=float(bacc.mean()),
        balanced_accuracy_std=float(bacc.std(ddof=0)),
    )
    return summary, fold_metrics


def _test_metrics(estimator: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, object]:
    y_pred = estimator.predict(x_test)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "classification_report": report,
    }


def _write_baseline_report(
    out_path: Path,
    input_path: str,
    feature_count: int,
    train_rows: int,
    test_rows: int,
    imbalance_ratio: float,
    class_weight_enabled: bool,
    cv_summary: MetricSummary,
    test_result: Dict[str, object],
) -> None:
    text = "\n".join(
        [
            "# Baseline Model Report",
            "",
            f"- Generated at (UTC): {datetime.now(timezone.utc).isoformat()}",
            f"- Input file: {input_path}",
            f"- Feature count: {feature_count}",
            f"- Train rows: {train_rows}",
            f"- Test rows: {test_rows}",
            f"- Train imbalance ratio (max/min): {imbalance_ratio:.2f}x",
            f"- Automatic class-weight handling enabled: {class_weight_enabled}",
            "",
            "## Baseline Model",
            "- Algorithm: DummyClassifier(strategy='most_frequent')",
            "- Purpose: lower-bound sanity baseline",
            "",
            "## GroupKFold CV On Train",
            f"- Accuracy: {cv_summary.accuracy_mean:.4f} +/- {cv_summary.accuracy_std:.4f}",
            f"- Macro F1: {cv_summary.f1_macro_mean:.4f} +/- {cv_summary.f1_macro_std:.4f}",
            (
                "- Balanced Accuracy: "
                f"{cv_summary.balanced_accuracy_mean:.4f} +/- {cv_summary.balanced_accuracy_std:.4f}"
            ),
            "",
            "## Final Test Evaluation",
            f"- Accuracy: {test_result['accuracy']:.4f}",
            f"- Macro F1: {test_result['f1_macro']:.4f}",
            f"- Balanced Accuracy: {test_result['balanced_accuracy']:.4f}",
            "",
            "## Interpretation",
            "- If your candidate model does not beat this baseline on macro F1, it is not learning useful class-discriminative structure.",
            "- Macro F1 is prioritized because label frequencies are imbalanced.",
            "",
        ]
    )
    out_path.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)
    _ensure_columns(df, [args.group_col, args.label_col, args.split_col])

    train_df = df[df[args.split_col] == "train"].copy()
    test_df = df[df[args.split_col] == "test"].copy()

    if train_df.empty or test_df.empty:
        raise RuntimeError("Both train and test splits must be non-empty.")

    feature_cols = _select_feature_columns(df, args.group_col, args.label_col, args.split_col)

    x_train = train_df[feature_cols]
    y_train = train_df[args.label_col].astype(str)
    g_train = train_df[args.group_col].astype(str)

    x_test = test_df[feature_cols]
    y_test = test_df[args.label_col].astype(str)

    imbalance_ratio, train_label_counts = _imbalance_ratio(y_train)
    class_weight_enabled = imbalance_ratio >= args.imbalance_threshold

    lr_class_weight = "balanced" if class_weight_enabled else None
    rf_class_weight = "balanced_subsample" if class_weight_enabled else None

    candidate_models: Dict[str, Pipeline] = {
        "logreg": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=4000,
                        solver="lbfgs",
                        class_weight=lr_class_weight,
                        random_state=args.seed,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=100,
                        max_depth=20,
                        min_samples_leaf=2,
                        class_weight=rf_class_weight,
                        random_state=args.seed,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }

    baseline_model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", DummyClassifier(strategy="most_frequent")),
        ]
    )

    baseline_cv_summary, baseline_folds = _cv_groupkfold_metrics(
        baseline_model, x_train, y_train, g_train, args.n_splits
    )
    baseline_fit = clone(baseline_model).fit(x_train, y_train)
    baseline_test = _test_metrics(baseline_fit, x_test, y_test)

    results: Dict[str, object] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "input": str(input_path),
            "output_dir": str(output_dir),
            "group_col": args.group_col,
            "label_col": args.label_col,
            "split_col": args.split_col,
            "n_splits": args.n_splits,
            "imbalance_threshold": args.imbalance_threshold,
            "seed": args.seed,
        },
        "data": {
            "rows_total": int(len(df)),
            "rows_train": int(len(train_df)),
            "rows_test": int(len(test_df)),
            "groups_train": int(g_train.nunique()),
            "groups_test": int(test_df[args.group_col].nunique()),
            "features": feature_cols,
            "feature_count": int(len(feature_cols)),
            "train_label_counts": train_label_counts,
            "train_imbalance_ratio": imbalance_ratio,
            "class_weight_enabled": class_weight_enabled,
        },
        "baseline": {
            "model": "dummy_most_frequent",
            "cv": asdict(baseline_cv_summary),
            "cv_folds": baseline_folds,
            "test": baseline_test,
        },
        "candidates": {},
    }

    best_name = ""
    best_score = -np.inf
    best_model: Pipeline | None = None

    for name, estimator in candidate_models.items():
        cv_summary, cv_folds = _cv_groupkfold_metrics(estimator, x_train, y_train, g_train, args.n_splits)
        fitted = clone(estimator).fit(x_train, y_train)
        test_result = _test_metrics(fitted, x_test, y_test)

        results["candidates"][name] = {
            "cv": asdict(cv_summary),
            "cv_folds": cv_folds,
            "test": test_result,
        }

        if cv_summary.f1_macro_mean > best_score:
            best_score = cv_summary.f1_macro_mean
            best_name = name
            best_model = fitted

    if best_model is None:
        raise RuntimeError("No candidate model was trained.")

    final_test = _test_metrics(best_model, x_test, y_test)
    results["selected_model"] = {
        "name": best_name,
        "selection_metric": "cv_f1_macro_mean",
        "selection_score": float(best_score),
        "test": final_test,
    }

    metrics_path = output_dir / "training_results.json"
    metrics_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    baseline_report_path = output_dir / "baseline_model_report.md"
    _write_baseline_report(
        out_path=baseline_report_path,
        input_path=str(input_path),
        feature_count=len(feature_cols),
        train_rows=len(train_df),
        test_rows=len(test_df),
        imbalance_ratio=imbalance_ratio,
        class_weight_enabled=class_weight_enabled,
        cv_summary=baseline_cv_summary,
        test_result=baseline_test,
    )

    print(f"Saved metrics JSON: {metrics_path}")
    print(f"Saved baseline report: {baseline_report_path}")
    print(f"Class-weight handling enabled: {class_weight_enabled} (ratio={imbalance_ratio:.2f}x)")
    print(f"Selected model: {best_name} (CV macro F1={best_score:.4f})")
    print(
        "Final test metrics | "
        f"accuracy={final_test['accuracy']:.4f}, "
        f"f1_macro={final_test['f1_macro']:.4f}, "
        f"balanced_accuracy={final_test['balanced_accuracy']:.4f}"
    )


if __name__ == "__main__":
    main()
