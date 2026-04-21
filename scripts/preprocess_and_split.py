#!/usr/bin/env python3
"""Preprocess EEG tabular data and create leakage-safe subject-wise splits.

This script is designed for the current repository schema:
- input EEG table: EEG/eeg-data.csv
- group key: column `id` (subject identifier)
- label column: `label`

Outputs are written to `outputs/processed/` by default.
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd


BAND_NAMES = [
    "delta",
    "theta",
    "low_alpha",
    "high_alpha",
    "low_beta",
    "high_beta",
    "low_gamma",
    "mid_gamma",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess EEG data and split by subject.")
    parser.add_argument(
        "--input",
        default="EEG/eeg-data.csv",
        help="Path to EEG CSV file (default: EEG/eeg-data.csv)",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/processed",
        help="Output directory (default: outputs/processed)",
    )
    parser.add_argument(
        "--group-col",
        default="id",
        help="Group column for leakage-safe split (default: id)",
    )
    parser.add_argument(
        "--label-col",
        default="label",
        help="Label column name (default: label)",
    )
    parser.add_argument(
        "--drop-unlabeled",
        action="store_true",
        help="Drop rows whose label is 'unlabeled'.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Training group ratio (default: 0.7)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Validation group ratio (default: 0.15)",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="Test group ratio (default: 0.15)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for group shuffle (default: 42)",
    )
    return parser.parse_args()


def parse_list_cell(value: object) -> List[float]:
    if isinstance(value, list):
        return [float(v) for v in value]
    if pd.isna(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, (list, tuple)):
            return [float(v) for v in parsed]
    except (ValueError, SyntaxError):
        pass
    return []


def summarize_vector(values: Sequence[float], prefix: str) -> Dict[str, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return {
            f"{prefix}_len": 0.0,
            f"{prefix}_mean": np.nan,
            f"{prefix}_std": np.nan,
            f"{prefix}_min": np.nan,
            f"{prefix}_max": np.nan,
            f"{prefix}_median": np.nan,
            f"{prefix}_q05": np.nan,
            f"{prefix}_q95": np.nan,
            f"{prefix}_abs_mean": np.nan,
            f"{prefix}_clip_ratio": np.nan,
        }

    return {
        f"{prefix}_len": float(arr.size),
        f"{prefix}_mean": float(arr.mean()),
        f"{prefix}_std": float(arr.std()),
        f"{prefix}_min": float(arr.min()),
        f"{prefix}_max": float(arr.max()),
        f"{prefix}_median": float(np.median(arr)),
        f"{prefix}_q05": float(np.quantile(arr, 0.05)),
        f"{prefix}_q95": float(np.quantile(arr, 0.95)),
        f"{prefix}_abs_mean": float(np.mean(np.abs(arr))),
        f"{prefix}_clip_ratio": float(np.mean(np.abs(arr) >= 2047.0)),
    }


def bandpower_features(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        return {f"bp_{name}": np.nan for name in BAND_NAMES}

    out = {f"bp_{name}": np.nan for name in BAND_NAMES}
    for i, value in enumerate(values[: len(BAND_NAMES)]):
        out[f"bp_{BAND_NAMES[i]}"] = float(value)
    return out


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    for _, row in df.iterrows():
        raw_vals = parse_list_cell(row.get("raw_values"))
        power_vals = parse_list_cell(row.get("eeg_power"))

        feature_row: Dict[str, float] = {}
        feature_row.update(summarize_vector(raw_vals, prefix="raw"))
        feature_row.update(bandpower_features(power_vals))
        rows.append(feature_row)

    feature_df = pd.DataFrame(rows)
    keep_cols = [
        "id",
        "indra_time",
        "browser_latency",
        "reading_time",
        "attention_esense",
        "meditation_esense",
        "signal_quality",
        "createdAt",
        "updatedAt",
        "label",
    ]
    keep_existing = [c for c in keep_cols if c in df.columns]
    return pd.concat([df[keep_existing].reset_index(drop=True), feature_df], axis=1)


def grouped_split(
    df: pd.DataFrame,
    group_col: str,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> pd.Series:
    total = train_ratio + val_ratio + test_ratio
    if not np.isclose(total, 1.0):
        raise ValueError(f"Split ratios must sum to 1.0, got {total}")

    groups = df[group_col].dropna().astype(str).unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)

    n_groups = len(groups)
    n_train = int(round(n_groups * train_ratio))
    n_val = int(round(n_groups * val_ratio))

    if n_train <= 0 or n_val <= 0 or n_train + n_val >= n_groups:
        raise ValueError(
            "Invalid split sizes after rounding. Try adjusting ratios or use more groups."
        )

    train_groups = set(groups[:n_train])
    val_groups = set(groups[n_train : n_train + n_val])
    test_groups = set(groups[n_train + n_val :])

    split = []
    for value in df[group_col].astype(str):
        if value in train_groups:
            split.append("train")
        elif value in val_groups:
            split.append("val")
        elif value in test_groups:
            split.append("test")
        else:
            split.append("unknown")

    split_series = pd.Series(split, index=df.index, name="split")
    if (split_series == "unknown").any():
        raise RuntimeError("Some rows were not assigned to any split.")
    return split_series


def make_summary(df: pd.DataFrame, group_col: str, label_col: str) -> Dict[str, object]:
    return {
        "rows": int(len(df)),
        "groups": int(df[group_col].nunique()),
        "labels": df[label_col].value_counts(dropna=False).to_dict(),
        "split_rows": df["split"].value_counts(dropna=False).to_dict(),
        "split_groups": (
            df.groupby("split")[group_col].nunique().sort_index().to_dict()
        ),
    }


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    raw_df = pd.read_csv(input_path)

    if args.group_col not in raw_df.columns:
        raise KeyError(f"Group column not found: {args.group_col}")
    if args.label_col not in raw_df.columns:
        raise KeyError(f"Label column not found: {args.label_col}")

    df = raw_df.copy()
    df[args.label_col] = df[args.label_col].astype(str)

    if args.drop_unlabeled:
        df = df[df[args.label_col].str.lower() != "unlabeled"].copy()

    if df.empty:
        raise RuntimeError("No rows left after filtering. Check label filters.")

    features_df = build_features(df)

    features_df["split"] = grouped_split(
        features_df,
        group_col=args.group_col,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    features_df.to_csv(output_dir / "features_with_split.csv", index=False)
    features_df[features_df["split"] == "train"].to_csv(output_dir / "train.csv", index=False)
    features_df[features_df["split"] == "val"].to_csv(output_dir / "val.csv", index=False)
    features_df[features_df["split"] == "test"].to_csv(output_dir / "test.csv", index=False)

    summary = make_summary(features_df, group_col=args.group_col, label_col=args.label_col)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Saved outputs to: {output_dir}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
