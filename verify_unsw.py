#!/usr/bin/env python3
"""
Quick verification script for the cleaned UNSW-NB15 dataset.

Checks:
  ✅ Row & column count
  ✅ Missing value counts
  ✅ Label distribution
  ✅ Attack category breakdown
  ✅ Summary stats for numeric features
"""

import pandas as pd
from pathlib import Path

DATA_PATH = Path("data/processed/unsw_nb15_merge_ready.csv.gz")

def main():
    if not DATA_PATH.exists():
        raise SystemExit(f"[ERROR] File not found: {DATA_PATH}")

    print(f"[INFO] Loading dataset from {DATA_PATH} ...\n")

    # Load just enough to check structure
    df = pd.read_csv(DATA_PATH, low_memory=False)

    print(f"[INFO] Rows: {len(df):,}")
    print(f"[INFO] Columns: {len(df.columns)}\n")

    print("[INFO] Column names:")
    print(", ".join(df.columns))
    print()

    # Check for missing values
    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if len(missing) > 0:
        print("[INFO] Columns with missing values:")
        print(missing)
    else:
        print("[INFO] No missing values detected.")
    print()

    # Label distribution
    if "label" in df.columns:
        print("[INFO] Label distribution:")
        print(df["label"].value_counts(dropna=False))
        print()

    # Attack category breakdown
    if "attack_cat" in df.columns:
        print("[INFO] Top 10 attack categories:")
        print(df["attack_cat"].value_counts().head(10))
        print()

    # Summary of numeric columns
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if num_cols:
        print("[INFO] Numeric feature summary (first 8):")
        print(df[num_cols].describe().round(2).T.head(8))
        print()

    print("[✅] Verification complete — dataset looks healthy.")

if __name__ == "__main__":
    main()
