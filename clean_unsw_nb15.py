#!/usr/bin/env python3
"""
Clean and standardize UNSW-NB15 dataset into one merge-ready CSV.

✅ No need for UNSW-NB15_features.csv — header is built in.
✅ Combines UNSW-NB15_1.csv through UNSW-NB15_4.csv.
✅ Normalizes column names, maps to a consistent schema,
   and prepares the file for cross-dataset use with CIC-IDS2017.

Output: data/processed/unsw_nb15_merge_ready.csv.gz
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

# -----------------------------
# Canonical UNSW-NB15 header (49 columns)
# -----------------------------
UNSW_HEADER = [
    "srcip","sport","dstip","dsport","proto","state","dur",
    "sbytes","dbytes","sttl","dttl","sload","dload","spkts","dpkts",
    "swin","dwin","stcpb","dtcpb","smeansz","dmeansz","trans_depth",
    "response_body_len","sjit","djit","stime","ltime","sintpkt","dintpkt",
    "tcprtt","synack","ackdat","is_sm_ips_ports","ct_state_ttl",
    "ct_flw_http_mthd","is_ftp_login","ct_ftp_cmd","ct_srv_src","ct_srv_dst",
    "ct_dst_ltm","ct_src_ltm","ct_src_dport_ltm","ct_dst_sport_ltm",
    "ct_dst_src_ltm","attack_cat","label"
]

# Standardized output schema (for merging later)
STANDARD_COLS = [
    "timestamp", "src_ip", "src_port", "dst_ip", "dst_port", "proto",
    "duration", "bytes_out", "bytes_in", "pkts_out", "pkts_in",
    "flow_bytes_total", "flow_pkts_total", "state", "service",
    "label", "attack_cat", "dataset"
]

# -----------------------------
# Utility functions
# -----------------------------
def coerce_numeric(s, dtype=float):
    """Convert safely to numeric, replacing invalids with NaN."""
    return pd.to_numeric(s, errors="coerce").astype(dtype, errors="ignore")

def unify_label_attack(df):
    """Standardize label and attack category."""
    if "label" in df.columns:
        df["label"] = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int)
    else:
        df["label"] = 0

    if "attack_cat" not in df.columns:
        df["attack_cat"] = np.where(df["label"] == 1, "attack", "benign")

    df["attack_cat"] = (
        df["attack_cat"].astype(str).str.strip().str.lower()
        .replace({"normal": "benign", "nan": "benign"})
        .fillna("benign")
    )
    return df

def map_unsw_to_standard(df):
    """Map UNSW columns to standardized schema."""
    out = pd.DataFrame(index=df.index, columns=STANDARD_COLS)

    out["timestamp"] = coerce_numeric(df.get("stime", pd.NA), float)
    out["src_ip"] = df.get("srcip", pd.Series(pd.NA, index=df.index))
    out["dst_ip"] = df.get("dstip", pd.Series(pd.NA, index=df.index))

    out["src_port"] = coerce_numeric(df.get("sport", pd.NA), "Int64")
    out["dst_port"] = coerce_numeric(df.get("dsport", pd.NA), "Int64")

    out["proto"] = df.get("proto", pd.Series(pd.NA, index=df.index)).astype(str).str.lower()
    out["duration"] = coerce_numeric(df.get("dur", pd.NA), float)

    out["bytes_out"] = coerce_numeric(df.get("sbytes", pd.NA), "Int64")
    out["bytes_in"] = coerce_numeric(df.get("dbytes", pd.NA), "Int64")
    out["pkts_out"] = coerce_numeric(df.get("spkts", pd.NA), "Int64")
    out["pkts_in"] = coerce_numeric(df.get("dpkts", pd.NA), "Int64")

    out["flow_bytes_total"] = (
        out["bytes_out"].fillna(0).astype(float) + out["bytes_in"].fillna(0).astype(float)
    )
    out["flow_pkts_total"] = (
        out["pkts_out"].fillna(0).astype(float) + out["pkts_in"].fillna(0).astype(float)
    )

    out["state"] = df.get("state", pd.Series(pd.NA, index=df.index)).astype(str).str.lower()
    out["service"] = df.get("service", pd.Series("unknown", index=df.index))

    df2 = unify_label_attack(df)
    out["label"] = df2["label"]
    out["attack_cat"] = df2["attack_cat"]
    out["dataset"] = "UNSW-NB15"

    # Remove impossible negatives
    for c in ["duration", "bytes_out", "bytes_in", "pkts_out", "pkts_in",
              "flow_bytes_total", "flow_pkts_total"]:
        out.loc[out[c] < 0, c] = pd.NA

    return out

# -----------------------------
# Main processing
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Clean and merge UNSW-NB15 dataset")
    parser.add_argument("--input-dir", required=True, help="Path to UNSW-NB15 folder")
    parser.add_argument("--pattern", default="UNSW-NB15_[1-4].csv", help="File pattern")
    parser.add_argument("--out", required=True, help="Output file path (.csv.gz)")
    parser.add_argument("--chunksize", type=int, default=250000)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    files = sorted(input_dir.glob(args.pattern))
    files = [f for f in files if "feature" not in f.name.lower()]
    if not files:
        raise SystemExit(f"No UNSW-NB15 CSV files found in {input_dir}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Found {len(files)} parts. Starting merge...")
    total_rows = 0
    first_write = True

    for fp in files:
        print(f"[INFO] Processing {fp.name} ...")
        for chunk in pd.read_csv(
            fp,
            header=None,
            names=UNSW_HEADER,
            chunksize=args.chunksize,
            low_memory=False,
            na_values=["?", "-", "NaN", "nan", ""]
        ):
            # Clean text fields
            for c in chunk.select_dtypes(include=["object"]).columns:
                chunk[c] = chunk[c].astype(str).str.strip()

            chunk.replace([np.inf, -np.inf], np.nan, inplace=True)

            mapped = map_unsw_to_standard(chunk)

            mode = "w" if first_write else "a"
            mapped.to_csv(out_path, mode=mode, header=first_write, index=False, compression="gzip")
            first_write = False

            total_rows += len(mapped)

            print(f"  Chunk processed: {len(mapped)} rows")

    print(f"[DONE] Wrote {total_rows:,} rows to {out_path}")
    print(f"[INFO] Columns: {STANDARD_COLS}")

if __name__ == "__main__":
    main()
