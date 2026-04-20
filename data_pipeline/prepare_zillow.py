"""
Zillow ZHVI preprocessing: wide → long format.

Input  : data_raw/Zip_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv
Output : data/zillow_prices_long.parquet

The raw CSV has one column per month (2000-01-31 through the latest).
Dashboard queries are far easier against a long-format table:
    (zip_code, date, zhvi) — one row per ZIP per month.

Run from the repo root:
    python data_pipeline/prepare_zillow.py
"""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


# Paths resolved relative to repo root (assumes script is run from repo root or
# anywhere — we anchor to the repo root by walking up from this file).
REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_CSV = REPO_ROOT / "data_raw" / "Zip_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv"
OUTPUT = REPO_ROOT / "data" / "zillow_prices_long.parquet"


def is_date_column(col: str) -> bool:
    """A Zillow date column looks like '2024-09-30' — 10 chars with dashes."""
    return isinstance(col, str) and len(col) == 10 and col.count("-") == 2


def main() -> int:
    if not RAW_CSV.exists():
        print(f"ERROR: raw file not found: {RAW_CSV}")
        print("Place the Zillow ZHVI CSV in data_raw/ and re-run.")
        return 1

    print(f"Reading: {RAW_CSV.name} "
          f"({RAW_CSV.stat().st_size / 1024 / 1024:.1f} MB)")
    df = pd.read_csv(RAW_CSV)
    print(f"  loaded: {df.shape[0]:,} ZIPs × {df.shape[1]:,} columns")

    date_cols = [c for c in df.columns if is_date_column(c)]
    if not date_cols:
        print("ERROR: no date-like columns detected.")
        return 1
    print(f"  date columns: {len(date_cols)} "
          f"(from {min(date_cols)} to {max(date_cols)})")

    # Melt wide to long — keep only RegionName + date columns.
    long_df = df[["RegionName"] + date_cols].melt(
        id_vars="RegionName",
        var_name="date",
        value_name="zhvi",
    )

    # Clean up types and drop rows with no price.
    long_df["zip_code"] = long_df["RegionName"].astype(str).str.zfill(5)
    long_df["date"] = pd.to_datetime(long_df["date"])
    long_df["zhvi"] = pd.to_numeric(long_df["zhvi"], errors="coerce")
    long_df = long_df.dropna(subset=["zhvi"])
    long_df = long_df[["zip_code", "date", "zhvi"]].sort_values(
        ["zip_code", "date"]
    ).reset_index(drop=True)

    print(f"  long-form rows (after dropna): {len(long_df):,}")
    print(f"  distinct ZIPs: {long_df['zip_code'].nunique():,}")

    # Write parquet.
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    long_df.to_parquet(OUTPUT, index=False, compression="zstd")
    size_mb = OUTPUT.stat().st_size / 1024 / 1024
    print(f"\n[OK] Saved: {OUTPUT.relative_to(REPO_ROOT)}  ({size_mb:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
