"""
Forecast preprocessing: CSV → long-format parquet aligned with zillow_prices_long.

Input  : data_raw/zipcode_price_forecast_2026.csv
         Columns: Zipcode, Date, Forecasted_Price, Predicted_Return, Cumulative_Growth
Output : data/zillow_forecast_long.parquet
         Columns: zip_code, date, zhvi_forecast   (aligned with zillow_prices_long)

The forecast comes from the Module 2 Random Forest (Part B-1), which produced
month-ahead predictions for Feb–Jun 2026 for every ZIP code Zillow covers.
We drop the Predicted_Return and Cumulative_Growth columns — the dashboard
only needs the level forecast, not the return decomposition.

Run from the repo root:
    python data_pipeline/prepare_forecast.py
"""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_CSV = REPO_ROOT / "data_raw" / "zipcode_price_forecast_2026.csv"
OUTPUT = REPO_ROOT / "data" / "zillow_forecast_long.parquet"


def main() -> int:
    if not RAW_CSV.exists():
        print(f"ERROR: raw file not found: {RAW_CSV}")
        print("Place the forecast CSV in data_raw/ and re-run.")
        return 1

    print(f"Reading: {RAW_CSV.name} "
          f"({RAW_CSV.stat().st_size / 1024:.1f} KB)")
    df = pd.read_csv(RAW_CSV)
    print(f"  loaded: {len(df):,} rows")

    # Normalize column names to match the historical long table.
    df = df.rename(columns={
        "Zipcode": "zip_code",
        "Date": "date",
        "Forecasted_Price": "zhvi_forecast",
    })
    df["zip_code"] = df["zip_code"].astype(str).str.zfill(5)
    df["date"] = pd.to_datetime(df["date"])
    df["zhvi_forecast"] = pd.to_numeric(df["zhvi_forecast"], errors="coerce")

    # Keep only the columns the dashboard needs.
    df = df[["zip_code", "date", "zhvi_forecast"]].dropna(subset=["zhvi_forecast"])
    df = df.sort_values(["zip_code", "date"]).reset_index(drop=True)

    print(f"  distinct ZIPs: {df['zip_code'].nunique():,}")
    print(f"  date range: {df['date'].min().strftime('%Y-%m')} "
          f"to {df['date'].max().strftime('%Y-%m')}")
    print(f"  rows after clean: {len(df):,}")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT, index=False, compression="zstd")
    size_kb = OUTPUT.stat().st_size / 1024
    print(f"\n[OK] Saved: {OUTPUT.relative_to(REPO_ROOT)}  ({size_kb:.1f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
