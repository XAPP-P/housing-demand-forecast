"""
ZCTA geometry preprocessing: simplify + split by state.

Input  : data_raw/tl_2024_us_zcta520.shp  (~800 MB, ~33k polygons)
Output : data/zcta_by_state/<STATE>.parquet  (one file per state, ~5-10 MB each)

Why this step exists
--------------------
Streamlit Community Cloud gives you ~1 GB of RAM. Loading the entire national
shapefile blows through that. Instead, we:

1. Simplify every polygon (tolerance=0.001 deg ≈ 100 m) so the browser doesn't
   have to render dense coastline vertices the user can't see at state zoom.
2. Join with Zillow's ZIP→state mapping so we know which state each ZCTA
   belongs to. ZCTAs not covered by Zillow are dropped (they have no price
   data to show anyway).
3. Split into one parquet per state. The running app then only loads the
   state(s) the user has selected.

Run from the repo root:
    python data_pipeline/prepare_zcta.py
"""

from __future__ import annotations

from pathlib import Path
import sys

import geopandas as gpd
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_SHP = REPO_ROOT / "data_raw" / "tl_2024_us_zcta520.shp"
RAW_ZILLOW = REPO_ROOT / "data_raw" / "Zip_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv"
OUTPUT_DIR = REPO_ROOT / "data" / "zcta_by_state"

# Simplification tolerance in degrees (~111 km per degree at equator, so
# 0.001 ≈ 100 m — more than enough detail for state-level visualization).
SIMPLIFY_TOL = 0.001


def main() -> int:
    # ------------------------------------------------------------------
    # Sanity checks
    # ------------------------------------------------------------------
    for f in [RAW_SHP, RAW_ZILLOW]:
        if not f.exists():
            print(f"ERROR: missing {f}")
            return 1

    # ------------------------------------------------------------------
    # Load Zillow metadata — we only need the ZIP → state mapping + a few
    # human-readable fields for tooltips. Skip the monthly price columns.
    # ------------------------------------------------------------------
    print("Reading Zillow metadata for ZIP→state mapping…")
    meta_cols = ["RegionName", "State", "StateName", "City", "Metro", "CountyName"]
    available_meta = [c for c in meta_cols if c in
                      pd.read_csv(RAW_ZILLOW, nrows=0).columns]
    zillow_meta = pd.read_csv(RAW_ZILLOW, usecols=available_meta)
    zillow_meta["zip_code"] = zillow_meta["RegionName"].astype(str).str.zfill(5)
    zillow_meta = zillow_meta.drop(columns=["RegionName"]).drop_duplicates("zip_code")
    print(f"  Zillow ZIPs with state metadata: {len(zillow_meta):,}")

    # ------------------------------------------------------------------
    # Load shapefile — this is the slow step (~30–60 s)
    # ------------------------------------------------------------------
    print(f"\nReading shapefile: {RAW_SHP.name} "
          f"({RAW_SHP.stat().st_size / 1024 / 1024:.0f} MB)…")
    gdf = gpd.read_file(RAW_SHP)
    print(f"  loaded: {len(gdf):,} ZCTAs   CRS: {gdf.crs}")

    # ------------------------------------------------------------------
    # Identify the ZCTA column — column name varies by TIGER vintage
    # (ZCTA5CE10, ZCTA5CE20, GEOID20, etc.)
    # ------------------------------------------------------------------
    candidates = [c for c in gdf.columns if "ZCTA" in c.upper()]
    if not candidates:
        candidates = [c for c in gdf.columns if "GEOID" in c.upper()]
    if not candidates:
        print("ERROR: could not find a ZCTA or GEOID column in shapefile.")
        return 1
    zcta_col = candidates[0]
    print(f"  ZCTA column: {zcta_col}")

    gdf = gdf.rename(columns={zcta_col: "zip_code"})
    gdf["zip_code"] = gdf["zip_code"].astype(str).str.zfill(5)

    # ------------------------------------------------------------------
    # Simplify geometries in-place
    # ------------------------------------------------------------------
    print(f"\nSimplifying geometries (tolerance={SIMPLIFY_TOL})…")
    gdf["geometry"] = gdf["geometry"].simplify(
        tolerance=SIMPLIFY_TOL, preserve_topology=True
    )

    # ------------------------------------------------------------------
    # Merge with state metadata — inner join keeps only ZIPs Zillow covers
    # ------------------------------------------------------------------
    merged = gdf[["zip_code", "geometry"]].merge(
        zillow_meta, on="zip_code", how="inner"
    )
    merged = merged.dropna(subset=["State"])
    print(f"  ZCTAs with both geometry and state info: {len(merged):,}")

    # ------------------------------------------------------------------
    # Split by state → one parquet per state
    # ------------------------------------------------------------------
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nWriting per-state parquet files to {OUTPUT_DIR.relative_to(REPO_ROOT)}/")
    total_bytes = 0
    for state, group in merged.groupby("State"):
        out_path = OUTPUT_DIR / f"{state}.parquet"
        group.to_parquet(out_path, index=False, compression="zstd")
        size_kb = out_path.stat().st_size / 1024
        total_bytes += out_path.stat().st_size
        print(f"  {state}: {len(group):>5} ZIPs → {out_path.name} ({size_kb:>6.1f} KB)")

    print(f"\n[OK] Done. {len(merged):,} ZCTAs split across "
          f"{merged['State'].nunique()} states "
          f"(total: {total_bytes / 1024 / 1024:.1f} MB).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
