# US Housing Demand Forecast Dashboard

Interactive, ZIP-code-level dashboard integrating housing price forecasts with residential construction supply-chain signals.

**Course:** UC Berkeley INDENG 243 — Analytics Lab (Spring 2026)
**Team:** Group 9
**Module 3 Deliverable:** Interactive communication tool

---

## Overview

This dashboard translates the models built in Modules 1–2 into a decision-support tool for two distinct audiences:

- **Homebuyers and sellers** — view historical ZIP-code-level median home values across the United States, play through history to see how markets have moved, and (optionally) overlay school-district ratings.
- **Supply-chain actors** — builders, material suppliers, and contractors can explore how mortgage rates, tariff regimes, and material-cost regimes interact with the construction pipeline (permits → starts → completions) to inform production planning.

Rather than surfacing a single national headline forecast, the dashboard emphasizes **geographic granularity** and **scenario exploration**, reflecting the reality that housing decisions are deeply local and supply-chain decisions are deeply conditional.

## Live Demo

Deployed via Streamlit Community Cloud: *(link coming)*

## Model Architecture

The underlying forecasting system is a two-layer parallel pipeline:

- **Part A — Macro baseline.** Ridge regression on national and Census-division HPI, capturing aggregate demand signals.
- **Part B-1 — ZIP-level price model.** Random Forest predicting local home values, using Zillow ZHVI, FRED mortgage rates, and engineered features (Affordability Index, Relative Spread, Price Momentum) with Part A outputs as upstream inputs.
- **Part B-2 — Division-level sales model.** Random Forest predicting sales volume, parallel to B-1 and drawing on Realtor sales data alongside Part A outputs.

B-1 and B-2 are **parallel, not sequential** — they share upstream inputs from Part A but produce independent predictions at different geographic granularities.

## Repository Layout

```
housing-demand-forecast/
├── app.py                      # Streamlit entry point
├── requirements.txt            # Python dependencies
├── packages.txt                # System libs for Streamlit Cloud (GDAL)
├── .streamlit/
│   └── config.toml             # Theme / server config
├── components/
│   ├── price_map.py            # Tab 1: ZIP-level price map
│   └── supply_chain.py         # Tab 2: Supply-chain decision tool
├── data_pipeline/
│   ├── prepare_zillow.py       # Wide → long transformation of Zillow ZHVI
│   └── prepare_zcta.py         # Shapefile simplification + per-state split
├── data/
│   ├── zillow_prices_long.parquet
│   └── zcta_by_state/*.parquet
├── notebooks/                  # Research notebooks from earlier modules
└── data_raw/                   # (git-ignored) raw source files
```

## Data Sources

| Source | Used for | Notes |
|---|---|---|
| Zillow ZHVI | ZIP-level monthly home values, 2000–present | `Zip_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv` |
| Census TIGER/Line | ZCTA boundary geometries | `tl_2024_us_zcta520.shp`, ~800 MB raw |
| FRED | Mortgage rates, HPI, macro indicators | Feature engineering and B-1/B-2 |
| Realtor | Sales volume | Input to Part B-2 |
| Census Building Permits, FHFA HPI, BLS, USITC | Supply-chain signals | Used in Module 2 construction model |

## Running Locally

**Prerequisites:** Python 3.10+, Git.

```bash
# 1. Clone
git clone https://github.com/XAPP-P/housing-demand-forecast.git
cd housing-demand-forecast

# 2. Create virtual environment and install deps
python -m venv .venv
source .venv/bin/activate                  # macOS / Linux
# .\.venv\Scripts\Activate.ps1             # Windows PowerShell
pip install -r requirements.txt

# 3. Place raw data in data_raw/ (not in git)
#    - Zip_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv
#    - tl_2024_us_zcta520.shp (and .dbf, .shx, .prj, .cpg)

# 4. Run preprocessing (one-time)
python data_pipeline/prepare_zillow.py
python data_pipeline/prepare_zcta.py

# 5. Launch dashboard
streamlit run app.py
```

## Limitations

- ZHVI reflects Zillow's model of typical home values, not actual transaction prices.
- ZCTAs do not exactly correspond to USPS ZIP codes; some PO-box ZIPs have no geometry.
- School-district ratings (where included) are point-in-time snapshots and may lag current conditions.
- The supply-chain decision tool is scenario-based, not a causal counterfactual engine.

## Team

Group 9, INDENG 243 (Spring 2026). Coordinated model integration and dashboard delivery.

## License

See `LICENSE`.
