"""
US Housing Demand Forecast Dashboard
INDENG 243 — Group 9 — Spring 2026
"""

import streamlit as st

# ---------------------------------------------------------------------
# Page configuration — must be the first Streamlit call
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="US Housing Demand Forecast",
    page_icon="🏘️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------
st.title("🏘️ US Housing Demand Forecast")
st.caption("UC Berkeley INDENG 243 · Group 9 · Spring 2026")

# ---------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------
tab_price, tab_supply, tab_about = st.tabs(
    ["🗺️  Price Map", "🏗️  Supply Chain", "ℹ️  About"]
)

with tab_price:
    from components.price_map import render_price_map
    render_price_map()

with tab_supply:
    from components.supply_chain import render_supply_chain
    render_supply_chain()

with tab_about:
    st.header("About this dashboard")
    st.markdown(
        """
        This dashboard translates the housing-demand forecasting models built in
        INDENG 243 Modules 1–2 into a decision-support tool for two audiences:

        - **Buyers and sellers** use the *Price Map* tab to see how ZIP-code-level
          home values have moved, compare regions, and (optionally) weigh school
          quality alongside price.
        - **Supply-chain actors** — builders, material suppliers, contractors —
          use the *Supply Chain* tab to explore how rates, tariffs, and material
          costs interact with the permit → start → completion pipeline.

        **Model architecture.** A two-layer parallel pipeline: a macro Ridge
        baseline (Part A) produces division-level HPI forecasts that feed two
        parallel downstream Random Forest models — ZIP-level prices (B-1) and
        division-level sales volume (B-2).

        **Data sources.** Zillow ZHVI, Census TIGER/Line ZCTA shapefiles, FRED
        (mortgage rates, HPI, macro series), Realtor sales data, Census building
        permits, BLS employment, USITC tariff data.

        **Source code:** https://github.com/XAPP-P/housing-demand-forecast
        """
    )
