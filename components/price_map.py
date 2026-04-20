"""
Tab 1: ZIP-code-level housing price map.

This is the scaffold — the full interactive map (state selector, time slider,
play button, school-district overlay) will be built incrementally. This
placeholder verifies the data pipeline produced what we need and lets the
app boot end-to-end.
"""

from pathlib import Path

import streamlit as st

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"


def render_price_map() -> None:
    st.header("🗺️ ZIP-Code-Level Housing Prices")
    st.markdown(
        "Explore median home values by ZIP code across the United States."
    )

    prices_path = DATA_DIR / "zillow_prices_long.parquet"
    zcta_dir = DATA_DIR / "zcta_by_state"

    prices_ok = prices_path.exists()
    zcta_ok = zcta_dir.exists() and any(zcta_dir.glob("*.parquet"))

    col1, col2 = st.columns(2)
    with col1:
        if prices_ok:
            size = prices_path.stat().st_size / 1024 / 1024
            st.success(f"✓ Price data ready ({size:.1f} MB)")
        else:
            st.error("✗ Price data not found")
            st.caption("Run: `python data_pipeline/prepare_zillow.py`")

    with col2:
        if zcta_ok:
            n_states = len(list(zcta_dir.glob("*.parquet")))
            st.success(f"✓ Geometry data ready ({n_states} states)")
        else:
            st.error("✗ Geometry data not found")
            st.caption("Run: `python data_pipeline/prepare_zcta.py`")

    if prices_ok and zcta_ok:
        st.info(
            "🚧 **Map view coming in the next build.** "
            "Data pipeline is ready — we'll plug in the interactive map, "
            "time slider, and layer controls next."
        )
    else:
        st.warning(
            "Complete the preprocessing steps shown above before the map "
            "can be rendered."
        )
