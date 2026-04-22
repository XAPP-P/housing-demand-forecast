"""
Tab 2: Supply-chain decision tool.

Originally built by Jiajun as a standalone Streamlit app
(https://github.com/yizhouzheng358/Supply-Side-Dashboard). Integrated
here as a function invoked by the main app's tab system.

Integration changes from the standalone version
-----------------------------------------------
- Removed st.set_page_config: the main app already calls it, and
  Streamlit forbids calling it twice.
- Moved the date-range filter from st.sidebar into the tab body, so
  it does not leak into the Price Map and About tabs (Streamlit's
  sidebar is global).
- Wrapped the full body in render_supply_chain() so app.py can invoke it.
- Broke the four sub-tab bodies out into helper functions for readability.
- Anchored the parquet path to the project's data/ directory and wrapped
  the load in st.cache_data so re-tabbing doesn't re-read from disk.

Data source
-----------
data/residential_panel.parquet — monthly panel built by the Module 2
construction base model (FRED + BLS + Census NRC + USITC tariff data).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
DATA_FILE = DATA_DIR / "residential_panel.parquet"


# ---------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def _load_panel() -> pd.DataFrame:
    df = pd.read_parquet(DATA_FILE)
    df.index = pd.to_datetime(df.index)
    return df


def _latest_value(df: pd.DataFrame, col: str):
    if col in df.columns and df[col].dropna().shape[0] > 0:
        return df[col].dropna().iloc[-1]
    return np.nan


# ---------------------------------------------------------------------
# Sub-tab renderers
# ---------------------------------------------------------------------

def _render_overview(df: pd.DataFrame) -> None:
    st.subheader("Data Overview")
    col1, col2, col3, col4 = st.columns(4)

    latest_permit = _latest_value(df, "PERMIT1")
    latest_starts = _latest_value(df, "HOUST1F")
    latest_completion = _latest_value(df, "COMPU1USA")
    latest_mortgage = _latest_value(df, "MORTGAGE30US")

    col1.metric(
        "Latest Single-Family Permits",
        f"{latest_permit:,.1f}" if pd.notna(latest_permit) else "NA",
    )
    col2.metric(
        "Latest Single-Family Starts",
        f"{latest_starts:,.1f}" if pd.notna(latest_starts) else "NA",
    )
    col3.metric(
        "Latest Single-Family Completions",
        f"{latest_completion:,.1f}" if pd.notna(latest_completion) else "NA",
    )
    col4.metric(
        "Latest Mortgage Rate",
        f"{latest_mortgage:.2f}%" if pd.notna(latest_mortgage) else "NA",
    )

    st.markdown("### Recent Data")
    st.dataframe(df.tail(12), width="stretch")


def _render_pipeline(df: pd.DataFrame) -> None:
    st.subheader("Residential Construction Pipeline")

    pipeline_cols = [c for c in ["PERMIT1", "HOUST1F", "COMPU1USA"]
                     if c in df.columns]
    label_map = {
        "PERMIT1": "Single-Family Permits",
        "HOUST1F": "Single-Family Starts",
        "COMPU1USA": "Single-Family Completions",
    }

    if len(pipeline_cols) > 0:
        plot_df = df[pipeline_cols].reset_index().rename(columns=label_map)
        friendly_cols = [label_map[c] for c in pipeline_cols]

        fig = px.line(
            plot_df,
            x="date",
            y=friendly_cols,
            title="Permits, Starts, and Completions Over Time",
        )
        fig.update_layout(
            legend_title_text="Series",
            xaxis_title="Date",
            yaxis_title="Units (SAAR)",
        )
        st.plotly_chart(fig, config={"responsive": True})
    else:
        st.warning("Could not find pipeline columns in the dataset.")

    st.markdown("### Interpretation")
    st.write(
        "This chart shows the residential construction pipeline from "
        "permits to starts to completions. Permits act as an early signal, "
        "starts reflect construction activity, and completions show "
        "finished housing supply."
    )


def _render_cost_pressures(df: pd.DataFrame) -> None:
    st.subheader("Construction Cost Pressures")

    candidates = [
        "WPU081",
        "WPU101",
        "lumber_canada_tariff_pct",
        "steel_section232_tariff_pct",
        "signal_tariff_cost_score",
    ]
    label_map = {
        "WPU081": "Lumber Prices",
        "WPU101": "Steel Prices",
        "lumber_canada_tariff_pct": "Lumber Tariff (%)",
        "steel_section232_tariff_pct": "Steel Tariff (%)",
        "signal_tariff_cost_score": "Tariff Cost Pressure Index",
    }
    material_cols = [c for c in candidates if c in df.columns]

    if len(material_cols) > 0:
        plot_df = df[material_cols].reset_index().rename(columns=label_map)
        friendly_cols = [label_map[c] for c in material_cols]

        fig = px.line(
            plot_df,
            x="date",
            y=friendly_cols,
            title="Material Cost Trends and Tariff Shocks",
        )
        fig.update_layout(legend_title_text="Series")
        st.plotly_chart(fig, config={"responsive": True})
    else:
        st.warning("Material and tariff data not available.")

    st.markdown("### Interpretation")
    st.write(
        "Higher material prices and tariff shocks can raise construction "
        "costs, which may delay or reduce housing starts over time."
    )


def _render_rates_market(df: pd.DataFrame) -> None:
    st.subheader("Interest Rates & Housing Activity")

    if "PERMIT1" in df.columns and "MORTGAGE30US" in df.columns:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df["PERMIT1"],
            name="Single-Family Permits",
        ))
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df["MORTGAGE30US"],
            name="Mortgage Rate",
            yaxis="y2",
        ))
        fig.update_layout(
            title="Mortgage Rates vs Housing Permits",
            xaxis_title="Date",
            yaxis=dict(title="Permits"),
            yaxis2=dict(title="Mortgage Rate (%)", overlaying="y", side="right"),
            legend=dict(orientation="h"),
        )
        st.plotly_chart(fig, config={"responsive": True})
    else:
        st.warning("Need PERMIT1 and MORTGAGE30US columns for this chart.")

    st.markdown("### Correlation Analysis")

    corr_cols = [c for c in [
        "PERMIT1", "HOUST1F", "COMPU1USA",
        "MORTGAGE30US", "WPU081", "WPU101",
    ] if c in df.columns]

    if len(corr_cols) >= 2:
        corr = df[corr_cols].corr(numeric_only=True)
        fig_corr = px.imshow(
            corr,
            text_auto=True,
            aspect="auto",
            title="Correlation Matrix",
        )
        st.plotly_chart(fig_corr, config={"responsive": True})
    else:
        st.warning("Not enough variables available for the correlation matrix.")


# ---------------------------------------------------------------------
# Public entry point (called by app.py)
# ---------------------------------------------------------------------

def render_supply_chain() -> None:
    st.header("🏗️ Supply-Side Analysis")
    st.caption(
        "Residential construction indicators — permits, starts, completions, "
        "material costs, interest rates, and tariffs."
    )

    if not DATA_FILE.exists():
        st.error(f"File not found: `{DATA_FILE.relative_to(REPO_ROOT)}`")
        st.info(
            "Generate this file from the Module 2 construction model notebook "
            "using: `panel_df.to_parquet('data/residential_panel.parquet')`"
        )
        return

    df = _load_panel()

    # Filter in the tab body (not sidebar — sidebar would leak to other tabs).
    min_date = df.index.min().date()
    max_date = df.index.max().date()
    date_range = st.date_input(
        "Date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
        key="sc_date_range",
    )

    if isinstance(date_range, tuple) and len(date_range) == 2:
        start, end = date_range
        filtered_df = df.loc[
            (df.index.date >= start) & (df.index.date <= end)
        ].copy()
    else:
        filtered_df = df.copy()

    sub_tab1, sub_tab2, sub_tab3, sub_tab4 = st.tabs([
        "Data Overview",
        "Pipeline",
        "Cost Pressures",
        "Interest Rates & Market",
    ])
    with sub_tab1:
        _render_overview(filtered_df)
    with sub_tab2:
        _render_pipeline(filtered_df)
    with sub_tab3:
        _render_cost_pressures(filtered_df)
    with sub_tab4:
        _render_rates_market(filtered_df)
