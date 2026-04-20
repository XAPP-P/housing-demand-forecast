"""
Tab 2: Supply-chain decision tool.

Target audience: builders, material suppliers, contractors.
Planned features (iteration order TBD):
  * Scenario controls — mortgage rate, tariff regime, material cost shock.
  * Outputs — expected permit level, pipeline timing (permits → starts →
    completions lag), material-cost pressure index.
  * Feature importance panel from the Module 2 construction base model.
"""

import streamlit as st


def render_supply_chain() -> None:
    st.header("🏗️ Supply-Chain Decision Tool")
    st.markdown(
        "Scenario exploration for builders, material suppliers, and "
        "contractors."
    )
    st.info(
        "🚧 **Under construction.** "
        "This tab will surface the construction pipeline (permits → starts "
        "→ completions), tariff-regime scenarios, and material-cost "
        "pressure signals from the Module 2 construction base model."
    )
