"""
Tab 1: ZIP-code-level housing price map — client-side interactive version.

Previous revision re-rendered the Folium map on the Python side every time
the user moved the slider. Each frame was a full server→browser round-trip,
which made "Play" animation choppy and even manual scrubbing sluggish on
large states like CA (~2,500 ZIPs).

This revision builds a single self-contained interactive HTML page per state,
with **all monthly prices baked into the GeoJSON features** and all
interactivity (slider + dropdowns + play/pause + hover tooltips + live
summary stats) implemented in client-side JavaScript. After the first
state-level load, every subsequent interaction runs entirely in the
browser — no server round-trip, no Streamlit rerun.

Trade-offs
----------
* First load per state rebuilds the HTML from parquet and ships ~5-10 MB
  across the wire. Shown with a loading spinner; cached in-process so
  subsequent visits to the same state are instant.
* Uses Leaflet's canvas renderer (preferCanvas: true) — ~5× faster redraw
  than the default SVG renderer for states with thousands of polygons.
* Cache is per-session; on Streamlit Cloud this survives across users
  of the same running container.
"""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components


REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"

STATE_NAMES: dict[str, str] = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas",
    "CA": "California", "CO": "Colorado", "CT": "Connecticut",
    "DE": "Delaware", "DC": "District of Columbia",
    "FL": "Florida", "GA": "Georgia", "HI": "Hawaii", "ID": "Idaho",
    "IL": "Illinois", "IN": "Indiana", "IA": "Iowa", "KS": "Kansas",
    "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine", "MD": "Maryland",
    "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota",
    "MS": "Mississippi", "MO": "Missouri", "MT": "Montana",
    "NE": "Nebraska", "NV": "Nevada", "NH": "New Hampshire",
    "NJ": "New Jersey", "NM": "New Mexico", "NY": "New York",
    "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio",
    "OK": "Oklahoma", "OR": "Oregon", "PA": "Pennsylvania",
    "RI": "Rhode Island", "SC": "South Carolina", "SD": "South Dakota",
    "TN": "Tennessee", "TX": "Texas", "UT": "Utah", "VT": "Vermont",
    "VA": "Virginia", "WA": "Washington", "WV": "West Virginia",
    "WI": "Wisconsin", "WY": "Wyoming",
    "PR": "Puerto Rico", "VI": "US Virgin Islands",
}


# ---------------------------------------------------------------------
# Cached data loaders
# ---------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def list_available_states() -> list[str]:
    return [f.stem for f in sorted((DATA_DIR / "zcta_by_state").glob("*.parquet"))]


@st.cache_data(show_spinner=False)
def _load_state_geometry(state_abbr: str) -> gpd.GeoDataFrame:
    return gpd.read_parquet(DATA_DIR / "zcta_by_state" / f"{state_abbr}.parquet")


@st.cache_data(show_spinner=False)
def _load_state_prices(state_abbr: str) -> pd.DataFrame:
    zcta = _load_state_geometry(state_abbr)
    zips = set(zcta["zip_code"])
    df = pd.read_parquet(DATA_DIR / "zillow_prices_long.parquet")
    df = df[df["zip_code"].isin(zips)].copy()
    df["date"] = pd.to_datetime(df["date"])
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------
# HTML builder — heavy work, cached per state
# ---------------------------------------------------------------------

def _safe_json(obj) -> str:
    """json.dumps with HTML-injection safety (escape </script> etc.)."""
    return (
        json.dumps(obj, separators=(",", ":"))
        .replace("</", "<\\/")
        .replace("<!--", "<\\!--")
    )


@st.cache_data(show_spinner="Building interactive map for this state…")
def build_state_html(state_abbr: str) -> str:
    """Render the self-contained interactive HTML for one state.

    Bakes all monthly prices into GeoJSON feature properties. Returns an
    HTML string ready to hand to st.components.v1.html().
    """
    zcta = _load_state_geometry(state_abbr)
    prices = _load_state_prices(state_abbr)

    # Pivot to wide: zip × month → easy per-ZIP extraction
    wide = prices.pivot_table(
        index="zip_code", columns="date", values="zhvi", aggfunc="first"
    )
    months = sorted(wide.columns.tolist())
    month_keys = [pd.Timestamp(m).strftime("%Y-%m") for m in months]

    # Build GeoJSON, embedding a compact prices array aligned to month_keys.
    # Rounding to integer dollars keeps the JSON small (~40% smaller than
    # floats) without any visible loss of precision for a color scale.
    features = []
    wide_idx = set(wide.index)
    for _, row in zcta.iterrows():
        zip_code = row["zip_code"]
        if zip_code in wide_idx:
            vals = wide.loc[zip_code, months].values
            prices_arr = [None if pd.isna(v) else int(round(v)) for v in vals]
        else:
            prices_arr = [None] * len(months)

        features.append({
            "type": "Feature",
            "geometry": row["geometry"].__geo_interface__,
            "properties": {
                "zip": zip_code,
                "city": row.get("City", "") if pd.notna(row.get("City", "")) else "",
                "metro": row.get("Metro", "") if pd.notna(row.get("Metro", "")) else "",
                "county": row.get("CountyName", "") if pd.notna(row.get("CountyName", "")) else "",
                "prices": prices_arr,
            },
        })

    geojson = {"type": "FeatureCollection", "features": features}

    # Color-scale anchors: 2nd and 98th percentile across all time.
    flat = prices["zhvi"].dropna()
    vmin = int(round(float(flat.quantile(0.02))))
    vmax = int(round(float(flat.quantile(0.98))))

    years = sorted({pd.Timestamp(m).year for m in months})

    # Fit bounds (lat/lng order for Leaflet)
    b = zcta.total_bounds  # [minx, miny, maxx, maxy] == [lng_min, lat_min, lng_max, lat_max]
    bounds = [[float(b[1]), float(b[0])], [float(b[3]), float(b[2])]]

    replacements = {
        "__STATE_ABBR__": _safe_json(state_abbr),
        "__STATE_NAME__": _safe_json(STATE_NAMES.get(state_abbr, state_abbr)),
        "__MONTHS__": _safe_json(month_keys),
        "__YEARS__": _safe_json(years),
        "__GEOJSON__": _safe_json(geojson),
        "__BOUNDS__": _safe_json(bounds),
        "__VMIN__": str(vmin),
        "__VMAX__": str(vmax),
        "__LATEST_IDX__": str(len(months) - 1),
    }

    html = _HTML_TEMPLATE
    for k, v in replacements.items():
        html = html.replace(k, v)
    return html


# ---------------------------------------------------------------------
# Streamlit render
# ---------------------------------------------------------------------

def render_price_map() -> None:
    st.header("🗺️ ZIP-Code-Level Housing Prices")
    st.caption(
        "Pick a state. Inside the widget you can drag the time slider, "
        "jump to a specific year/month, or press **Play** to watch the "
        "market move. Hover any ZIP for details."
    )

    states = list_available_states()
    if not states:
        st.error(
            "No state parquet files found in `data/zcta_by_state/`. "
            "Run `python data_pipeline/prepare_zcta.py` first."
        )
        return

    default_idx = states.index("CA") if "CA" in states else 0
    state = st.selectbox(
        "State",
        options=states,
        index=default_idx,
        format_func=lambda s: f"{s} — {STATE_NAMES.get(s, s)}",
        key="pm_state",
    )

    html = build_state_html(state)
    components.html(html, height=1060, scrolling=False)


# ---------------------------------------------------------------------
# HTML / JavaScript template (self-contained, runs in an iframe)
# ---------------------------------------------------------------------

_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
<style>
  :root {
    --primary: #2E86AB;
    --text: #262730;
    --muted: #6b7280;
    --tile-bg: #f5f5f7;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; padding: 0;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    color: var(--text);
    font-size: 14px;
  }
  .container { display: flex; flex-direction: column; gap: 6px; padding: 2px; }

  .controls-row {
    display: flex; align-items: center; gap: 14px;
    padding: 6px 2px; flex-wrap: wrap;
  }
  .control-group { display: flex; align-items: center; gap: 6px; }
  .control-group label { font-size: 13px; color: var(--muted); }
  button, select {
    font-size: 14px; padding: 6px 12px;
    border: 1px solid #d1d5db; background: white; border-radius: 4px;
    cursor: pointer; font-family: inherit;
  }
  button { min-width: 96px; font-weight: 500; }
  button.playing { background: var(--primary); color: white; border-color: var(--primary); }
  select { cursor: pointer; }
  .month-display {
    font-size: 20px; font-weight: 600;
    color: var(--primary); margin-left: auto;
    font-variant-numeric: tabular-nums;
  }

  .slider-wrap {
    position: relative;
    padding: 8px 0 2px 0;
  }
  #time-slider {
    width: 100%;
    -webkit-appearance: none;
    appearance: none;
    height: 6px;
    background: #e5e7eb;
    border-radius: 3px;
    outline: none;
    cursor: pointer;
  }
  #time-slider::-webkit-slider-thumb {
    -webkit-appearance: none;
    appearance: none;
    width: 18px; height: 18px;
    border-radius: 50%;
    background: var(--primary);
    cursor: pointer;
    box-shadow: 0 1px 3px rgba(0,0,0,0.2);
  }
  #time-slider::-moz-range-thumb {
    width: 18px; height: 18px;
    border-radius: 50%;
    background: var(--primary);
    cursor: pointer;
    border: none;
  }
  .slider-tooltip {
    position: absolute;
    bottom: 28px;
    display: none;
    background: #1f2937; color: white;
    padding: 4px 10px;
    border-radius: 4px;
    font-size: 12px;
    pointer-events: none;
    white-space: nowrap;
    font-variant-numeric: tabular-nums;
  }
  .slider-tooltip::after {
    content: '';
    position: absolute;
    left: 50%; bottom: -4px;
    transform: translateX(-50%) rotate(45deg);
    width: 6px; height: 6px;
    background: #1f2937;
  }

  .legend-row {
    display: flex; align-items: center; gap: 10px;
    font-size: 12px; color: var(--muted);
    padding: 4px 0;
    font-variant-numeric: tabular-nums;
  }
  .legend-gradient {
    flex-grow: 1; height: 10px;
    border-radius: 3px;
    background: linear-gradient(to right,
      #ffffcc 0%, #ffeda0 25%, #fd8d3c 50%, #e31a1c 75%, #800026 100%);
  }

  #map { height: 440px; border-radius: 4px; background: #eee; }

  .feature-tooltip {
    position: absolute;
    display: none;
    background: white;
    padding: 8px 12px;
    border-radius: 4px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.18);
    font-size: 12px;
    pointer-events: none;
    z-index: 1000;
    white-space: nowrap;
    line-height: 1.5;
  }
  .feature-tooltip div { margin: 1px 0; }
  .feature-tooltip .k { color: var(--muted); display: inline-block; min-width: 52px; }

  .stats-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 10px;
    padding: 8px 0 0 0;
  }
  .stat-tile {
    background: var(--tile-bg);
    padding: 10px 14px;
    border-radius: 6px;
  }
  .stat-tile .label { font-size: 12px; color: var(--muted); margin-bottom: 2px; }
  .stat-tile .value {
    font-size: 22px; font-weight: 600; color: var(--text);
    font-variant-numeric: tabular-nums;
  }

  /* --- Price history chart section --- */
  .chart-section {
    padding: 14px 0 4px 0;
    margin-top: 10px;
    border-top: 1px solid #e5e7eb;
  }
  .chart-header {
    display: flex;
    align-items: center;
    gap: 12px;
    flex-wrap: wrap;
    margin-bottom: 6px;
  }
  .chart-header h3 {
    margin: 0;
    font-size: 16px;
    font-weight: 600;
  }
  .chart-controls {
    display: flex;
    align-items: center;
    gap: 6px;
    margin-left: auto;
  }
  .chart-controls label { font-size: 13px; color: var(--muted); }
  #zip-input {
    width: 96px;
    font-family: inherit;
    font-size: 14px;
    padding: 6px 10px;
    border: 1px solid #d1d5db;
    border-radius: 4px;
    font-variant-numeric: tabular-nums;
    outline: none;
    letter-spacing: 0.5px;
  }
  #zip-input:focus { border-color: var(--primary); }
  #zip-input.error { border-color: #dc2626; background: #fef2f2; }
  #zip-clear {
    min-width: 60px;
    font-size: 13px;
    padding: 5px 10px;
    font-weight: normal;
  }
  .chart-error {
    color: #dc2626;
    font-size: 13px;
    padding: 10px 0;
    text-align: center;
  }
  .chart-hint {
    color: var(--muted);
    font-size: 13px;
    padding: 34px 0;
    text-align: center;
    font-style: italic;
  }
  .chart-wrap {
    display: none;
    position: relative;
    height: 230px;
  }
  .chart-caption {
    font-size: 12px;
    color: var(--muted);
    padding-top: 4px;
    text-align: center;
    min-height: 16px;
  }
</style>
</head>
<body>
<div class="container">
  <div class="controls-row">
    <button id="play-btn">▶ Play</button>
    <div class="control-group">
      <label>Jump to</label>
      <select id="year-select"></select>
      <select id="month-select"></select>
    </div>
    <div class="month-display" id="current-month">—</div>
  </div>

  <div class="slider-wrap">
    <input type="range" id="time-slider" min="0" max="__LATEST_IDX__" value="__LATEST_IDX__" step="1">
    <div class="slider-tooltip" id="slider-tooltip"></div>
  </div>

  <div class="legend-row">
    <span id="legend-min">—</span>
    <div class="legend-gradient" title="Median Home Value — color scale pinned to 2nd–98th percentile across this state's entire history"></div>
    <span id="legend-max">—</span>
  </div>

  <div id="map"></div>

  <div class="stats-row">
    <div class="stat-tile"><div class="label">ZIPs with data</div><div class="value" id="stat-count">—</div></div>
    <div class="stat-tile"><div class="label">Median</div><div class="value" id="stat-median">—</div></div>
    <div class="stat-tile"><div class="label">25th percentile</div><div class="value" id="stat-p25">—</div></div>
    <div class="stat-tile"><div class="label">75th percentile</div><div class="value" id="stat-p75">—</div></div>
  </div>

  <div class="chart-section">
    <div class="chart-header">
      <h3>📈 Price History for a Single ZIP</h3>
      <div class="chart-controls">
        <label for="zip-input">ZIP</label>
        <input type="text" id="zip-input" placeholder="5-digit" maxlength="5" inputmode="numeric" autocomplete="off" />
        <button id="zip-clear">Clear</button>
      </div>
    </div>
    <div class="chart-error" id="chart-error"></div>
    <div class="chart-hint" id="chart-hint">
      Click any ZIP on the map to load its price history, or type a 5-digit ZIP above.
    </div>
    <div class="chart-wrap" id="chart-wrap">
      <canvas id="price-chart"></canvas>
    </div>
    <div class="chart-caption" id="chart-caption"></div>
  </div>
</div>

<div class="feature-tooltip" id="feature-tooltip"></div>

<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.js"></script>
<script>
(function() {
  // ----- Injected data -----
  const STATE_ABBR = __STATE_ABBR__;
  const STATE_NAME = __STATE_NAME__;
  const MONTHS     = __MONTHS__;
  const YEARS      = __YEARS__;
  const GEOJSON    = __GEOJSON__;
  const BOUNDS     = __BOUNDS__;
  const VMIN       = __VMIN__;
  const VMAX       = __VMAX__;

  const PLAY_INTERVAL_MS = 180;
  const MONTH_NAMES = ['Jan','Feb','Mar','Apr','May','Jun',
                       'Jul','Aug','Sep','Oct','Nov','Dec'];

  // ----- DOM refs -----
  const playBtn         = document.getElementById('play-btn');
  const yearSelect      = document.getElementById('year-select');
  const monthSelect     = document.getElementById('month-select');
  const currentMonthEl  = document.getElementById('current-month');
  const slider          = document.getElementById('time-slider');
  const sliderTooltip   = document.getElementById('slider-tooltip');
  const featureTooltip  = document.getElementById('feature-tooltip');
  const legendMinEl     = document.getElementById('legend-min');
  const legendMaxEl     = document.getElementById('legend-max');
  const statCount       = document.getElementById('stat-count');
  const statMedian      = document.getElementById('stat-median');
  const statP25         = document.getElementById('stat-p25');
  const statP75         = document.getElementById('stat-p75');
  const zipInput        = document.getElementById('zip-input');
  const zipClear        = document.getElementById('zip-clear');
  const chartError      = document.getElementById('chart-error');
  const chartHint       = document.getElementById('chart-hint');
  const chartWrap       = document.getElementById('chart-wrap');
  const chartCaption    = document.getElementById('chart-caption');
  const chartCanvas     = document.getElementById('price-chart');

  // ----- State -----
  let currentIdx = MONTHS.length - 1;
  let isPlaying = false;
  let playTimer = null;
  let hoveredFeature = null;   // tracks which feature the cursor is over, so
                               // tooltip content can refresh on each play tick
  let chart = null;             // Chart.js instance, null when no ZIP selected
  let currentZip = null;        // ZIP currently charted (or null)

  // ----- Formatting -----
  function fmtPrice(v) {
    return (v === null || v === undefined || isNaN(v))
      ? '—'
      : '$' + Math.round(v).toLocaleString('en-US');
  }
  function fmtMonthYear(idx) {
    const [y, m] = MONTHS[idx].split('-').map(Number);
    return MONTH_NAMES[m-1] + ' ' + y;
  }

  // ----- Color scale (5-stop linear, YlOrRd) -----
  const STOPS = [
    [0.00, [255, 255, 204]],
    [0.25, [255, 237, 160]],
    [0.50, [253, 141,  60]],
    [0.75, [227,  26,  28]],
    [1.00, [128,   0,  38]],
  ];
  function colorFor(value) {
    if (value === null || value === undefined || isNaN(value)) return '#e5e5e5';
    let t = (value - VMIN) / (VMAX - VMIN);
    if (t < 0) t = 0;
    if (t > 1) t = 1;
    for (let i = 0; i < STOPS.length - 1; i++) {
      if (t <= STOPS[i+1][0]) {
        const t0 = STOPS[i][0], t1 = STOPS[i+1][0];
        const c0 = STOPS[i][1], c1 = STOPS[i+1][1];
        const f = (t1 === t0) ? 0 : (t - t0) / (t1 - t0);
        const r = Math.round(c0[0] + f * (c1[0] - c0[0]));
        const g = Math.round(c0[1] + f * (c1[1] - c0[1]));
        const b = Math.round(c0[2] + f * (c1[2] - c0[2]));
        return 'rgb(' + r + ',' + g + ',' + b + ')';
      }
    }
    return 'rgb(128,0,38)';
  }

  // ----- Map -----
  const map = L.map('map', {
    preferCanvas: true,  // ~5× faster with many polygons
    zoomControl: true,
    attributionControl: true,
  });
  L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png', {
    attribution: '© OpenStreetMap · © CARTO',
    subdomains: 'abcd',
    maxZoom: 19,
  }).addTo(map);
  map.fitBounds(BOUNDS);

  function styleFeature(feature) {
    const price = feature.properties.prices[currentIdx];
    return {
      fillColor: colorFor(price),
      weight: 0.3,
      color: 'white',
      fillOpacity: (price === null || price === undefined) ? 0.35 : 0.78,
    };
  }

  const geojsonLayer = L.geoJSON(GEOJSON, {
    style: styleFeature,
    onEachFeature: function(feature, layer) {
      layer.on('mouseover', function(e) { showFeatureTooltip(feature, e); });
      layer.on('mousemove', function(e) { positionFeatureTooltip(e); });
      layer.on('mouseout',  function()  { hideFeatureTooltip(); });
      layer.on('click',     function()  { drawZipChart(feature.properties.zip); });
    }
  }).addTo(map);

  // ----- Feature tooltip -----
  function showFeatureTooltip(feature, e) {
    hoveredFeature = feature;
    renderFeatureTooltip();
    featureTooltip.style.display = 'block';
    positionFeatureTooltip(e);
  }
  function renderFeatureTooltip() {
    // Render (or re-render) tooltip HTML using the live currentIdx. This
    // is called both on mouseover and on every updateMap() so that during
    // play the hovered ZIP's Value updates along with everything else.
    if (!hoveredFeature) return;
    const p = hoveredFeature.properties;
    const price = p.prices[currentIdx];
    featureTooltip.innerHTML =
      '<div><span class="k">ZIP</span> '    + p.zip    + '</div>' +
      '<div><span class="k">City</span> '   + (p.city   || '—') + '</div>' +
      '<div><span class="k">Metro</span> '  + (p.metro  || '—') + '</div>' +
      '<div><span class="k">County</span> ' + (p.county || '—') + '</div>' +
      '<div><span class="k">Value</span> '  + fmtPrice(price)   + '</div>';
  }
  function positionFeatureTooltip(e) {
    // pageX/pageY account for scroll within the iframe
    featureTooltip.style.left = (e.originalEvent.pageX + 14) + 'px';
    featureTooltip.style.top  = (e.originalEvent.pageY + 10) + 'px';
  }
  function hideFeatureTooltip() {
    hoveredFeature = null;
    featureTooltip.style.display = 'none';
  }

  // ----- Summary stats -----
  function updateStats() {
    const vals = [];
    for (let i = 0; i < GEOJSON.features.length; i++) {
      const v = GEOJSON.features[i].properties.prices[currentIdx];
      if (v !== null && v !== undefined && !isNaN(v)) vals.push(v);
    }
    if (vals.length === 0) {
      statCount.textContent  = '0';
      statMedian.textContent = '—';
      statP25.textContent    = '—';
      statP75.textContent    = '—';
      return;
    }
    vals.sort(function(a, b) { return a - b; });
    function pct(p) {
      const i = Math.min(vals.length - 1, Math.floor(p * vals.length));
      return vals[i];
    }
    statCount.textContent  = vals.length.toLocaleString('en-US');
    statMedian.textContent = fmtPrice(pct(0.5));
    statP25.textContent    = fmtPrice(pct(0.25));
    statP75.textContent    = fmtPrice(pct(0.75));
  }

  // ----- Main: repaint for a new month index -----
  function updateMap(newIdx) {
    currentIdx = newIdx;
    geojsonLayer.setStyle(styleFeature);
    currentMonthEl.textContent = fmtMonthYear(currentIdx);
    slider.value = String(currentIdx);
    const parts = MONTHS[currentIdx].split('-');
    yearSelect.value  = parts[0];
    monthSelect.value = String(parseInt(parts[1], 10));
    updateStats();
    renderFeatureTooltip();   // keep hover tooltip's Value in sync with time
    refreshChartMarker();     // move the red dot along the line chart
  }

  // ----- Populate dropdowns -----
  for (let i = 0; i < YEARS.length; i++) {
    const opt = document.createElement('option');
    opt.value = YEARS[i]; opt.textContent = YEARS[i];
    yearSelect.appendChild(opt);
  }
  for (let i = 0; i < MONTH_NAMES.length; i++) {
    const opt = document.createElement('option');
    opt.value = i + 1; opt.textContent = MONTH_NAMES[i];
    monthSelect.appendChild(opt);
  }

  function jumpToYearMonth(year, month) {
    const key = year + '-' + String(month).padStart(2, '0');
    const idx = MONTHS.indexOf(key);
    if (idx >= 0) {
      stopPlaying();
      updateMap(idx);
    } else {
      // Requested month not in data (edge months). Find nearest.
      let best = 0, bestDiff = Infinity;
      for (let i = 0; i < MONTHS.length; i++) {
        const [y, m] = MONTHS[i].split('-').map(Number);
        const diff = Math.abs((y * 12 + m) - (year * 12 + month));
        if (diff < bestDiff) { bestDiff = diff; best = i; }
      }
      stopPlaying();
      updateMap(best);
    }
  }
  yearSelect.addEventListener('change', function() {
    jumpToYearMonth(parseInt(yearSelect.value, 10), parseInt(monthSelect.value, 10));
  });
  monthSelect.addEventListener('change', function() {
    jumpToYearMonth(parseInt(yearSelect.value, 10), parseInt(monthSelect.value, 10));
  });

  // ----- Slider: drag + hover tooltip -----
  slider.addEventListener('input', function() {
    stopPlaying();
    updateMap(parseInt(slider.value, 10));
  });
  slider.addEventListener('mousemove', function(e) {
    const rect = slider.getBoundingClientRect();
    const x = e.clientX - rect.left;
    let ratio = x / rect.width;
    if (ratio < 0) ratio = 0; if (ratio > 1) ratio = 1;
    const idx = Math.round(ratio * (MONTHS.length - 1));
    sliderTooltip.textContent = fmtMonthYear(idx);
    sliderTooltip.style.display = 'block';
    // Position tooltip centered on cursor (approx)
    sliderTooltip.style.left = (x - 36) + 'px';
  });
  slider.addEventListener('mouseleave', function() {
    sliderTooltip.style.display = 'none';
  });

  // ----- Play / pause -----
  function startPlaying() {
    isPlaying = true;
    playBtn.textContent = '⏸ Pause';
    playBtn.classList.add('playing');
    playTimer = setInterval(function() {
      let next = currentIdx + 1;
      if (next >= MONTHS.length) next = 0;
      updateMap(next);
    }, PLAY_INTERVAL_MS);
  }
  function stopPlaying() {
    if (!isPlaying) return;
    isPlaying = false;
    playBtn.textContent = '▶ Play';
    playBtn.classList.remove('playing');
    if (playTimer) clearInterval(playTimer);
    playTimer = null;
  }
  playBtn.addEventListener('click', function() {
    if (isPlaying) stopPlaying(); else startPlaying();
  });

  // ----- Initial render -----
  legendMinEl.textContent = fmtPrice(VMIN);
  legendMaxEl.textContent = fmtPrice(VMAX);
  updateMap(MONTHS.length - 1);

  // ===================================================================
  // Price-history chart (bottom section)
  // ===================================================================

  function findFeatureByZip(zip) {
    const z = String(zip).padStart(5, '0');
    for (let i = 0; i < GEOJSON.features.length; i++) {
      if (GEOJSON.features[i].properties.zip === z) return GEOJSON.features[i];
    }
    return null;
  }

  function hideChart() {
    chartWrap.style.display = 'none';
    chartHint.style.display = 'block';
    chartError.textContent = '';
    chartError.style.display = 'none';
    chartCaption.textContent = '';
    currentZip = null;
    zipInput.classList.remove('error');
    if (chart) { chart.destroy(); chart = null; }
  }

  function showChartError(msg) {
    chartError.textContent = msg;
    chartError.style.display = 'block';
    chartWrap.style.display = 'none';
    chartHint.style.display = 'none';
    chartCaption.textContent = '';
    zipInput.classList.add('error');
    currentZip = null;
    if (chart) { chart.destroy(); chart = null; }
  }

  function currentMonthMarkerData() {
    // Array of nulls except at currentIdx — for the red-dot "current month"
    // marker dataset. Updated on every updateMap() call.
    const arr = new Array(MONTHS.length).fill(null);
    if (currentZip) {
      const feature = findFeatureByZip(currentZip);
      if (feature) {
        const v = feature.properties.prices[currentIdx];
        if (v !== null && v !== undefined) arr[currentIdx] = v;
      }
    }
    return arr;
  }

  function refreshChartMarker() {
    if (chart && currentZip) {
      chart.data.datasets[1].data = currentMonthMarkerData();
      chart.update('none');
    }
  }

  function drawZipChart(zip) {
    const z = String(zip).padStart(5, '0');
    if (!/^\d{5}$/.test(z)) {
      showChartError('Please enter a 5-digit ZIP code.');
      return;
    }
    const feature = findFeatureByZip(z);
    if (!feature) {
      showChartError('ZIP ' + z + ' isn\'t in ' + STATE_NAME + '. Click a polygon on the map, or switch state.');
      return;
    }

    // Good to go — hide hint/error, show chart container
    chartHint.style.display = 'none';
    chartError.style.display = 'none';
    chartWrap.style.display = 'block';
    zipInput.classList.remove('error');
    zipInput.value = z;
    currentZip = z;

    const p = feature.properties;
    const captionParts = [p.city, p.county ? p.county : null, STATE_ABBR]
      .filter(function(s) { return s; });
    chartCaption.textContent = captionParts.join(', ');

    if (chart) chart.destroy();

    chart = new Chart(chartCanvas.getContext('2d'), {
      type: 'line',
      data: {
        labels: MONTHS,
        datasets: [
          {
            label: 'ZIP ' + z,
            data: p.prices,
            borderColor: '#2E86AB',
            backgroundColor: 'rgba(46,134,171,0.08)',
            borderWidth: 2,
            pointRadius: 0,
            pointHoverRadius: 4,
            tension: 0.15,
            fill: true,
            spanGaps: false,
          },
          {
            label: 'Current month',
            data: currentMonthMarkerData(),
            pointRadius: 6,
            pointBackgroundColor: '#e31a1c',
            pointBorderColor: 'white',
            pointBorderWidth: 2,
            showLine: false,
            pointHoverRadius: 6,
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: 'index', intersect: false },
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              title: function(items) {
                if (!items.length) return '';
                const parts = items[0].label.split('-').map(Number);
                return MONTH_NAMES[parts[1]-1] + ' ' + parts[0];
              },
              label: function(ctx) {
                if (ctx.parsed.y === null || ctx.parsed.y === undefined) return null;
                if (ctx.datasetIndex === 1) return null;  // hide duplicate for marker
                return '$' + Math.round(ctx.parsed.y).toLocaleString('en-US');
              }
            }
          }
        },
        scales: {
          x: {
            ticks: {
              maxTicksLimit: 10,
              callback: function(val) {
                const label = this.getLabelForValue(val);
                return label ? label.split('-')[0] : '';
              },
              font: { size: 11 }
            },
            grid: { display: false }
          },
          y: {
            ticks: {
              callback: function(v) { return '$' + Math.round(v/1000) + 'k'; },
              font: { size: 11 }
            },
            grid: { color: '#f0f0f0' }
          }
        }
      }
    });
  }

  // ----- Input box handlers -----
  zipInput.addEventListener('keydown', function(e) {
    if (e.key === 'Enter') { e.preventDefault(); zipInput.blur(); }
  });
  zipInput.addEventListener('blur', function() {
    const v = zipInput.value.trim();
    if (!v) { hideChart(); return; }
    drawZipChart(v);
  });
  zipClear.addEventListener('click', function() {
    zipInput.value = '';
    hideChart();
  });

  // ----- Default: 94704 if present (Berkeley core), else blank -----
  if (findFeatureByZip('94704')) {
    drawZipChart('94704');
  }
})();
</script>
</body>
</html>
"""
